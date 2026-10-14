"""Invariants for gravity realignment and BA write-back (no model downloads)."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from splg_slam.data.euroc import ImuCalibration
from splg_slam.map.keyframe import KeyFrame
from splg_slam.map.world_map import WorldMap
from splg_slam.mapping import imu_init, local_ba
from splg_slam.mapping.imu_preintegration import make_preintegration_params, preintegrate

CALIB = ImuCalibration(2e-4, 2e-6, 8e-4, 2e-5, np.eye(4))

def frame(i):
    pose = np.eye(4)
    pose[:3, :3] = gtsam.Rot3.Ry(np.pi / 2).matrix().T  # camera forward = world +X
    return KeyFrame(frame_id=i, timestamp_ns=i*100_000_000,
                    keypoints=np.zeros((2,2)), descriptors=np.zeros((2,256)),
                    depths=np.full(2,3.), pose_cw=pose,
                    velocity=np.array([1.,2.,3.]), imu_bias=np.arange(6)*.001)

def world():
    w=WorldMap()
    for i in range(4): w.add_keyframe(frame(i))
    w.keyframes[3].segment_id=2
    w.add_imu_factor(1,2,np.zeros((2,7)))
    return w

class RealignmentTests(unittest.TestCase):
    def solve(self, w, gravity=(.1,0,-9.81), scale=1., **kw):
        with patch.object(imu_init, '_estimate_gyro_bias', return_value=np.zeros(3)), \
             patch.object(imu_init, '_estimate_gravity_bias_and_velocities',
                 return_value=(np.array(gravity), np.array([.01,.02,.03]), {0:np.ones(3),1:np.ones(3)*2}, scale)):
            return imu_init._solve_and_realign(w,[1,2],CALIB,None,9.81,apply_kf_ids=[0,1,2],**kw)

    def test_window_exterior_velocity_and_scale_follow_pose(self):
        w=world(); old=w.keyframes[0].pose_cw.copy(); v=w.keyframes[0].velocity.copy()
        bias=w.keyframes[0].imu_bias.copy(); other=w.keyframes[3].pose_cw.copy()
        self.solve(w,scale=2.,solve_scale=True)
        rotation=w.keyframes[0].pose_cw[:3,:3].T@old[:3,:3]
        np.testing.assert_allclose(w.keyframes[0].velocity, 2*rotation@v, atol=1e-12)
        np.testing.assert_allclose(w.keyframes[1].velocity,rotation@np.ones(3),atol=1e-12)
        np.testing.assert_array_equal(w.keyframes[0].imu_bias,bias)
        np.testing.assert_array_equal(w.keyframes[3].pose_cw,other)
        np.testing.assert_array_equal(w.keyframes[3].velocity,v)

    def test_sliding_window_does_not_redefine_segment_heading(self):
        w=world(); yaw=gtsam.Rot3.Rz(.7).matrix()
        for i in (1,2):w.keyframes[i].pose_cw[:3,:3] = w.keyframes[i].pose_cw[:3,:3]@yaw.T
        old=w.keyframes[0].pose_cw.copy()
        self.solve(w,gravity=(0,0,-9.81))
        np.testing.assert_allclose(w.keyframes[0].pose_cw,old,atol=1e-12)

    def test_nonfinite_estimate_never_mutates_map(self):
        w=world(); before={i:k.pose_cw.copy() for i,k in w.keyframes.items()}
        result=self.solve(w,gravity=(np.nan,0,-9.81))
        self.assertTrue(result is None or not result['accepted'])
        for i in before:np.testing.assert_array_equal(w.keyframes[i].pose_cw,before[i])

class CovarianceTests(unittest.TestCase):
    def test_default_preserves_library_weighting(self):
        actual=make_preintegration_params(CALIB,9.81,1e-4)
        default=gtsam.PreintegrationCombinedParams(np.array([0.,0.,-9.81]))
        np.testing.assert_array_equal(actual.getBiasAccOmegaInit(),default.getBiasAccOmegaInit())

    def test_invalid_covariance_is_rejected(self):
        for value in [-1.,np.nan,np.inf]:
            with self.assertRaises(ValueError):
                make_preintegration_params(CALIB,9.81,1e-4,bias_init_covariance=value)

    def test_integrated_rotation_uncertainty_matches_sensor_density(self):
        params=make_preintegration_params(CALIB,9.81,1e-4,bias_init_covariance=0.)
        samples=np.zeros((41,7));samples[:,0]=np.arange(41)*5e6;samples[:,6]=9.81
        pim=preintegrate(samples,gtsam.imuBias.ConstantBias(),params)
        variance=np.diag(pim.preintMeasCov())[:3]
        np.testing.assert_allclose(variance,np.full(3,CALIB.gyro_noise_density**2*.2),rtol=1e-4)
        self.assertGreater(np.linalg.eigvalsh(pim.preintMeasCov()).min(),0)

class BAWritebackTests(unittest.TestCase):
    def make_map(self):
        w=WorldMap()
        for i in range(2):
            k=frame(i);k.pose_cw=np.eye(4);w.add_keyframe(k)
        shared=w.new_map_point(np.array([0.,0.,3.]),np.zeros(256))
        for i in range(2):w.add_observation(shared,i,0)
        orphan=w.new_map_point(np.array([.2,.3,3.]),np.zeros(256));w.add_observation(orphan,1,1)
        return w,orphan

    def optimize(self,w,limit=10):
        def factory(graph,values,params):
            result=gtsam.Values(values)
            result.update(X(1),gtsam.Pose3(gtsam.Rot3.Ry(.1),np.array([.2,0,0])))
            class Optimizer:
                def optimize(self):return result
            return Optimizer()
        with patch.object(local_ba.gtsam,'LevenbergMarquardtOptimizer',side_effect=factory):
            return local_ba.local_bundle_adjustment(w,[0,1],np.diag([300.,300.,1.]),max_pose_shift_m=limit,outlier_reproj_threshold_px=1e6)

    def test_single_observer_point_preserves_camera_coordinates(self):
        w,pid=self.make_map();old=(w.keyframes[1].pose_cw@np.r_[w.map_points[pid].position,1])[:3]
        self.optimize(w)
        new=(w.keyframes[1].pose_cw@np.r_[w.map_points[pid].position,1])[:3]
        np.testing.assert_allclose(new,old,atol=1e-12)

    def test_visual_only_ba_transports_existing_velocity(self):
        w,pid=self.make_map();v=w.keyframes[1].velocity.copy()
        self.optimize(w)
        np.testing.assert_allclose(w.keyframes[1].velocity,w.keyframes[1].pose_cw[:3,:3].T@v,atol=1e-12)

    def test_rejected_ba_leaves_point_and_pose_unchanged(self):
        w,pid=self.make_map();p=w.map_points[pid].position.copy();pose=w.keyframes[1].pose_cw.copy()
        self.assertTrue(self.optimize(w,limit=.01)['rejected'])
        np.testing.assert_array_equal(w.map_points[pid].position,p)
        np.testing.assert_array_equal(w.keyframes[1].pose_cw,pose)

class InitializationTransactionTests(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.pid = self.w.new_map_point(np.array([0.,0.,3.]),np.zeros(256))
        self.w.add_observation(self.pid,1,0)
        self.before = self.w.keyframes[1].pose_cw.copy()
        self.bias = self.w.keyframes[1].imu_bias.copy()
        self.point = self.w.map_points[self.pid].position.copy()

    def initialize(self,*args,**kwargs):
        self.w.keyframes[1].pose_cw[0,3] += 1
        self.w.keyframes[1].velocity[:] = 7
        self.w.keyframes[1].imu_bias[:] = .2
        self.w.map_points[self.pid].position[:] = 4
        return dict(accepted=True,accel_bias=np.zeros(3),gyro_bias=np.zeros(3))

    def run_init(self,refine):
        with patch.object(imu_init,'run_dynamic_imu_init',side_effect=self.initialize), \
             patch.object(local_ba,'local_bundle_adjustment',side_effect=refine):
            return imu_init.initialize_imu_with_refinement(self.w,[1,2],CALIB,None,9.81,k_rect=np.eye(3))

    def check_restored(self):
        np.testing.assert_array_equal(self.w.keyframes[1].pose_cw,self.before)
        np.testing.assert_array_equal(self.w.keyframes[1].imu_bias,self.bias)
        np.testing.assert_array_equal(self.w.keyframes[1].velocity,[1,2,3])
        np.testing.assert_array_equal(self.w.map_points[self.pid].position,self.point)
        self.assertEqual(self.w.map_points[self.pid].observations,{1:0})
        self.assertEqual(self.w.keyframes[1].map_point_ids[0],self.pid)

    def test_disabled_refinement_preserves_legacy_schedule(self):
        with patch.object(imu_init,'run_dynamic_imu_init',side_effect=self.initialize), \
             patch.object(local_ba,'local_bundle_adjustment') as refine:
            diag=imu_init.initialize_imu_with_refinement(self.w,[1,2],CALIB,None,9.81,k_rect=np.eye(3),enabled=False)
        self.assertTrue(diag['accepted'])
        refine.assert_not_called()

    def test_rejected_refinement_rolls_back_seed(self):
        self.assertIsNone(self.run_init(lambda *a,**kw:dict(rejected=True)))
        self.check_restored()

    def test_missing_inertial_constraints_restore_pruned_observations(self):
        def refine(*a,**kw):
            self.w.remove_observation(self.pid,1)
            return dict(rejected=False,num_imu_states=0)
        self.assertIsNone(self.run_init(refine));self.check_restored()

    def test_optimizer_exception_restores_before_propagating(self):
        def refine(*a,**kw):raise RuntimeError('indeterminate graph')
        with self.assertRaises(RuntimeError):self.run_init(refine)
        self.check_restored()

    def test_accepted_diagnostics_use_refined_bias(self):
        def refine(*a,**kw):
            self.w.keyframes[2].imu_bias=np.full(6,.03)
            return dict(rejected=False,num_imu_states=2)
        diag=self.run_init(refine)
        np.testing.assert_array_equal(diag['accel_bias'],np.full(3,.03))
        np.testing.assert_array_equal(diag['proposed_accel_bias'],np.zeros(3))

if __name__=='__main__':unittest.main()
