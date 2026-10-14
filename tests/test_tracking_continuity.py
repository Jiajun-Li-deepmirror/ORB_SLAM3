"""Regression cases for stale visual references and all failure routes."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from splg_slam.map.frame import Frame
from splg_slam.mapping.tracker import OfflineMapper


def pose(x=0., angle=0.):
    p=np.eye(4);p[:3,:3]=Rotation.from_euler('z',angle,degrees=True).as_matrix()
    p[:3,3]=-p[:3,:3]@np.array([x,0.,0.]);return p


def frame(i,t,x=0.):
    return Frame(i,int(t*1e9),np.zeros((40,2)),np.zeros((40,256)),np.ones(40),pose(x))


def mapper():
    m=OfflineMapper.__new__(OfflineMapper)
    m.cfg=NS(tracking=NS(pose_jump_min_floor_m=.5,reinit_after_lost_frames=3))
    m.ref_keyframe=frame(1646,1,0);m.ref_feats={};m.last_frame=frame(1679,2.65,2.56)
    m._pose_jump_m_fixed=False;m._effective_max_speed_mps=1.35;m.max_pose_jump_deg=90.
    m._effective_max_pose_jump_m=None;m._pose_history=[pose(2.50),pose(2.56)]
    m._consecutive_lost=0;m._frame_diagnostics={};m._tracking_route='reference'
    m._next_frame_id=1680;m.imu_tight_fusion=False;m.mono_mode=False
    m.track_stats=[];m._speed_window=[];m._speed_last_printed=None
    m.splg=NS(match=Mock(return_value={'matches':np.zeros((40,2),int)}))
    m._need_new_keyframe=Mock(return_value=(False,''));m._try_relocalize=Mock(return_value=None)
    m._start_new_map_segment=Mock(return_value=(frame(1682,2.8),True))
    m._try_motion_model_tracking=Mock(return_value=None)
    return m


def process(m,p,t):
    tracked=None if p is None else (p,list(range(40)),list(range(40)),np.ones(40,bool),0)
    m._try_track_against_ref=Mock(return_value=tracked)
    return m._process_frame_impl(np.zeros((2,2)),{},np.zeros((40,2)),np.zeros((40,256)),np.ones(40),int(t*1e9),None)


class TrackingContinuityTests(unittest.TestCase):
    def test_stale_reference_does_not_reject_small_actual_step(self):
        m=mapper();result,is_kf=process(m,pose(2.626),2.7)
        self.assertIsNotNone(result);self.assertFalse(is_kf);m._try_relocalize.assert_not_called()
        d=m._frame_diagnostics['motion_continuity']
        self.assertEqual(d['reference_frame_id'],1679);self.assertAlmostEqual(d['dt_s'],.05)
        self.assertAlmostEqual(d['translation_m'],.066);self.assertAlmostEqual(d['translation_limit_m'],.5)
        # Insertion geometry remains relative to the keyframe, not the previous frame.
        self.assertAlmostEqual(m._need_new_keyframe.call_args.args[0],2.626)
        self.assertAlmostEqual(m._speed_window[0],1.32)

    def test_actual_translation_and_rotation_jumps_remain_rejected(self):
        for p in [pose(3.56),pose(2.56,100.)]:
            with self.subTest(pose=p):
                m=mapper();self.assertEqual(process(m,p,2.7),(None,False))
                self.assertEqual(m._consecutive_lost,1)
                self.assertFalse(m._frame_diagnostics['pose_jump_rejection']['accepted'])
                self.assertEqual(m.last_frame.frame_id,1679)

    def test_repeated_rejected_pnp_reaches_atlas_recovery(self):
        m=mapper()
        for i in range(3):result=process(m,pose(10),2.7+i*.05)
        self.assertEqual(m._consecutive_lost,3);m._start_new_map_segment.assert_called_once()
        self.assertTrue(result[1]);self.assertEqual(m._try_relocalize.call_count,3)

    def test_mixed_no_pnp_and_rejected_pnp_count_once_each(self):
        m=mapper();process(m,None,2.7);process(m,pose(10),2.75);process(m,None,2.8)
        self.assertEqual(m._consecutive_lost,3);m._start_new_map_segment.assert_called_once()

    def test_success_resets_streak_only_after_acceptance(self):
        m=mapper();m._consecutive_lost=2
        process(m,pose(2.62),2.7)
        self.assertEqual(m._consecutive_lost,0);m._start_new_map_segment.assert_not_called()

    def test_relocalization_uses_current_frame_time_not_candidate_time(self):
        m=mapper();m._consecutive_lost=2;m.n_relocalizations=0
        m.world_map=NS(add_relocalization_event=Mock())
        candidate=frame(100,.1,2.5)
        m._try_relocalize=Mock(return_value=(candidate,{},40,pose(2.62)))
        process(m,None,2.7)
        self.assertEqual(m._consecutive_lost,0);self.assertEqual(m.ref_keyframe.frame_id,100)
        self.assertEqual(len(m._pose_history),1)
        d=m._motion_continuity(pose(2.68),int(2.75e9))
        self.assertEqual(d['reference_frame_id'],1680);self.assertAlmostEqual(d['dt_s'],.05)
        self.assertTrue(d['accepted'])

    def test_prediction_chain_never_becomes_accepted_temporal_reference(self):
        m=mapper();original=m.last_frame
        for i in range(2):process(m,None,2.7+i*.05)
        self.assertIs(m.last_frame,original)
        self.assertGreater(-m._pose_history[-1][0,3],2.56)
        self.assertAlmostEqual(m._motion_continuity(pose(2.7),int(2.8e9))['dt_s'],.15)

    def test_threshold_uses_actual_gap_and_honors_fixed_value(self):
        m=mapper();self.assertAlmostEqual(m._pose_jump_threshold_m(int(3.65e9)),1.35)
        m._pose_jump_m_fixed=True;m._effective_max_pose_jump_m=.12
        self.assertAlmostEqual(m._pose_jump_threshold_m(int(3.65e9)),.12)
        self.assertFalse(m._motion_continuity(pose(2.8),int(2.7e9))['accepted'])

    def test_backend_updated_pose_is_used_without_stale_history(self):
        m=mapper();m.last_frame.pose_cw=pose(12.56)
        # This is the assignment build_map performs after BA/IMU/loop correction.
        m._record_tracked_pose(m.last_frame.pose_cw,reset=True)
        self.assertTrue(m._motion_continuity(pose(12.62),int(2.7e9))['accepted'])

    def test_imu_only_recovery_does_not_reset_visual_failure_streak(self):
        m=mapper();m._consecutive_lost=2;m.cfg.tracking.continuous_imu_tracking=True
        kf=frame(1680,2.7,2.62);kf.imu_only=True
        m._insert_imu_only_keyframe=Mock(return_value=(kf,True))
        result=process(m,None,2.7)
        self.assertIs(result[0],kf);self.assertEqual(m._consecutive_lost,3)
        m._start_new_map_segment.assert_not_called()
        m.last_frame=kf
        self.assertTrue(m._motion_continuity(pose(2.68),int(2.75e9))['reference_imu_only'])

    def test_nonfinite_pnp_is_rejected_and_diagnostics_are_json_safe(self):
        import json
        m=mapper();p=pose();p[0,3]=np.nan
        self.assertEqual(process(m,p,2.7),(None,False))
        self.assertEqual(m._consecutive_lost,1);json.dumps(m._frame_diagnostics,allow_nan=False)

if __name__=='__main__':unittest.main()
