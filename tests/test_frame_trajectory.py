import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from splg_slam.map.frame_trajectory import FrameTrajectory
from splg_slam.map.world_map import WorldMap
from splg_slam.map.keyframe import KeyFrame
from splg_slam.map.io import load_map, save_map
from splg_slam.mapping.tracker import OfflineMapper


def pose(x=0., angle=0.):
    p = np.eye(4)
    p[:3,:3] = Rotation.from_euler("z", angle).as_matrix()
    p[0,3] = x
    return p


def world():
    w = WorldMap()
    for i in range(3):
        w.add_keyframe(KeyFrame(frame_id=i, timestamp_ns=i*10**9,
                               keypoints=np.empty((0,2)), descriptors=np.empty((0,256)),
                               depths=np.empty(0), pose_cw=pose(i)))
    w.frame_trajectory = FrameTrajectory()
    return w


class TrajectoryTests(unittest.TestCase):
    def test_rigid_map_correction_follows_anchor_and_preserves_online(self):
        w=world(); f=SimpleNamespace(pose_cw=pose(.3,.1))
        w.frame_trajectory.append(w,4,4*10**9,f,False,0,"tracked")
        old=f.pose_cw.copy(); transform=pose(2.,.7)
        w.transform_keyframes(set(w.keyframes),transform)
        rec=w.frame_trajectory.records[0]
        np.testing.assert_allclose(w.frame_trajectory.resolve(w,rec)[0],old@np.linalg.inv(transform),atol=1e-12)
        np.testing.assert_array_equal(rec["pose_online_cw"],old)

    def test_repeated_culling_keeps_pose_and_tracks_surviving_anchor(self):
        w=world(); f=SimpleNamespace(pose_cw=pose(.5,.2))
        w.frame_trajectory.append(w,4,4*10**9,f,False,0,"tracked")
        w.remove_keyframe(0); w.remove_keyframe(1)
        rec=w.frame_trajectory.records[0]
        np.testing.assert_allclose(w.frame_trajectory.resolve(w,rec)[0],f.pose_cw,atol=1e-12)
        transform=pose(.7,.3);w.transform_keyframes({2},transform)
        np.testing.assert_allclose(w.frame_trajectory.resolve(w,rec)[0],f.pose_cw@np.linalg.inv(transform),atol=1e-12)
        self.assertEqual(w.frame_trajectory.resolve(w,rec)[1],2)

    def test_no_cross_segment_parent_or_silent_fill(self):
        w=world();w.keyframes[0].segment_id=2
        w.frame_trajectory.append(w,4,4*10**9,SimpleNamespace(pose_cw=pose()),False,0,"tracked")
        w.remove_keyframe(0)
        self.assertEqual(w.frame_trajectory.resolve(w,w.frame_trajectory.records[0]),(None,None,None))

    def test_export_keeps_lost_timestamps_and_pickle_references(self):
        w=world(); trajectory=w.frame_trajectory
        trajectory.append(w,4,4*10**9,SimpleNamespace(pose_cw=pose(.4)),False,0,"tracked")
        trajectory.append(w,5,5*10**9,None,False,0,"lost")
        w.remove_keyframe(0)
        with tempfile.TemporaryDirectory() as d:
            save_map(w,Path(d)/"map.pkl"); restored=load_map(Path(d)/"map.pkl")
            stats=restored.frame_trajectory.export(restored,d)
            self.assertEqual(stats["frames"],2);self.assertEqual(stats["final_valid"],1)
            data=np.load(Path(d)/"trajectory_frames.npz")
            np.testing.assert_array_equal(data["timestamp_ns"],[4*10**9,5*10**9])
            self.assertTrue(np.isnan(data["pose_final_cw"][1]).all())
            rows=[json.loads(x) for x in (Path(d)/"tracking_frames.jsonl").read_text().splitlines()]
            self.assertEqual(rows[1]["status"],"lost");self.assertFalse(rows[1]["final_valid"])

    def test_keyframe_record_exactly_matches_final_keyframe(self):
        w=world(); trajectory=w.frame_trajectory;k=w.keyframes[1]
        trajectory.append(w,1,10**9,k,True,1,"tracked")
        k.pose_cw=pose(4.,.6)
        np.testing.assert_allclose(trajectory.resolve(w,trajectory.records[0])[0],k.pose_cw,atol=1e-12)

    def test_bootstrap_and_mono_are_not_reported_as_final_estimates(self):
        w=world();t=w.frame_trajectory;f=SimpleNamespace(pose_cw=pose())
        t.append(w,4,4*10**9,f,False,0,"bootstrap_wait")
        t.append(w,5,5*10**9,f,False,0,"tracked",final_supported=False)
        self.assertIsNone(t.records[0]["pose_online_cw"])
        self.assertIsNotNone(t.records[1]["pose_online_cw"])
        self.assertIsNone(t.resolve(w,t.records[1])[0])

    def test_common_frontend_wrapper_records_early_failure_and_relocalization(self):
        w=world();mapper=OfflineMapper.__new__(OfflineMapper)
        mapper.world_map=w;mapper._next_frame_id=4;mapper.ref_keyframe=w.keyframes[0]
        mapper.n_map_segments=1;mapper.n_relocalizations=0;mapper.mono_mode=False
        mapper.track_stats=[];mapper._consecutive_lost=0
        args=(np.empty((1,1)),{},np.empty((0,2)),np.empty((0,256)),np.empty(0),4*10**9,None)
        with patch.object(mapper,"_process_frame_impl",return_value=(None,False)):
            mapper._process_frame(*args)
        self.assertEqual(w.frame_trajectory.records[-1]["status"],"lost")
        def recover(*a):
            mapper.n_relocalizations+=1;mapper.ref_keyframe=w.keyframes[2]
            return SimpleNamespace(pose_cw=pose(.1)),False
        mapper._next_frame_id=5
        with patch.object(mapper,"_process_frame_impl",side_effect=recover):
            mapper._process_frame(*args[:5],5*10**9,None)
        self.assertEqual(w.frame_trajectory.records[-1]["status"],"relocalized")
        self.assertEqual(w.frame_trajectory.records[-1]["original_anchor_id"],2)

class EvaluationTests(unittest.TestCase):
    def test_rigid_alignment_does_not_hide_scale_error(self):
        from splg_slam.geometry.frame_evaluation import rigid_alignment
        pts=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        rotated=pts@Rotation.from_euler("z",.6).as_matrix().T+np.array([1.,2.,3.])
        aligned,_,_=rigid_alignment(pts,rotated)
        np.testing.assert_allclose(aligned,rotated,atol=1e-12)
        scaled,_,_=rigid_alignment(pts,pts*2.)
        self.assertGreater(np.linalg.norm(scaled-pts*2.),.5)

    def test_exact_pairing_excludes_missing_and_imu_only_without_interpolation(self):
        from splg_slam.geometry.frame_evaluation import paired_frame_indices
        def data(ts,status):
            return dict(timestamp_ns=np.array(ts,dtype=np.int64),status=np.array(status),
                        pose_final_cw=np.tile(np.eye(4),(len(ts),1,1)),final_segment_id=np.zeros(len(ts),dtype=int))
        a=data([10,20,30,40],["tracked","lost","imu_only","tracked"])
        b=data([10,25,30,40],["tracked"]*4)
        ts,ia,ib=paired_frame_indices(a,b)
        np.testing.assert_array_equal(ts,[10,40]);np.testing.assert_array_equal(ia,[0,3])

    def test_gt_body_camera_offset_and_missing_intervals(self):
        from splg_slam.geometry.frame_evaluation import euroc_camera_gt
        import yaml
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"cam0").mkdir();(root/"state_groundtruth_estimate0").mkdir()
            ext=np.eye(4);ext[0,3]=1.
            (root/"cam0/sensor.yaml").write_text(yaml.safe_dump(dict(T_BS=dict(data=ext.ravel().tolist()))))
            # A 90 degree yaw makes the camera lever arm point along body-world +Y.
            q=np.sqrt(.5)
            (root/"state_groundtruth_estimate0/data.csv").write_text(
                "# ts,p,q\n"+"".join(f"{t},2,3,4,{q},0,0,{q}\n" for t in (0,10_000_000,1_000_000_000)))
            xyz,valid=euroc_camera_gt(root,np.array([5_000_000,15_000_000,1_000_000_000,1_000_000_001]))
            np.testing.assert_array_equal(valid,[True,False,True,False])
            np.testing.assert_allclose(xyz[valid],[[2.,4.,4.],[2.,4.,4.]],atol=1e-12)

if __name__ == "__main__": unittest.main()
