"""Run with: python -m unittest discover -s tests -v (no model downloads)."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import cv2
import gtsam
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import build_map
import localize
from splg_slam.data.euroc import ImuCalibration, StereoRig
from splg_slam.data.loader import iter_keyframe_stereo_images, read_frame_images, set_dataset_dir
from splg_slam.geometry.camera import PinholeCamera
from splg_slam.geometry.stereo import MonoRectifier, StereoRectifier, rectify_imu_calibration
from splg_slam.localization.continuous_localizer import ContinuousLocalizer
from splg_slam.map.keyframe import KeyFrame
from splg_slam.map.world_map import WorldMap
from splg_slam.mapping.gtsam_utils import pose_cw_to_body_gtsam
from splg_slam.mapping.imu_preintegration import imu_samples_between, preintegrate
from splg_slam.mapping.loop_closure import LoopConsistencyTracker, verify_loop_candidate, fuse_loop_matches
from splg_slam.mapping.pose_graph import optimize_pose_graph, relative_pose
from splg_slam.mapping.tracker import OfflineMapper


def keyframe(i, n=1, timestamp=None):
    return KeyFrame(
        frame_id=i, timestamp_ns=i * 25_000_000 if timestamp is None else timestamp,
        keypoints=np.zeros((n, 2)), descriptors=np.zeros((n, 256)),
        depths=np.full(n, 2.), pose_cw=np.eye(4), velocity=np.array([1., 0., 0.]),
        imu_bias=np.arange(6, dtype=float),
    )


def two_frame_map():
    m = WorldMap()
    for i in (0, 100):
        m.add_keyframe(keyframe(i))
        pid = m.new_map_point(np.array([0., 0., 2.]), np.zeros(256))
        m.add_observation(pid, i, 0)
    return m


def camera():
    return PinholeCamera(fx=300., fy=300., cx=160., cy=120.,
                         dist_coeffs=np.zeros(4), width=320, height=240)


class ImuIntervalTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.zeros((21, 7))
        self.raw[:, 0] = np.arange(21) * 5_000_000
        self.raw[:, 3] = 1.0
        self.raw[:, 6] = 9.81
        self.params = gtsam.PreintegrationCombinedParams(np.array([0., 0., -9.81]))

    def integrate(self, samples):
        return preintegrate(samples, gtsam.imuBias.ConstantBias(), self.params)

    def test_exact_endpoints_and_angular_motion(self):
        samples = imu_samples_between(self.raw, 0, 50_000_000)
        p = self.integrate(samples)
        self.assertAlmostEqual(p.deltaTij(), .05, places=12)
        np.testing.assert_allclose(gtsam.Rot3.Logmap(p.deltaRij()), [0, 0, .05], atol=1e-10)

    def test_off_grid_and_sub_sample_interval(self):
        for start, end in ((2_000_000, 48_000_000), (1_000_000, 2_000_000)):
            samples = imu_samples_between(self.raw, start, end)
            self.assertEqual(samples[0, 0], start)
            self.assertEqual(samples[-1, 0], end)
            self.assertAlmostEqual(self.integrate(samples).deltaTij(), (end - start) / 1e9)

    def test_epoch_timestamps_and_constant_acceleration(self):
        self.raw[:, 0] += 1_790_000_000_000_000_000
        self.raw[:, 1:4] = 0
        self.raw[:, 4] = 2.0
        start = int(self.raw[0, 0]) + 1_234_567
        end = start + 50_000_000
        samples = imu_samples_between(self.raw, start, end)
        p = self.integrate(samples)
        self.assertAlmostEqual(p.deltaTij(), .05, delta=512e-9)
        bias = gtsam.imuBias.ConstantBias()
        predicted = p.predict(gtsam.NavState(gtsam.Pose3(), np.zeros(3)), bias)
        np.testing.assert_allclose(predicted.velocity(), [.1, 0, 0], atol=2e-6)
        np.testing.assert_allclose(predicted.pose().translation(), [.0025, 0, 0], atol=2e-6)

    def test_boundary_uses_previous_measurement_without_future_leak(self):
        self.raw[:, 3] = np.arange(len(self.raw))
        samples = imu_samples_between(self.raw, 2_000_000, 8_000_000)
        np.testing.assert_array_equal(samples[:, 0], [2_000_000, 5_000_000, 8_000_000])
        np.testing.assert_array_equal(samples[:, 3], [0, 1, 1])

    def test_no_extrapolation_or_empty_interval(self):
        for start, end in ((-1, 50_000_000), (0, 100_000_001), (1, 1), (2, 1)):
            self.assertIsNone(imu_samples_between(self.raw, start, end))
        self.assertIsNone(imu_samples_between(self.raw[:0], 0, 10))

    def test_tracker_accumulates_complete_intervals(self):
        mapper = OfflineMapper.__new__(OfflineMapper)
        mapper.imu_measurements = self.raw
        mapper._imu_last_ts_ns = None
        mapper._pending_imu_samples = []
        mapper.ref_keyframe = keyframe(0)
        for t in (0, 23_000_000, 50_000_000):
            mapper._pull_imu_samples(t)
        samples = mapper._pending_imu_for_keyframe(50_000_000)
        self.assertAlmostEqual(self.integrate(samples).deltaTij(), .05)
        mapper.ref_keyframe = keyframe(9, timestamp=-1_000_000_000)
        self.assertIsNone(mapper._pending_imu_for_keyframe(50_000_000))

    def test_culling_preserves_full_imu_chain(self):
        m = WorldMap()
        for i in range(3):
            m.add_keyframe(keyframe(i))
        m.add_imu_factor(0, 1, imu_samples_between(self.raw, 0, 25_000_000))
        m.add_imu_factor(1, 2, imu_samples_between(self.raw, 25_000_000, 50_000_000))
        m.remove_keyframe(1)
        self.assertEqual(len(m.imu_factors), 1)
        a, b, samples = m.imu_factors[0]
        self.assertEqual((a, b), (0, 2))
        self.assertAlmostEqual(self.integrate(samples).deltaTij(), .05)

    def test_continuous_localizer_uses_same_interval(self):
        localizer = ContinuousLocalizer.__new__(ContinuousLocalizer)
        localizer.imu_enabled = True
        localizer.imu_measurements = self.raw
        samples = localizer._pull_imu_samples(2_000_000, 48_000_000)
        self.assertAlmostEqual(self.integrate(samples).deltaTij(), .046)


class CoordinateTests(unittest.TestCase):
    def test_rectified_extrinsics_preserve_physical_body_pose(self):
        relative = np.eye(4)
        relative[:3, :3] = cv2.Rodrigues(np.array([.02, -.01, .03]))[0]
        relative[:3, 3] = [-.1, .003, .002]
        rectifier = StereoRectifier(StereoRig(camera(), camera(), relative))
        raw_extrinsic = np.eye(4)
        raw_extrinsic[:3, :3] = cv2.Rodrigues(np.array([.1, .2, -.1]))[0]
        raw_extrinsic[:3, 3] = [.01, .02, .03]
        calibration = ImuCalibration(.01, .001, .1, .01, raw_extrinsic.copy())
        corrected = rectify_imu_calibration(calibration, rectifier)
        raw_pose = np.eye(4)
        raw_pose[:3, 3] = [1., 2., 3.]
        rect_pose = rectifier.T_rect_cam0 @ raw_pose
        expected = pose_cw_to_body_gtsam(raw_pose, raw_extrinsic).matrix()
        actual = pose_cw_to_body_gtsam(rect_pose, corrected.T_cam0_body).matrix()
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_array_equal(calibration.T_cam0_body, raw_extrinsic)
        mono = rectify_imu_calibration(calibration, MonoRectifier(camera()))
        np.testing.assert_array_equal(mono.T_cam0_body, raw_extrinsic)

    def test_segment_transform_rotates_velocity_and_keeps_sensor_bias(self):
        m = two_frame_map()
        m.keyframes[100].segment_id = 2
        original_bias = m.keyframes[100].imu_bias.copy()
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(np.array([0., 0., np.pi / 2]))[0]
        transform[:3, 3] = [2., 3., 4.]
        m.transform_segment(2, transform, 1)
        np.testing.assert_allclose(m.keyframes[100].velocity, [0, 1, 0], atol=1e-12)
        np.testing.assert_array_equal(m.keyframes[100].imu_bias, original_bias)
        np.testing.assert_array_equal(m.keyframes[0].velocity, [1, 0, 0])
        p = m.map_points[int(m.keyframes[100].map_point_ids[0])].position
        np.testing.assert_allclose(m.keyframes[100].pose_cw @ np.r_[p, 1], [0, 0, 2, 1], atol=1e-12)

    def test_pose_graph_rotates_velocity_with_accepted_correction(self):
        m = WorldMap()
        for i in range(3):
            kf = keyframe(i)
            kf.pose_cw[0, 3] = -i
            m.add_keyframe(kf)
        old_pose = m.keyframes[2].pose_cw.copy()
        desired = old_pose.copy()
        desired[:3, :3] = cv2.Rodrigues(np.array([0., 0., .2]))[0]
        m.add_loop_edge(0, 2, relative_pose(m.keyframes[0].pose_cw, desired), 100)
        stats = optimize_pose_graph(m)
        self.assertFalse(stats['rejected'])
        correction = m.keyframes[2].pose_cw[:3, :3].T @ old_pose[:3, :3]
        self.assertGreater(np.linalg.norm(correction - np.eye(3)), .01)
        np.testing.assert_allclose(m.keyframes[2].velocity, correction @ [1., 0., 0.])


class LoopTests(unittest.TestCase):
    def test_confirmations_require_distinct_later_queries(self):
        tracker = LoopConsistencyTracker(2)
        self.assertFalse(tracker.observe(10, 100, np.eye(4)))
        self.assertFalse(tracker.observe(11, 100, np.eye(4)))
        self.assertFalse(tracker.observe(12, 99, np.eye(4)))
        self.assertTrue(tracker.observe(12, 105, np.eye(4)))

    def test_timeout_and_single_confirmation(self):
        tracker = LoopConsistencyTracker(2, pending_timeout_kf=10)
        self.assertFalse(tracker.observe(10, 100, np.eye(4)))
        self.assertFalse(tracker.observe(10, 111, np.eye(4)))
        self.assertTrue(LoopConsistencyTracker(1).observe(10, 100, np.eye(4)))

    def test_real_pnp_outliers_are_not_fused(self):
        cv2.setRNGSeed(0)
        rng = np.random.default_rng(12)
        n = 40
        positions = rng.uniform([-1, -1, 3], [1, 1, 6], (n, 3))
        K = camera().K
        uv = (K @ positions.T).T
        uv = uv[:, :2] / uv[:, 2:]
        uv[-8:] += [100, 80]
        m = WorldMap()
        for i in (0, 100):
            m.add_keyframe(keyframe(i, n))
            for j, position in enumerate(positions):
                pid = m.new_map_point(position.copy(), np.zeros(256))
                m.add_observation(pid, i, j)
        # Missing candidate landmarks must not shift the match/inlier indexing.
        m.remove_map_point(int(m.keyframes[0].map_point_ids[0]))
        matches = np.column_stack([np.arange(n), np.arange(n)])
        matcher = NS(match=lambda *_: {'matches': matches})
        with patch('splg_slam.mapping.loop_closure.SPLG.to_frame_arrays', return_value=(uv, None)):
            accepted, pose, count, inliers = verify_loop_candidate(
                m, NS(K_rect=K), matcher, 100, 0, min_inliers=15,
                feats_cache={0: {}, 100: {}},
            )
        self.assertTrue(accepted)
        self.assertEqual(count, 31)
        self.assertEqual(set(inliers[:, 0]), set(range(1, 32)))
        before = len(m.map_points)
        self.assertEqual(fuse_loop_matches(m, 0, 100, inliers), 31)
        self.assertEqual(len(m.map_points), before - 31)
        self.assertNotEqual(m.keyframes[0].map_point_ids[-1], m.keyframes[100].map_point_ids[-1])

    def call_loop(self, m, rejected):
        mapper = NS(world_map=m, cfg=NS(tracking=NS(keyframe_translation_m=.3), mapping=NS()),
                    reloc_index=None, splg=None)
        cfg = NS(min_id_gap=30, top_k=3, min_similarity=.5, min_inliers=1,
                 max_consistency_trans_m=2.5, max_consistency_rot_deg=25., fusion_enabled=True)
        stats = {'rejected': rejected, 'max_pose_shift_m': 21., 'initial_error': 1., 'final_error': 0.}
        with patch.object(build_map, 'detect_loop_candidates', return_value=[(0, 1.)]), \
             patch.object(build_map, 'verify_loop_candidate', return_value=(True, np.eye(4), 1, np.array([[0, 0]]))), \
             patch.object(build_map, 'optimize_pose_graph', return_value=stats):
            return build_map.try_close_loops(mapper, None, NS(build=lambda _: None), cfg, {}, LoopConsistencyTracker(1), 100)

    def test_rejected_loop_leaves_points_edges_and_observations_unchanged(self):
        m = two_frame_map()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.call_loop(m, True), 0)
        self.assertEqual(len(m.loop_edges), 0)
        self.assertEqual(len(m.map_points), 2)
        np.testing.assert_array_equal(m.keyframes[100].map_point_ids, [1])
        self.assertEqual(m.map_points[1].observations, {100: 0})

    def test_accepted_loop_commits_edge_and_fuses(self):
        m = two_frame_map()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.call_loop(m, False), 1)
        self.assertEqual(len(m.loop_edges), 1)
        self.assertEqual(len(m.map_points), 1)

    def test_rigid_rejection_and_exception_restore_state(self):
        for error in (False, True):
            m = two_frame_map()
            transform = np.eye(4)
            transform[:3, :3] = cv2.Rodrigues(np.array([0., 0., .5]))[0]
            transform[:3, 3] = [4, 5, 6]
            kwargs = {'side_effect': RuntimeError('solver failure')} if error else {'return_value': {'rejected': True}}
            with patch.object(build_map, 'optimize_pose_graph', **kwargs):
                if error:
                    with self.assertRaisesRegex(RuntimeError, 'solver failure'):
                        build_map.optimize_loop_update(m, 0, 100, np.eye(4), 60, {100}, transform)
                else:
                    build_map.optimize_loop_update(m, 0, 100, np.eye(4), 60, {100}, transform)
            self.assertEqual(len(m.loop_edges), 0)
            np.testing.assert_array_equal(m.keyframes[100].pose_cw, np.eye(4))
            np.testing.assert_array_equal(m.keyframes[100].velocity, [1, 0, 0])
            np.testing.assert_array_equal(m.map_points[1].position, [0, 0, 2])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.cfg = NS(dataset=NS(kind='rosbag2', bag_dir='/unused'))
        self.image = np.zeros((4, 5), dtype=np.uint8)

    def test_stream_keyframe_images_without_file_paths(self):
        m = WorldMap()
        m.add_keyframe(keyframe(0, timestamp=100))
        m.add_keyframe(keyframe(1, timestamp=200))
        closed = []
        def entries():
            try:
                for t in (50, 100, 200, 300):
                    yield NS(timestamp_ns=t, left_image=self.image, right_image=self.image)
            finally:
                closed.append(True)
        with patch('splg_slam.data.loader.dataset_module', return_value=NS(load_stereo_frames=lambda _: entries())):
            result = list(iter_keyframe_stereo_images(m, self.cfg, [0, 1], color_left=True))
        self.assertEqual([r[0] for r in result], [0, 1])
        self.assertEqual(result[0][1].shape, (4, 5, 3))
        self.assertTrue(closed)

    def test_missing_bag_keyframe_fails_instead_of_empty_cloud(self):
        m = WorldMap()
        m.add_keyframe(keyframe(0, timestamp=100))
        with patch('splg_slam.data.loader.dataset_module', return_value=NS(load_stereo_frames=lambda _: iter([]))):
            with self.assertRaisesRegex(ValueError, 'missing 1'):
                list(iter_keyframe_stereo_images(m, self.cfg, [0]))

    def test_directory_override_uses_dataset_kind(self):
        for kind, field in [('rosbag2', 'bag_dir'), ('euroc', 'mav0_dir'), ('kitti', 'sequence_dir')]:
            cfg = NS(dataset=NS(kind=kind))
            set_dataset_dir(cfg, '/new')
            self.assertEqual(getattr(cfg.dataset, field), '/new')

    def test_file_and_memory_image_readers(self):
        left, right = read_frame_images(NS(timestamp_ns=1, left_image=self.image), stereo=False)
        self.assertIs(left, self.image)
        self.assertIsNone(right)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'image.png'
            cv2.imwrite(str(path), self.image)
            left, right = read_frame_images(NS(timestamp_ns=1, left_path=path, right_path=path))
            np.testing.assert_array_equal(left, self.image)
            np.testing.assert_array_equal(right, self.image)

    def test_localize_rosbag_without_ground_truth(self):
        cfg = NS(dataset=NS(kind='rosbag2', bag_dir='/nonexistent-test-data'),
                 tracking=NS(relocalization_early_exit_inliers=120, relocalization_max_keypoints=1024))
        entries = [NS(index=i, timestamp_ns=i, left_image=self.image) for i in range(2)]
        info = {'num_inliers': 100, 'candidate_kf': 0, 'retrieval_sim': 1.}
        localizer = NS(localize=lambda *_args, **_kwargs: (True, np.eye(4), info))
        rig = NS()
        output = io.StringIO()
        with patch.object(sys, 'argv', ['localize.py', 'cfg', 'map', '--count', '2']), \
             patch.object(localize, 'load_config', return_value=cfg), \
             patch.object(localize, 'load_map', return_value=two_frame_map()), \
             patch.object(localize, 'load_dataset_rig', return_value=rig), \
             patch.object(localize, 'StereoRectifier', return_value=NS(rectify_left=lambda x: x)), \
             patch.object(localize, 'Relocalizer', return_value=localizer), \
             patch.object(localize, 'dataset_module', return_value=NS(load_stereo_frames=lambda _: iter(entries))), \
             contextlib.redirect_stdout(output):
            localize.main()
        self.assertIn('2/2 localized', output.getvalue())
        self.assertIn('No ground truth supplied', output.getvalue())


if __name__ == '__main__':
    unittest.main()
