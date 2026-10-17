import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from splg_slam.data.ground_depth import decode_depth, GroundDepthReader
from splg_slam.map.keyframe import KeyFrame
from splg_slam.map.world_map import WorldMap
from splg_slam.mapping.ground_plane import (
    GroundPlaneOptions, add_ground_factors, associate_ground, detect_ground, plane_error,
)
from splg_slam.mapping.gtsam_utils import gtsam_pose_to_cw
from splg_slam.mapping.pose_graph import optimize_pose_graph


def keyframe(i, position=(0., 0., 1.5), segment=1):
    pose = gtsam.Pose3(gtsam.Rot3(), np.asarray(position))
    return KeyFrame(i, i * 100_000_000, np.empty((0, 2)), np.empty((0, 256)), np.empty(0),
                    pose_cw=gtsam_pose_to_cw(pose), segment_id=segment)


def observation(height=1.5, group=0):
    xy = np.mgrid[-1:1:10j, -1:1:10j].reshape(2, -1).T
    support = np.column_stack((xy, np.full(len(xy), -height)))
    return {"normal": np.array([0., 0., 1.]), "centroid": np.array([0., 0., -height]),
            "support": support, "group_id": group}


class GroundPlaneTests(unittest.TestCase):
    def test_pose_and_plane_jacobians_match_finite_difference(self):
        pose = gtsam.Pose3(gtsam.Rot3.RzRyRx(.2, -.1, .3), np.array([2., 3., 1.]))
        plane = np.array([.04, -.03, .7])
        normal = np.array([.1, .2, .97]); normal /= np.linalg.norm(normal)
        center = np.array([.3, -.8, 2.])
        error, jp, jl = plane_error(pose, plane, normal, center, True)
        eps = 1e-6
        for i in range(6):
            delta = np.eye(6)[i] * eps
            fd = (plane_error(pose.retract(delta), plane, normal, center)
                  - plane_error(pose.retract(-delta), plane, normal, center)) / (2 * eps)
            np.testing.assert_allclose(jp[:, i], fd, atol=2e-8)
        for i in range(3):
            delta = np.eye(3)[i] * eps
            fd = (plane_error(pose, plane + delta, normal, center)
                  - plane_error(pose, plane - delta, normal, center)) / (2 * eps)
            np.testing.assert_allclose(jl[:, i], fd, atol=2e-8)

    def test_floor_extraction_rejects_wall_and_ceiling(self):
        rng = np.random.default_rng(3)
        floor = np.column_stack((rng.uniform(-2, 2, 1000), rng.uniform(-2, 2, 1000),
                                 -1.5 + rng.normal(0, .005, 1000)))
        wall = np.column_stack((np.full(2000, 2.), rng.uniform(-2, 2, 2000), rng.uniform(-2, 2, 2000)))
        ceiling = floor.copy(); ceiling[:, 2] += 3
        options = GroundPlaneOptions()
        result, diag = detect_ground(np.vstack((floor, wall, ceiling)), [0., 0., 1.], options)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['height_m'], 1.5, places=2)
        self.assertGreater(result['normal'][2], .999)
        self.assertIsNone(detect_ground(wall, [0., 0., 1.], options)[0])
        self.assertIsNone(detect_ground(ceiling, [0., 0., 1.], options)[0])

    def test_narrow_stair_tread_is_rejected(self):
        x, y = np.meshgrid(np.linspace(-.5, .5, 30), np.linspace(-.12, .12, 30))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.3)))
        self.assertIsNone(detect_ground(points, [0., 0., 1.], GroundPlaneOptions())[0])

    def test_normal_bias_is_removed_in_camera_frame(self):
        rng = np.random.default_rng(5)
        floor = np.column_stack((rng.uniform(-2, 2, 1500), 1.4 + rng.normal(0, .003, 1500),
                                 rng.uniform(.5, 4, 1500)))  # camera frame: y down, floor below
        bias = (0.6, -0.3)
        tilted = floor @ Rotation.from_rotvec(np.radians([bias[0], 0., bias[1]])).as_matrix().T
        up = np.array([0., -1., 0.])
        raw, _ = detect_ground(tilted, up, GroundPlaneOptions())
        np.testing.assert_allclose(np.degrees(np.cross(up, raw['normal']))[[0, 2]], bias, atol=0.05)
        fixed, _ = detect_ground(tilted, up, GroundPlaneOptions.from_config({'normal_bias_deg': list(bias)}))
        np.testing.assert_allclose(fixed['normal'], up, atol=1e-3)
        # Only the normal is corrected; the measured centroid (used by the distance factor) stays.
        np.testing.assert_allclose(fixed['centroid'], raw['centroid'])
        self.assertAlmostEqual(fixed['height_m'], -fixed['normal'] @ fixed['centroid'])
        self.assertLess(abs(fixed['height_m'] - 1.4), 0.03)
        with self.assertRaises(ValueError):
            GroundPlaneOptions.from_config({'normal_bias_deg': [10., 0.]})
        with self.assertRaises(ValueError):
            GroundPlaneOptions.from_config({'normal_bias_deg': [0.5]})

    def test_camera_height_can_change_while_floor_stays_same(self):
        w = WorldMap(); options = GroundPlaneOptions()
        a, b = keyframe(0), keyframe(1, (.2, 0., 1.8))
        w.add_keyframe(a); w.add_keyframe(b)
        associate_ground(w, a, observation(1.5), options)
        associate_ground(w, b, observation(1.8), options)
        self.assertEqual(a.ground_plane['group_id'], b.ground_plane['group_id'])

    def test_other_floor_disjoint_patch_and_segment_do_not_merge(self):
        w = WorldMap(); options = GroundPlaneOptions()
        for i, position, segment in [(0, (0, 0, 1.5), 1), (1, (0, 0, 4.5), 1),
                                     (2, (8, 0, 1.5), 1), (3, (0, 0, 1.5), 2)]:
            kf = keyframe(i, position, segment); w.add_keyframe(kf)
            associate_ground(w, kf, observation(), options)
        self.assertEqual(len({k.ground_plane['group_id'] for k in w.keyframes.values()}), 4)

    def test_plane_factor_preserves_camera_bob_and_floor_separation(self):
        w = WorldMap(); w.ground_plane_options = GroundPlaneOptions(enabled=True)
        for i in range(6):
            floor = 0 if i < 3 else 3
            camera_height = 1.5 + .05 * i
            kf = keyframe(i, (i * .2, 0, floor + camera_height))
            kf.ground_plane = observation(camera_height, 0 if i < 3 else 1)
            w.add_keyframe(kf)
        before = {i: k.pose_cw.copy() for i, k in w.keyframes.items()}
        stats = optimize_pose_graph(w)
        self.assertEqual(stats['ground_plane_factors'], 6)
        self.assertEqual(stats['ground_plane_groups'], 2)
        for i, pose in before.items():
            np.testing.assert_allclose(w.keyframes[i].pose_cw, pose, atol=1e-8)

    def test_ground_factor_reduces_synthetic_vertical_drift(self):
        w = WorldMap(); w.ground_plane_options = GroundPlaneOptions(enabled=True, distance_sigma_m=.02)
        for i in range(8):
            kf = keyframe(i, (i * .2, 0, 1.5 + i * .04))
            kf.ground_plane = observation(); w.add_keyframe(kf)
        before = abs(-w.keyframes[7].pose_cw[2, 3] - 1.5)
        stats = optimize_pose_graph(w)
        after = abs(-w.keyframes[7].pose_cw[2, 3] - 1.5)
        self.assertFalse(stats['rejected'])
        self.assertLess(after, before / 3)

    def test_disabled_and_old_maps_leave_graph_untouched(self):
        w = WorldMap(); graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        self.assertEqual(add_ground_factors(w, graph, values, set(), set())['ground_plane_factors'], 0)
        w.ground_plane_options = GroundPlaneOptions(enabled=False)
        self.assertEqual(add_ground_factors(w, graph, values, set(), set())['ground_plane_factors'], 0)
        self.assertEqual(graph.size(), 0)
        self.assertEqual(values.size(), 0)

    def test_culling_and_rigid_gauge_change_keep_measurements_valid(self):
        w = WorldMap(); w.ground_plane_options = GroundPlaneOptions(enabled=True)
        for i in range(5):
            kf = keyframe(i, (i*.2, 0, 1.5)); kf.ground_plane = observation(); w.add_keyframe(kf)
        w.remove_keyframe(2)
        transform = gtsam.Pose3(gtsam.Rot3.Rz(.3), np.array([2., -1., 3.])).matrix()
        w.transform_keyframes(set(w.keyframes), transform)
        before = {i: k.pose_cw.copy() for i, k in w.keyframes.items()}
        self.assertEqual(optimize_pose_graph(w)['ground_plane_factors'], 4)
        for i, pose in before.items():
            np.testing.assert_allclose(w.keyframes[i].pose_cw, pose, atol=1e-8)

    def test_depth_decoding_honors_padding_and_endianness(self):
        data = np.array([[1000, 2000, 9999], [3000, 0, 9999]], dtype='>u2').tobytes()
        msg = SimpleNamespace(encoding='16UC1', is_bigendian=True, height=2, width=2, step=6, data=data)
        actual = decode_depth(msg, .001)
        np.testing.assert_allclose(actual, [[1, 2], [3, np.nan]], equal_nan=True)
        msg = SimpleNamespace(encoding='32FC1', is_bigendian=False, height=1, width=3, step=12,
                              data=np.array([1.5, np.inf, 0], dtype='<f4').tobytes())
        np.testing.assert_allclose(decode_depth(msg, .001), [[1.5, np.nan, np.nan]], equal_nan=True)

    def test_stale_depth_is_rejected_before_decoding(self):
        reader = GroundDepthReader.__new__(GroundDepthReader)
        reader.options = GroundPlaneOptions()
        reader.last_request = -1
        reader.previous = (0, None); reader.current = (100_000_000, None)
        points, diag = reader.points_at(50_000_000)
        self.assertIsNone(points); self.assertEqual(diag['reason'], 'depth_time_gap')
        with self.assertRaises(ValueError): reader.points_at(40_000_000)

    def test_invalid_config_fails_early(self):
        for config in [{'distance_sigma_m': 0}, {'normal_sigma_deg': np.nan},
                       {'min_observations': 1}, {'pixel_stride': 1.5}, {'enable': True}]:
            with self.assertRaises(ValueError): GroundPlaneOptions.from_config(config)


if __name__ == '__main__':
    unittest.main()
