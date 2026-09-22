from dataclasses import dataclass

import cv2
import numpy as np

from splg_slam.geometry.triangulation import triangulate_points


@dataclass
class MonoInitResult:
    pose_cw_b: np.ndarray  # second keyframe's pose (world -> camera); world frame is the first keyframe (identity)
    median_parallax_deg: float
    num_points: int


def try_initialize(
    ref_kpts: np.ndarray, cur_kpts: np.ndarray, matches: list[tuple[int, int]], k: np.ndarray,
    min_matches: int = 100, min_parallax_deg: float = 1.5, min_triangulated: int = 50,
    ransac_reproj_threshold_px: float = 1.0, ransac_confidence: float = 0.999,
    assumed_scene_depth_m: float = 4.0,
) -> MonoInitResult | None:
    """Classic two-view monocular bootstrap: essential-matrix pose recovery (up to an
    arbitrary, unit-norm translation scale) between the first keyframe (pose_cw = identity,
    defines the world frame) and the current frame, then triangulates.

    Returns None if there aren't enough matches, the essential matrix can't be recovered,
    or the recovered motion has too little parallax/too few points survive triangulation -
    the caller should just keep calling this on each new frame (with more accumulated
    camera motion) until it succeeds, mirroring ORB-SLAM's monocular bootstrap loop.

    assumed_scene_depth_m: the whole rest of the config (keyframe_translation_m,
    max_pose_jump_m, mapping/loop-closure distance thresholds, ...) is written in real
    meters, but a monocular reconstruction has no metric reference at all until IMU data
    resolves it (see imu_init.py's solve_scale) - normalizing the arbitrary bootstrap scale
    to an assumed *typical* scene depth (rather than an arbitrary "1.0") keeps those
    meter-denominated thresholds behaving sanely from frame one, instead of firing on
    every single frame (arbitrary units too large) or almost never (too small). It's a
    rough prior, not a calibration - accuracy doesn't depend on getting it exactly right,
    only on the same order of magnitude as the real scene."""
    if len(matches) < min_matches:
        return None

    ref_idx = np.array([m[0] for m in matches])
    cur_idx = np.array([m[1] for m in matches])
    pts_ref = ref_kpts[ref_idx]
    pts_cur = cur_kpts[cur_idx]

    e_mat, inlier_mask = cv2.findEssentialMat(
        pts_ref, pts_cur, k, method=cv2.RANSAC,
        prob=ransac_confidence, threshold=ransac_reproj_threshold_px,
    )
    if e_mat is None or e_mat.shape != (3, 3) or inlier_mask is None:
        return None
    inlier_mask = inlier_mask.ravel().astype(bool)
    if inlier_mask.sum() < min_matches:
        return None

    n_inliers, r_rel, t_rel, pose_mask = cv2.recoverPose(e_mat, pts_ref[inlier_mask], pts_cur[inlier_mask], k)
    if n_inliers < min_triangulated:
        return None
    pose_mask = pose_mask.ravel().astype(bool)

    pose_cw_a = np.eye(4)
    pose_cw_b = np.eye(4)
    pose_cw_b[:3, :3] = r_rel
    pose_cw_b[:3, 3] = t_rel.ravel()  # unit-norm translation: arbitrary initial scale

    points_world, valid = triangulate_points(
        pose_cw_a, pose_cw_b, k, pts_ref[inlier_mask][pose_mask], pts_cur[inlier_mask][pose_mask],
        min_parallax_deg=min_parallax_deg,
    )
    if valid.sum() < min_triangulated:
        return None

    # Normalize scale so the median triangulated depth matches assumed_scene_depth_m - see
    # the assumed_scene_depth_m docstring above for why "1.0" (arbitrary units) is the
    # wrong target here.
    depths_a = points_world[valid, 2]  # pose_cw_a is identity, so world Z == depth in cam A
    median_depth = float(np.median(depths_a))
    if median_depth <= 1e-6:
        return None
    norm = assumed_scene_depth_m / median_depth
    points_world = points_world * norm
    pose_cw_b[:3, 3] *= norm

    center_b = -pose_cw_b[:3, :3].T @ pose_cw_b[:3, 3]
    ray_a = points_world[valid]  # center_a is the origin
    ray_b = points_world[valid] - center_b
    cos_parallax = np.sum(ray_a * ray_b, axis=1) / (
        np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1) + 1e-9
    )
    # Median of the per-point ANGLE, not arccos of the median cosine - arccos is
    # nonlinear, so the two only agree when the median falls on a single point (odd
    # count); for an even count (median = average of two middle values) they diverge.
    parallax_deg = np.degrees(np.arccos(np.clip(cos_parallax, -1.0, 1.0)))
    median_parallax_deg = float(np.median(parallax_deg))

    return MonoInitResult(
        pose_cw_b=pose_cw_b,
        median_parallax_deg=median_parallax_deg,
        num_points=int(valid.sum()),
    )
