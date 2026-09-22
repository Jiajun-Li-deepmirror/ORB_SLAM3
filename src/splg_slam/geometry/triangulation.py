import cv2
import numpy as np


def triangulate_points(
    pose_cw_a: np.ndarray, pose_cw_b: np.ndarray, k: np.ndarray,
    pts_a: np.ndarray, pts_b: np.ndarray,
    min_parallax_deg: float = 1.0, max_reproj_err_px: float = 4.0, min_depth: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-view DLT triangulation of matched pixel pairs (pts_a/pts_b are the same
    physical points, seen in cameras A/B's already-rectified pixel grid), given each
    camera's own already-known pose_cw (world -> camera) and shared intrinsics k.

    Returns (points_world Nx3, valid_mask N bool). valid requires: positive depth in
    both cameras (cheirality - same check as local_ba.py's outlier gate), a parallax
    angle above min_parallax_deg (near-zero-parallax pairs triangulate to numerically
    garbage points - the classic two-view degeneracy, e.g. a camera holding still), and
    reprojection error below max_reproj_err_px in both views."""
    p_a = k @ pose_cw_a[:3, :]
    p_b = k @ pose_cw_b[:3, :]

    pts4d = cv2.triangulatePoints(p_a, p_b, pts_a.T.astype(np.float64), pts_b.T.astype(np.float64))
    pts3d = (pts4d[:3] / pts4d[3]).T  # world frame, Nx3

    depth_a = pts3d @ pose_cw_a[:3, :3].T[:, 2] + pose_cw_a[2, 3]
    depth_b = pts3d @ pose_cw_b[:3, :3].T[:, 2] + pose_cw_b[2, 3]
    cheirality = (depth_a > min_depth) & (depth_b > min_depth)

    center_a = -pose_cw_a[:3, :3].T @ pose_cw_a[:3, 3]
    center_b = -pose_cw_b[:3, :3].T @ pose_cw_b[:3, 3]
    ray_a = pts3d - center_a
    ray_b = pts3d - center_b
    cos_parallax = np.sum(ray_a * ray_b, axis=1) / (
        np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1) + 1e-9
    )
    parallax_deg = np.degrees(np.arccos(np.clip(cos_parallax, -1.0, 1.0)))
    enough_parallax = parallax_deg > min_parallax_deg

    pts3d_h = np.hstack([pts3d, np.ones((len(pts3d), 1))])
    proj_a = (p_a @ pts3d_h.T).T
    proj_a = proj_a[:, :2] / proj_a[:, 2:3]
    err_a = np.linalg.norm(proj_a - pts_a, axis=1)
    proj_b = (p_b @ pts3d_h.T).T
    proj_b = proj_b[:, :2] / proj_b[:, 2:3]
    err_b = np.linalg.norm(proj_b - pts_b, axis=1)
    low_err = (err_a < max_reproj_err_px) & (err_b < max_reproj_err_px)

    valid = cheirality & enough_parallax & low_err
    return pts3d, valid
