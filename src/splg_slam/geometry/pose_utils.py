import numpy as np


def camera_center(pose_cw: np.ndarray) -> np.ndarray:
    r = pose_cw[:3, :3]
    t = pose_cw[:3, 3]
    return -r.T @ t


def invert_pose(pose_cw: np.ndarray) -> np.ndarray:
    r = pose_cw[:3, :3]
    t = pose_cw[:3, 3]
    pose_wc = np.eye(4)
    pose_wc[:3, :3] = r.T
    pose_wc[:3, 3] = -r.T @ t
    return pose_wc


def pose_delta(pose_a_cw: np.ndarray, pose_b_cw: np.ndarray) -> tuple[float, float]:
    """Translation (m) between camera centers and rotation (deg) between orientations."""
    translation = float(np.linalg.norm(camera_center(pose_b_cw) - camera_center(pose_a_cw)))

    r_rel = pose_a_cw[:3, :3].T @ pose_b_cw[:3, :3]
    cos_angle = np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0)
    rotation_deg = float(np.degrees(np.arccos(cos_angle)))
    return translation, rotation_deg
