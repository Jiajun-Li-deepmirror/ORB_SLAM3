import numpy as np


def camera_center(pose_cw: np.ndarray) -> np.ndarray:
    r = pose_cw[:3, :3]
    t = pose_cw[:3, 3]
    return -r.T @ t


def invert_pose(pose_cw: np.ndarray) -> np.ndarray:
    r = pose_cw[:3, :3]
    pose_wc = np.eye(4)
    pose_wc[:3, :3] = r.T
    pose_wc[:3, 3] = camera_center(pose_cw)
    return pose_wc


def rotation_angle_deg(r_rel: np.ndarray) -> float:
    """Angle (deg) of a 3x3 relative rotation matrix, via the standard trace formula."""
    cos_angle = np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def pose_delta(pose_a_cw: np.ndarray, pose_b_cw: np.ndarray) -> tuple[float, float]:
    """Translation (m) between camera centers and rotation (deg) between orientations."""
    translation = float(np.linalg.norm(camera_center(pose_b_cw) - camera_center(pose_a_cw)))
    rotation_deg = rotation_angle_deg(pose_a_cw[:3, :3].T @ pose_b_cw[:3, :3])
    return translation, rotation_deg
