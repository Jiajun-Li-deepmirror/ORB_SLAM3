import numpy as np
from gtsam import Pose3, Rot3

from splg_slam.geometry.pose_utils import invert_pose


def matrix_to_gtsam_pose3(mat: np.ndarray) -> Pose3:
    return Pose3(Rot3(mat[:3, :3]), mat[:3, 3])


def pose_cw_to_gtsam(pose_cw: np.ndarray) -> Pose3:
    """gtsam's Pose3 is camera->world, opposite of our world->camera pose_cw convention."""
    return matrix_to_gtsam_pose3(invert_pose(pose_cw))


def gtsam_pose_to_cw(pose: Pose3) -> np.ndarray:
    return invert_pose(pose.matrix())


def pose_cw_to_body_gtsam(pose_cw: np.ndarray, t_cam_from_body: np.ndarray) -> Pose3:
    """Composes a camera pose_cw with the fixed cam<-body extrinsic to get the body/IMU
    pose, in gtsam's body->world convention (same convention pose_cw_to_gtsam uses for the
    camera)."""
    return pose_cw_to_gtsam(pose_cw).compose(matrix_to_gtsam_pose3(t_cam_from_body))


def confidence_scaled_sigma(
    base_sigma: float, num_inliers: int, min_inliers: int, floor_scale: float = 0.3
) -> float:
    """Scales a loop-edge noise sigma down as verification confidence goes up, instead of
    trusting every accepted loop edge (61 inliers or 600) at the same fixed sigma. Scale
    is 1.0 right at the acceptance threshold and shrinks like 1/sqrt(inliers) above it,
    floored at `floor_scale` so an extreme inlier count doesn't make one edge dominate
    the whole graph outright."""
    scale = max(floor_scale, min(1.0, (min_inliers / max(num_inliers, 1)) ** 0.5))
    return base_sigma * scale
