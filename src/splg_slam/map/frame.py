from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    frame_id: int
    timestamp_ns: int
    keypoints: np.ndarray  # Nx2, pixel coords in the rectified left image
    descriptors: np.ndarray  # Nx256, SuperPoint descriptors
    depths: np.ndarray  # N, stereo depth in meters, NaN where invalid
    pose_cw: np.ndarray | None = None  # 4x4, world -> camera; None until estimated
    map_point_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    velocity: np.ndarray | None = None  # 3, world-frame body linear velocity (IMU fusion only)
    # True if this keyframe's pose came from pure IMU-only prediction (tracking.
    # continuous_imu_tracking) rather than a verified vision correspondence - i.e. tracking
    # was "lost" by the usual vision-only definition, but kept going on IMU dead reckoning
    # instead of cutting into a new Atlas segment. Not a verified pose the way a normal
    # keyframe's is; kept for visualization and pose-graph sigma-scaling decisions.
    imu_only: bool = False

    def __post_init__(self):
        if self.map_point_ids.size == 0 and len(self.keypoints) > 0:
            self.map_point_ids = np.full(len(self.keypoints), -1, dtype=np.int64)

    def valid_depth_mask(self) -> np.ndarray:
        return ~np.isnan(self.depths)

    def pose_wc(self) -> np.ndarray:
        from splg_slam.geometry.pose_utils import invert_pose

        return invert_pose(self.pose_cw)
