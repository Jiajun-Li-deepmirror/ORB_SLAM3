from dataclasses import dataclass

import numpy as np

from splg_slam.map.frame import Frame


@dataclass
class KeyFrame(Frame):
    global_descriptor: np.ndarray | None = None  # for image retrieval (loop closure / relocalization)
    image_path: str | None = None  # left rectified image path, kept for debugging/visualization
    imu_bias: np.ndarray | None = None  # 6, [ax,ay,az,gx,gy,gz] (IMU fusion only)
    # Atlas-style multi-segment mapping: which map segment this keyframe belongs to (1 for
    # the whole run until a re-init happens - see OfflineMapper._start_new_map_segment).
    # A segment's own poses are self-consistent, but the *first* keyframe of segment N>1
    # only has a guessed (dead-reckoned/IMU-bridged) pose bridging the tracking-loss gap -
    # not a real measurement - so a relative pose between two keyframes in *different*
    # segments isn't backed by the same "already locally-accurate odometry" assumption a
    # same-segment relative pose is. See build_map.py's loop-closure consistency check.
    segment_id: int = 1
