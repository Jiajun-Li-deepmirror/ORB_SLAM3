from dataclasses import dataclass, field

import numpy as np


@dataclass
class MapPoint:
    point_id: int
    position: np.ndarray  # 3, world coordinates
    descriptor: np.ndarray  # 256, representative SuperPoint descriptor
    observations: dict[int, int] = field(default_factory=dict)  # keyframe_id -> keypoint_index

    def add_observation(self, keyframe_id: int, kp_idx: int) -> None:
        self.observations[keyframe_id] = kp_idx

    def num_observations(self) -> int:
        return len(self.observations)
