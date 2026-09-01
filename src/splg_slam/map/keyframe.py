from dataclasses import dataclass

import numpy as np

from splg_slam.map.frame import Frame


@dataclass
class KeyFrame(Frame):
    global_descriptor: np.ndarray | None = None  # for image retrieval (loop closure / relocalization)
    image_path: str | None = None  # left rectified image path, kept for debugging/visualization
