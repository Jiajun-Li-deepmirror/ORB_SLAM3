from dataclasses import dataclass

import numpy as np


def build_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@dataclass
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray  # radial-tangential: k1, k2, p1, p2[, k3]
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return build_K(self.fx, self.fy, self.cx, self.cy)
