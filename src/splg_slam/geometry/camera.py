from dataclasses import dataclass

import numpy as np


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
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
