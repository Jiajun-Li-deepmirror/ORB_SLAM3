import cv2
import numpy as np

from splg_slam.data.euroc import StereoRig


class StereoRectifier:
    """Computes rectification maps once from calibration and reuses them per frame."""

    def __init__(self, rig: StereoRig):
        k_l, d_l = rig.cam0.K, rig.cam0.dist_coeffs
        k_r, d_r = rig.cam1.K, rig.cam1.dist_coeffs
        size = (rig.cam0.width, rig.cam0.height)

        r_rel = np.ascontiguousarray(rig.T_cam1_cam0[:3, :3])
        t_rel = np.ascontiguousarray(rig.T_cam1_cam0[:3, 3].reshape(3, 1))

        r_l, r_r, p_l, p_r, q, _, _ = cv2.stereoRectify(
            k_l, d_l, k_r, d_r, size, r_rel, t_rel,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )

        self.map_l = cv2.initUndistortRectifyMap(k_l, d_l, r_l, p_l, size, cv2.CV_32FC1)
        self.map_r = cv2.initUndistortRectifyMap(k_r, d_r, r_r, p_r, size, cv2.CV_32FC1)

        self.p_l = p_l
        self.p_r = p_r
        self.q = q
        self.fx_rect = float(p_l[0, 0])
        self.fy_rect = float(p_l[1, 1])
        self.cx_rect = float(p_l[0, 2])
        self.cy_rect = float(p_l[1, 2])
        self.baseline = float(-p_r[0, 3] / p_r[0, 0])  # meters, positive

    @property
    def K_rect(self) -> np.ndarray:
        return np.array(
            [[self.fx_rect, 0.0, self.cx_rect],
             [0.0, self.fy_rect, self.cy_rect],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def rectify(self, img_left: np.ndarray, img_right: np.ndarray):
        rect_l = cv2.remap(img_left, *self.map_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_right, *self.map_r, cv2.INTER_LINEAR)
        return rect_l, rect_r

    def backproject(self, points_xy: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """points_xy: Nx2 pixel coords, depths: N meters. Returns Nx3 points in the rectified left camera frame."""
        x = (points_xy[:, 0] - self.cx_rect) * depths / self.fx_rect
        y = (points_xy[:, 1] - self.cy_rect) * depths / self.fy_rect
        return np.stack([x, y, depths], axis=1)


class StereoDepthEstimator:
    """Dense SGBM disparity, sampled at sparse SuperPoint keypoint locations."""

    def __init__(self, rectifier: StereoRectifier, min_disp=0, num_disp=128, block_size=5):
        self.rectifier = rectifier
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * block_size ** 2,
            P2=32 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute_disparity(self, rect_left_gray: np.ndarray, rect_right_gray: np.ndarray) -> np.ndarray:
        return self.matcher.compute(rect_left_gray, rect_right_gray).astype(np.float32) / 16.0

    def depths_at_points(self, disparity: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        """points_xy: Nx2 pixel coords in the rectified left image. Returns depth in meters, NaN if invalid."""
        h, w = disparity.shape
        depths = np.full(len(points_xy), np.nan, dtype=np.float32)
        xi = np.round(points_xy[:, 0]).astype(np.int64)
        yi = np.round(points_xy[:, 1]).astype(np.int64)
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        d = np.full(len(points_xy), -1.0, dtype=np.float32)
        d[valid] = disparity[yi[valid], xi[valid]]
        ok = valid & (d > 0.5)
        depths[ok] = self.rectifier.fx_rect * self.rectifier.baseline / d[ok]
        return depths
