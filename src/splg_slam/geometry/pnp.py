import cv2
import numpy as np


def solve_pnp_ransac(
    object_points: np.ndarray,
    image_points: np.ndarray,
    k: np.ndarray,
    reproj_threshold_px: float = 3.0,
    iterations_count: int = 200,
    confidence: float = 0.999,
):
    """object_points: Nx3 (world/reference frame), image_points: Nx2 pixel coords.

    Returns (success, pose_cw [4x4, world->camera], inlier_mask [N bool]).
    """
    if len(object_points) < 6:
        return False, None, None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points.astype(np.float64),
        image_points.astype(np.float64),
        k, None,
        reprojectionError=reproj_threshold_px,
        iterationsCount=iterations_count,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 6:
        return False, None, None

    inlier_idx = inliers.ravel()
    ok, rvec, tvec = cv2.solvePnP(
        object_points[inlier_idx].astype(np.float64),
        image_points[inlier_idx].astype(np.float64),
        k, None,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return False, None, None

    r_mat, _ = cv2.Rodrigues(rvec)
    pose_cw = np.eye(4)
    pose_cw[:3, :3] = r_mat
    pose_cw[:3, 3] = tvec.ravel()

    inlier_mask = np.zeros(len(object_points), dtype=bool)
    inlier_mask[inlier_idx] = True
    return True, pose_cw, inlier_mask
