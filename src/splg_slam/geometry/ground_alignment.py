import numpy as np
import open3d as o3d

from splg_slam.mapping.imu_preintegration import rotation_aligning


def ransac_ground_alignment(
    points: np.ndarray, distance_threshold: float = 0.03, ransac_n: int = 3, num_iterations: int = 2000,
) -> tuple[np.ndarray, dict]:
    """Estimates a gravity-alignment rotation from point-cloud geometry alone (no IMU): finds
    the dominant plane via RANSAC (Open3D's segment_plane) and rotates its normal onto world
    +Z. The dominant plane in an indoor scene is usually the floor, but isn't guaranteed to be
    (a large wall or ceiling structure could be bigger) - this is a purely geometric guess,
    unlike the IMU-registration route which is grounded in an actual gravity measurement.

    Disambiguates which way the normal should point (a plane has no inherent orientation) by
    checking that non-floor points end up above the floor's inlier points once rotated - a
    room has more stuff above the ground than below it.

    Returns (R (3x3), diagnostics: {num_inliers, num_points, inlier_ratio})."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inlier_idx = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    a, b, c, _ = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    r_align = rotation_aligning(normal, np.array([0.0, 0.0, 1.0]))
    rotated = (r_align @ points.T).T
    inlier_mask = np.zeros(len(points), dtype=bool)
    inlier_mask[inlier_idx] = True
    if np.median(rotated[inlier_mask, 2]) > np.median(rotated[~inlier_mask, 2]):
        # picked the wrong sign - most of the scene ended up BELOW the "floor", flip it
        r_align = rotation_aligning(-normal, np.array([0.0, 0.0, 1.0]))

    return r_align, {
        "num_inliers": len(inlier_idx),
        "num_points": len(points),
        "inlier_ratio": len(inlier_idx) / max(len(points), 1),
    }
