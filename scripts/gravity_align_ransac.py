import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import open3d as o3d

from splg_slam.geometry.ground_alignment import ransac_ground_alignment


def main():
    parser = argparse.ArgumentParser(
        description="Dataset-agnostic RANSAC ground-plane gravity alignment for a dense point "
        "cloud: finds the dominant plane and rotates it onto world +Z, no IMU required. Reusable "
        "wherever a map was built vision-only (e.g. KITTI, which has no IMU stream in this "
        "odometry release) - unlike gravity_align_compare.py, which is a one-off script hardcoded "
        "to compare this against the EuRoC IMU-registration method."
    )
    parser.add_argument("cloud_path", type=str)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--distance_threshold", type=float, default=0.03, help="RANSAC plane-inlier distance in meters")
    args = parser.parse_args()

    out_path = args.out_path or str(Path(args.cloud_path).with_suffix("")) + "_gravity_ransac.ply"

    cloud = o3d.io.read_point_cloud(args.cloud_path)
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    print(f"Loaded {len(points)} points from {args.cloud_path}")

    r_align, diag = ransac_ground_alignment(points, distance_threshold=args.distance_threshold)
    print(f"RANSAC plane: {diag['num_inliers']}/{diag['num_points']} inliers ({diag['inlier_ratio']:.1%})")

    pts_aligned = (r_align @ points.T).T
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts_aligned)
    if len(colors):
        pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, pc)
    print(f"Saved gravity-aligned cloud to {out_path}")

    rot_path = str(Path(out_path).with_suffix("")) + "_rotation.npy"
    np.save(rot_path, r_align)
    print(f"Saved rotation matrix to {rot_path}")


if __name__ == "__main__":
    main()
