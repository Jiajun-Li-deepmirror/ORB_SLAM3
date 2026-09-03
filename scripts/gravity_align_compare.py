import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import open3d as o3d

from splg_slam.config import load_config
from splg_slam.data.euroc import load_stereo_rig
from splg_slam.geometry.ground_alignment import ransac_ground_alignment
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.relocalizer import Relocalizer
from splg_slam.map.io import load_map
from splg_slam.mapping.map_merge import register_map_b_into_a


def rotation_angle_diff_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    r_rel = r1.T @ r2
    cos_angle = np.clip((np.trace(r_rel) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def main():
    target_map_name = "mh_multi_map"  # Approach B's concatenated map
    dense_cloud_path = f"results/{target_map_name}/dense_cloud.ply"

    print("=== Method 1: IMU-registration alignment (mh01_imu_map as gravity-aligned reference) ===")
    imu_cfg = load_config("configs/euroc_mh01_imu.yaml")
    rig = load_stereo_rig(imu_cfg.dataset.mav0_dir)
    rectifier = StereoRectifier(rig)
    map_a = load_map("results/mh01_imu_map/map.pkl")
    relocalizer = Relocalizer(map_a, rectifier, imu_cfg)

    map_b = load_map(f"results/{target_map_name}/map.pkl")
    result = register_map_b_into_a(relocalizer, map_b, rectifier, sample_stride=10)
    if not result["accepted"]:
        print(f"IMU-registration alignment FAILED: {result}")
        r_imu = None
    else:
        t_wa_wb = result["T_worldA_worldB"]
        r_imu = t_wa_wb[:3, :3]
        print(
            f"Registered ({result['cluster_size']}/{result['num_candidates']} agreeing matches), "
            f"translation={t_wa_wb[:3, 3]}"
        )

    print("\n=== Method 2: RANSAC ground-plane alignment (geometry only, no IMU) ===")
    cloud = o3d.io.read_point_cloud(dense_cloud_path)
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    r_ransac, diag = ransac_ground_alignment(points)
    print(f"RANSAC plane: {diag['num_inliers']}/{diag['num_points']} inliers ({diag['inlier_ratio']:.1%})")

    if r_imu is not None:
        angle_diff = rotation_angle_diff_deg(r_imu, r_ransac)
        print(f"\nAngle between the two methods' rotations: {angle_diff:.1f} deg")
        up_imu = r_imu @ np.array([0, 0, 1.0])
        up_ransac = r_ransac @ np.array([0, 0, 1.0])
        print(f"'up' direction (in the multi-map's own original frame) implied by each method:")
        print(f"  IMU-registration: {up_imu}")
        print(f"  RANSAC ground-plane: {up_ransac}")

    for name, r in [("imu_registration", r_imu), ("ransac_ground_plane", r_ransac)]:
        if r is None:
            continue
        pts_aligned = (r @ points.T).T
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts_aligned)
        pc.colors = o3d.utility.Vector3dVector(colors)
        out_path = f"results/{target_map_name}/dense_cloud_gravity_{name}.ply"
        o3d.io.write_point_cloud(out_path, pc)
        print(f"Saved {name}-aligned cloud to {out_path}")
        np.save(f"results/{target_map_name}/gravity_rotation_{name}.npy", r)
        print(f"Saved rotation matrix to results/{target_map_name}/gravity_rotation_{name}.npy")


if __name__ == "__main__":
    main()
