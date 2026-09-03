import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splg_slam.config import load_config
from splg_slam.data.loader import dataset_dir, dataset_module, right_image_path
from splg_slam.geometry.stereo import StereoDepthEstimator, StereoRectifier
from splg_slam.map.io import load_map


def main():
    parser = argparse.ArgumentParser(
        description="Fuse per-keyframe dense stereo depth (standing in for a RealSense depth "
        "stream) into a single world-frame point cloud, using the already-optimized keyframe "
        "poses from a built map. Backproject -> transform by pose_wc() -> voxel filter, same "
        "mechanism that will later run on real RealSense depth images."
    )
    parser.add_argument("config", type=str)
    parser.add_argument("map_path", type=str)
    parser.add_argument("out_path", type=str, help="output point cloud path (.ply or .pcd)")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="meters, voxel filter resolution")
    parser.add_argument("--stride", type=int, default=2, help="use every Nth pixel per image (speed/density tradeoff)")
    parser.add_argument(
        "--max_depth_m", type=float, default=6.0,
        help="narrow-baseline stereo (EuRoC's ~11cm baseline) has depth error that grows "
        "quadratically with range - a small disparity quantization error translates to a huge "
        "depth swing past a few meters, showing up as radial streaking noise fanning out from "
        "each keyframe. Keep this tight; a real depth sensor (e.g. RealSense) won't need it as low.",
    )
    parser.add_argument("--every_n_kf", type=int, default=1, help="use every Nth keyframe")
    parser.add_argument(
        "--outlier_nb_neighbors", type=int, default=20,
        help="statistical outlier removal: number of neighbors to check per point (0 disables)",
    )
    parser.add_argument("--outlier_std_ratio", type=float, default=1.5, help="statistical outlier removal threshold")
    args = parser.parse_args()

    cfg = load_config(args.config)
    world_map = load_map(args.map_path)
    rig = dataset_module(cfg).load_stereo_rig(dataset_dir(cfg))
    rectifier = StereoRectifier(rig)
    depth_est = StereoDepthEstimator(
        rectifier, min_disp=cfg.stereo.min_disp, num_disp=cfg.stereo.num_disp, block_size=cfg.stereo.block_size,
    )

    kf_ids = world_map.keyframe_ids_sorted()[:: args.every_n_kf]
    print(f"Fusing dense depth from {len(kf_ids)} keyframes (pixel stride={args.stride})...")

    pixel_xy = None
    global_cloud = o3d.geometry.PointCloud()
    n_used = 0

    for count, kf_id in enumerate(kf_ids):
        kf = world_map.keyframes[kf_id]
        if kf.image_path is None:
            continue
        left_path = Path(kf.image_path)
        right_path = right_image_path(left_path, cfg)
        img_l = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        img_l_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        if img_l is None or img_r is None:
            continue

        rect_l, rect_r = rectifier.rectify(img_l, img_r)
        rect_l_bgr, _ = rectifier.rectify(img_l_bgr, img_l_bgr)
        disp = depth_est.compute_disparity(rect_l, rect_r)

        if pixel_xy is None:
            h, w = disp.shape
            ys, xs = np.mgrid[0:h:args.stride, 0:w:args.stride]
            pixel_xy = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

        depths = depth_est.depths_at_points(disp, pixel_xy)
        valid = np.isfinite(depths) & (depths > 0) & (depths < args.max_depth_m)
        if not valid.any():
            continue

        pts_cam = rectifier.backproject(pixel_xy[valid], depths[valid])
        pose_wc = kf.pose_wc()
        pts_world = (pose_wc[:3, :3] @ pts_cam.T).T + pose_wc[:3, 3]

        xi = pixel_xy[valid, 0].astype(np.int64)
        yi = pixel_xy[valid, 1].astype(np.int64)
        colors = rect_l_bgr[yi, xi][:, ::-1].astype(np.float64) / 255.0  # BGR -> RGB

        frame_cloud = o3d.geometry.PointCloud()
        frame_cloud.points = o3d.utility.Vector3dVector(pts_world)
        frame_cloud.colors = o3d.utility.Vector3dVector(colors)
        global_cloud += frame_cloud
        n_used += 1

        if n_used % 10 == 0:
            global_cloud = global_cloud.voxel_down_sample(args.voxel_size)
            print(f"  [{count + 1}/{len(kf_ids)}] fused, {len(global_cloud.points)} points after downsample")

    global_cloud = global_cloud.voxel_down_sample(args.voxel_size)
    if args.outlier_nb_neighbors > 0:
        n_before = len(global_cloud.points)
        global_cloud, _ = global_cloud.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors, std_ratio=args.outlier_std_ratio,
        )
        print(f"Statistical outlier removal: {n_before} -> {len(global_cloud.points)} points")

    print(f"Final point cloud: {len(global_cloud.points)} points from {n_used} keyframes")
    o3d.io.write_point_cloud(args.out_path, global_cloud)
    print(f"Saved to {args.out_path}")


if __name__ == "__main__":
    main()
