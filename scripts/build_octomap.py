import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import octomap
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splg_slam.config import load_config
from splg_slam.data.loader import dataset_dir, dataset_module, right_image_path
from splg_slam.geometry.pose_utils import camera_center
from splg_slam.geometry.stereo import StereoDepthEstimator, StereoRectifier
from splg_slam.map.io import load_map


def main():
    parser = argparse.ArgumentParser(
        description="Build a real Octomap (ray-cast 3D occupancy octree: occupied/free/unknown, "
        "not just 'every point is occupied') from per-keyframe dense stereo depth + camera "
        "poses. Unlike the point-cloud fusion in dense_map.py, this needs each keyframe's own "
        "(points, sensor origin) pair - insertPointCloud() ray-casts from the origin to each "
        "point, marking the traversed voxels FREE and the endpoint OCCUPIED, which a plain "
        "point cloud can't distinguish (a point cloud only ever says 'something was seen here', "
        "never 'this space was seen through and is empty')."
    )
    parser.add_argument("config", type=str)
    parser.add_argument("map_path", type=str)
    parser.add_argument("out_path", type=str, help="output octomap path (.bt)")
    parser.add_argument("--resolution", type=float, default=0.1, help="octree leaf voxel size in meters")
    parser.add_argument("--stride", type=int, default=3, help="use every Nth pixel per image")
    parser.add_argument("--max_depth_m", type=float, default=6.0)
    parser.add_argument("--every_n_kf", type=int, default=1)
    parser.add_argument(
        "--outlier_nb_neighbors", type=int, default=20,
        help="per-keyframe statistical outlier removal before ray-casting insertion (same "
        "technique dense_map.py applies to the merged point cloud) - without this, raw dense "
        "stereo noise gets inserted as real OCCUPIED voxels, not just noisy points sitting "
        "unused in a point cloud. 0 disables.",
    )
    parser.add_argument("--outlier_std_ratio", type=float, default=1.5)
    parser.add_argument(
        "--gravity_rotation", type=str, default=None,
        help="path to a saved gravity-alignment rotation .npy (3x3) - applied to both points "
        "and each keyframe's camera-center origin before insertion, so the resulting octomap "
        "has a true vertical Z axis (see gravity_align_compare.py). Omit to keep the map's own "
        "(possibly not gravity-aligned) frame as-is.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    world_map = load_map(args.map_path)
    rig = dataset_module(cfg).load_stereo_rig(dataset_dir(cfg))
    rectifier = StereoRectifier(rig)
    depth_est = StereoDepthEstimator(
        rectifier, min_disp=cfg.stereo.min_disp, num_disp=cfg.stereo.num_disp, block_size=cfg.stereo.block_size,
    )

    r_align = np.load(args.gravity_rotation) if args.gravity_rotation else None

    tree = octomap.OcTree(args.resolution)
    kf_ids = world_map.keyframe_ids_sorted()[:: args.every_n_kf]
    print(f"Building octomap ({args.resolution}m resolution) from {len(kf_ids)} keyframes...")

    pixel_xy = None
    n_used = 0
    for count, kf_id in enumerate(kf_ids):
        kf = world_map.keyframes[kf_id]
        if kf.image_path is None:
            continue
        left_path = Path(kf.image_path)
        right_path = right_image_path(left_path, cfg)
        img_l = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if img_l is None or img_r is None:
            continue

        rect_l, rect_r = rectifier.rectify(img_l, img_r)
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
        origin = camera_center(kf.pose_cw)

        if r_align is not None:
            pts_world = (r_align @ pts_world.T).T
            origin = r_align @ origin

        if args.outlier_nb_neighbors > 0 and len(pts_world) > args.outlier_nb_neighbors:
            frame_cloud = o3d.geometry.PointCloud()
            frame_cloud.points = o3d.utility.Vector3dVector(pts_world)
            frame_cloud, _ = frame_cloud.remove_statistical_outlier(
                nb_neighbors=args.outlier_nb_neighbors, std_ratio=args.outlier_std_ratio,
            )
            pts_world = np.asarray(frame_cloud.points)
        if len(pts_world) == 0:
            continue

        tree.insertPointCloud(pts_world.astype(np.float64), origin.astype(np.float64), maxrange=args.max_depth_m, lazy_eval=True)
        n_used += 1

        if n_used % 50 == 0:
            print(f"  [{count + 1}/{len(kf_ids)}] inserted, tree nodes={tree.calcNumNodes()}")

    tree.updateInnerOccupancy()
    print(f"Final octree: {tree.calcNumNodes()} nodes from {n_used} keyframes")
    tree.writeBinary(args.out_path.encode())
    print(f"Saved to {args.out_path}")


if __name__ == "__main__":
    main()
