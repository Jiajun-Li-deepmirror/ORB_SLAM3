import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import octomap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splg_slam.config import load_config
from splg_slam.data.loader import dataset_dir, dataset_module, right_image_path
from splg_slam.geometry.stereo import StereoDepthEstimator, StereoRectifier
from splg_slam.map.io import load_map
from splg_slam.mapping.octomap_utils import insert_keyframe_into_octree


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
    parser.add_argument(
        "--min_depth_m", type=float, default=None,
        help="rejects depth points below this range before insertion - dense stereo saturates "
        "at its matcher's max-disparity limit for pixels it fails to match, reporting a "
        "hard-clamped near-camera depth instead of failing outright, which otherwise gets "
        "ray-cast in as a spurious OCCUPIED voxel sitting on the camera's own trajectory. "
        "Default (omit this flag): auto-derived from the rig's own calibration as "
        "fx_rect*baseline/max_disparity - the exact depth the matcher's disparity search "
        "range saturates at, not an empirical guess (verified: matches the observed clamped "
        "value to 3 decimal places on the EuRoC rig).",
    )
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

    if args.min_depth_m is not None:
        min_depth_m = args.min_depth_m
    else:
        max_disparity = cfg.stereo.min_disp + cfg.stereo.num_disp - 1
        min_depth_m = rectifier.fx_rect * rectifier.baseline / max_disparity
        print(f"Auto-derived min_depth_m={min_depth_m:.4f} from fx_rect={rectifier.fx_rect:.2f}, baseline={rectifier.baseline:.4f}, max_disparity={max_disparity}")

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

        pixel_xy = insert_keyframe_into_octree(
            tree, img_l, img_r, kf.pose_cw, rectifier, depth_est, pixel_xy,
            args.stride, args.max_depth_m, args.outlier_nb_neighbors, args.outlier_std_ratio, r_align,
            min_depth_m=min_depth_m,
        )
        n_used += 1

        if n_used % 50 == 0:
            print(f"  [{count + 1}/{len(kf_ids)}] inserted, tree nodes={tree.calcNumNodes()}")

    tree.updateInnerOccupancy()
    print(f"Final octree: {tree.calcNumNodes()} nodes from {n_used} keyframes")
    tree.writeBinary(args.out_path.encode())
    print(f"Saved to {args.out_path}")


if __name__ == "__main__":
    main()
