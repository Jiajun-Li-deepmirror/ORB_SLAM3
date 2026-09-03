import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splg_slam.config import load_config
from splg_slam.data.euroc import load_stereo_rig
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.relocalizer import Relocalizer
from splg_slam.map.io import load_map
from splg_slam.mapping.map_merge import register_map_b_into_a


def main():
    parser = argparse.ArgumentParser(
        description="Approach A: register MH02-05's already-built maps against a reference map "
        "via cross-map Relocalizer-based localization, then transform and merge each dataset's "
        "pre-built dense point cloud into one. Defaults to mh01_imu_map (gravity-aligned, via "
        "IMU tight coupling) as the reference, NOT the plain vision-only mh01_map - registering "
        "against a non-gravity-aligned reference silently produces a non-gravity-aligned merge "
        "with no indication in the output filename (this bit us once already: the original "
        "merged_registration.ply was built against mh01_map and is stale/non-gravity-aligned)."
    )
    parser.add_argument("--voxel_size", type=float, default=0.03)
    parser.add_argument("--sample_stride", type=int, default=5, help="use every Nth keyframe of the target map as a cross-localization query")
    parser.add_argument("--ref_config", type=str, default="configs/euroc_mh01_imu.yaml")
    parser.add_argument("--ref_map", type=str, default="results/mh01_imu_map/map.pkl")
    parser.add_argument(
        "--ref_dense_cloud", type=str, default="results/mh01_imu_map/dense_cloud.ply",
        help="MUST be built (via dense_map.py) from --ref_map's own poses, not reused from a "
        "different map's dense cloud - mh01_map vs mh01_imu_map differ by a real, non-negligible "
        "world-frame ROTATION (gravity alignment), not just small per-keyframe pose noise, so "
        "reusing mh01_map's cloud here would silently reintroduce the exact non-gravity-aligned "
        "mismatch this change is meant to fix.",
    )
    parser.add_argument("--out", type=str, default="results/merged_registration_gravity_aligned.ply")
    args = parser.parse_args()

    other_names = ["mh02", "mh03", "mh04", "mh05"]

    ref_cfg = load_config(args.ref_config)
    rig = load_stereo_rig(ref_cfg.dataset.mav0_dir)
    rectifier = StereoRectifier(rig)

    map_a = load_map(args.ref_map)
    print(f"Reference map: {args.ref_map} ({len(map_a.keyframes)} keyframes)")
    relocalizer = Relocalizer(map_a, rectifier, ref_cfg)

    merged = o3d.io.read_point_cloud(args.ref_dense_cloud)
    print(f"reference ({args.ref_dense_cloud}): {len(merged.points)} points, identity transform")

    transforms = {"ref": np.eye(4)}
    for name in other_names:
        map_b = load_map(f"results/{name}_map/map.pkl")
        result = register_map_b_into_a(relocalizer, map_b, rectifier, sample_stride=args.sample_stride)
        if not result["accepted"]:
            print(f"{name}: REGISTRATION FAILED ({result.get('reason')}, {result.get('num_candidates', 0)} candidates)")
            continue

        t_wa_wb = result["T_worldA_worldB"]
        transforms[name] = t_wa_wb
        print(
            f"{name}: registered ({result['cluster_size']}/{result['num_candidates']} agreeing matches), "
            f"translation={t_wa_wb[:3, 3]}, "
        )

        cloud_b = o3d.io.read_point_cloud(f"results/{name}_map/dense_cloud.ply")
        cloud_b.transform(t_wa_wb)
        merged += cloud_b
        print(f"  -> merged, running total before downsample: {len(merged.points)} points")
        merged = merged.voxel_down_sample(args.voxel_size)
        print(f"  -> after downsample: {len(merged.points)} points")

    print(f"\nFinal merged cloud: {len(merged.points)} points")
    o3d.io.write_point_cloud(args.out, merged)
    print(f"Saved to {args.out}")

    np.savez(args.out.replace(".ply", "_transforms.npz"), **transforms)
    print(f"Saved per-dataset transforms to {args.out.replace('.ply', '_transforms.npz')}")


if __name__ == "__main__":
    main()
