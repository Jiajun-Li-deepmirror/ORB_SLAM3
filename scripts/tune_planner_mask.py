import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import open3d as o3d
from scipy import ndimage

from splg_slam.planning.planner_2d import build_cost_and_blocked


def analyze(elevation: dict, block_cost_threshold: float, robot_radius_m: float, inflate_radius_m: float, close_radius_m: float) -> dict:
    """Builds the exact same blocked/cost mask plan_2d/DStarLite2D would search over for this
    configuration, and reports its connectivity: number of disconnected free-space
    components, and what fraction of all free space the single largest one covers. This is
    the number that matters for an online replanning loop - a query landing in a small
    fragment separate from the goal's component fails outright, no matter how good the
    planner is."""
    blocked, _ = build_cost_and_blocked(
        elevation, block_cost_threshold=block_cost_threshold, robot_radius_m=robot_radius_m,
        inflate_radius_m=inflate_radius_m, close_radius_m=close_radius_m,
    )
    free = ~blocked
    lbl, n = ndimage.label(free, structure=np.ones((3, 3)))
    if n == 0:
        return {"n_components": 0, "largest_frac": 0.0, "total_free": 0}
    sizes = ndimage.sum(free, lbl, range(1, n + 1))
    return {"n_components": int(n), "largest_frac": float(sizes.max() / free.sum()), "total_free": int(free.sum())}


def rebuild_with_smoothing(cloud_path: str, resolution: float, ground_percentile: float, smooth_size: int) -> dict:
    """Rebuilds the elevation map from the source dense cloud with a NaN-aware median filter
    on the height grid before slope/step/roughness are computed from it - fixes noise at the
    source instead of papering over its symptom (fragmented traversability) downstream.
    Reimplements build_elevation_map.py's pipeline rather than importing it (that script has
    no importable module boundary) - keep in sync if that script's math changes."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_elevation_map import build_elevation_map, compute_traversability, smooth_height

    cloud = o3d.io.read_point_cloud(cloud_path)
    points = np.asarray(cloud.points)
    elev = build_elevation_map(points, resolution, ground_percentile)
    if smooth_size > 1:
        elev["height"] = smooth_height(elev["height"], smooth_size)
    trav = compute_traversability(elev)
    return {
        "height": elev["height"], "cost": trav["cost"], "traversable": trav["traversable"],
        "x_min": elev["x_min"], "y_min": elev["y_min"], "resolution": elev["resolution"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline tuning pass: pick ONE fixed (close_radius_m, block_cost_threshold, "
        "smooth_size) configuration for a static elevation map, instead of an online escalating "
        "fallback ladder. Rationale: closing and smoothing fix the map's actual noise-driven "
        "fragmentation and should be decided once, offline, before deployment; only "
        "block_cost_threshold is a real accuracy/permissiveness trade-off worth leaving as a "
        "single tunable safety margin. Doing this at query time instead (try stricter, retry "
        "looser, retry looser still...) multiplies cost per failed query - severely so for an "
        "incremental planner (D* Lite) where each distinct mask needs its own from-scratch "
        "persistent search (observed: 92s for one query across 4 fallback stages)."
    )
    parser.add_argument("elevation_npz", type=str)
    parser.add_argument("--robot_radius_m", type=float, default=1.0)
    parser.add_argument("--inflate_radius_m", type=float, default=1.5)
    parser.add_argument("--target_largest_frac", type=float, default=0.97, help="stop searching once the largest connected component covers at least this fraction of all free space")
    parser.add_argument("--close_radii", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.4])
    parser.add_argument("--block_cost_thresholds", type=float, nargs="+", default=[0.85, 0.9, 0.95])
    parser.add_argument("--dense_cloud_path", type=str, default=None, help="if given, and closing+threshold alone can't reach the target, rebuilds the elevation map with --smooth_sizes to fix noise at the source")
    parser.add_argument("--smooth_sizes", type=int, nargs="+", default=[3, 5, 7, 9])
    parser.add_argument("--resolution", type=float, default=None, help="only needed with --dense_cloud_path; defaults to the input elevation map's own resolution")
    parser.add_argument("--ground_percentile", type=float, default=5.0)
    args = parser.parse_args()

    data = np.load(args.elevation_npz)
    elevation = {k: data[k] for k in ["height", "cost", "traversable", "x_min", "y_min", "resolution"]}
    elevation["x_min"], elevation["y_min"], elevation["resolution"] = (
        float(elevation["x_min"]), float(elevation["y_min"]), float(elevation["resolution"]),
    )

    print(f"=== Pass 1: closing sweep at default block_cost_threshold={args.block_cost_thresholds[0]} ===")
    best = None
    for close_r in args.close_radii:
        stats = analyze(elevation, args.block_cost_thresholds[0], args.robot_radius_m, args.inflate_radius_m, close_r)
        hit = stats["largest_frac"] >= args.target_largest_frac
        print(f"  close_radius_m={close_r:.1f}: n_components={stats['n_components']:5d}, largest_frac={stats['largest_frac']:.3f} {'<- TARGET MET' if hit else ''}")
        if hit and best is None:
            best = {"close_radius_m": close_r, "block_cost_threshold": args.block_cost_thresholds[0], "smooth_size": None, **stats}

    if best is not None:
        print(f"\nRECOMMENDATION (closing alone suffices): {best}")
        return

    print(f"\n=== Pass 2: closing did not reach target alone - sweeping block_cost_threshold too ===")
    for bct in args.block_cost_thresholds:
        for close_r in args.close_radii:
            stats = analyze(elevation, bct, args.robot_radius_m, args.inflate_radius_m, close_r)
            hit = stats["largest_frac"] >= args.target_largest_frac
            print(f"  block_cost_threshold={bct:.2f}, close_radius_m={close_r:.1f}: n_components={stats['n_components']:5d}, largest_frac={stats['largest_frac']:.3f} {'<- TARGET MET' if hit else ''}")
            if hit and best is None:
                best = {"close_radius_m": close_r, "block_cost_threshold": bct, "smooth_size": None, **stats}
        if best is not None:
            break

    if best is not None:
        print(f"\nRECOMMENDATION (needed a looser block_cost_threshold too): {best}")
        return

    if not args.dense_cloud_path:
        print(
            "\nNo (close_radius_m, block_cost_threshold) combination tried reached the target, and "
            "no --dense_cloud_path was given to try rebuilding with --smooth_sizes. Either widen "
            "--close_radii / --block_cost_thresholds, lower --target_largest_frac, or pass "
            "--dense_cloud_path to try fixing the underlying noise instead."
        )
        return

    print(f"\n=== Pass 3: rebuilding from {args.dense_cloud_path} with height smoothing ===")
    resolution = args.resolution or elevation["resolution"]
    for smooth_size in args.smooth_sizes:
        rebuilt = rebuild_with_smoothing(args.dense_cloud_path, resolution, args.ground_percentile, smooth_size)
        for close_r in args.close_radii:
            stats = analyze(rebuilt, args.block_cost_thresholds[0], args.robot_radius_m, args.inflate_radius_m, close_r)
            hit = stats["largest_frac"] >= args.target_largest_frac
            print(f"  smooth_size={smooth_size}, close_radius_m={close_r:.1f}: n_components={stats['n_components']:5d}, largest_frac={stats['largest_frac']:.3f} {'<- TARGET MET' if hit else ''}")
            if hit and best is None:
                best = {"close_radius_m": close_r, "block_cost_threshold": args.block_cost_thresholds[0], "smooth_size": smooth_size, **stats}
        if best is not None:
            break

    if best is not None:
        print(f"\nRECOMMENDATION (needed a height-smoothed rebuild): {best}")
        print(f"Rebuild your elevation map with: python3 scripts/build_elevation_map.py {args.dense_cloud_path} --smooth_size {best['smooth_size']} ...")
    else:
        print(f"\nTarget not reached by any combination tried. Consider a lower --target_largest_frac, or accept --allow_unknown for the remaining gap (a real, deliberate risk trade-off, not a tuning fix).")


if __name__ == "__main__":
    main()
