import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import octomap

from splg_slam.planning.dense_grid_3d import extract_dense_grid


def main():
    parser = argparse.ArgumentParser(
        description="Converts a sparse Octomap (.bt/.ot) into a fixed dense 3D grid (.npz), "
        "the same one-time offline step build_elevation_map.py does for a 2D dense-stereo "
        "point cloud. The planner (plan_3d_dense / DStarLite3D) then loads this file once and "
        "reuses it across every replan - the octree itself is never touched again after this "
        "runs, same as plan_2d never re-touches the original point cloud."
    )
    parser.add_argument("octomap_path", type=str, help=".bt or .ot octomap file")
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--resolution", type=float, default=None, help="defaults to the octomap's own native leaf resolution")
    parser.add_argument(
        "--bbx_min", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="bound the extracted region - required for a map too large to hold densely in "
        "full (an outdoor, KITTI-scale octomap). Omit for a room/building-scale map (EuRoC) "
        "to extract the whole thing.",
    )
    parser.add_argument("--bbx_max", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    args = parser.parse_args()

    out_path = args.out_path or str(Path(args.octomap_path).with_suffix("")) + "_grid.npz"

    tree = octomap.OcTree(str(args.octomap_path).encode())
    resolution = args.resolution or tree.getResolution()
    bbx_min = np.array(args.bbx_min) if args.bbx_min else None
    bbx_max = np.array(args.bbx_max) if args.bbx_max else None

    tmin, tmax = tree.getMetricMin(), tree.getMetricMax()
    print(f"Octomap bounds: min={tmin} max={tmax}, native resolution={tree.getResolution()}m")
    print(f"Extracting dense grid at {resolution}m resolution over {'the whole map' if bbx_min is None else (bbx_min, bbx_max)}...")

    occupied, unknown, origin = extract_dense_grid(tree, resolution, bbx_min, bbx_max)
    print(
        f"Grid shape {occupied.shape} ({occupied.size:,} cells): "
        f"{occupied.sum():,} occupied ({100*occupied.sum()/occupied.size:.1f}%), "
        f"{unknown.sum():,} unknown ({100*unknown.sum()/occupied.size:.1f}%), "
        f"{occupied.size - occupied.sum() - unknown.sum():,} free"
    )

    np.savez(out_path, occupied=occupied, unknown=unknown, origin=origin, resolution=resolution)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
