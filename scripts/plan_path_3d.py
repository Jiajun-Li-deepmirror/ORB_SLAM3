import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import octomap

from splg_slam.planning.planner_3d import OccupancyChecker, astar_3d, build_edt, make_segment_free_check
from splg_slam.planning.trajectory import build_trajectory


def main():
    parser = argparse.ArgumentParser(
        description="3D shortest-path planning on an Octomap for drone-style navigation, "
        "then turned into an executable trajectory: safety-inflated obstacle clearance "
        "(DynamicEDTOctomap), visibility-based shortcutting + Chaikin smoothing to remove "
        "voxel-grid zig-zag, and a curvature/acceleration-limited velocity profile."
    )
    parser.add_argument("octomap_path", type=str, help=".bt or .ot octomap file")
    parser.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"), help="world meters")
    parser.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--allow_unknown", action="store_true", help="allow flying through never-observed space (default: blocked, safer)")
    parser.add_argument("--robot_radius_m", type=float, default=0.3, help="hard clearance: voxels closer than this to any obstacle become impassable")
    parser.add_argument("--inflate_radius_m", type=float, default=0.0, help="soft buffer beyond robot_radius_m adding cost (needs --cost_weight > 0 to have any effect). 0 disables - default keeps the search a pure/optimal shortest path.")
    parser.add_argument("--cost_weight", type=float, default=0.0, help="how strongly to penalize flying near the inflation boundary vs pure distance (0 = provably-shortest path in the safety-inflated free space)")
    parser.add_argument("--edt_padding_m", type=float, default=None, help="EDT is computed only in a box around start/goal padded by this much (default: max(10, 1.5x start-goal distance)) - the whole-map EDT does not finish in reasonable time on a large outdoor map")
    parser.add_argument("--shortcut_iters", type=int, default=3)
    parser.add_argument("--chaikin_iters", type=int, default=3)
    parser.add_argument("--resample_ds", type=float, default=0.2, help="arc-length spacing (m) of the final trajectory samples")
    parser.add_argument("--max_vel", type=float, default=2.0, help="m/s")
    parser.add_argument("--max_accel", type=float, default=1.0, help="m/s^2, applied to both acceleration and braking")
    parser.add_argument("--max_lateral_accel", type=float, default=2.0, help="m/s^2, caps cornering speed via v <= sqrt(a_lat_max / curvature)")
    parser.add_argument("--out", type=str, default=None, help="output path viewer HTML (reuses view_octomap.py's renderer, with the path overlaid)")
    args = parser.parse_args()

    tree = octomap.OcTree(str(args.octomap_path).encode())
    resolution = tree.getResolution()
    checker = OccupancyChecker(tree)

    start = np.array(args.start, dtype=np.float64)
    goal = np.array(args.goal, dtype=np.float64)

    edt_padding_m = args.edt_padding_m if args.edt_padding_m is not None else max(10.0, 1.5 * float(np.linalg.norm(goal - start)))
    tree_min, tree_max = np.array(tree.getMetricMin()), np.array(tree.getMetricMax())
    bbx_min = np.maximum(np.minimum(start, goal) - edt_padding_m, tree_min)
    bbx_max = np.minimum(np.maximum(start, goal) + edt_padding_m, tree_max)
    edt_max_dist = max(0.5, args.robot_radius_m + args.inflate_radius_m + resolution)
    build_edt(tree, bbx_min, bbx_max, edt_max_dist, treat_unknown_as_occupied=not args.allow_unknown)
    print(
        f"EDT built over box {bbx_min.round(1).tolist()} .. {bbx_max.round(1).tolist()} "
        f"(padding={edt_padding_m:.1f}m, max_dist={edt_max_dist:.2f}m)"
    )

    origin = tree_min

    def world_to_idx(p):
        return tuple(int(round(v)) for v in (np.array(p) - origin) / resolution)

    start_idx = world_to_idx(start)
    goal_idx = world_to_idx(goal)
    print(f"start world={tuple(args.start)} -> voxel {start_idx}; goal world={tuple(args.goal)} -> voxel {goal_idx}")
    print(f"octomap resolution={resolution}m, bounds min={tree.getMetricMin()} max={tree.getMetricMax()}")

    for name, p in [("start", start), ("goal", goal)]:
        state = checker.state(p)
        if state != "free":
            print(f"ERROR: {name} point is {state}, not free space - pick a different point")
            return
        clearance = checker.clearance(p)
        if clearance < args.robot_radius_m:
            print(f"ERROR: {name} point is only {clearance:.2f}m from an obstacle, less than --robot_radius_m={args.robot_radius_m} - pick a different point")
            return

    path_idx, total_dist_cells = astar_3d(
        checker, resolution, start_idx, goal_idx, origin, args.allow_unknown,
        args.robot_radius_m, args.inflate_radius_m, args.cost_weight,
    )
    if path_idx is None:
        print("NO PATH FOUND")
        return

    world_path = np.array([origin + np.array(idx) * resolution for idx in path_idx])
    path_len_m = sum(
        float(np.linalg.norm(world_path[i + 1] - world_path[i])) for i in range(len(world_path) - 1)
    )
    straight_line_m = float(np.linalg.norm(goal - start))
    print(
        f"A* path found: {len(path_idx)} voxels, {path_len_m:.2f}m flight path "
        f"(straight-line distance {straight_line_m:.2f}m, detour factor {path_len_m / max(straight_line_m, 1e-6):.2f}x)"
    )

    is_segment_free = make_segment_free_check(checker, resolution, args.robot_radius_m, args.allow_unknown)
    traj = build_trajectory(
        world_path, is_segment_free,
        shortcut_iters=args.shortcut_iters, chaikin_iters=args.chaikin_iters, resample_ds=args.resample_ds,
        v_max=args.max_vel, a_max=args.max_accel, a_lat_max=args.max_lateral_accel,
    )
    print(
        f"Trajectory: {len(path_idx)} A* voxels -> {len(traj['shortcut'])} shortcut waypoints -> "
        f"{len(traj['positions'])} trajectory samples @ {args.resample_ds}m spacing, "
        f"{traj['distance_m'][-1]:.2f}m, {traj['time_s'][-1]:.1f}s, peak v={traj['velocity_mps'].max():.2f}m/s"
    )

    out_path = args.out or str(Path(args.octomap_path).with_suffix("")) + "_3dpath.npz"
    np.savez(
        out_path,
        path_world=world_path, path_voxel=np.array(path_idx),
        trajectory_xyz=traj["positions"], trajectory_v=traj["velocity_mps"],
        trajectory_t=traj["time_s"], trajectory_s=traj["distance_m"],
    )
    print(f"Saved {out_path}")

    fig_path = str(Path(out_path).with_suffix("")) + "_trajectory.png"
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 1]})

    ax = axes[0]
    ax.plot(world_path[:, 0], world_path[:, 1], color="0.6", linestyle="--", linewidth=1, label="raw A* path (voxel zig-zag)")
    sc = ax.scatter(
        traj["positions"][:, 0], traj["positions"][:, 1], c=traj["velocity_mps"], cmap="turbo",
        vmin=0, vmax=max(args.max_vel, 1e-6), s=10, label="smoothed trajectory (color=speed)",
    )
    plt.colorbar(sc, ax=ax, fraction=0.03, label="speed (m/s)")
    ax.plot(start[0], start[1], "g^", markersize=12, label="start")
    ax.plot(goal[0], goal[1], "r*", markersize=14, label="goal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"3D trajectory (top-down view): {traj['distance_m'][-1]:.1f}m, {traj['time_s'][-1]:.1f}s")

    ax2 = axes[1]
    ax2.plot(traj["distance_m"], traj["velocity_mps"], color="tab:blue")
    ax2.set_xlabel("arc length (m)")
    ax2.set_ylabel("speed (m/s)")
    ax2.set_title("Velocity profile (curvature + accel/decel limited)")
    ax2.grid(alpha=0.3)
    ax2b = ax2.twiny()
    ax2b.plot(traj["time_s"], traj["velocity_mps"], alpha=0)
    ax2b.set_xlabel("time (s)")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=130)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
