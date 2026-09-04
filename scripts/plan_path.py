import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from splg_slam.planning.planner_2d import (
    astar,
    close_free_space_gaps,
    grid_to_world,
    inflate_obstacles,
    make_segment_free_check,
    sample_height_at,
    world_to_grid,
)
from splg_slam.planning.trajectory import build_trajectory


def main():
    parser = argparse.ArgumentParser(
        description="A* global path planning on a traversability cost map produced by "
        "build_elevation_map.py, then turned into an executable trajectory: safety-inflated "
        "obstacle clearance, visibility-based shortcutting + Chaikin smoothing to remove grid "
        "zig-zag, and a curvature/acceleration-limited velocity profile."
    )
    parser.add_argument("elevation_npz", type=str)
    parser.add_argument("--start", type=float, nargs=2, required=True, metavar=("X", "Y"), help="world (map-frame) meters")
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--cost_weight", type=float, default=5.0, help="how strongly to penalize rough/steep cells vs pure distance")
    parser.add_argument("--allow_unknown", action="store_true", help="allow planning through never-observed cells (default: blocked)")
    parser.add_argument(
        "--block_cost_threshold", type=float, default=0.85,
        help="block a cell only once its continuous cost exceeds this, instead of the strict "
        "binary `traversable` flag (which requires slope AND step AND roughness to ALL pass "
        "at once - noisy dense-stereo reconstruction fails that AND every so often even on "
        "genuinely flat floor, fragmenting it into many disconnected traversable islands with "
        "no path between them). Set to >=1.0 or pass --use_strict_traversable to fall back to "
        "the binary flag.",
    )
    parser.add_argument("--use_strict_traversable", action="store_true", help="block on the binary `traversable` flag instead of --block_cost_threshold")
    parser.add_argument("--close_radius_m", type=float, default=0.0, help="morphological closing of the free-space mask: bridges gaps narrower than this (noisy dense-stereo dropouts, thin unobserved seams) before planning. 0 disables. Fixes traversable-region fragmentation directly instead of the blunter --allow_unknown.")
    parser.add_argument("--robot_radius_m", type=float, default=0.3, help="hard clearance: cells within this distance of any obstacle become impassable")
    parser.add_argument("--inflate_radius_m", type=float, default=0.5, help="soft buffer beyond robot_radius_m: extra cost ramping to 0, encouraging (not requiring) more clearance. 0 disables.")
    parser.add_argument("--shortcut_iters", type=int, default=3, help="visibility-shortcutting passes to remove grid zig-zag before smoothing")
    parser.add_argument("--chaikin_iters", type=int, default=3, help="Chaikin corner-cutting smoothing passes")
    parser.add_argument("--resample_ds", type=float, default=0.2, help="arc-length spacing (m) of the final trajectory samples")
    parser.add_argument("--max_vel", type=float, default=1.0, help="m/s")
    parser.add_argument("--max_accel", type=float, default=0.5, help="m/s^2, applied to both acceleration and braking")
    parser.add_argument("--max_lateral_accel", type=float, default=1.0, help="m/s^2, caps cornering speed via v <= sqrt(a_lat_max / curvature)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    data = np.load(args.elevation_npz)
    height, cost, traversable = data["height"], data["cost"], data["traversable"]
    x_min, y_min, resolution = float(data["x_min"]), float(data["y_min"]), float(data["resolution"])
    observed = ~np.isnan(height)

    if args.use_strict_traversable:
        blocked = ~traversable
    else:
        blocked = np.nan_to_num(cost, nan=1.0) > args.block_cost_threshold
    if not args.allow_unknown:
        blocked = blocked | ~observed

    if args.close_radius_m > 0:
        blocked = close_free_space_gaps(blocked, resolution, args.close_radius_m)

    blocked, cost = inflate_obstacles(blocked, cost, resolution, args.robot_radius_m, args.inflate_radius_m)

    start_grid = world_to_grid(args.start[0], args.start[1], x_min, y_min, resolution)
    goal_grid = world_to_grid(args.goal[0], args.goal[1], x_min, y_min, resolution)
    print(f"start world={tuple(args.start)} -> grid={start_grid}; goal world={tuple(args.goal)} -> grid={goal_grid}")

    ny, nx = height.shape
    for name, (gx, gy) in [("start", start_grid), ("goal", goal_grid)]:
        if not (0 <= gx < nx and 0 <= gy < ny):
            print(f"ERROR: {name} {(gx, gy)} is outside the map grid ({nx}x{ny})")
            return
        if blocked[gy, gx]:
            print(f"ERROR: {name} cell is blocked (not traversable, unobserved, or within robot_radius_m of an obstacle) - pick a different point")
            return

    path, total_cost = astar(cost, blocked, start_grid, goal_grid, args.cost_weight)
    if path is None:
        print("NO PATH FOUND")
        return

    world_path = [grid_to_world(ix, iy, x_min, y_min, resolution) for ix, iy in path]
    path_len_m = sum(
        float(np.hypot(world_path[i + 1][0] - world_path[i][0], world_path[i + 1][1] - world_path[i][1]))
        for i in range(len(world_path) - 1)
    )
    print(f"A* path found: {len(path)} cells, {path_len_m:.2f}m path length, total weighted cost={total_cost:.2f}")

    is_segment_free = make_segment_free_check(blocked, x_min, y_min, resolution)
    traj = build_trajectory(
        np.array(world_path), is_segment_free,
        shortcut_iters=args.shortcut_iters, chaikin_iters=args.chaikin_iters, resample_ds=args.resample_ds,
        v_max=args.max_vel, a_max=args.max_accel, a_lat_max=args.max_lateral_accel,
    )
    z = sample_height_at(traj["positions"], height, x_min, y_min, resolution)
    trajectory_xyz = np.column_stack([traj["positions"], z])
    print(
        f"Trajectory: {len(path)} A* cells -> {len(traj['shortcut'])} shortcut waypoints -> "
        f"{len(trajectory_xyz)} trajectory samples @ {args.resample_ds}m spacing, "
        f"{traj['distance_m'][-1]:.2f}m, {traj['time_s'][-1]:.1f}s, peak v={traj['velocity_mps'].max():.2f}m/s"
    )

    out_prefix = str(Path(args.elevation_npz).with_suffix(""))
    npz_out = f"{out_prefix}_path.npz"
    np.savez(
        npz_out,
        grid_path=np.array(path), world_path=np.array(world_path),
        trajectory_xyz=trajectory_xyz, trajectory_v=traj["velocity_mps"],
        trajectory_t=traj["time_s"], trajectory_s=traj["distance_m"],
    )
    print(f"Saved {npz_out}")

    out_path = args.out or f"{out_prefix}_path.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    display_cost = np.where(observed, cost, np.nan)
    im = ax.imshow(np.ma.masked_invalid(display_cost), origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, label="traversability cost (post-inflation)")
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, "b-", linewidth=2, label="A* grid path")
    ax.plot(start_grid[0], start_grid[1], "g^", markersize=12, label="start")
    ax.plot(goal_grid[0], goal_grid[1], "r*", markersize=14, label="goal")
    ax.legend()
    ax.set_title(f"A* path: {path_len_m:.2f}m, {len(path)} cells")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")

    traj_out = f"{out_prefix}_trajectory.png"
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    im = ax.imshow(np.ma.masked_invalid(display_cost), origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.6)
    ax.plot(px, py, color="0.5", linestyle="--", linewidth=1, label="raw A* path (grid zig-zag)")
    traj_grid = np.array([world_to_grid(x, y, x_min, y_min, resolution) for x, y in traj["positions"]])
    sc = ax.scatter(
        traj_grid[:, 0], traj_grid[:, 1], c=traj["velocity_mps"], cmap="turbo",
        vmin=0, vmax=max(args.max_vel, 1e-6), s=8, label="smoothed trajectory (color=speed)",
    )
    plt.colorbar(sc, ax=ax, fraction=0.03, label="speed (m/s)")
    ax.plot(start_grid[0], start_grid[1], "g^", markersize=12, label="start")
    ax.plot(goal_grid[0], goal_grid[1], "r*", markersize=14, label="goal")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Trajectory: {traj['distance_m'][-1]:.1f}m, {traj['time_s'][-1]:.1f}s")

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
    plt.savefig(traj_out, dpi=130)
    print(f"Saved {traj_out}")


if __name__ == "__main__":
    main()
