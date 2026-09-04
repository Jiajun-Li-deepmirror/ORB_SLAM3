import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import octomap

from splg_slam.config import load_config
from splg_slam.data.loader import dataset_dir, dataset_module
from splg_slam.geometry.pose_utils import camera_center
from splg_slam.geometry.stereo import StereoDepthEstimator, StereoRectifier
from splg_slam.mapping.octomap_utils import insert_keyframe_into_octree
from splg_slam.mapping.tracker import OfflineMapper
from splg_slam.planning.planner_3d import OccupancyChecker, make_segment_free_check, plan_3d
from splg_slam.planning.trajectory import build_trajectory
from splg_slam.utils import set_global_seed


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop online planning: runs the SLAM frontend frame-by-frame "
        "(simulated real-time via dataset playback), grows a persistent Octomap incrementally "
        "as new keyframes land, and periodically replans a trajectory from the robot's *current* "
        "estimated position to a fixed goal - so the plan can be checked for updating sensibly "
        "as the map fills in (no path -> path found -> path stabilizes/shortens), instead of "
        "planning once against a map that was already complete."
    )
    parser.add_argument("config", type=str)
    parser.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"), help="fixed goal in the SLAM's own world frame (camera frame of keyframe 0)")
    parser.add_argument("--replan_every_n_kf", type=int, default=15)
    parser.add_argument("--octomap_resolution", type=float, default=0.15)
    parser.add_argument("--octomap_stride", type=int, default=4, help="dense-depth sampling stride for octomap insertion (coarser than the tracker's own sparse keypoints)")
    parser.add_argument("--octomap_max_depth_m", type=float, default=6.0)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--allow_unknown", action="store_true", help="allow planning through never-observed space (default: blocked - the interesting case, since it means no path exists until enough area is actually mapped)")
    parser.add_argument("--max_expansions", type=int, default=300_000, help="bounds A* search effort per replan attempt, so a genuinely-still-disconnected goal fails fast instead of exhausting the whole known-free region every single replan")
    parser.add_argument("--shortcut_iters", type=int, default=3)
    parser.add_argument("--chaikin_iters", type=int, default=3)
    parser.add_argument("--resample_ds", type=float, default=0.2)
    parser.add_argument("--max_vel", type=float, default=1.0)
    parser.add_argument("--max_accel", type=float, default=0.5)
    parser.add_argument("--max_lateral_accel", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None, help="defaults to <config's output.map_dir>_online")
    args = parser.parse_args()

    set_global_seed(args.seed)
    cfg = load_config(args.config)
    goal = np.array(args.goal, dtype=np.float64)
    out_dir = Path(args.out_dir or f"{cfg.output.map_dir}_online")
    out_dir.mkdir(parents=True, exist_ok=True)

    dmod = dataset_module(cfg)
    ddir = dataset_dir(cfg)
    rig = dmod.load_stereo_rig(ddir)
    rectifier = StereoRectifier(rig)
    octomap_depth_est = StereoDepthEstimator(
        rectifier, min_disp=cfg.stereo.min_disp, num_disp=cfg.stereo.num_disp, block_size=cfg.stereo.block_size,
    )

    mapper = OfflineMapper(cfg, rectifier, global_extractor=None)
    tree = octomap.OcTree(args.octomap_resolution)
    pixel_xy = None

    entries = dmod.load_stereo_frames(ddir)
    stride = cfg.dataset.frame_stride or 1
    entries = entries[::stride]
    if args.max_frames:
        entries = entries[: args.max_frames]
    print(f"Processing {len(entries)} stereo pairs, replanning every {args.replan_every_n_kf} keyframes toward goal {goal.tolist()}")

    checkpoints = []
    n_keyframes = 0
    n_tracked = 0
    n_lost = 0
    t_wall_start = time.monotonic()

    for i, e in enumerate(entries):
        img_l = cv2.imread(str(e.left_path), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(e.right_path), cv2.IMREAD_GRAYSCALE)

        result, is_kf = mapper.process_stereo_pair(img_l, img_r, e.timestamp_ns, image_path=e.left_path)
        if result is None:
            n_lost += 1
            continue
        n_tracked += 1
        if not is_kf:
            continue
        n_keyframes += 1

        pixel_xy = insert_keyframe_into_octree(
            tree, img_l, img_r, result.pose_cw, rectifier, octomap_depth_est, pixel_xy,
            args.octomap_stride, args.octomap_max_depth_m,
        )

        if n_keyframes % args.replan_every_n_kf != 0:
            continue

        current_pos = camera_center(result.pose_cw)
        t0 = time.monotonic()
        path_world, path_idx, err = plan_3d(
            tree, current_pos, goal, robot_radius_m=args.robot_radius_m, allow_unknown=args.allow_unknown,
            max_expansions=args.max_expansions,
        )
        traj = None
        if path_world is not None:
            # Same safety-inflation + shortcut/Chaikin smoothing + curvature-limited velocity
            # profile as the offline plan_path_3d.py CLI - the online loop's replanned path
            # goes through the identical pipeline, not just the raw A* grid path, so "does the
            # plan update reasonably" is judged on the same executable trajectory a real run
            # would fly, not a rougher stand-in. The tree's EDT (built inside plan_3d, above)
            # is still valid here since it lives on the C++ tree object, not this Python call.
            checker = OccupancyChecker(tree)
            is_segment_free = make_segment_free_check(checker, tree.getResolution(), args.robot_radius_m, args.allow_unknown)
            traj = build_trajectory(
                path_world, is_segment_free, shortcut_iters=args.shortcut_iters, chaikin_iters=args.chaikin_iters,
                resample_ds=args.resample_ds, v_max=args.max_vel, a_max=args.max_accel, a_lat_max=args.max_lateral_accel,
            )
        replan_time_s = time.monotonic() - t0

        n_occupied = int(tree.calcNumNodes())
        dist_to_goal = float(np.linalg.norm(goal - current_pos))
        record = {
            "frame_idx": i, "n_keyframes": n_keyframes, "wall_time_s": time.monotonic() - t_wall_start,
            "current_pos": current_pos.copy(), "dist_to_goal_m": dist_to_goal, "n_octree_nodes": n_occupied,
            "replan_time_s": replan_time_s, "path_found": path_world is not None, "error": err,
        }
        if traj is not None:
            record["path_len_m"] = float(traj["distance_m"][-1])
            record["path_world"] = traj["positions"]  # smoothed trajectory, not the raw voxel path
            record["path_time_s"] = float(traj["time_s"][-1])
        else:
            record["path_len_m"] = None
            record["path_world"] = None
            record["path_time_s"] = None
        checkpoints.append(record)

        status = f"path={record['path_len_m']:.1f}m ({record['path_time_s']:.1f}s)" if path_world is not None else f"NO PATH ({err})"
        print(
            f"[kf {n_keyframes}, frame {i}/{len(entries)}] pos={current_pos.round(2)} "
            f"dist_to_goal={dist_to_goal:.1f}m octree_nodes={n_occupied} replan={replan_time_s:.2f}s -> {status}"
        )

    print(f"Done. tracked={n_tracked} lost={n_lost} keyframes={n_keyframes} checkpoints={len(checkpoints)}")

    np.savez(
        out_dir / "checkpoints.npz",
        n_keyframes=np.array([c["n_keyframes"] for c in checkpoints]),
        frame_idx=np.array([c["frame_idx"] for c in checkpoints]),
        wall_time_s=np.array([c["wall_time_s"] for c in checkpoints]),
        dist_to_goal_m=np.array([c["dist_to_goal_m"] for c in checkpoints]),
        n_octree_nodes=np.array([c["n_octree_nodes"] for c in checkpoints]),
        replan_time_s=np.array([c["replan_time_s"] for c in checkpoints]),
        path_found=np.array([c["path_found"] for c in checkpoints]),
        path_len_m=np.array([c["path_len_m"] if c["path_len_m"] is not None else np.nan for c in checkpoints]),
        current_pos=np.array([c["current_pos"] for c in checkpoints]),
        goal=goal,
    )
    print(f"Saved {out_dir / 'checkpoints.npz'}")

    tree.writeBinary(str(out_dir / "octomap_final.bt").encode())
    print(f"Saved {out_dir / 'octomap_final.bt'}")

    plot_evolution(checkpoints, goal, out_dir)
    plot_metrics(checkpoints, out_dir)


def plot_evolution(checkpoints: list[dict], goal: np.ndarray, out_dir: Path) -> None:
    """Top-down snapshots at a handful of checkpoints spread across the run, so the map
    filling in and the plan appearing/changing is visible at a glance rather than just in a
    table of numbers."""
    if not checkpoints:
        return
    n_panels = min(6, len(checkpoints))
    picks = np.linspace(0, len(checkpoints) - 1, n_panels).astype(int)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.5), squeeze=False)
    axes = axes[0]

    all_pos = np.array([c["current_pos"] for c in checkpoints])
    xlim = (min(all_pos[:, 0].min(), goal[0]) - 1, max(all_pos[:, 0].max(), goal[0]) + 1)
    ylim = (min(all_pos[:, 1].min(), goal[1]) - 1, max(all_pos[:, 1].max(), goal[1]) + 1)

    for ax, idx in zip(axes, picks):
        c = checkpoints[idx]
        if c["path_world"] is not None:
            ax.plot(c["path_world"][:, 0], c["path_world"][:, 1], "b-", linewidth=2, label="planned path")
        ax.plot(c["current_pos"][0], c["current_pos"][1], "g^", markersize=10, label="robot")
        ax.plot(goal[0], goal[1], "r*", markersize=12, label="goal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        status = f"{c['path_len_m']:.1f}m" if c["path_found"] else "no path"
        ax.set_title(f"kf {c['n_keyframes']}: {status}\n{c['n_octree_nodes']} octree nodes", fontsize=9)
        ax.tick_params(labelsize=7)
        if idx == picks[0]:
            ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    out_path = out_dir / "plan_evolution.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


def plot_metrics(checkpoints: list[dict], out_dir: Path) -> None:
    """Path length and distance-to-goal over the run: distance-to-goal should trend down as
    the robot flies (it's on a real trajectory through the scene, not driven toward the goal
    by the planner), path_found should flip from False to True once enough is mapped and stay
    True, and once found, path length should track distance-to-goal reasonably closely (not
    wildly diverge) rather than being some incoherent, unstable, no-relation quantity."""
    if not checkpoints:
        return
    n_kf = [c["n_keyframes"] for c in checkpoints]
    dist = [c["dist_to_goal_m"] for c in checkpoints]
    path_len = [c["path_len_m"] for c in checkpoints]
    found = [c["path_found"] for c in checkpoints]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_kf, dist, "o-", color="tab:gray", label="straight-line distance to goal")
    found_kf = [k for k, f in zip(n_kf, found) if f]
    found_len = [p for p, f in zip(path_len, found) if f]
    ax.plot(found_kf, found_len, "o-", color="tab:blue", label="planned path length (path found)")
    not_found_kf = [k for k, f in zip(n_kf, found) if not f]
    if not_found_kf:
        ax.scatter(not_found_kf, [0] * len(not_found_kf), marker="x", color="tab:red", label="no path found", zorder=5)
    ax.set_xlabel("keyframes inserted")
    ax.set_ylabel("meters")
    ax.set_title("Online replanning: does the plan track a sensible, improving trajectory?")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "plan_metrics.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
