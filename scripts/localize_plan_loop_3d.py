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

from splg_slam.config import load_config
from splg_slam.data.loader import dataset_dir, dataset_module
from splg_slam.geometry.pose_utils import camera_center
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.continuous_localizer import ContinuousLocalizer, estimate_max_track_jump_m
from splg_slam.map.io import load_map
from splg_slam.planning.dense_grid_3d import (
    build_cost_and_blocked_3d, grid_to_world_3d, make_segment_free_check_3d, plan_3d_dense, world_to_grid_3d,
)
from splg_slam.planning.dijkstra_field_3d import DijkstraField3DPlanner
from splg_slam.planning.dstar_lite_3d import DStarLite3DPlanner
from splg_slam.planning.trajectory import build_trajectory
from splg_slam.geometry.alignment import umeyama


def load_euroc_gt(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import csv
    ts, xyz = [], []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            ts.append(int(row[0]))
            xyz.append([float(row[1]), float(row[2]), float(row[3])])
    return np.array(ts), np.array(xyz)


def fit_slam_to_gt(world_map, gt_ts: np.ndarray, gt_xyz: np.ndarray):
    """Fits the same Umeyama similarity eval_trajectory.py uses to score ATE (aligning
    estimated keyframe centers onto the official GT trajectory, in the map's own RAW frame -
    before any r_align gravity rotation). Returns (r, s, t): a SLAM-frame point `p` maps to
    the GT frame via `s*(r@p)+t`."""
    kf_ids = world_map.keyframe_ids_sorted()
    centers = np.array([camera_center(world_map.keyframes[i].pose_cw) for i in kf_ids])
    timestamps = np.array([world_map.keyframes[i].timestamp_ns for i in kf_ids])
    gt_idx = np.clip(np.searchsorted(gt_ts, timestamps), 0, len(gt_ts) - 1)
    r, s, t = umeyama(centers, gt_xyz[gt_idx])
    print(f"Fitted SLAM-frame -> GT-frame similarity: scale={s:.4f}")
    return r, s, t


def pick_last_frame_goal(r: np.ndarray, s: float, t: np.ndarray, gt_xyz: np.ndarray, r_align: np.ndarray | None) -> np.ndarray:
    """The dataset's last frame's position: the last point of the official GT trajectory,
    brought back into the SLAM frame via the inverse SLAM->GT fit, then rotated by r_align
    (if given) to land in the same frame the octomap grid/live positions use."""
    gt_point = gt_xyz[-1]
    goal_slam = (r.T @ (gt_point - t)) / s
    goal = r_align @ goal_slam if r_align is not None else goal_slam
    print(f"Last GT point = {gt_point.round(2)} -> SLAM frame {goal_slam.round(2)} -> grid frame {goal.round(2)}")
    return goal


def snap_to_free_3d(pos: np.ndarray, grid: dict, mask_kwargs: dict, max_cells: int = 20) -> np.ndarray | None:
    """Snaps to the center of the nearest free grid cell, searching outward ring by ring -
    never returns an arbitrary nearby float coordinate, which on this codebase's grids has
    repeatedly landed exactly on a cell boundary and round-tripped into the wrong cell."""
    blocked, _ = build_cost_and_blocked_3d(grid, **mask_kwargs)
    origin, resolution = grid["origin"], grid["resolution"]
    idx0 = world_to_grid_3d(pos, origin, resolution)
    nx, ny, nz = blocked.shape

    def in_bounds_and_free(idx):
        return 0 <= idx[0] < nx and 0 <= idx[1] < ny and 0 <= idx[2] < nz and not blocked[idx]

    if in_bounds_and_free(idx0):
        return grid_to_world_3d(idx0, origin, resolution)
    for r in range(1, max_cells + 1):
        best = None
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if max(abs(dx), abs(dy), abs(dz)) != r:
                        continue
                    idx = (idx0[0] + dx, idx0[1] + dy, idx0[2] + dz)
                    if in_bounds_and_free(idx):
                        d = dx * dx + dy * dy + dz * dz
                        if best is None or d < best[0]:
                            best = (d, idx)
        if best is not None:
            return grid_to_world_3d(best[1], origin, resolution)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="3D analog of localize_plan_loop.py: localization-only online replanning "
        "against a static, pre-built dense octomap grid (build_octomap_grid.py) - no new "
        "mapping happens here. Two-tier localization (ContinuousLocalizer: cheap frame-to-"
        "keyframe tracking every frame, falling back to full retrieval-based relocalization "
        "only on tracking loss) feeds a two-tier planner (cheap per-frame local path-follow "
        "against the cached route, falling back to a full global D*Lite/A* replan only when "
        "the position has drifted off that route or the cached route is no longer collision-"
        "free) - mirroring how real AMR/robot stacks separate high-frequency tracking/local "
        "control from rare global relocalization/replanning."
    )
    parser.add_argument("config", type=str)
    parser.add_argument("map_path", type=str)
    parser.add_argument("grid_npz", type=str, help="dense grid from build_octomap_grid.py")
    parser.add_argument(
        "--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="override: use this world (grid-frame) point directly. Omit to use the "
        "dataset's own last frame's position (from official GT, aligned into the grid frame) "
        "as the fixed goal.",
    )
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument(
        "--max_track_jump_m", type=float, default=None,
        help="rejects a track-mode pose whose position jumps further than this from the "
        "previous successful localization, falling back to relocalization instead - catches "
        "a rare but real failure mode where a track-mode PnP solve is 'valid' (enough "
        "inliers, passes RANSAC) but physically implausible (observed: a single-frame "
        "teleport that snaps back the very next query). Default (omit this flag): auto-"
        "derived from the map's own keyframe speed distribution and the actual interval "
        "between queried frames (see estimate_max_track_jump_m) - a fixed constant tuned for "
        "one dataset's speed does not transfer (an EuRoC-tuned 1.0m rejected 27% of KITTI's "
        "genuine car-speed motion as false positives). Pass 0 to disable the check entirely, "
        "or a positive value to override the auto-derived one.",
    )
    parser.add_argument(
        "--force_relocalize_every", type=int, default=0,
        help="opt-in, disabled by default (0): forces a full relocalization after this many "
        "consecutive track-mode successes anchored to the same reference keyframe, even "
        "though tracking hasn't failed. Catches a different failure mode than "
        "--max_track_jump_m: a run of visually ambiguous frames (repetitive structure) can "
        "produce a smoothly-varying but systematically WRONG pose that stays self-consistent "
        "for many frames (no single-step discontinuity to catch) until the anchor happens to "
        "change - observed on MH04 as an ~18-frame, ~1.9m sustained bias. Costs an extra "
        "expensive relocalization periodically even when tracking looks fine.",
    )
    parser.add_argument("--num_checkpoints", type=int, default=5)
    parser.add_argument(
        "--plan_every", type=int, default=1,
        help="only attempt a replan every this many queries (localization still runs on "
        "EVERY query regardless, so localization-only timing/accuracy stats aren't diluted "
        "by skipped queries) - set e.g. 20 to decouple planning frequency from localization "
        "frequency for a pure localization-timing/accuracy measurement. 1 (default): replan "
        "every successful localization, the original behavior.",
    )
    parser.add_argument(
        "--planner", type=str, default="dijkstra", choices=["astar", "dstar_lite", "dijkstra"],
        help="dijkstra (default): one full Dijkstra distance-to-goal field precomputed once "
        "up front - correct for this codebase's fixed-map/fixed-goal/moving-start access "
        "pattern with no D* Lite-style queue-growth spikes, since it never needs incremental "
        "repair. dstar_lite: incremental, cheaper if the goal/map ever change mid-run, but "
        "carries the queue-growth cost D* Lite SPIKE monitoring reports. astar: a fresh "
        "search every replan (predictable but slow in a loop).",
    )
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--inflate_radius_m", type=float, default=0.3)
    parser.add_argument("--close_radius_m", type=float, default=0.2)
    parser.add_argument("--cost_weight", type=float, default=0.0)
    parser.add_argument("--allow_unknown", action="store_true")
    parser.add_argument(
        "--gravity_rotation", type=str, default=None,
        help="path to a saved gravity-alignment rotation .npy (3x3) - MUST match whatever "
        "rotation (if any) was passed to build_octomap.py when building grid_npz's source "
        "octomap. The relocalizer returns positions in the map's own raw frame; if the octomap "
        "was built in a rotated frame (see build_octomap.py --gravity_rotation) and this is "
        "omitted, every query silently plans against the wrong frame - same failure mode as an "
        "un-gravity-aligned map, just self-inflicted.",
    )
    parser.add_argument("--max_expansions", type=int, default=1_000_000)
    parser.add_argument(
        "--rebuild_interval", type=int, default=0,
        help="D* Lite only, OPT-IN (disabled by default): every this many start moves, "
        "discard the accumulated queue/km state and do one fresh full sweep. Measured on "
        "both KITTI and MH01: an unconditional call-count trigger doesn't reduce the number "
        "of spikes and adds net overhead (a periodic full sweep costs about as much as one "
        "of the spikes it's meant to prevent, whether or not the queue has actually grown "
        "problematic yet) - kept only for experimentation with a smarter (queue-size-based) "
        "trigger. 0/negative disables.",
    )
    parser.add_argument(
        "--local_replan_trigger_m", type=float, default=1.0,
        help="global layer (D*Lite/A*) only reruns when the localized position drifts more "
        "than this far from the cached route or that route is no longer collision-free; "
        "otherwise the cheap local layer just follows the cached route from the nearest point "
        "onward. 0 disables local-follow (global replan every frame, the old behavior).",
    )
    parser.add_argument(
        "--shortcut_iters", type=int, default=0,
        help="disabled by default: shortcut_path's visibility scan is O(N^2)-ish over the "
        "raw path's waypoint count, and became the dominant per-replan cost once Dijkstra-"
        "field/D* Lite made the search itself nearly free on a long path. Opt in only if a "
        "specific route needs its corners straightened.",
    )
    parser.add_argument("--chaikin_iters", type=int, default=0, help="disabled by default, see --shortcut_iters")
    parser.add_argument("--resample_ds", type=float, default=0.2)
    parser.add_argument("--max_vel", type=float, default=2.0)
    parser.add_argument("--max_accel", type=float, default=1.0)
    parser.add_argument("--max_lateral_accel", type=float, default=2.0)
    parser.add_argument("--out_dir", type=str, default="results/mh_multi_map_localize_plan")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r_align = np.load(args.gravity_rotation) if args.gravity_rotation else None

    cfg = load_config(args.config)
    ddir = dataset_dir(cfg)
    dmod = dataset_module(cfg)

    print("Loading pre-built map (this is the ONLY map used - no new mapping happens)...")
    world_map = load_map(args.map_path)
    print(f"Map: {len(world_map.keyframes)} keyframes, {len(world_map.map_points)} map points")

    gt_ts, gt_xyz = load_euroc_gt(ddir / "state_groundtruth_estimate0" / "data.csv")
    r_fit, s_fit, t_fit = fit_slam_to_gt(world_map, gt_ts, gt_xyz)
    if args.goal is not None:
        goal = np.array(args.goal, dtype=np.float64)
    else:
        goal = pick_last_frame_goal(r_fit, s_fit, t_fit, gt_xyz, r_align)

    grid_data = np.load(args.grid_npz)
    grid = {k: grid_data[k] for k in ["occupied", "unknown", "origin", "resolution"]}
    grid["resolution"] = float(grid["resolution"])
    mask_kwargs = dict(
        allow_unknown=args.allow_unknown, close_radius_m=args.close_radius_m,
        robot_radius_m=args.robot_radius_m, inflate_radius_m=args.inflate_radius_m,
    )

    snapped = snap_to_free_3d(goal, grid, mask_kwargs)
    if snapped is None:
        print(f"ERROR: no free cell found near goal {goal.round(2)}")
        return
    if np.linalg.norm(snapped - goal) > 1e-6:
        print(f"Goal {goal.round(2)} snapped to nearest free cell {snapped.round(2)}")
    goal = snapped

    rebuild_interval = args.rebuild_interval if args.rebuild_interval > 0 else None
    planner_obj = None
    if args.planner == "dstar_lite":
        planner_obj = DStarLite3DPlanner(
            grid, goal, cost_weight=args.cost_weight, rebuild_interval=rebuild_interval, **mask_kwargs,
        )
        print(f"Using D* Lite: one persistent search reused across every replan below (rebuild_interval={rebuild_interval}).")
    elif args.planner == "dijkstra":
        planner_obj = DijkstraField3DPlanner(grid, goal, cost_weight=args.cost_weight, **mask_kwargs)
        print("Using Dijkstra field: one full distance-to-goal sweep precomputed once, every replan is a pure lookup.")

    rig = dmod.load_stereo_rig(ddir)
    rectifier = StereoRectifier(rig)

    entries = dmod.load_stereo_frames(ddir)
    base_stride = cfg.dataset.frame_stride or 1
    entries = entries[::base_stride][:: args.stride]
    print(f"Querying {len(entries)} frames, goal fixed at {goal.tolist()}")

    print("Building continuous localizer (retrieval index + warmup)...")
    if args.max_track_jump_m is None:
        query_dt_s = float(np.median(np.diff([e.timestamp_ns for e in entries])) * 1e-9)
        max_track_jump_m = estimate_max_track_jump_m(world_map, query_dt_s)
        print(f"Auto-derived max_track_jump_m={max_track_jump_m:.3f}m (query interval={query_dt_s*1000:.0f}ms)")
    elif args.max_track_jump_m == 0:
        max_track_jump_m = None
    else:
        max_track_jump_m = args.max_track_jump_m
    force_relocalize_every = args.force_relocalize_every if args.force_relocalize_every > 0 else None
    localizer = ContinuousLocalizer(
        world_map, rectifier, cfg, max_track_jump_m=max_track_jump_m, force_relocalize_every=force_relocalize_every,
    )

    checkpoint_marks = set(int(round((k / args.num_checkpoints) * len(entries))) for k in range(1, args.num_checkpoints + 1))
    if args.plan_every > 1:
        # snap each checkpoint down to the nearest query that actually attempts a plan, so a
        # checkpoint never lands on a localization-only query with nothing to plot
        checkpoint_marks = {max(args.plan_every, (m // args.plan_every) * args.plan_every) for m in checkpoint_marks}

    records = []
    last_localize_t = None
    localize_intervals = []
    localize_errors = []  # localization-only accuracy vs GT, one entry per successful localization
    n_success = 0
    n_global_replan = 0
    n_local_follow = 0
    cached_traj = None  # last trajectory dict from a global replan, reused by the local layer
    cached_blocked = None
    q_last_global = None
    spike_threshold_s = 1.0

    for q, e in enumerate(entries, start=1):
        img_l = cv2.imread(str(e.left_path), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(e.right_path), cv2.IMREAD_GRAYSCALE)
        rect_l, _ = rectifier.rectify(img_l, img_r)

        t_loc0 = time.monotonic()
        ok, pose_cw, info = localizer.localize(rect_l)
        t_loc1 = time.monotonic()

        record = {
            "query_idx": q, "frame_idx": e.index, "localize_ok": ok, "localize_time_s": t_loc1 - t_loc0,
            "localize_mode": info.get("mode") if ok else None,
        }
        if ok:
            n_success += 1
            now = time.monotonic()
            if last_localize_t is not None:
                localize_intervals.append(now - last_localize_t)
            last_localize_t = now

            pos_raw = camera_center(pose_cw)
            pos = r_align @ pos_raw if r_align is not None else pos_raw

            gt_i = int(np.clip(np.searchsorted(gt_ts, e.timestamp_ns), 0, len(gt_ts) - 1))
            est_gt = s_fit * (r_fit @ pos_raw) + t_fit
            localize_accuracy_m = float(np.linalg.norm(est_gt - gt_xyz[gt_i]))
            localize_errors.append(localize_accuracy_m)

            if q % args.plan_every != 0:
                record.update({
                    "pos": pos, "dist_to_goal_m": float(np.linalg.norm(goal - pos)),
                    "replan_time_s": None, "path_found": None, "error": None, "plan_mode": None,
                    "path_len_m": None, "path_xyz": None, "localize_accuracy_m": localize_accuracy_m,
                })
                print(
                    f"[q {q}/{len(entries)}, frame {e.index}] loc={info.get('mode')} pos={pos.round(2)} "
                    f"localize={record['localize_time_s']*1000:.0f}ms gt_err={localize_accuracy_m:.2f}m (planning skipped this query)"
                )
                records.append(record)
                continue

            t_plan0 = time.monotonic()
            plan_mode = "global"
            traj, err = None, None
            if args.local_replan_trigger_m > 0 and cached_traj is not None:
                idx, deviation = nearest_point_on_path(pos, cached_traj["positions"])
                if deviation <= args.local_replan_trigger_m:
                    is_segment_free = make_segment_free_check_3d(cached_blocked, grid["origin"], grid["resolution"])
                    if is_segment_free(pos, cached_traj["positions"][idx]):
                        plan_mode = "local"
                        remaining_xyz = np.vstack([pos[None, :], cached_traj["positions"][idx:]])
                        traj = build_trajectory(
                            remaining_xyz, is_segment_free, shortcut_iters=0, chaikin_iters=0,
                            resample_ds=args.resample_ds, v_max=args.max_vel, a_max=args.max_accel,
                            a_lat_max=args.max_lateral_accel,
                        )

            if plan_mode == "global":
                if planner_obj is not None:
                    path_world, path_idx, blocked, _origin, err = planner_obj.plan(pos, max_expansions=args.max_expansions)
                else:
                    path_world, path_idx, blocked, _origin, err = plan_3d_dense(
                        grid, pos, goal, cost_weight=args.cost_weight, max_expansions=args.max_expansions, **mask_kwargs,
                    )
                if path_world is not None:
                    is_segment_free = make_segment_free_check_3d(blocked, grid["origin"], grid["resolution"])
                    traj = build_trajectory(
                        path_world, is_segment_free, shortcut_iters=args.shortcut_iters, chaikin_iters=args.chaikin_iters,
                        resample_ds=args.resample_ds, v_max=args.max_vel, a_max=args.max_accel, a_lat_max=args.max_lateral_accel,
                    )
                    cached_traj, cached_blocked = traj, blocked
                else:
                    cached_traj, cached_blocked = None, None

            if plan_mode == "global":
                n_global_replan += 1
                gap = (q - q_last_global) if q_last_global is not None else q
                q_last_global = q
            else:
                n_local_follow += 1
            replan_time_s = time.monotonic() - t_plan0
            if plan_mode == "global" and isinstance(planner_obj, DStarLite3DPlanner) and replan_time_s > spike_threshold_s:
                dsl = planner_obj.dsl
                print(
                    f"  [D* Lite SPIKE] query {q}: {replan_time_s*1000:.0f}ms, "
                    f"{dsl.last_num_expansions} real expansions + {dsl.last_num_requeues} stale-key "
                    f"requeues, {gap} queries since the last global replan, km={dsl.km:.1f}, "
                    f"queue size after={len(dsl._in_queue)}, rebuilt_this_call={dsl.last_rebuilt}"
                )

            record.update({
                "pos": pos, "dist_to_goal_m": float(np.linalg.norm(goal - pos)),
                "replan_time_s": replan_time_s, "path_found": traj is not None, "error": err,
                "plan_mode": plan_mode, "localize_accuracy_m": localize_accuracy_m,
                "path_len_m": float(traj["distance_m"][-1]) if traj is not None else None,
                "path_xyz": traj["positions"] if traj is not None else None,
            })
            status = f"path={record['path_len_m']:.1f}m" if traj is not None else f"NO PATH ({err})"
            print(
                f"[q {q}/{len(entries)}, frame {e.index}] loc={info.get('mode')} pos={pos.round(2)} "
                f"dist_to_goal={record['dist_to_goal_m']:.1f}m localize={record['localize_time_s']*1000:.0f}ms "
                f"plan={plan_mode} replan={replan_time_s*1000:.0f}ms -> {status}"
            )
        else:
            record.update({
                "pos": None, "dist_to_goal_m": None, "replan_time_s": None, "path_found": False,
                "path_len_m": None, "path_xyz": None, "plan_mode": None, "localize_accuracy_m": None,
            })
            print(f"[q {q}/{len(entries)}, frame {e.index}] LOCALIZATION FAILED ({info.get('reason')})")
        records.append(record)

        if q in checkpoint_marks:
            frac = q / len(entries)
            plot_checkpoint(grid, record, goal, frac, out_dir)

    print(f"\n=== Localization (every query, {len(entries)} total) ===")
    if localize_intervals:
        mean_interval = float(np.mean(localize_intervals))
        print(
            f"Success: {n_success}/{len(entries)}, mean interval = {mean_interval*1000:.0f}ms "
            f"({1.0/mean_interval:.2f} Hz)"
        )
    print(
        f"Tier: {localizer.n_track} tracked (cheap, frame-to-keyframe), "
        f"{localizer.n_relocalize} full-relocalized (expensive, global retrieval), "
        f"{localizer.n_track_lost} tracking-loss events, "
        f"{localizer.n_track_jump_rejected} implausible-jump rejections, "
        f"{localizer.n_forced_relocalize} forced periodic refreshes"
    )
    loc_times_ms = np.array([r["localize_time_s"] * 1000 for r in records if r["localize_ok"]])
    if len(loc_times_ms):
        print(
            f"Localization-only time (ms, excludes planning): mean={loc_times_ms.mean():.1f} "
            f"median={np.median(loc_times_ms):.1f} min={loc_times_ms.min():.1f} max={loc_times_ms.max():.1f}"
        )
    if localize_errors:
        errs = np.array(localize_errors)
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        print(
            f"Localization accuracy vs GT (m): mean={errs.mean():.3f} median={np.median(errs):.3f} "
            f"rmse={rmse:.3f} min={errs.min():.3f} max={errs.max():.3f}"
        )

    planned = [r for r in records if r["plan_mode"] is not None]
    print(f"\n=== Planning (every {args.plan_every} queries, {len(planned)} attempted) ===")
    print(f"Tier: {n_global_replan} global replans, {n_local_follow} local path-follows (no search)")
    n_found = sum(1 for r in planned if r["path_found"])
    if planned:
        print(f"Success rate: {n_found}/{len(planned)} ({100*n_found/len(planned):.0f}%)")
    plan_times_ms = np.array([r["replan_time_s"] * 1000 for r in planned])
    if len(plan_times_ms):
        print(
            f"Planning-only time (ms, excludes localization): mean={plan_times_ms.mean():.1f} "
            f"median={np.median(plan_times_ms):.1f} min={plan_times_ms.min():.1f} max={plan_times_ms.max():.1f} "
            f"total={plan_times_ms.sum()/1000:.1f}s"
        )
    plot_summary(records, out_dir)


def nearest_point_on_path(pos: np.ndarray, path_xyz: np.ndarray) -> tuple[int, float]:
    d = np.linalg.norm(path_xyz - pos[None, :], axis=1)
    idx = int(np.argmin(d))
    return idx, float(d[idx])


def plot_checkpoint(grid: dict, record: dict, goal: np.ndarray, frac: float, out_dir: Path) -> None:
    occupied = grid["occupied"]
    origin, resolution = grid["origin"], grid["resolution"]
    ys, xs, _ = np.where(occupied[:, :, ::4])  # coarse top-down projection for a quick background
    wx = origin[0] + xs * resolution
    wy = origin[1] + ys * resolution

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(wx, wy, s=0.5, c="0.75", label="occupied (top-down projection)")
    ax.plot(goal[0], goal[1], "r*", markersize=16, label="fixed goal")
    if record["pos"] is not None:
        ax.plot(record["pos"][0], record["pos"][1], "g^", markersize=13, label="current localized position")
    if record["path_xyz"] is not None:
        ax.plot(record["path_xyz"][:, 0], record["path_xyz"][:, 1], "b-", linewidth=2.5, label="replanned route")
    title = f"{frac:.0%} through dataset (frame {record['frame_idx']}), loc={record['localize_mode']}, plan={record['plan_mode']}: "
    title += f"{record['path_len_m']:.1f}m" if record["path_found"] else "no path"
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    plt.tight_layout()
    out_path = out_dir / f"checkpoint_{int(round(frac*100)):03d}pct.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


def plot_summary(records: list[dict], out_dir: Path) -> None:
    ok_records = [r for r in records if r["localize_ok"]]
    if not ok_records:
        return
    q = [r["query_idx"] for r in ok_records]
    dist = [r["dist_to_goal_m"] for r in ok_records]

    planned_records = [r for r in ok_records if r["plan_mode"] is not None]
    plan_q = [r["query_idx"] for r in planned_records]
    path_len = [r["path_len_m"] for r in planned_records]
    found = [r["path_found"] for r in planned_records]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(q, dist, "o-", color="tab:gray", markersize=3, label="straight-line distance to goal")
    found_q = [k for k, f in zip(plan_q, found) if f]
    found_len = [p for p, f in zip(path_len, found) if f]
    ax.plot(found_q, found_len, "o-", color="tab:blue", markersize=3, label="replanned path length")
    not_found_q = [k for k, f in zip(plan_q, found) if not f]
    if not_found_q:
        ax.scatter(not_found_q, [0] * len(not_found_q), marker="x", color="tab:red", label="no path found", zorder=5)
    failed_q = [r["query_idx"] for r in records if not r["localize_ok"]]
    if failed_q:
        ax.scatter(failed_q, [-max(dist) * 0.03] * len(failed_q), marker="|", color="black", label="localization failed", zorder=5)
    ax.set_xlabel("query index (localization/replan calls)")
    ax.set_ylabel("meters")
    ax.set_title("3D localization-only online replanning against a static EuRoC octomap")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "summary.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
