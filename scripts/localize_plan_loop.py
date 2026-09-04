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
from splg_slam.data import kitti as kitti_data
from splg_slam.data.loader import dataset_dir, dataset_module
from splg_slam.geometry.alignment import umeyama
from splg_slam.geometry.pose_utils import camera_center
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.continuous_localizer import ContinuousLocalizer, estimate_max_track_jump_m
from splg_slam.map.io import load_map
from splg_slam.planning.dijkstra_field_2d import DijkstraField2DPlanner
from splg_slam.planning.dstar_lite_2d import DStarLiteFallbackPlanner
from splg_slam.planning.planner_2d import make_segment_free_check, plan_2d_with_fallback, world_to_grid
from splg_slam.planning.trajectory import build_trajectory


def fit_slam_to_gt(world_map, ddir: Path):
    """Fits the same Umeyama similarity eval_trajectory.py uses to score ATE (aligning
    estimated keyframe centers onto the official GT trajectory). Returns (r, s, t, gt_ts,
    gt_xyz) - the transform maps a SLAM-frame point `p` to the GT frame via `s*(r@p)+t`, and
    the caller can look up the GT point matching any timestamp via gt_ts/gt_xyz."""
    kf_ids = world_map.keyframe_ids_sorted()
    centers = np.array([camera_center(world_map.keyframes[i].pose_cw) for i in kf_ids])
    timestamps = np.array([world_map.keyframes[i].timestamp_ns for i in kf_ids])

    gt_ts, gt_xyz = kitti_data.load_gt_as_xyz(ddir)
    gt_idx = np.clip(np.searchsorted(gt_ts, timestamps), 0, len(gt_ts) - 1)
    gt_matched = gt_xyz[gt_idx]

    r, s, t = umeyama(centers, gt_matched)
    print(f"Fitted SLAM-frame -> GT-frame similarity: scale={s:.4f}")
    return r, s, t, gt_ts, gt_xyz


def pick_gt_goal(r: np.ndarray, s: float, t: np.ndarray, gt_xyz: np.ndarray, r_align: np.ndarray, gt_frac: float) -> np.ndarray:
    """Picks a point at `gt_frac` along the *official* GT trajectory (not the SLAM estimate)
    as the fixed goal - "put the goal on an arbitrary point of the mapping GT" (gt_frac=1.0
    picks the last point, i.e. the dataset's last frame). Inverts the SLAM->GT fit to bring
    a GT point back into the SLAM frame, then applies `r_align` (the gravity/RANSAC rotation
    baked into the elevation map) to land in the elevation map's own frame."""
    goal_gt_idx = int(round(gt_frac * (len(gt_xyz) - 1)))
    gt_point = gt_xyz[goal_gt_idx]
    goal_slam = (r.T @ (gt_point - t)) / s
    goal_aligned = r_align @ goal_slam
    print(f"GT point #{goal_gt_idx}/{len(gt_xyz)} (frac={gt_frac}) = {gt_point.round(2)} -> SLAM frame {goal_slam.round(2)} -> elevation-map frame {goal_aligned.round(2)}")
    return goal_aligned


def snap_to_traversable(xy: np.ndarray, elevation: dict, max_cells: int = 60) -> np.ndarray | None:
    """Snaps to the center of the nearest traversable grid cell, searching outward ring by
    ring - never returns an arbitrary nearby float coordinate, which on this codebase's grids
    has repeatedly landed exactly on a cell boundary and round-tripped into the wrong cell."""
    x_min, y_min, res = elevation["x_min"], elevation["y_min"], elevation["resolution"]
    traversable = elevation["traversable"]
    ny, nx = traversable.shape
    gx0, gy0 = world_to_grid(xy[0], xy[1], x_min, y_min, res)
    if 0 <= gx0 < nx and 0 <= gy0 < ny and traversable[gy0, gx0]:
        return np.array([x_min + gx0 * res, y_min + gy0 * res])
    for r in range(1, max_cells + 1):
        best = None
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                gx, gy = gx0 + dx, gy0 + dy
                if 0 <= gx < nx and 0 <= gy < ny and traversable[gy, gx]:
                    d = dx * dx + dy * dy
                    if best is None or d < best[0]:
                        best = (d, gx, gy)
        if best is not None:
            _, gx, gy = best
            return np.array([x_min + gx * res, y_min + gy * res])
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Localization-only online replanning: NO new mapping happens here - a "
        "pre-built static map and elevation grid stay fixed the whole run. Each frame is "
        "independently relocalized against that map (global retrieval + LightGlue + PnP, no "
        "temporal tracking state carried between frames - purely 'where am I right now'), and "
        "a route is replanned from that fresh position to a fixed goal every single time a "
        "localization succeeds (replanning frequency = localization output frequency, by "
        "construction - there is nothing else driving the replan). Nothing here closes the "
        "loop toward the goal (no controller), so the position wanders wherever the recorded "
        "dataset actually drove; the point is to see the *plan* update sensibly against a "
        "moving current position and a static map, not to reach the goal."
    )
    parser.add_argument("config", type=str)
    parser.add_argument("map_path", type=str)
    parser.add_argument("elevation_npz", type=str)
    parser.add_argument("rotation_npy", type=str, help="gravity/RANSAC rotation (.npy, 3x3) baked into the elevation map, e.g. dense_cloud_gravity_ransac_rotation.npy")
    parser.add_argument("--gt_frac", type=float, default=0.7, help="fraction along the official KITTI GT trajectory to use as the fixed goal")
    parser.add_argument("--goal", type=float, nargs=2, default=None, help="override: use this world (elevation-map-frame) point directly instead of deriving one from GT")
    parser.add_argument("--stride", type=int, default=8, help="query every Nth frame (on top of cfg.dataset.frame_stride) - controls the localization/replanning rate")
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
        "change. Costs an extra expensive relocalization periodically even when tracking "
        "looks fine.",
    )
    parser.add_argument("--num_checkpoints", type=int, default=3, help="how many progress checkpoints (evenly spaced, e.g. 3 = every 1/3) to save a route plot for")
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
        help="dijkstra (default): one full distance-to-goal field precomputed once up front - "
        "correct for this loop's fixed-map/fixed-goal/moving-start pattern, no incremental "
        "queue to maintain so no D* Lite-style growth spikes. dstar_lite: one incremental "
        "search reused across replans - after an expensive first call, later calls are "
        "typically cheap, but the pending-node queue grows over a long run and occasionally "
        "spikes (see the SPIKE diagnostics) since it's built to repair a moving start/changing "
        "map, machinery this fixed-goal loop doesn't need. astar: plan_2d_with_fallback, a "
        "fresh global search every replan - much lower success rate and higher total time in "
        "a repeated-replan loop (no fallback ladder support for dijkstra: single fixed mask "
        "only).",
    )
    parser.add_argument("--robot_radius_m", type=float, default=1.0, help="KITTI is a car, not a legged robot - wider default clearance")
    parser.add_argument("--inflate_radius_m", type=float, default=1.5)
    parser.add_argument("--cost_weight", type=float, default=5.0)
    parser.add_argument("--close_radius_m", type=float, default=0.9, help="morphological closing of the free-space mask (default on, unlike plan_path.py) - this KITTI elevation map has 6051 disconnected traversable fragments at 0 (dense-stereo noise splitting the road); 0.9m closes most of that (down to ~340 fragments, largest covering 97.7% of free space) without the search-space blowup --allow_unknown causes")
    parser.add_argument("--allow_unknown", action="store_true", help="NOT recommended here - blows up the search space on a map this large (every failed query then explores up to --max_expansions nodes, ~6s each); prefer --close_radius_m")
    parser.add_argument("--max_expansions", type=int, default=150_000, help="bounds A* effort per replan so a genuinely unreachable goal fails in a bounded time instead of exhausting a huge allow_unknown-expanded search space")
    parser.add_argument(
        "--local_replan_trigger_m", type=float, default=1.0,
        help="global layer (D*Lite/A*) only reruns when the localized position drifts more "
        "than this far from the cached route or that route is no longer collision-free; "
        "otherwise the cheap local layer just follows the cached route from the nearest point "
        "onward. 0 disables local-follow (global replan every frame, the old behavior).",
    )
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
        "--use_fallback_ladder", action="store_true",
        help="opt-in only, NOT recommended for normal use. Default behavior is a single fixed "
        "mask (robot_radius_m/inflate_radius_m/close_radius_m/block_cost_threshold as given) "
        "decided ONCE offline - see scripts/tune_planner_mask.py - since closing/smoothing fix "
        "the map's actual noise-driven fragmentation and should be tuned before deployment, not "
        "re-decided per query. Passing this instead escalates through DEFAULT_FALLBACK_STAGES at "
        "query time on failure; for --planner dstar_lite especially, each stage is a SEPARATE "
        "persistent search paying its own full first-exploration cost the first time it's needed, "
        "so a query failing every stage pays that cost N times over (observed: 92s for one query "
        "across 4 stages). Only worth it if you haven't tuned a working fixed mask and need "
        "per-query robustness more than predictable latency.",
    )
    parser.add_argument(
        "--shortcut_iters", type=int, default=0,
        help="disabled by default: shortcut_path's visibility scan is O(N^2)-ish over the "
        "raw path's waypoint count, and became the dominant per-replan cost once Dijkstra-"
        "field/D* Lite made the search itself nearly free (measured ~660ms for one 300m "
        "KITTI path). Opt in only if a specific route needs its corners straightened.",
    )
    parser.add_argument("--chaikin_iters", type=int, default=0, help="disabled by default, see --shortcut_iters")
    parser.add_argument("--resample_ds", type=float, default=0.5)
    parser.add_argument("--max_vel", type=float, default=8.0, help="m/s, car-speed default")
    parser.add_argument("--max_accel", type=float, default=2.0)
    parser.add_argument("--max_lateral_accel", type=float, default=3.0)
    parser.add_argument("--out_dir", type=str, default="results/kitti00_localize_plan")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    ddir = dataset_dir(cfg)
    dmod = dataset_module(cfg)

    print("Loading pre-built map (this is the ONLY map used - no new mapping happens)...")
    world_map = load_map(args.map_path)
    print(f"Map: {len(world_map.keyframes)} keyframes, {len(world_map.map_points)} map points")

    r_align = np.load(args.rotation_npy)
    elevation_data = np.load(args.elevation_npz)
    elevation = {k: elevation_data[k] for k in ["height", "cost", "traversable", "x_min", "y_min", "resolution"]}
    elevation["x_min"], elevation["y_min"], elevation["resolution"] = (
        float(elevation["x_min"]), float(elevation["y_min"]), float(elevation["resolution"]),
    )

    r_fit, s_fit, t_fit, gt_ts, gt_xyz = fit_slam_to_gt(world_map, ddir)
    if args.goal is not None:
        goal_xy = np.array(args.goal, dtype=np.float64)
    else:
        goal_3d = pick_gt_goal(r_fit, s_fit, t_fit, gt_xyz, r_align, args.gt_frac)
        goal_xy = goal_3d[:2]
    snapped = snap_to_traversable(goal_xy, elevation)
    if snapped is None:
        print(f"ERROR: no traversable cell found near goal {goal_xy.round(2)}")
        return
    if np.linalg.norm(snapped - goal_xy) > 1e-6:
        print(f"Goal {goal_xy.round(2)} snapped to nearest traversable cell {snapped.round(2)}")
    goal_xy = snapped

    rig = dmod.load_stereo_rig(ddir)
    rectifier = StereoRectifier(rig)

    rebuild_interval = args.rebuild_interval if args.rebuild_interval > 0 else None
    planner_obj = None
    if args.planner == "dstar_lite":
        planner_obj = DStarLiteFallbackPlanner(
            elevation, goal_xy, fallback_stages=None if args.use_fallback_ladder else [],
            cost_weight=args.cost_weight, allow_unknown=args.allow_unknown,
            robot_radius_m=args.robot_radius_m, inflate_radius_m=args.inflate_radius_m,
            close_radius_m=args.close_radius_m, rebuild_interval=rebuild_interval,
        )
        print(f"Using D* Lite: one persistent search reused across every replan below (fallback ladder {'enabled' if args.use_fallback_ladder else 'disabled - single fixed mask'}, rebuild_interval={rebuild_interval}).")
    elif args.planner == "dijkstra":
        planner_obj = DijkstraField2DPlanner(
            elevation, goal_xy, cost_weight=args.cost_weight, allow_unknown=args.allow_unknown,
            robot_radius_m=args.robot_radius_m, inflate_radius_m=args.inflate_radius_m,
            close_radius_m=args.close_radius_m,
        )
        print("Using Dijkstra field: one full distance-to-goal sweep precomputed once, every replan is a pure lookup.")

    entries = dmod.load_stereo_frames(ddir)
    base_stride = cfg.dataset.frame_stride or 1
    entries = entries[::base_stride][:: args.stride]
    print(
        f"Querying {len(entries)} frames (base stride {base_stride} x query stride {args.stride}), "
        f"goal fixed at {goal_xy.round(2)} (elevation-map frame)"
    )

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

    checkpoint_marks = set(
        int(round((k / args.num_checkpoints) * len(entries))) for k in range(1, args.num_checkpoints + 1)
    )
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

            pos_slam = camera_center(pose_cw)
            pos_xy = (r_align @ pos_slam)[:2]

            gt_i = int(np.clip(np.searchsorted(gt_ts, e.timestamp_ns), 0, len(gt_ts) - 1))
            est_gt = s_fit * (r_fit @ pos_slam) + t_fit
            localize_accuracy_m = float(np.linalg.norm(est_gt - gt_xyz[gt_i]))
            localize_errors.append(localize_accuracy_m)

            if q % args.plan_every != 0:
                record.update({
                    "pos_xy": pos_xy, "dist_to_goal_m": float(np.linalg.norm(goal_xy - pos_xy)),
                    "replan_time_s": None, "path_found": None, "error": None, "fallback_stage": None,
                    "plan_mode": None, "path_len_m": None, "path_xy": None, "localize_accuracy_m": localize_accuracy_m,
                })
                print(
                    f"[q {q}/{len(entries)}, frame {e.index}] loc={info.get('mode')} pos={pos_xy.round(1)} "
                    f"localize={record['localize_time_s']*1000:.0f}ms gt_err={localize_accuracy_m:.2f}m (planning skipped this query)"
                )
                records.append(record)
                continue

            t_plan0 = time.monotonic()
            plan_mode = "global"
            traj, err, stage = None, None, None
            if args.local_replan_trigger_m > 0 and cached_traj is not None:
                idx, deviation = nearest_point_on_path(pos_xy, cached_traj["positions"])
                if deviation <= args.local_replan_trigger_m:
                    is_segment_free = make_segment_free_check(cached_blocked, elevation["x_min"], elevation["y_min"], elevation["resolution"])
                    if is_segment_free(pos_xy, cached_traj["positions"][idx]):
                        plan_mode = "local"
                        remaining_xy = np.vstack([pos_xy[None, :], cached_traj["positions"][idx:]])
                        traj = build_trajectory(
                            remaining_xy, is_segment_free, shortcut_iters=0, chaikin_iters=0,
                            resample_ds=args.resample_ds, v_max=args.max_vel, a_max=args.max_accel,
                            a_lat_max=args.max_lateral_accel,
                        )

            if plan_mode == "global":
                if planner_obj is not None:
                    path_world, path_grid, blocked, err, stage = planner_obj.plan(pos_xy, max_expansions=args.max_expansions)
                else:
                    path_world, path_grid, blocked, err, stage = plan_2d_with_fallback(
                        elevation, pos_xy, goal_xy, fallback_stages=None if args.use_fallback_ladder else [],
                        cost_weight=args.cost_weight, allow_unknown=args.allow_unknown,
                        robot_radius_m=args.robot_radius_m, inflate_radius_m=args.inflate_radius_m,
                        close_radius_m=args.close_radius_m, max_expansions=args.max_expansions,
                    )
                if path_world is not None:
                    # `blocked` is exactly what the winning attempt (base settings or a fallback
                    # stage) searched over, so the shortcut check can't pass through a cell A*
                    # would never have allowed even when a fallback stage relaxed things.
                    is_segment_free = make_segment_free_check(blocked, elevation["x_min"], elevation["y_min"], elevation["resolution"])
                    traj = build_trajectory(
                        path_world, is_segment_free, shortcut_iters=args.shortcut_iters, chaikin_iters=args.chaikin_iters,
                        resample_ds=args.resample_ds, v_max=args.max_vel, a_max=args.max_accel, a_lat_max=args.max_lateral_accel,
                    )
                    cached_traj, cached_blocked = traj, blocked
                else:
                    cached_traj, cached_blocked = None, None

            if plan_mode == "global":
                n_global_replan += 1
            else:
                n_local_follow += 1
            replan_time_s = time.monotonic() - t_plan0

            if plan_mode == "global" and isinstance(planner_obj, DStarLiteFallbackPlanner) and stage is not None and replan_time_s > spike_threshold_s:
                dsl = planner_obj._get_stage(stage)[0]
                print(
                    f"  [D* Lite SPIKE] query {q}: {replan_time_s*1000:.0f}ms, "
                    f"{dsl.last_num_expansions} real expansions + {dsl.last_num_requeues} stale-key "
                    f"requeues, km={dsl.km:.1f}, queue size after={len(dsl._in_queue)}, "
                    f"rebuilt_this_call={dsl.last_rebuilt}"
                )

            record.update({
                "pos_xy": pos_xy, "dist_to_goal_m": float(np.linalg.norm(goal_xy - pos_xy)),
                "replan_time_s": replan_time_s, "path_found": traj is not None, "error": err, "fallback_stage": stage,
                "plan_mode": plan_mode, "localize_accuracy_m": localize_accuracy_m,
                "path_len_m": float(traj["distance_m"][-1]) if traj is not None else None,
                "path_xy": traj["positions"] if traj is not None else None,
            })
            fallback_note = f" [fallback stage {stage}]" if stage else ""
            status = f"path={record['path_len_m']:.1f}m{fallback_note}" if traj is not None else f"NO PATH ({err})"
            print(
                f"[q {q}/{len(entries)}, frame {e.index}] loc={info.get('mode')} pos={pos_xy.round(1)} "
                f"dist_to_goal={record['dist_to_goal_m']:.1f}m localize={record['localize_time_s']*1000:.0f}ms "
                f"plan={plan_mode} replan={replan_time_s*1000:.0f}ms -> {status}"
            )
        else:
            record.update({
                "pos_xy": None, "dist_to_goal_m": None, "replan_time_s": None, "path_found": False,
                "path_len_m": None, "path_xy": None, "fallback_stage": None, "plan_mode": None,
            })
            print(f"[q {q}/{len(entries)}, frame {e.index}] LOCALIZATION FAILED ({info.get('reason')})")
        records.append(record)

        if q in checkpoint_marks:
            frac = q / len(entries)
            plot_checkpoint(elevation, record, goal_xy, frac, out_dir)

    print(f"\n=== Localization (every query, {len(entries)} total) ===")
    if localize_intervals:
        mean_interval = float(np.mean(localize_intervals))
        print(
            f"Success: {n_success}/{len(entries)}, mean interval between successful "
            f"localizations = {mean_interval*1000:.0f}ms ({1.0/mean_interval:.2f} Hz)"
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


def nearest_point_on_path(pos: np.ndarray, path_xy: np.ndarray) -> tuple[int, float]:
    d = np.linalg.norm(path_xy - pos[None, :], axis=1)
    idx = int(np.argmin(d))
    return idx, float(d[idx])


def plot_checkpoint(elevation: dict, record: dict, goal_xy: np.ndarray, frac: float, out_dir: Path) -> None:
    observed = ~np.isnan(elevation["height"])
    display_cost = np.where(observed, elevation["cost"], np.nan)
    x_min, y_min, res = elevation["x_min"], elevation["y_min"], elevation["resolution"]
    ny, nx = elevation["height"].shape
    extent = [x_min, x_min + nx * res, y_min, y_min + ny * res]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(np.ma.masked_invalid(display_cost), origin="lower", extent=extent, cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, label="traversability cost")
    ax.plot(goal_xy[0], goal_xy[1], "r*", markersize=16, label="fixed goal (from mapping GT)")
    if record["pos_xy"] is not None:
        ax.plot(record["pos_xy"][0], record["pos_xy"][1], "g^", markersize=13, label="current localized position")
    if record["path_xy"] is not None:
        ax.plot(record["path_xy"][:, 0], record["path_xy"][:, 1], "b-", linewidth=2.5, label="replanned route")
    title = f"{frac:.0%} through dataset (frame {record['frame_idx']}), loc={record['localize_mode']}, plan={record['plan_mode']}: "
    title += f"{record['path_len_m']:.1f}m" if record["path_found"] else "no path"
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
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
    ax.set_title("Localization-only online replanning against a static KITTI map")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "summary.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
