import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splg_slam.config import load_config
from splg_slam.data.euroc import load_imu_calibration, load_imu_measurements
from splg_slam.data.loader import dataset_dir, dataset_module
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.retrieval import GlobalDescriptorExtractor, GlobalDescriptorIndex
from splg_slam.map.io import save_map
from splg_slam.mapping.imu_init import run_dynamic_imu_init, run_periodic_imu_reinit
from splg_slam.mapping.imu_preintegration import make_preintegration_params
from splg_slam.mapping.local_ba import local_bundle_adjustment
from splg_slam.mapping.loop_closure import (
    LoopConsistencyTracker,
    detect_loop_candidates,
    fuse_loop_matches,
    verify_loop_candidate,
)
from splg_slam.mapping.pose_graph import (
    odometry_arc_length_m,
    optimize_pose_graph,
    relative_pose,
    relative_pose_discrepancy,
)
from splg_slam.mapping.tracker import OfflineMapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)

    dmod = dataset_module(cfg)
    ddir = dataset_dir(cfg)
    rig = dmod.load_stereo_rig(ddir)
    rectifier = StereoRectifier(rig)
    global_extractor = GlobalDescriptorExtractor() if getattr(cfg, "retrieval", None) and cfg.retrieval.enabled else None
    stereo_ba_baseline = rectifier.baseline if getattr(cfg.mapping, "stereo_ba_enabled", False) else None

    imu_enabled = bool(getattr(cfg, "imu", None) and cfg.imu.enabled)
    imu_calib = load_imu_calibration(cfg.dataset.mav0_dir) if imu_enabled else None
    imu_measurements = load_imu_measurements(cfg.dataset.mav0_dir) if imu_enabled else None
    imu_params = (
        make_preintegration_params(imu_calib, cfg.imu.gravity_norm, cfg.imu.integration_sigma) if imu_enabled else None
    )

    mapper = OfflineMapper(
        cfg, rectifier, global_extractor=global_extractor,
        imu_measurements=imu_measurements, imu_calib=imu_calib,
    )
    # Instead of a hand-picked list of keyframe counts, keep checking periodically and only
    # keep a correction that's at least as self-consistent (by gravity-magnitude error, a
    # ground-truth-free sanity signal) as the last one applied - see run_periodic_imu_reinit.
    next_reinit_check_kf = getattr(cfg.imu, "reinit_min_kf", 20) if imu_enabled else None
    reinit_check_every_kf = getattr(cfg.imu, "reinit_check_every_kf", 15)
    reinit_gravity_tolerance = getattr(cfg.imu, "reinit_gravity_tolerance", 0.08)
    best_reinit_gravity_error: float | None = None

    entries = dmod.load_stereo_frames(ddir)
    stride = cfg.dataset.frame_stride or 1
    entries = entries[::stride]
    if cfg.dataset.max_frames:
        entries = entries[: cfg.dataset.max_frames]

    print(f"Processing {len(entries)} stereo pairs (stride={stride})")

    loop_cfg = getattr(cfg, "loop_closure", None)
    loop_enabled = bool(loop_cfg and loop_cfg.enabled)
    retrieval_index = GlobalDescriptorIndex() if loop_enabled else None
    loop_feats_cache: dict = {}
    loop_consistency = LoopConsistencyTracker(
        required_confirmations=loop_cfg.required_confirmations,
        group_radius_kf=loop_cfg.group_radius_kf,
        pending_timeout_kf=loop_cfg.pending_timeout_kf,
    ) if loop_enabled else None

    n_keyframes = 0
    n_tracked = 0
    n_lost = 0
    n_loops = 0
    n_culled_keyframes = 0
    t0 = time.monotonic()
    for i, e in enumerate(entries):
        t_frame_start = time.monotonic()

        img_l = cv2.imread(str(e.left_path), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(e.right_path), cv2.IMREAD_GRAYSCALE)

        result, is_kf = mapper.process_stereo_pair(img_l, img_r, e.timestamp_ns, image_path=e.left_path)

        if result is None:
            n_lost += 1
        else:
            n_tracked += 1
            if is_kf:
                n_keyframes += 1

                just_reinitialized = False

                if imu_enabled and mapper.imu_init_pending and n_keyframes >= cfg.imu.init_dynamic_window_kf:
                    # Use ALL keyframes so far, not a fixed [:init_dynamic_window_kf] slice: on
                    # the first attempt (n_keyframes just crossed the threshold) this is the
                    # same window as before, but if it fails (e.g. a reloc jump broke the
                    # imu_factor chain inside that window) a fixed slice would retry with the
                    # exact same input forever, deterministically failing every time and
                    # silently disabling IMU for the whole run. Growing the window each retry
                    # gives each subsequent attempt a real chance to route around the gap.
                    init_window = mapper.world_map.keyframe_ids_sorted()
                    diag = run_dynamic_imu_init(mapper.world_map, init_window, imu_calib, imu_params, cfg.imu.gravity_norm)
                    if diag is None:
                        print(f"  dynamic IMU init: IMU-factor chain incomplete over {len(init_window)} keyframes, will retry with more data later")
                    else:
                        mapper.imu_init_pending = False
                        just_reinitialized = True
                        print(
                            f"  dynamic IMU init done over {len(init_window)} keyframes: "
                            f"|g|={diag['gravity_norm_estimated']:.3f} (expected {diag['gravity_norm_expected']:.3f}), "
                            f"gyro_bias={diag['gyro_bias']}"
                        )

                if (
                    imu_enabled and not mapper.imu_init_pending
                    and next_reinit_check_kf is not None and n_keyframes >= next_reinit_check_kf
                ):
                    # ORB-SLAM3-style staged re-optimization (VIBA1/VIBA2): the bootstrap
                    # solve above only sees a small, low-motion-diversity window, and never
                    # solves accelerometer bias at all. Redo the same solve later with much
                    # more accumulated data (and now including accel bias) - runs regardless
                    # of whether the bootstrap took the static or dynamic path. No fixed
                    # window size: keep checking every reinit_check_every_kf keyframes and
                    # only keep a correction that's at least as self-consistent (by gravity-
                    # magnitude error) as the last one applied - a longer window isn't
                    # reliably better, since the solve trusts the vision trajectory as ground
                    # truth and that trajectory's own drift grows with the window too.
                    next_reinit_check_kf += reinit_check_every_kf
                    all_kf_ids = mapper.world_map.keyframe_ids_sorted()
                    diag = run_periodic_imu_reinit(
                        mapper.world_map, all_kf_ids, imu_calib, imu_params, cfg.imu.gravity_norm,
                        gravity_error_tolerance=reinit_gravity_tolerance, best_gravity_error_so_far=best_reinit_gravity_error,
                    )
                    if diag is None:
                        print(f"  IMU reinit check @ {n_keyframes}kf: no surviving IMU-factor chain, skipped")
                    elif diag["accepted"]:
                        best_reinit_gravity_error = diag["gravity_error"]
                        just_reinitialized = True
                        print(
                            f"  IMU reinit ACCEPTED @ {n_keyframes}kf, over {diag['num_keyframes']} keyframes "
                            f"({diag['num_pairs']} pairs): |g|={diag['gravity_norm_estimated']:.3f} "
                            f"(expected {diag['gravity_norm_expected']:.3f}, error {diag['gravity_error']:.1%}), "
                            f"gyro_bias={diag['gyro_bias']}, accel_bias={diag['accel_bias']}"
                        )
                    else:
                        print(
                            f"  IMU reinit check @ {n_keyframes}kf: not accepted (error {diag['gravity_error']:.1%}, "
                            f"best so far {best_reinit_gravity_error if best_reinit_gravity_error is not None else float('nan'):.1%})"
                        )

                if cfg.mapping.periodic_local_ba_enabled and n_keyframes % cfg.mapping.local_ba_every_n_kf == 0:
                    window = mapper.world_map.covisible_window(
                        result.frame_id, window_size=cfg.mapping.local_ba_window_size,
                        min_shared=cfg.mapping.local_ba_min_shared,
                    )
                    if imu_enabled and not mapper.imu_init_pending:
                        # covisible_window() picks keyframes by shared map points, not
                        # temporal order - most periodic local BA calls would otherwise
                        # include only one end of most imu_factors and skip them entirely
                        # (see local_bundle_adjustment's "both ends touched" requirement).
                        # Pull in the missing temporal neighbor for any factor straddling
                        # the window's boundary, so the IMU chain is actually used
                        # throughout tracking instead of only in the one final global BA.
                        window_set = set(window)
                        imu_neighbors = set()
                        for a, b, _ in mapper.world_map.imu_factors:
                            if a in window_set and b not in window_set and b in mapper.world_map.keyframes:
                                imu_neighbors.add(b)
                            elif b in window_set and a not in window_set and a in mapper.world_map.keyframes:
                                imu_neighbors.add(a)
                        window = window + list(imu_neighbors)
                    if len(window) >= 3:
                        # Skip imu_factors while dynamic init is still pending (world frame
                        # isn't gravity-aligned yet) or right after a (re)init just fired
                        # this same iteration (poses/velocities/points were just rotated in
                        # place - run BA against that on the NEXT keyframe, not immediately).
                        use_imu = imu_enabled and not mapper.imu_init_pending and not just_reinitialized
                        ba_stats = local_bundle_adjustment(
                            mapper.world_map, window, rectifier.K_rect, baseline=stereo_ba_baseline,
                            imu_factors=mapper.world_map.imu_factors if use_imu else None,
                            imu_calib=imu_calib if use_imu else None, imu_params=imu_params if use_imu else None,
                        )
                        if ba_stats["rejected"]:
                            print(
                                f"  local BA around kf{result.frame_id} REJECTED (would move a keyframe "
                                f"{ba_stats['max_pose_shift_m']:.1f}m - degenerate/underconstrained window)"
                            )

                if loop_enabled and n_keyframes % loop_cfg.every_n_kf == 0:
                    retrieval_index.build(mapper.world_map)
                    candidates = detect_loop_candidates(
                        mapper.world_map, retrieval_index, result.frame_id,
                        min_id_gap=loop_cfg.min_id_gap, top_k=loop_cfg.top_k,
                        min_similarity=loop_cfg.min_similarity,
                    )
                    for cand_id, sim in candidates:
                        accepted, pose_cw_loop, n_inliers, loop_matches = verify_loop_candidate(
                            mapper.world_map, rectifier, mapper.splg, result.frame_id, cand_id,
                            min_inliers=loop_cfg.min_inliers,
                            min_inlier_ratio=getattr(loop_cfg, "min_inlier_ratio", 0.0),
                            feats_cache=loop_feats_cache,
                        )
                        if not accepted:
                            continue

                        rel = relative_pose(mapper.world_map.keyframes[cand_id].pose_cw, pose_cw_loop)
                        current_rel = relative_pose(
                            mapper.world_map.keyframes[cand_id].pose_cw, mapper.world_map.keyframes[result.frame_id].pose_cw
                        )
                        trans_diff, rot_diff = relative_pose_discrepancy(rel, current_rel)
                        # A fixed-meters tolerance is only sized right for one specific
                        # trajectory scale (2.5m suits EuRoC's ~80m room-scale loops); scale
                        # it up with the actual arc length traveled since the candidate, so a
                        # multi-km outdoor loop gets a proportionally larger allowance for
                        # genuine accumulated drift instead of vetoing every real closure.
                        arc_length_m = odometry_arc_length_m(mapper.world_map, cand_id, result.frame_id)
                        trans_tol = max(
                            loop_cfg.max_consistency_trans_m,
                            getattr(loop_cfg, "max_consistency_trans_ratio", 0.0) * arc_length_m,
                        )
                        if trans_diff > trans_tol or rot_diff > loop_cfg.max_consistency_rot_deg:
                            print(
                                f"  loop candidate kf{result.frame_id} <-> kf{cand_id} rejected "
                                f"(sim={sim:.3f}, inliers={n_inliers}): disagrees with odometry by "
                                f"{trans_diff:.2f}m / {rot_diff:.1f}deg (tolerance {trans_tol:.2f}m over "
                                f"{arc_length_m:.0f}m arc) - likely perceptual aliasing"
                            )
                            continue

                        if not loop_consistency.observe(cand_id, result.frame_id, rel):
                            print(
                                f"  loop candidate kf{result.frame_id} <-> kf{cand_id} "
                                f"(sim={sim:.3f}, inliers={n_inliers}) verified but awaiting re-confirmation"
                            )
                            continue

                        n_fused = 0
                        if getattr(loop_cfg, "fusion_enabled", True):
                            n_fused = fuse_loop_matches(mapper.world_map, cand_id, result.frame_id, loop_matches)

                        mapper.world_map.add_loop_edge(cand_id, result.frame_id, rel, n_inliers)
                        pg_stats = optimize_pose_graph(mapper.world_map, loop_min_inliers=loop_cfg.min_inliers)
                        if pg_stats["rejected"]:
                            print(
                                f"  loop candidate kf{result.frame_id} <-> kf{cand_id} confirmed but pose-graph "
                                f"update REJECTED (would move a keyframe {pg_stats['max_pose_shift_m']:.1f}m - "
                                f"likely a degenerate solve elsewhere in the graph)"
                            )
                            continue
                        n_loops += 1
                        print(
                            f"  loop closure: kf{result.frame_id} <-> kf{cand_id} "
                            f"(sim={sim:.3f}, inliers={n_inliers}, fused={n_fused} points) -> pose graph optimized, "
                            f"error {pg_stats['initial_error']:.1f} -> {pg_stats['final_error']:.1f}"
                        )
                        break

                if cfg.mapping.keyframe_culling_enabled and n_keyframes % cfg.mapping.keyframe_culling_every_n_kf == 0:
                    window = mapper.world_map.covisible_window(
                        result.frame_id, window_size=cfg.mapping.local_ba_window_size,
                        min_shared=cfg.mapping.local_ba_min_shared,
                    )
                    # Not protecting imu_factors' endpoints here (unlike loop_edges): with IMU
                    # enabled, nearly every consecutive keyframe pair has one, so doing so
                    # would protect almost the whole map and defeat culling entirely. A
                    # culled keyframe just leaves a gap in the IMU factor chain instead (the
                    # same graceful-degradation the reloc/loop-closure gaps already rely on) -
                    # local_bundle_adjustment skips any imu_factor whose endpoint is gone.
                    protected = {0, result.frame_id, mapper.ref_keyframe.frame_id}
                    for a, b, _, _ in mapper.world_map.loop_edges:
                        protected.add(a)
                        protected.add(b)
                    candidates = [kf_id for kf_id in window if kf_id not in protected]
                    redundant = mapper.world_map.find_redundant_keyframes(
                        candidates,
                        min_observers=cfg.mapping.keyframe_culling_min_observers,
                        redundancy_ratio=cfg.mapping.keyframe_culling_redundancy_ratio,
                    )
                    for kf_id in redundant:
                        mapper.world_map.remove_keyframe(kf_id)
                    if redundant:
                        n_culled_keyframes += len(redundant)
                        if mapper.reloc_index is not None:
                            mapper.reloc_index.build(mapper.world_map)
                        print(f"  culled {len(redundant)} redundant keyframe(s): {redundant}")

        mapper.world_map.frame_processing_times_s.append(time.monotonic() - t_frame_start)

        if (i + 1) % 200 == 0 or i == len(entries) - 1:
            elapsed = time.monotonic() - t0
            fps = (i + 1) / elapsed
            print(
                f"[{i + 1}/{len(entries)}] keyframes={n_keyframes} (culled={n_culled_keyframes}) "
                f"map_points={len(mapper.world_map.map_points)} lost={n_lost} loops={n_loops} "
                f"reloc={mapper.n_relocalizations} ({fps:.1f} fps)"
            )

    print(
        f"Done. tracked={n_tracked} lost={n_lost} keyframes_inserted={n_keyframes} "
        f"culled={n_culled_keyframes} keyframes_remaining={len(mapper.world_map.keyframes)} "
        f"map_points={len(mapper.world_map.map_points)} loops={n_loops} reloc={mapper.n_relocalizations}"
    )

    if imu_enabled and mapper.imu_init_pending:
        print("  dynamic IMU init never completed (chain never reached the required window) - "
              "final BA will run vision-only, without IMU factors")

    if getattr(cfg.mapping, "final_global_ba", False) and n_keyframes >= 3:
        t0 = time.monotonic()
        all_kf_ids = mapper.world_map.keyframe_ids_sorted()
        use_imu = imu_enabled and not mapper.imu_init_pending
        stats = local_bundle_adjustment(
            mapper.world_map, all_kf_ids, rectifier.K_rect, baseline=stereo_ba_baseline,
            loop_edges=mapper.world_map.loop_edges, loop_min_inliers=loop_cfg.min_inliers if loop_enabled else 60,
            imu_factors=mapper.world_map.imu_factors if use_imu else None,
            imu_calib=imu_calib if use_imu else None, imu_params=imu_params if use_imu else None,
        )
        print(
            f"Final global BA ({len(all_kf_ids)} keyframes, {stats['num_points']} points, "
            f"{len(mapper.world_map.loop_edges)} loop edges, {stats['num_outliers_removed']} outlier obs removed): "
            f"error {stats['initial_error']:.1f} -> {stats['final_error']:.1f} ({time.monotonic() - t0:.1f}s)"
        )

    out_path = Path(cfg.output.map_dir) / "map.pkl"
    save_map(mapper.world_map, out_path)
    print(f"Map saved to {out_path}")


if __name__ == "__main__":
    main()
