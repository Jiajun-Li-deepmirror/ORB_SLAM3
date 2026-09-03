import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splg_slam.config import load_config
from splg_slam.data.euroc import load_stereo_frames, load_stereo_rig
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.retrieval import GlobalDescriptorExtractor, GlobalDescriptorIndex
from splg_slam.map.io import save_map
from splg_slam.mapping.local_ba import local_bundle_adjustment
from splg_slam.mapping.loop_closure import (
    LoopConsistencyTracker,
    detect_loop_candidates,
    fuse_loop_matches,
    verify_loop_candidate,
)
from splg_slam.mapping.pose_graph import optimize_pose_graph, relative_pose, relative_pose_discrepancy
from splg_slam.mapping.tracker import OfflineMapper

# Approach B: instead of building 5 independent maps and registering them post-hoc (see
# merge_maps_registration.py), feed all 5 sequences through ONE continuous OfflineMapper run,
# back to back. At each dataset boundary the camera "teleports" to wherever the next sequence
# starts - ordinary frame-to-frame tracking against the last keyframe will fail there, which
# is exactly what triggers the existing relocalization fallback (_on_tracking_failure ->
# _try_relocalize, global-descriptor retrieval against the WHOLE map built so far). If the new
# sequence's early flight path crosses anything already mapped, relocalization stitches it
# into the same map automatically; frames before that succeeds are just dropped as "lost",
# same graceful degradation already used for genuine mid-sequence tracking loss. No IMU here -
# concatenating independent recordings' IMU streams has its own discontinuity to solve
# (a real jump in the raw signal at the boundary) that's out of scope for this comparison.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=str, nargs="+", help="configs to concatenate, in order (e.g. euroc_mh01.yaml euroc_mh02.yaml ...)")
    parser.add_argument("--out_dir", type=str, default="results/mh_multi_map")
    args = parser.parse_args()

    configs = [load_config(c) for c in args.configs]
    base_cfg = configs[0]  # calibration/tracking/mapping params shared across the same physical rig

    rig = load_stereo_rig(base_cfg.dataset.mav0_dir)
    rectifier = StereoRectifier(rig)
    global_extractor = (
        GlobalDescriptorExtractor() if getattr(base_cfg, "retrieval", None) and base_cfg.retrieval.enabled else None
    )
    stereo_ba_baseline = rectifier.baseline if getattr(base_cfg.mapping, "stereo_ba_enabled", False) else None

    mapper = OfflineMapper(base_cfg, rectifier, global_extractor=global_extractor)

    # (dataset_index, entry) tuples so we can log dataset boundaries and, later, recover which
    # source dataset a given keyframe/frame came from via its image_path.
    all_entries = []
    for idx, cfg in enumerate(configs):
        entries = load_stereo_frames(cfg.dataset.mav0_dir)
        stride = cfg.dataset.frame_stride or 1
        entries = entries[::stride]
        if cfg.dataset.max_frames:
            entries = entries[: cfg.dataset.max_frames]
        print(f"[{args.configs[idx]}] {len(entries)} stereo pairs")
        all_entries.extend((idx, e) for e in entries)

    print(f"Total {len(all_entries)} stereo pairs across {len(configs)} sequences")

    loop_cfg = getattr(base_cfg, "loop_closure", None)
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
    current_dataset_idx = -1
    t0 = time.monotonic()
    for i, (dataset_idx, e) in enumerate(all_entries):
        if dataset_idx != current_dataset_idx:
            current_dataset_idx = dataset_idx
            print(f"--- switching to {args.configs[dataset_idx]} at global frame {i} ---")

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

                if base_cfg.mapping.periodic_local_ba_enabled and n_keyframes % base_cfg.mapping.local_ba_every_n_kf == 0:
                    window = mapper.world_map.covisible_window(
                        result.frame_id, window_size=base_cfg.mapping.local_ba_window_size,
                        min_shared=base_cfg.mapping.local_ba_min_shared,
                    )
                    if len(window) >= 3:
                        ba_stats = local_bundle_adjustment(mapper.world_map, window, rectifier.K_rect, baseline=stereo_ba_baseline)
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
                        if trans_diff > loop_cfg.max_consistency_trans_m or rot_diff > loop_cfg.max_consistency_rot_deg:
                            print(
                                f"  loop candidate kf{result.frame_id} <-> kf{cand_id} rejected "
                                f"(sim={sim:.3f}, inliers={n_inliers}): disagrees with odometry by "
                                f"{trans_diff:.2f}m / {rot_diff:.1f}deg - likely perceptual aliasing"
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

                if base_cfg.mapping.keyframe_culling_enabled and n_keyframes % base_cfg.mapping.keyframe_culling_every_n_kf == 0:
                    window = mapper.world_map.covisible_window(
                        result.frame_id, window_size=base_cfg.mapping.local_ba_window_size,
                        min_shared=base_cfg.mapping.local_ba_min_shared,
                    )
                    protected = {0, result.frame_id, mapper.ref_keyframe.frame_id}
                    for a, b, _, _ in mapper.world_map.loop_edges:
                        protected.add(a)
                        protected.add(b)
                    candidates = [kf_id for kf_id in window if kf_id not in protected]
                    redundant = mapper.world_map.find_redundant_keyframes(
                        candidates,
                        min_observers=base_cfg.mapping.keyframe_culling_min_observers,
                        redundancy_ratio=base_cfg.mapping.keyframe_culling_redundancy_ratio,
                    )
                    for kf_id in redundant:
                        mapper.world_map.remove_keyframe(kf_id)
                    if redundant:
                        n_culled_keyframes += len(redundant)
                        if mapper.reloc_index is not None:
                            mapper.reloc_index.build(mapper.world_map)
                        print(f"  culled {len(redundant)} redundant keyframe(s): {redundant}")

        mapper.world_map.frame_processing_times_s.append(time.monotonic() - t_frame_start)

        if (i + 1) % 500 == 0 or i == len(all_entries) - 1:
            elapsed = time.monotonic() - t0
            fps = (i + 1) / elapsed
            print(
                f"[{i + 1}/{len(all_entries)}] keyframes={n_keyframes} (culled={n_culled_keyframes}) "
                f"map_points={len(mapper.world_map.map_points)} lost={n_lost} loops={n_loops} "
                f"reloc={mapper.n_relocalizations} ({fps:.1f} fps)"
            )

    print(
        f"Done. tracked={n_tracked} lost={n_lost} keyframes_inserted={n_keyframes} "
        f"culled={n_culled_keyframes} keyframes_remaining={len(mapper.world_map.keyframes)} "
        f"map_points={len(mapper.world_map.map_points)} loops={n_loops} reloc={mapper.n_relocalizations}"
    )

    if getattr(base_cfg.mapping, "final_global_ba", False) and n_keyframes >= 3:
        t0 = time.monotonic()
        all_kf_ids = mapper.world_map.keyframe_ids_sorted()
        stats = local_bundle_adjustment(
            mapper.world_map, all_kf_ids, rectifier.K_rect, baseline=stereo_ba_baseline,
            loop_edges=mapper.world_map.loop_edges, loop_min_inliers=loop_cfg.min_inliers if loop_enabled else 60,
        )
        print(
            f"Final global BA ({len(all_kf_ids)} keyframes, {stats['num_points']} points, "
            f"{len(mapper.world_map.loop_edges)} loop edges, {stats['num_outliers_removed']} outlier obs removed): "
            f"error {stats['initial_error']:.1f} -> {stats['final_error']:.1f} ({time.monotonic() - t0:.1f}s)"
        )

    out_path = Path(args.out_dir) / "map.pkl"
    save_map(mapper.world_map, out_path)
    print(f"Map saved to {out_path}")


if __name__ == "__main__":
    main()
