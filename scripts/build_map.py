import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import cv2
import gtsam
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splg_slam.config import load_config
from splg_slam.data.loader import dataset_dir, dataset_module
from splg_slam.geometry.pose_utils import camera_center, invert_pose, pose_delta
from splg_slam.geometry.stereo import MonoRectifier, StereoRectifier
from splg_slam.localization.retrieval import GlobalDescriptorExtractor, GlobalDescriptorIndex
from splg_slam.map.io import save_map
from splg_slam.mapping.gtsam_utils import gtsam_pose_to_cw, matrix_to_gtsam_pose3, pose_cw_to_body_gtsam
from splg_slam.mapping.imu_init import run_dynamic_imu_init, run_periodic_imu_reinit
from splg_slam.mapping.imu_preintegration import (
    bias_from_vector,
    make_preintegration_params,
    preintegrate,
)
from splg_slam.mapping.local_ba import local_bundle_adjustment
from splg_slam.mapping.loop_closure import (
    LoopConsistencyTracker,
    detect_loop_candidates,
    fuse_loop_matches,
    verify_loop_candidate,
)
from splg_slam.mapping.pose_graph import (
    apply_vertical_axis_correction,
    estimate_vertical_axis_pca,
    odometry_arc_length_m,
    optimize_pose_graph,
    relative_pose,
    relative_pose_discrepancy,
)
from splg_slam.mapping.tracker import OfflineMapper
from splg_slam.utils import set_global_seed


def try_close_loops(
    mapper, rectifier, retrieval_index, loop_cfg, loop_feats_cache, loop_consistency,
    target_frame_id: int, require_confirmation: bool = True, verbose_rejections: bool = False,
) -> int:
    """Detects and (if verified) closes loops for `target_frame_id`. Returns the number of
    loops actually closed (0 or 1 - stops at the first accepted candidate, same as the
    original inline loop this was extracted from).

    require_confirmation=False skips loop_consistency's "seen this candidate twice"
    requirement entirely - used for the one extra pass run right after the whole sequence
    finishes (see main()'s "final loop closure pass"): a candidate discovered only on (or
    near) the very last keyframe can never get a second, independent confirmation, because
    there's no next keyframe left to provide it - normally that means a real, high-
    confidence closure right at the end of a run just silently never fires. Everything
    else (min_inliers/inlier_ratio geometric verification, the same/cross-segment
    consistency check) still applies unchanged - this only removes the "seen twice" bar.

    verbose_rejections=True prints why each candidate failed *geometric* verification
    (retrieval found it visually similar enough, but too few/too sparse a slice of 2D-3D
    correspondences survived LightGlue+PnP+RANSAC) - normally silent, since during regular
    per-frame operation most candidates are close, harmless near-misses and printing every
    one would drown the log. Turned on for the orphan-segment-reconciliation pass (see
    main()): when a whole Atlas segment never merges, "retrieval never found anything
    similar enough" and "found good candidates but they all failed geometric verification"
    look identical from the outside (both end in 0 merges) without this - worth the
    verbosity for a pass that, by construction, only runs on keyframes already suspected of
    being a real problem."""
    retrieval_index.build(mapper.world_map)
    # min_arc_length_m default derived from the tracking config's own keyframe-density
    # floor instead of a flat 0.0 (disabled unless a dataset opts in): the problem this
    # filters - a candidate far apart in keyframe ID but only a few meters away in real
    # distance, because keyframes get inserted densely during slow/repetitive local motion
    # - scales directly with how many meters apart consecutive keyframes typically are, so
    # deriving it as a multiple of that (3x, chosen to reproduce realsense's own previously
    # hand-tuned 3.0m given its 1.0m translation fallback) adapts automatically instead of
    # needing a per-dataset guess: ~0.9m for EuRoC (0.3m keyframe_translation_m), ~3.0m for
    # KITTI/realsense (1.0m fallback floor) - all small enough to be a no-op for any
    # dataset's genuine loop closures (tens to hundreds of meters), only filtering the
    # near-zero-distance false ones this was designed for.
    # Reads the mapper's own EFFECTIVE fallback (post auto-estimation, if that's active -
    # see tracker.py's __init__), not the static config field: once keyframe_translation_
    # fallback_m itself is left unset to auto-derive, the config field alone would read as
    # 0.0 here and silently under-derive this too, even though the mapper has long since
    # settled on a real, non-zero effective value from its own depth-based bootstrap.
    default_min_arc_length_m = 3.0 * max(
        getattr(mapper.cfg.tracking, "keyframe_translation_m", 0.3),
        getattr(mapper, "_effective_keyframe_translation_fallback_m", 0.0),
    )
    configured_min_arc_length_m = getattr(loop_cfg, "min_arc_length_m", None)
    candidates = detect_loop_candidates(
        mapper.world_map, retrieval_index, target_frame_id,
        min_id_gap=loop_cfg.min_id_gap, top_k=loop_cfg.top_k,
        min_similarity=loop_cfg.min_similarity,
        min_arc_length_m=(
            configured_min_arc_length_m if configured_min_arc_length_m is not None else default_min_arc_length_m
        ),
    )
    for cand_id, sim in candidates:
        accepted, pose_cw_loop, n_inliers, loop_matches = verify_loop_candidate(
            mapper.world_map, rectifier, mapper.splg, target_frame_id, cand_id,
            min_inliers=loop_cfg.min_inliers,
            min_inlier_ratio=getattr(loop_cfg, "min_inlier_ratio", 0.0),
            feats_cache=loop_feats_cache,
        )
        if not accepted:
            if verbose_rejections:
                # verify_loop_candidate's returned n_inliers means one of three different
                # things depending on which check tripped (see its docstring): the raw
                # 2D-3D correspondence count (rejected before PnP even ran), 0 (PnP itself
                # failed to converge), or a genuine post-RANSAC inlier count (ran fine, just
                # under min_inliers/min_inlier_ratio) - report the raw LightGlue match
                # count too so it's clear which case this was.
                print(
                    f"    candidate kf{target_frame_id} <-> kf{cand_id} (sim={sim:.3f}) failed "
                    f"geometric verification: {len(loop_matches)} LightGlue match(es), "
                    f"n_inliers={n_inliers} (need >={loop_cfg.min_inliers} 2D-3D "
                    f"correspondences with a map point to even attempt PnP, then that many "
                    f"post-RANSAC inliers with ratio >={getattr(loop_cfg, 'min_inlier_ratio', 0.0)} "
                    "to accept)"
                )
            continue

        rel = relative_pose(mapper.world_map.keyframes[cand_id].pose_cw, pose_cw_loop)
        same_segment = (
            mapper.world_map.keyframes[cand_id].segment_id
            == mapper.world_map.keyframes[target_frame_id].segment_id
        )
        if same_segment:
            current_rel = relative_pose(
                mapper.world_map.keyframes[cand_id].pose_cw, mapper.world_map.keyframes[target_frame_id].pose_cw
            )
            trans_diff, rot_diff = relative_pose_discrepancy(rel, current_rel)
            # A fixed tolerance is only sized right for one specific trajectory scale (2.5m/
            # 25deg suits EuRoC's ~80m room-scale loops); scale both up with the actual arc
            # length traveled since the candidate, so a long exploration - hundreds of
            # meters of real stereo-only drift, not just a multi-km outdoor loop - gets a
            # proportionally larger allowance instead of vetoing every real closure once
            # enough distance has accumulated (a fixed few-meter/few-degree budget is right
            # for a short loop and hopelessly tight for a long one).
            arc_length_m = odometry_arc_length_m(mapper.world_map, cand_id, target_frame_id)
            # max_consistency_trans_ratio default changed from 0.0 (disabled unless a dataset
            # opts in) to 0.02 (2%) - this isn't really per-dataset tuning, it's a standard
            # stereo-VO drift-rate budget (the same convention KITTI's own official benchmark
            # reports error in) that degrades to a no-op for a short trajectory anyway: the
            # max() with the fixed floor means it only ever binds once arc_length_m is large
            # enough for 2% of it to exceed the fixed tolerance, which never happens on a
            # room-scale EuRoC/realsense trajectory but matters on KITTI's km-scale ones -
            # previously required KITTI to opt in explicitly, now happens automatically for
            # any long trajectory without hurting short ones. max_consistency_rot_ratio stays
            # 0.0 (no dataset here has ever needed a rotation-vs-arc-length budget).
            trans_tol = max(
                loop_cfg.max_consistency_trans_m,
                getattr(loop_cfg, "max_consistency_trans_ratio", 0.02) * arc_length_m,
            )
            rot_tol = max(
                loop_cfg.max_consistency_rot_deg,
                getattr(loop_cfg, "max_consistency_rot_ratio", 0.0) * arc_length_m,
            )
            if trans_diff > trans_tol or rot_diff > rot_tol:
                print(
                    f"  loop candidate kf{target_frame_id} <-> kf{cand_id} rejected "
                    f"(sim={sim:.3f}, inliers={n_inliers}): disagrees with odometry by "
                    f"{trans_diff:.2f}m / {rot_diff:.1f}deg (tolerance {trans_tol:.2f}m / "
                    f"{rot_tol:.1f}deg over {arc_length_m:.0f}m arc) - likely perceptual aliasing"
                )
                continue
            # EXPERIMENTAL: rather than a soft pose-graph edge (which can dump the
            # correction wherever the graph is least constrained - see optimize_pose_graph's
            # comment), a same-segment closure whose arc is long enough gets the same
            # treatment as a cross-Atlas-segment merge below: weld the whole unanchored
            # excursion since the candidate as one rigid block. See rigid_excursion_ids.
            rigid_transform_threshold = getattr(loop_cfg, "rigid_transform_arc_threshold_m", None)
            rigid_excursion = rigid_transform_threshold is not None and arc_length_m > rigid_transform_threshold
        # Cross-Atlas-segment candidate: skip the odometry-agreement check entirely.
        # Unlike a same-segment relative pose (backed by continuous tracking, so
        # "disagrees with odometry" is real evidence of aliasing), the "odometry" between
        # two different segments is only as good as the dead-reckoned/IMU-bridged guess
        # that seeded the newer segment - it can legitimately be very wrong (see
        # _start_new_map_segment), so comparing a well-verified visual match against it
        # would reject genuine closures for the wrong reason. verify_loop_candidate's
        # min_inliers/inlier_ratio (already checked above) plus loop_consistency's
        # required-confirmations below remain as the safety net instead.
        else:
            rigid_excursion = False
            print(
                f"  loop candidate kf{target_frame_id} <-> kf{cand_id} spans map "
                f"segments {mapper.world_map.keyframes[target_frame_id].segment_id} and "
                f"{mapper.world_map.keyframes[cand_id].segment_id} - skipping odometry "
                f"consistency check (sim={sim:.3f}, inliers={n_inliers})"
            )

        if require_confirmation and not loop_consistency.observe(cand_id, target_frame_id, rel):
            print(
                f"  loop candidate kf{target_frame_id} <-> kf{cand_id} "
                f"(sim={sim:.3f}, inliers={n_inliers}) verified but awaiting re-confirmation"
            )
            continue

        if not same_segment:
            # ORB-SLAM3-Atlas-style map merge instead of a pose-graph edge: the new
            # segment never claimed to share a coordinate frame with this one (see
            # _start_new_map_segment), so there's no "drift" for a graph edge to correct -
            # weld it on with one rigid transform.
            new_seg_id = mapper.world_map.keyframes[target_frame_id].segment_id
            old_seg_id = mapper.world_map.keyframes[cand_id].segment_id
            t_correction = np.linalg.inv(pose_cw_loop) @ mapper.world_map.keyframes[target_frame_id].pose_cw
            mapper.world_map.transform_segment(new_seg_id, t_correction, old_seg_id)
            if mapper._current_segment_id == new_seg_id:
                # Keep tracking under the surviving segment id from now on, so keyframes
                # created after this merge join it directly instead of each separately
                # re-triggering their own merge one at a time as the map-merge loop
                # catches up to them.
                mapper._current_segment_id = old_seg_id
            n_fused = 0
            if getattr(loop_cfg, "fusion_enabled", True):
                n_fused = fuse_loop_matches(mapper.world_map, cand_id, target_frame_id, loop_matches)
            if mapper.reloc_index is not None:
                mapper.reloc_index.build(mapper.world_map)
            mapper.world_map.add_segment_merge(target_frame_id, cand_id)
            print(
                f"  map merge: kf{target_frame_id} (segment {new_seg_id}) <-> kf{cand_id} "
                f"(segment {old_seg_id}) (sim={sim:.3f}, inliers={n_inliers}, fused={n_fused} "
                f"points) -> segment {new_seg_id} welded onto {old_seg_id}"
            )
            return 1

        if rigid_excursion:
            # EXPERIMENTAL: same idea as the cross-segment merge above, but there's no
            # segment_id boundary to key off since tracking never actually broke - the
            # excursion is just "every keyframe since we last stood here" (cand_id itself
            # excluded and left untouched as the fixed anchor). NOTE this doesn't try to
            # detect a keyframe in that range that's *also* already fused with map
            # keyframes before cand_id via some earlier, independent closure (would exist
            # if the trajectory backtracked more intricately than a simple there-and-back)
            # - fine for this experiment's out-and-back excursion shape, not a general
            # solution.
            kf_ids_sorted = mapper.world_map.keyframe_ids_sorted()
            excursion_ids = {k for k in kf_ids_sorted if cand_id < k <= target_frame_id}
            t_correction = np.linalg.inv(pose_cw_loop) @ mapper.world_map.keyframes[target_frame_id].pose_cw
            mapper.world_map.transform_keyframes(excursion_ids, t_correction)
            n_fused = 0
            if getattr(loop_cfg, "fusion_enabled", True):
                n_fused = fuse_loop_matches(mapper.world_map, cand_id, target_frame_id, loop_matches)
            mapper.world_map.add_loop_edge(cand_id, target_frame_id, rel, n_inliers)
            pg_stats = optimize_pose_graph(mapper.world_map, loop_min_inliers=loop_cfg.min_inliers)
            if mapper.reloc_index is not None:
                mapper.reloc_index.build(mapper.world_map)
            mapper.world_map.add_segment_merge(target_frame_id, cand_id)
            print(
                f"  rigid excursion weld: kf{target_frame_id} <-> kf{cand_id} "
                f"(arc {arc_length_m:.0f}m, {len(excursion_ids)} kf transformed, sim={sim:.3f}, "
                f"inliers={n_inliers}, fused={n_fused} points) -> pose graph polish "
                f"{pg_stats['initial_error']:.1f} -> {pg_stats['final_error']:.1f}"
            )
            return 1

        n_fused = 0
        if getattr(loop_cfg, "fusion_enabled", True):
            n_fused = fuse_loop_matches(mapper.world_map, cand_id, target_frame_id, loop_matches)

        mapper.world_map.add_loop_edge(cand_id, target_frame_id, rel, n_inliers)
        pg_stats = optimize_pose_graph(
            mapper.world_map, loop_min_inliers=loop_cfg.min_inliers,
            odom_arc_scaling_enabled=getattr(mapper.cfg.mapping, "odom_arc_scaling_enabled", False),
            odom_arc_reference_m=getattr(mapper.cfg.mapping, "odom_arc_reference_m", 20.0),
            odom_imu_only_penalty_m=getattr(mapper.cfg.mapping, "odom_imu_only_penalty_m", 0.0),
        )
        if pg_stats["rejected"]:
            print(
                f"  loop candidate kf{target_frame_id} <-> kf{cand_id} confirmed but pose-graph "
                f"update REJECTED (would move a keyframe {pg_stats['max_pose_shift_m']:.1f}m - "
                f"likely a degenerate solve elsewhere in the graph)"
            )
            continue
        print(
            f"  loop closure: kf{target_frame_id} <-> kf{cand_id} "
            f"(sim={sim:.3f}, inliers={n_inliers}, fused={n_fused} points) -> pose graph optimized, "
            f"error {pg_stats['initial_error']:.1f} -> {pg_stats['final_error']:.1f}"
        )
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--seed", type=int, default=0, help="seeds numpy/torch/cv2-RANSAC so repeated runs are reproducible instead of drifting via unseeded RANSAC and GPU-kernel nondeterminism")
    args = parser.parse_args()

    set_global_seed(args.seed)
    cfg = load_config(args.config)

    tracking_mode = getattr(cfg.tracking, "mode", "stereo")
    mono_mode = tracking_mode == "mono"
    if tracking_mode not in ("stereo", "mono"):
        raise ValueError(f"Unknown tracking.mode: {tracking_mode!r}")

    dmod = dataset_module(cfg)
    ddir = dataset_dir(cfg)
    # Only meaningful for dataset.kind: rosbag2 - a bag recorded without /tf_static (e.g. a
    # topic-list slip) can reuse a known-good recording of the *same physical rig* instead,
    # since extrinsics are a fixed hardware property. euroc/kitti's load_stereo_rig/
    # load_imu_calibration don't accept this kwarg at all, so it's only ever passed when set.
    tf_kwargs = {"tf_static_from": cfg.dataset.tf_static_from} if getattr(cfg.dataset, "tf_static_from", None) else {}
    rig = dmod.load_stereo_rig(ddir, **tf_kwargs)  # still needed for cam0's own calibration/T_BS even in mono_mode
    rectifier = MonoRectifier(rig.cam0) if mono_mode else StereoRectifier(rig)
    global_extractor = GlobalDescriptorExtractor() if getattr(cfg, "retrieval", None) and cfg.retrieval.enabled else None
    stereo_ba_baseline = (
        # Default True: measured to help substantially wherever tested this session (real
        # stereo+depth reprojection constraints in BA instead of just monocular ones) - the
        # 17/95 configs that still explicitly set this false predate that finding or have a
        # dataset-specific reason (mono mode aside, where it's a no-op either way).
        rectifier.baseline if not mono_mode and getattr(cfg.mapping, "stereo_ba_enabled", True) else None
    )

    imu_enabled = bool(getattr(cfg, "imu", None) and cfg.imu.enabled)
    imu_calib = dmod.load_imu_calibration(ddir, **tf_kwargs) if imu_enabled and hasattr(dmod, "load_imu_calibration") else None
    if imu_calib is not None and getattr(cfg.imu, "t_cam0_body_override", None) is not None:
        # DIAGNOSTIC: override just the IMU->camera extrinsic (not the whole /tf_static
        # graph, unlike dataset.tf_static_from - that swaps the STEREO rig extrinsic too,
        # confirmed to corrupt tracking when the two recordings use different rectification
        # conventions). Used to test a real Kalibr-calibrated T_cam_imu (pulled directly off
        # the device, /app/calibration/calibr_info/imu_cam-camchain-imucam.yaml) against this
        # bag's own recorded IMU data, since this bag's own /tf_static publishes a suspect
        # value for its IMU frame (see imu_init discussion) - only touches imu_calib, leaves
        # rig/rectification untouched entirely.
        imu_calib.T_cam0_body = np.array(cfg.imu.t_cam0_body_override, dtype=np.float64)
        print(f"  IMU calib: T_cam0_body overridden by config (imu.t_cam0_body_override), ignoring this bag's own /tf_static value for it")

    # Noise-model override: euroc/kitti's load_imu_calibration reads real per-device
    # accel/gyro noise density + random walk from the dataset's own sensor.yaml; rosbag2's
    # doesn't have one to read (see splg_slam.data.rosbag_common's IMU_*_NOISE_DENSITY/
    # IMU_*_RANDOM_WALK module constants) and falls back to RealSense D4xx-family
    # datasheet placeholders regardless of which physical IMU actually recorded the bag.
    # Feeding CombinedImuFactor the wrong device's noise model isn't merely imprecise -
    # this codebase's own sibling project (tinynav_slam) measured directly that using
    # RealSense-tuned noise densities for a different IMU (EuRoC's ADIS16448, whose real
    # values are ~250-2950x smaller) made ATE *worse* than pure-stereo, not just neutral -
    # so this is worth overriding whenever a real per-device number is available, not just
    # a diagnostic curiosity.
    #
    # `calibration_yaml`: path to a flat YAML with any of accelerometer_noise_density/
    # accelerometer_random_walk/gyroscope_noise_density/gyroscope_random_walk (the exact
    # key names EuRoC's own imu0/sensor.yaml uses, and also the ones Kalibr's own INPUT
    # imu.yaml for its camera-imu calibration step uses - so a Kalibr imu.yaml can be
    # pointed at directly). Only these four noise-model keys are read; this does NOT
    # parse Kalibr's camchain-imucam.yaml *output* (a different, nested per-camera
    # schema) - use t_cam0_body_override above for an extrinsic pulled from there.
    # `*_noise_density_override`/`*_random_walk_override`: individual scalar overrides,
    # applied after calibration_yaml so they can fine-tune on top of (or fully replace)
    # a loaded file without needing a whole separate YAML for a single known-good number.
    if imu_calib is not None:
        calibration_yaml_path = getattr(cfg.imu, "calibration_yaml", None)
        if calibration_yaml_path:
            with open(calibration_yaml_path) as f:
                _ext_imu_yaml = yaml.safe_load(f)
            _applied = []
            for _yaml_key, _field in (
                ("gyroscope_noise_density", "gyro_noise_density"),
                ("gyroscope_random_walk", "gyro_random_walk"),
                ("accelerometer_noise_density", "accel_noise_density"),
                ("accelerometer_random_walk", "accel_random_walk"),
            ):
                if _yaml_key in _ext_imu_yaml:
                    setattr(imu_calib, _field, float(_ext_imu_yaml[_yaml_key]))
                    _applied.append(_field)
            print(f"  IMU calib: loaded {_applied or 'nothing recognized'} from imu.calibration_yaml={calibration_yaml_path}")

        for _cfg_name, _field in (
            ("gyro_noise_density_override", "gyro_noise_density"),
            ("gyro_random_walk_override", "gyro_random_walk"),
            ("accel_noise_density_override", "accel_noise_density"),
            ("accel_random_walk_override", "accel_random_walk"),
        ):
            _val = getattr(cfg.imu, _cfg_name, None)
            if _val is not None:
                setattr(imu_calib, _field, float(_val))
                print(f"  IMU calib: {_field} overridden by config (imu.{_cfg_name}) = {_val}")
    imu_measurements = dmod.load_imu_measurements(ddir) if imu_enabled and hasattr(dmod, "load_imu_measurements") else None
    accel_scale_correction = getattr(cfg.imu, "accel_scale_correction", None) if imu_enabled else None
    if imu_measurements is not None and accel_scale_correction is None:
        # Auto-detect instead of requiring a hand-measured per-device constant in every
        # config: an uncalibrated accelerometer's static-window magnitude directly IS the
        # scale error (originally found this way for a realsense D435i unit with no factory
        # IMU calibration at all - confirmed via rs-enumerate-devices -c: Motion Intrinsic
        # "Sensitivity" is the identity matrix - static accel norm read ~9.598-9.611 m/s^2
        # vs true gravity 9.81007, a consistent ~2.17-2.24% low scale across three separate
        # bags/a live capture, previously hand-entered as imu.accel_scale_correction per
        # config). Reuses the exact same leading-static-window search choose_imu_init_mode
        # uses for gravity alignment, so this costs nothing extra and stays consistent with
        # what the tracker itself will treat as "static" - if that window ISN'T actually
        # still enough (e.g. EuRoC's machine_hall sequences, handled/moved before takeoff),
        # there's no reliable reference to measure a scale error against, so this leaves the
        # accelerometer unscaled (1.0) rather than guessing - same behavior as not setting
        # this at all previously. An explicit imu.accel_scale_correction in the config still
        # overrides this entirely (checked above), for a device where auto-detection turns
        # out to be unreliable or a bag with no genuinely static leading window at all.
        # NOT the same gyro threshold choose_imu_init_mode uses for gravity ALIGNMENT (that
        # needs genuine staticness - any real rotation during the window corrupts the
        # DIRECTION estimate). Scale correction only needs the window's average accel
        # MAGNITUDE, which is far more tolerant of modest residual motion: found directly on
        # a real bag (rosbag2_2026_09_16-22_13_23) that NO window anywhere in the entire
        # ~28000-sample recording ever gets below init_gyro_static_threshold=0.05 (best
        # anywhere: 0.093, best in the leading search range: 0.131) - this device/handling
        # apparently never holds still enough for that strict a bar - yet the measured |g|
        # from that same "not static enough" window (9.596) matches the independently hand-
        # measured value for this exact bag (9.591-9.611) almost exactly, because averaging
        # accel magnitude over accel_window's many samples cancels out random handheld
        # jitter/rotation the way it never would for a direction estimate. A separate,
        # looser threshold (accel_scale_gyro_threshold, default 4x the alignment one) admits
        # this case while still rejecting genuinely dynamic handling (walking, swinging).
        # accel_wobble_tolerance still guards against the OTHER failure mode this alone
        # wouldn't catch - sustained net translation (not rotation) during the window, which
        # low gyro readings are blind to but would still bias the magnitude estimate.
        # Search the WHOLE recording, not just the leading window choose_imu_init_mode
        # itself is limited to (that one specifically needs a window near the START, since
        # it seeds the very first frame's orientation - gravity DIRECTION genuinely can't
        # come from later in the recording). Scale correction has no such constraint: the
        # sensor's own calibration bias is a constant hardware property true for the whole
        # recording, so a genuinely-still moment ANYWHERE (start, middle, or end) is equally
        # valid evidence. Measured directly on rosbag2_2026_09_16-22_13_23: the only window
        # in ~28000 samples clearing both the gyro and wobble bars sits at sample 27900 - the
        # very END of the recording - restricting the search to the leading window (as a
        # first version of this auto-detection did) missed it entirely and left this bag
        # permanently uncorrected.
        window_samples = cfg.imu.init_static_samples
        stride = max(window_samples // 4, 1)
        accel_scale_gyro_threshold = getattr(cfg.imu, "accel_scale_gyro_threshold", 4.0 * cfg.imu.init_gyro_static_threshold)
        accel_wobble_tolerance = getattr(cfg.imu, "accel_scale_wobble_tolerance", 0.4)
        gyro_mag_all = np.linalg.norm(imu_measurements[:, 1:4], axis=1)
        best = None  # (wobble_rms, start, gyro_mag_mean, mean_accel)
        for start in range(0, max(len(imu_measurements) - window_samples, 0) + 1, stride):
            g_win = gyro_mag_all[start:start + window_samples]
            gyro_mag_mean = float(g_win.mean())
            if gyro_mag_mean >= accel_scale_gyro_threshold:
                continue
            a_win = imu_measurements[start:start + window_samples, 4:7]
            mean_a = a_win.mean(axis=0)
            wobble_rms = float(np.sqrt(((a_win - mean_a) ** 2).sum(axis=1).mean()))
            if wobble_rms < accel_wobble_tolerance and (best is None or wobble_rms < best[0]):
                best = (wobble_rms, start, gyro_mag_mean, mean_a)
        if best is not None:
            wobble_rms, start, gyro_mag_mean, mean_a = best
            measured_g = float(np.linalg.norm(mean_a))
            accel_scale_correction = cfg.imu.gravity_norm / measured_g
            print(
                f"  IMU calib: auto-detected accel scale correction {accel_scale_correction:.5f} "
                f"from a window at sample {start}/{len(imu_measurements)} (gyro={gyro_mag_mean:.3f} "
                f"rad/s, wobble={wobble_rms:.3f} m/s^2, measured |g|={measured_g:.4f} vs expected "
                f"{cfg.imu.gravity_norm:.4f}) - override with imu.accel_scale_correction if unreliable"
            )
        else:
            accel_scale_correction = 1.0
            print(
                f"  IMU calib: no window anywhere in the recording still enough to auto-detect "
                f"accel scale correction (gyro<{accel_scale_gyro_threshold:.3f} rad/s and wobble<"
                f"{accel_wobble_tolerance:.3f} m/s^2) - leaving accelerometer unscaled; set "
                f"imu.accel_scale_correction explicitly if known"
            )
    if imu_measurements is not None and accel_scale_correction is not None and accel_scale_correction != 1.0:
        # This device's accelerometer has no factory scale/bias calibration loaded - correct
        # it uniformly here, before it reaches gravity init/preintegration/tight_fusion
        # factors, rather than papering over it by fudging cfg.imu.gravity_norm (which would
        # only patch the static magnitude check and leave every dynamic acceleration sample
        # still mis-scaled).
        imu_measurements[:, 4:7] *= accel_scale_correction
        print(f"  IMU calib: accel readings scaled by {accel_scale_correction} (imu.accel_scale_correction) to correct this unit's uncalibrated accelerometer")
    imu_params = (
        make_preintegration_params(imu_calib, cfg.imu.gravity_norm, cfg.imu.integration_sigma) if imu_enabled else None
    )

    use_sensor_depth = not mono_mode and bool(getattr(cfg.dataset, "use_sensor_depth", False))
    depth_lookup = dmod.load_depth_lookup(ddir) if use_sensor_depth and hasattr(dmod, "load_depth_lookup") else None
    if use_sensor_depth and depth_lookup is None:
        print("warning: dataset.use_sensor_depth=true but no depth0/data.csv found under the dataset dir; falling back to computed stereo disparity")
    elif depth_lookup is not None:
        print(f"using sensor-provided depth0/ ({len(depth_lookup)} frames) instead of computed stereo disparity")

    mapper = OfflineMapper(
        cfg, rectifier, global_extractor=global_extractor,
        imu_measurements=imu_measurements, imu_calib=imu_calib,
        depth_lookup=depth_lookup, mono_mode=mono_mode,
    )
    # Instead of a hand-picked list of keyframe counts, keep checking periodically over a
    # SLIDING window of the most recent reinit_window_kf keyframes (not the segment's whole
    # history - see run_periodic_imu_reinit and this call's comment below for why a growing,
    # unbounded window eventually self-corrupts on a long-enough run).
    next_reinit_check_kf = getattr(cfg.imu, "reinit_min_kf", 20) if mapper.imu_tight_fusion else None
    reinit_check_every_kf = getattr(cfg.imu, "reinit_check_every_kf", 15)
    reinit_window_kf = getattr(cfg.imu, "reinit_window_kf", 100)
    reinit_gravity_tolerance = getattr(cfg.imu, "reinit_gravity_tolerance", 0.08)
    reinit_max_rotation_deg = getattr(cfg.imu, "reinit_max_rotation_deg", 5.0)
    # Independent of gravity-magnitude/implied-rotation: reject a solve (bootstrap or
    # periodic) whose window didn't have enough attitude diversity (roll/pitch, not just
    # yaw or pure translation) to reliably separate gravity direction from accelerometer
    # bias in the first place - see imu_init._rotation_axis_diversity. None (default)
    # disables the check but the metric is still computed and printed either way, so a
    # threshold can be picked from real numbers on your own datasets before enabling it.
    min_rotation_axis_diversity = getattr(cfg.imu, "min_rotation_axis_diversity", None)
    reinit_segment_id = mapper._current_segment_id  # which segment next_reinit_check_kf applies to
    # Same idea as next_reinit_check_kf, but for the bootstrap-style dynamic init
    # (init_dynamic_window_kf keyframes into THIS segment, not a global count) - without
    # this, a re-init segment's very first keyframe would immediately (and repeatedly,
    # every subsequent keyframe) attempt-and-fail a solve with 0-1 pairs available, since
    # imu_init_pending now gets set for every new segment (see tracker.py's
    # _start_new_map_segment), not just the very first one.
    next_dynamic_init_check_kf = getattr(cfg.imu, "init_dynamic_window_kf", 10) if mapper.imu_tight_fusion else None
    # DEFAULT (was env-gated/opt-in until 2026-09-21): ORBSLAM3_STYLE_INIT reproduces
    # ORB-SLAM3's own actual IMU-initialization schedule more literally than
    # CUMULATIVE_REINIT_WINDOW alone did: ORB-SLAM3's LocalMapping::InitializeIMU is called
    # on a fixed, TIME-based schedule (not a keyframe-count cadence) - an initial bootstrap,
    # then VIBA1 at ~5s and VIBA2 at ~15s after map creation (matching the accel-bias-prior
    # staging already implemented in imu_init.py's _estimate_gravity_bias_and_velocities),
    # plus an extra kf20 pre-refine stage before VIBA1 - and then NEVER calls it again for
    # that map (mbIMU_BA2 latches true): from then on, bias/velocity only get refined by the
    # ongoing local-mapping/tracking joint optimization's own IMU factors, not by a fresh
    # closed-form re-solve. When enabled, forces CUMULATIVE_REINIT_WINDOW-style behavior
    # (whole-segment window, matching ORB-SLAM3's own GetAllKeyFrames() scope) AND gates the
    # periodic-reinit call itself to fire at most twice per segment (VIBA1/VIBA2), timed by
    # elapsed wall-clock time since the segment's first keyframe rather than keyframe count,
    # permanently skipping it for the rest of that segment's life afterward.
    #
    # Made the default flow for every dataset by explicit decision (2026-09-21), despite
    # measured evidence that it's a net accuracy REGRESSION on well-calibrated/feature-rich
    # EuRoC data (real GT ATE, all 5 MH sequences worse: MH01 +24.8%, MH02 +0.5%, MH03 +48.0%
    # [robust-trim; MH03's plain RMSE is separately confounded by a loop closure that only
    # fires on the sequence's last keyframe], MH04 +6.6%, MH05 +4.8%) - it was only a clear
    # win on the RealSense recording (uncalibrated IMU, long texture-poor stretches) that
    # motivated building it in the first place. Set imu.orbslam3_style_init: false in a
    # specific config to opt back into the conservative ratchet-gated default for that
    # dataset if this regression matters there.
    orbslam3_style_init = bool(getattr(cfg.imu, "orbslam3_style_init", True)) and mapper.imu_tight_fusion
    segment_start_ns: int | None = None
    segment_start_kf = 0
    kf20_refine_done = False
    reinit_call_count = 0
    # Per-segment "best accepted gravity error so far" ratchet: gates each periodic check so
    # only a check that IMPROVES on the best-ever accepted result for this segment gets
    # applied, instead of every check that merely clears the fixed gravity_error_tolerance.
    # Without this, a segment can accept many corrections over its lifetime, some
    # considerably worse than an earlier one (e.g. 3-7% error re-perturbing a solve that had
    # earlier locked onto 0.1% error), each one re-rotating/re-scaling the whole segment's
    # keyframes/map points away from an already-good state. Reset alongside
    # reinit_segment_id/next_reinit_check_kf whenever a new segment starts (a fresh segment's
    # best-so-far has no relation to a prior segment's).
    reinit_best_gravity_error: float | None = None
    # Tracked alongside reinit_best_gravity_error to make the ratchet a Pareto frontier over
    # BOTH gravity-magnitude accuracy and motion diversity, not magnitude alone - see
    # _solve_and_realign's own docstring on why a magnitude-only ratchet lets an early, low-
    # diversity "lucky" window permanently block every later, better-conditioned solve.
    reinit_best_axis_diversity: float | None = None
    # EXPERIMENTAL (env-gated, "Option D"): the ratchet above protects against accepting a
    # worse correction, but has no way to "try again" once a segment's best-ever result is
    # locked in - a long segment with no later comparably-diverse window just accumulates
    # unconstrained drift forever after that point (confirmed on one dataset: segment froze
    # at t=46s, 47s later off by 2.14m/27.7deg with no correction in between). RATCHET_RESET_EVERY_KF
    # (env var, keyframes), if set, periodically forgets reinit_best_gravity_error entirely so
    # the next periodic check is judged only against the fixed gravity_error_tolerance again,
    # instead of against a possibly very-lucky historical best.
    #
    # A softened variant (linearly relaxing the bar between the historical best and the full
    # tolerance over a window, instead of this all-or-nothing reset) was also tried, along
    # with also relaxing max_rotation_deg the same way - both measured WORSE than this plain
    # hard reset on one dataset (z_range 1.895m and 1.991m respectively vs 0.992m here).
    # Relaxing max_rotation_deg specifically let through corrections with clearly implausible
    # accel_bias (e.g. -0.35 m/s^2) that the un-relaxed 5deg cap exists precisely to catch -
    # that check stays fixed regardless of staleness. Reverted back to this simpler version.
    ratchet_reset_every_kf = int(os.environ["RATCHET_RESET_EVERY_KF"]) if os.environ.get("RATCHET_RESET_EVERY_KF") else None
    last_reinit_accept_kf = 0
    # Per-segment "last quality-gated bias" - the most recent accel/gyro bias from a
    # dynamic-init or periodic-reinit solve that actually PASSED its acceptance checks
    # (gravity-magnitude tolerance, rotation-axis-diversity, etc. - see imu_init.py's
    # _solve_and_realign) for that segment. Used by the fallback-stitch IMU bridge below in
    # preference to an anchor keyframe's own raw KeyFrame.imu_bias, which is instead
    # whatever the most recent *periodic local BA* pass wrote back (local_ba.py's
    # _ensure_imu_state, every local_ba_every_n_kf keyframes, no acceptance gate at all) -
    # confirmed on one dataset to drift to physically implausible values (e.g. -1.2 m/s^2 on
    # one axis) after a long segment with low rotation-axis diversity, since accel bias and
    # gravity/scale are only weakly observable without enough attitude diversity (see
    # _estimate_gravity_bias_and_velocities's docstring) and local BA's per-keyframe write-
    # back has none of the periodic solve's regularization or quality gates to catch that.
    last_accepted_bias: dict[int, np.ndarray] = {}

    entries = dmod.load_mono_frames(ddir) if mono_mode else dmod.load_stereo_frames(ddir)
    stride = cfg.dataset.frame_stride or 1
    if isinstance(entries, list):
        # euroc/kitti: a concrete, already-decoded-path list - slicing gives an exact count.
        entries = entries[::stride]
        if cfg.dataset.max_frames:
            entries = entries[: cfg.dataset.max_frames]
        n_entries = len(entries)
    else:
        # rosbag2: a generator streaming decoded frames straight from the bag (no
        # EuRoC-layout copy on disk) - itertools.islice applies the same stride/limit
        # without needing a len(). The frame count for the progress print below is only
        # a best-effort estimate from the bag's own metadata, since a generator has no len().
        entries = itertools.islice(entries, 0, None, stride)
        if cfg.dataset.max_frames:
            entries = itertools.islice(entries, cfg.dataset.max_frames)
        n_entries = None
        if hasattr(dmod, "expected_frame_count"):
            raw_count = dmod.expected_frame_count(ddir, mono_mode)
            if raw_count is not None:
                n_entries = len(range(0, raw_count, stride))
                if cfg.dataset.max_frames:
                    n_entries = min(n_entries, cfg.dataset.max_frames)

    if mapper.imu_enabled:
        print(f"IMU mode: {'tight fusion (BA factors + dynamic/periodic reinit)' if mapper.imu_tight_fusion else 'gravity-align only (Z axis from the leading static window; no BA coupling)'}")
    print(f"Processing {n_entries if n_entries is not None else 'an unknown number of'} {'mono' if mono_mode else 'stereo'} frames (stride={stride})")

    loop_cfg = getattr(cfg, "loop_closure", None)
    loop_enabled = bool(loop_cfg and loop_cfg.enabled)
    retrieval_index = GlobalDescriptorIndex() if loop_enabled else None
    # Share the mapper's own reloc feature cache instead of a separate dict: every
    # bag-sourced keyframe (image_path=None) is eagerly populated into it at insertion
    # time (see OfflineMapper._insert_keyframe) since there's no on-disk file
    # loop_closure._features_for_keyframe could otherwise re-read from - reusing the same
    # cache also means a keyframe already reloc'd against isn't redundantly
    # re-SPLG-extracted here for loop-closure verification, and vice versa.
    loop_feats_cache: dict = mapper.reloc_feats_cache
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

        if hasattr(e, "left_image"):
            # rosbag2: already-decoded array, no on-disk file to point a keyframe's
            # image_path at (see OfflineMapper._insert_keyframe's eager reloc-feats
            # caching for how relocalization copes with image_path=None).
            img_l = e.left_image
            image_path = None
        else:
            img_l = cv2.imread(str(e.left_path), cv2.IMREAD_GRAYSCALE)
            image_path = e.left_path

        if mono_mode:
            result, is_kf = mapper.process_mono_pair(img_l, e.timestamp_ns, image_path=image_path)
        else:
            img_r = e.right_image if hasattr(e, "right_image") else cv2.imread(str(e.right_path), cv2.IMREAD_GRAYSCALE)
            result, is_kf = mapper.process_stereo_pair(img_l, img_r, e.timestamp_ns, image_path=image_path)

        if result is None:
            n_lost += 1
        else:
            n_tracked += 1
            if is_kf:
                n_keyframes += 1
                if segment_start_ns is None:
                    segment_start_ns = e.timestamp_ns

                if mapper.imu_tight_fusion and mapper._current_segment_id != reinit_segment_id:
                    # A new Atlas segment started - its own IMU state (velocity/bias/gravity
                    # alignment) is being bootstrapped fresh (see tracker.py's
                    # _start_new_map_segment), so the periodic-reinit schedule needs to
                    # restart too (the sliding window below already only ever looks at the
                    # current segment's own recent keyframes, so it doesn't itself need a
                    # reset - only the *timing* of the next check does).
                    reinit_segment_id = mapper._current_segment_id
                    next_reinit_check_kf = n_keyframes - 1 + getattr(cfg.imu, "reinit_min_kf", 20)
                    reinit_best_gravity_error = None
                    reinit_best_axis_diversity = None
                    last_reinit_accept_kf = n_keyframes - 1
                    next_dynamic_init_check_kf = n_keyframes - 1 + getattr(cfg.imu, "init_dynamic_window_kf", 10)
                    segment_start_ns = e.timestamp_ns
                    segment_start_kf = n_keyframes - 1
                    kf20_refine_done = False
                    reinit_call_count = 0

                just_reinitialized = False

                if mapper.imu_tight_fusion and mapper.imu_init_pending and n_keyframes >= next_dynamic_init_check_kf:
                    # Use ALL of the CURRENT Atlas segment's keyframes so far, not a fixed
                    # [:init_dynamic_window_kf] slice: on the first attempt (n_keyframes just
                    # crossed the threshold) this is the same window as before, but if it
                    # fails (e.g. a reloc jump broke the imu_factor chain inside that window)
                    # a fixed slice would retry with the exact same input forever,
                    # deterministically failing every time and silently disabling IMU for the
                    # whole run. Growing the window each retry gives each subsequent attempt a
                    # real chance to route around the gap. Scoped to the current segment only
                    # (not mapper.world_map.keyframe_ids_sorted() globally) - mixing keyframes
                    # from a prior, unrelated-coordinate-frame Atlas segment into one solve
                    # that assumes a single shared gravity vector is meaningless once more
                    # than one segment exists; ORB-SLAM3's own IMU init is likewise always
                    # scoped to mpAtlas->GetCurrentMap() alone, never the whole Atlas.
                    init_window = [
                        kf_id for kf_id in mapper.world_map.keyframe_ids_sorted()
                        if mapper.world_map.keyframes[kf_id].segment_id == mapper._current_segment_id
                    ]
                    diag = run_dynamic_imu_init(
                        mapper.world_map, init_window, imu_calib, imu_params, cfg.imu.gravity_norm,
                        solve_scale=mono_mode,
                        # Mono's bootstrap solve also fixes *scale* (not just rotation), so an
                        # ill-conditioned early window can corrupt the whole map's metric size,
                        # not just its orientation - gate it the same way periodic reinit already
                        # is, rather than applying it unconditionally like stereo/RGBD do.
                        gravity_error_tolerance=reinit_gravity_tolerance if mono_mode else None,
                        min_rotation_axis_diversity=min_rotation_axis_diversity,
                    )
                    if diag is None:
                        print(f"  dynamic IMU init: IMU-factor chain incomplete, gravity sanity check failed, or insufficient rotation-axis diversity over {len(init_window)} keyframes, will retry with more data later")
                    else:
                        mapper.imu_init_pending = False
                        just_reinitialized = True
                        last_accepted_bias[mapper._current_segment_id] = np.concatenate(
                            [diag["accel_bias"], diag["gyro_bias"]]
                        )
                        print(
                            f"  dynamic IMU init done over {len(init_window)} keyframes: "
                            f"|g|={diag['gravity_norm_estimated']:.3f} (expected {diag['gravity_norm_expected']:.3f}), "
                            f"rotation={diag['rotation_angle_deg']:.2f}deg, axis_diversity={diag['rotation_axis_diversity']:.3f}, "
                            f"gyro_bias={diag['gyro_bias']}, accel_bias={diag['accel_bias']}"
                            + (f", scale={diag['scale']:.4f}" if mono_mode else "")
                        )

                if (
                    mapper.imu_tight_fusion and not mapper.imu_init_pending
                    and next_reinit_check_kf is not None and n_keyframes >= next_reinit_check_kf
                ):
                    # ORB-SLAM3-style staged re-optimization (VIBA1/VIBA2): the bootstrap
                    # solve above only sees a small, low-motion-diversity window, and never
                    # solves accelerometer bias at all. Redo the same solve later with much
                    # more accumulated data (and now including accel bias) - runs regardless
                    # of whether the bootstrap took the static or dynamic path. Keeps checking
                    # every reinit_check_every_kf keyframes, but over a SLIDING window of only
                    # the most recent reinit_window_kf keyframes (not the segment's whole
                    # history) - measured directly on a real run: an unbounded, ever-growing
                    # window's own gravity-magnitude error grew from 0.4% (early, short
                    # window) past 80%+ (once the window had grown to span several hundred
                    # keyframes), because the solve trusts the vision trajectory over the
                    # window as ground truth, and that trajectory's OWN accumulated drift
                    # grows right along with the window once it gets long enough - a bounded,
                    # recent window keeps that "trust vision as ground truth over this
                    # stretch" assumption valid regardless of how long the segment has been
                    # running in total, at the cost of losing the long-window's extra motion
                    # diversity (accel bias/gravity direction are that much less certain from
                    # fewer keyframes) - reinit_window_kf is the tunable tradeoff between the
                    # two. The window itself is independent each time (no growing/persistent
                    # state), but the ACCEPTANCE decision still keeps a per-segment "best
                    # gravity error so far" ratchet (reinit_best_gravity_error below): a check
                    # over a different, later window isn't inherently more trustworthy than an
                    # earlier one that already solved cleanly, so only accept a later check if
                    # it's at least as good as the best one already applied - otherwise a noisy
                    # later window (still real IMU data, just a less fortunate stretch of
                    # motion) can re-perturb a segment that was already well-aligned. Measured:
                    # without this ratchet (best_gravity_error_so_far always None), EuRoC MH04
                    # accepted many more corrections through the run (some at 3-7% gravity
                    # error) instead of locking onto an early, excellent one (~0.1%), costing
                    # ~0.015m of RMSE vs b119179 (which had this ratchet, pre-sliding-window).
                    next_reinit_check_kf += reinit_check_every_kf
                    if ratchet_reset_every_kf is not None and n_keyframes - last_reinit_accept_kf >= ratchet_reset_every_kf:
                        print(
                            f"  IMU reinit ratchet reset @ {n_keyframes}kf (RATCHET_RESET_EVERY_KF): "
                            f"{n_keyframes - last_reinit_accept_kf} keyframes since last accepted "
                            f"correction ({reinit_best_gravity_error}) - next check judged only "
                            "against the fixed gravity_error_tolerance"
                        )
                        reinit_best_gravity_error = None
                        reinit_best_axis_diversity = None
                    viba_stage = None
                    if (
                        orbslam3_style_init and not kf20_refine_done
                        and n_keyframes - segment_start_kf >= 20
                    ):
                        # EXTRA (env-gated, layered ON TOP of VIBA1/VIBA2, not counted against
                        # their own 2-call budget): an early, unconditional run_dynamic_imu_init
                        # refinement at kf20 - the segment already gets one bootstrap call at
                        # kf10 (init_dynamic_window_kf, unconditional for stereo - see above),
                        # but that window is tiny and typically low-diversity (measured
                        # axis_diversity=0.012 on one dataset's very first bootstrap). Re-
                        # solving again at kf20 with double the data gives VIBA1 (which won't
                        # fire for several more real-world seconds under the 5s time gate) a
                        # somewhat better-conditioned seed to build on than the raw kf10
                        # bootstrap alone, via the same seed-bias mechanism VIBA1->VIBA2 already
                        # uses (kf_ids_ordered[0]'s stored imu_bias). Keyframe-count-triggered
                        # (not time-gated) since its whole point is to run *before* VIBA1's own
                        # elapsed-time threshold is reached.
                        kf20_refine_done = True
                        segment_kf_ids = [
                            kf_id for kf_id in mapper.world_map.keyframe_ids_sorted()
                            if mapper.world_map.keyframes[kf_id].segment_id == mapper._current_segment_id
                        ]
                        viba_stage = "kf20 refine"
                        diag = run_dynamic_imu_init(
                            mapper.world_map, segment_kf_ids, imu_calib, imu_params, cfg.imu.gravity_norm,
                            solve_scale=mono_mode,
                            # Unlike the ORIGINAL kf10 bootstrap call above (intentionally
                            # unconditional - see its own docstring: a bad bootstrap there still
                            # gets a later correction chance), this kf20-refine/VIBA1/VIBA2 chain
                            # (below too) IS that later chance under orbslam3_style_init - the
                            # plain ratchet-gated periodic reinit path never runs once this mode
                            # is active (see the final `else` branch's own comment: unreachable
                            # here). With no gate at all, a single bad solve anywhere in this
                            # chain was permanent for the rest of the segment - measured directly
                            # on the RealSense run that validated this mode as the project
                            # default: one VIBA2 call landed on |g| error 16.0% with an
                            # accel_bias of [-0.21, -1.58, -0.05] m/s^2 (order-of-magnitude
                            # implausible) and got applied anyway, since nothing here ever
                            # checked. Reusing reinit_gravity_tolerance (the same physical
                            # gravity-magnitude sanity check the plain path already trusts)
                            # rejects a solve this bad instead: it's left pending, keeping
                            # whatever bias the segment already had (from the kf10 bootstrap, or
                            # nothing solved at all yet - not obviously worse than committing to
                            # a wrong one).
                            gravity_error_tolerance=reinit_gravity_tolerance,
                            min_rotation_axis_diversity=min_rotation_axis_diversity,
                            # Catches the OTHER half of the "bad solve" failure mode the
                            # gravity_error_tolerance fix above doesn't: magnitude agreement
                            # alone doesn't mean the DIRECTION is right (see run_dynamic_imu_init's
                            # own docstring) - measured directly on a real RealSense recording,
                            # a kf20-refine + VIBA1 pair that both passed a 0.3% gravity-magnitude
                            # check still baked in a ~3.2deg tilt (confirmed via a plane fit
                            # through the segment's own keyframe positions explaining 78% of its
                            # Z variance). run_periodic_imu_reinit already had this same check;
                            # it just wasn't wired through run_dynamic_imu_init until now.
                            max_rotation_deg=reinit_max_rotation_deg,
                        )
                    elif orbslam3_style_init and reinit_call_count >= 2:
                        # VIBA2 already fired for this segment - ORB-SLAM3 never calls
                        # InitializeIMU again after this point (mbIMU_BA2 latches true);
                        # from here on only the ongoing local-BA/tracking joint optimization's
                        # own IMU factors refine bias/velocity, no more closed-form re-solves.
                        diag = None
                    elif orbslam3_style_init and (e.timestamp_ns - segment_start_ns) / 1e9 < (5.0 if reinit_call_count == 0 else 15.0):
                        # Not yet VIBA1's (~5s) or VIBA2's (~15s) scheduled time - ORB-SLAM3
                        # times these by elapsed wall-clock since map creation, not keyframe
                        # count, so skip this keyframe-count-triggered check and wait for a
                        # later one that does fall past the time threshold.
                        diag = None
                    elif orbslam3_style_init:
                        # TRUE two-stage VIBA1/VIBA2 (not just the single-shot closed-form solve
                        # with a time-dependent Tikhonov weight this same math also supports -
                        # see imu_init.py's own elapsed_s staging comment): each stage is its own
                        # separate call to run_dynamic_imu_init - the SAME unconditional-
                        # application bootstrap function used for the segment's very first
                        # init, not the ratchet/tolerance-gated run_periodic_imu_reinit (ORB-
                        # SLAM3's InitializeIMU has no accept/reject gate at all; VIBA1 and
                        # VIBA2 both just apply whatever they compute at their scheduled time).
                        # VIBA2 automatically builds on VIBA1's OWN refined result as its
                        # starting point, not on whatever bias the segment inherited before
                        # either ran: _solve_and_realign seeds its Tikhonov prior from
                        # kf_ids_ordered[0]'s (the segment's first keyframe's) OWN stored
                        # imu_bias, and VIBA1's write-back already updated exactly that
                        # keyframe (it's always "touched" as the first pair's own left end) -
                        # so as long as both calls' windows start from the same first keyframe
                        # (they do: both are "every keyframe in this segment so far"), VIBA2
                        # reads VIBA1's answer as its seed automatically, with no extra state
                        # threading needed.
                        viba_stage = "VIBA1" if reinit_call_count == 0 else "VIBA2"
                        segment_kf_ids = [
                            kf_id for kf_id in mapper.world_map.keyframe_ids_sorted()
                            if mapper.world_map.keyframes[kf_id].segment_id == mapper._current_segment_id
                        ]
                        reinit_call_count += 1
                        diag = run_dynamic_imu_init(
                            mapper.world_map, segment_kf_ids, imu_calib, imu_params, cfg.imu.gravity_norm,
                            solve_scale=mono_mode,
                            # See the matching comment on the kf20-refine call above - same fix,
                            # same reason (this is VIBA2's own permanent-once-applied risk: it's
                            # the LAST closed-form correction this segment will ever get).
                            gravity_error_tolerance=reinit_gravity_tolerance,
                            min_rotation_axis_diversity=min_rotation_axis_diversity,
                            max_rotation_deg=reinit_max_rotation_deg,  # see matching comment on the kf20-refine call above
                        )
                    else:
                        # Scoped to the current segment only (same reasoning as init_window
                        # above) AND to just its most recent reinit_window_kf keyframes.
                        segment_kf_ids = [
                            kf_id for kf_id in mapper.world_map.keyframe_ids_sorted()
                            if mapper.world_map.keyframes[kf_id].segment_id == mapper._current_segment_id
                        ]
                        # EXPERIMENTAL (env-gated): CUMULATIVE_REINIT_WINDOW uses the segment's
                        # ENTIRE history instead of the bounded recent slice above - closer to
                        # ORB-SLAM3's own InitializeIMU, which solves over ALL keyframes in the
                        # current map (Atlas::GetCurrentMap()->GetAllKeyFrames()), not a sliding
                        # window. Re-testing on this dataset specifically rather than assuming the
                        # historical "grew past 80% gravity error" finding above (from a
                        # different dataset/run, pre-dating this session's other fixes) still
                        # applies as-is.
                        #
                        # NOT reachable at all when orbslam3_style_init is True: the elif chain
                        # above it (kf20 refine / VIBA-done / not-yet-time / VIBA1-VIBA2) has an
                        # unconditional `elif orbslam3_style_init:` arm, so this final `else` only
                        # ever executes for the plain (non-orbslam3_style_init) periodic reinit -
                        # VIBA1/VIBA2's own run_dynamic_imu_init calls above build their own
                        # `segment_kf_ids` directly (always the full, unbounded segment, by
                        # construction, independent of this env var) and never reach here.
                        window_kf_ids = (
                            segment_kf_ids if os.environ.get("CUMULATIVE_REINIT_WINDOW")
                            else segment_kf_ids[-reinit_window_kf:]
                        )
                        diag = run_periodic_imu_reinit(
                            mapper.world_map, window_kf_ids, imu_calib, imu_params, cfg.imu.gravity_norm,
                            gravity_error_tolerance=reinit_gravity_tolerance,
                            best_gravity_error_so_far=None if os.environ.get("DISABLE_RATCHET") else reinit_best_gravity_error,
                            best_axis_diversity_so_far=None if os.environ.get("DISABLE_RATCHET") else reinit_best_axis_diversity,
                            solve_scale=mono_mode, apply_kf_ids=segment_kf_ids,
                            max_rotation_deg=reinit_max_rotation_deg,
                            min_rotation_axis_diversity=min_rotation_axis_diversity,
                        )
                    if diag is None:
                        if viba_stage is not None:
                            # A kf20-refine/VIBA1/VIBA2 solve WAS just attempted this call (see
                            # gravity_error_tolerance added to those run_dynamic_imu_init calls
                            # above) and got rejected by the same sanity check the plain path
                            # trusts - report this distinctly from the "not yet due"/"already
                            # done" cases below (checked first: reinit_call_count is already
                            # bumped past 2 by the time a VIBA2 attempt lands here, so that
                            # branch would otherwise misreport a real rejection as routine
                            # silence). Segment's bias/gravity/scale are left exactly as they
                            # were before this call - not applying a bad solve, not "stuck", the
                            # segment simply keeps whatever it already had.
                            print(
                                f"  {viba_stage} @ {n_keyframes}kf: solve computed but REJECTED "
                                "(failed gravity-magnitude / implied-rotation-angle / "
                                "rotation-diversity sanity check) - not applied, segment's "
                                "bias/gravity left unchanged"
                            )
                        elif orbslam3_style_init and reinit_call_count >= 2:
                            pass  # VIBA1/VIBA2 both already done for this segment - silent from here on, matching ORB-SLAM3's own one-shot-per-stage behavior
                        elif orbslam3_style_init:
                            pass  # not yet VIBA1's/VIBA2's scheduled elapsed-time - silent, will fire once due
                        else:
                            print(f"  IMU reinit check @ {n_keyframes}kf: no surviving IMU-factor chain, skipped")
                    elif viba_stage is not None:
                        just_reinitialized = True
                        last_reinit_accept_kf = n_keyframes
                        last_accepted_bias[mapper._current_segment_id] = np.concatenate(
                            [diag["accel_bias"], diag["gyro_bias"]]
                        )
                        print(
                            f"  {viba_stage} @ {n_keyframes}kf, over {diag['num_keyframes']} keyframes "
                            f"({diag['num_pairs']} pairs): |g|={diag['gravity_norm_estimated']:.3f} "
                            f"(expected {diag['gravity_norm_expected']:.3f}, error {diag['gravity_error']:.1%}), "
                            f"rotation={diag['rotation_angle_deg']:.2f}deg, axis_diversity={diag['rotation_axis_diversity']:.3f}, "
                            f"gyro_bias={diag['gyro_bias']}, accel_bias={diag['accel_bias']}"
                            + (f", scale={diag['scale']:.4f}" if mono_mode else "")
                        )
                    elif diag["accepted"]:
                        just_reinitialized = True
                        # min/max (not plain overwrite): keeps both frontiers of the Pareto
                        # ratchet monotonically non-regressing regardless of WHICH axis let
                        # this particular solve in - see _solve_and_realign's docstring.
                        reinit_best_gravity_error = (
                            diag["gravity_error"] if reinit_best_gravity_error is None
                            else min(reinit_best_gravity_error, diag["gravity_error"])
                        )
                        reinit_best_axis_diversity = (
                            diag["rotation_axis_diversity"] if reinit_best_axis_diversity is None
                            else max(reinit_best_axis_diversity, diag["rotation_axis_diversity"])
                        )
                        last_reinit_accept_kf = n_keyframes
                        last_accepted_bias[mapper._current_segment_id] = np.concatenate(
                            [diag["accel_bias"], diag["gyro_bias"]]
                        )
                        print(
                            f"  IMU reinit ACCEPTED @ {n_keyframes}kf, over {diag['num_keyframes']} keyframes "
                            f"({diag['num_pairs']} pairs): |g|={diag['gravity_norm_estimated']:.3f} "
                            f"(expected {diag['gravity_norm_expected']:.3f}, error {diag['gravity_error']:.1%}), "
                            f"rotation={diag['rotation_angle_deg']:.2f}deg, axis_diversity={diag['rotation_axis_diversity']:.3f}, "
                            f"gyro_bias={diag['gyro_bias']}, accel_bias={diag['accel_bias']}"
                            + (f", scale={diag['scale']:.4f}" if mono_mode else "")
                        )
                    else:
                        print(
                            f"  IMU reinit check @ {n_keyframes}kf: not accepted (gravity error "
                            f"{diag['gravity_error']:.1%}, implied rotation {diag['rotation_angle_deg']:.2f}deg, "
                            f"axis_diversity={diag['rotation_axis_diversity']:.3f})"
                        )

                if cfg.mapping.periodic_local_ba_enabled and n_keyframes % cfg.mapping.local_ba_every_n_kf == 0:
                    window = mapper.world_map.covisible_window(
                        result.frame_id, window_size=cfg.mapping.local_ba_window_size,
                        min_shared=cfg.mapping.local_ba_min_shared,
                    )
                    if mapper.imu_tight_fusion and not mapper.imu_init_pending:
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
                        use_imu = mapper.imu_tight_fusion and not mapper.imu_init_pending and not just_reinitialized
                        ba_stats = local_bundle_adjustment(
                            mapper.world_map, window, rectifier.K_rect, baseline=stereo_ba_baseline,
                            imu_factors=mapper.world_map.imu_factors if use_imu else None,
                            imu_calib=imu_calib if use_imu else None, imu_params=imu_params if use_imu else None,
                            # This same window gets re-optimized again in local_ba_every_n_kf
                            # keyframes and fully re-converged once more by the final global BA -
                            # a periodic pass doesn't need GTSAM's default tight convergence every
                            # time (measured: ~12 LM iterations/call otherwise, ~65% of this
                            # function's own time). See local_bundle_adjustment's docstring.
                            lm_max_iterations=getattr(cfg.mapping, "local_ba_max_iterations", 6),
                            lm_relative_error_tol=getattr(cfg.mapping, "local_ba_relative_error_tol", 1e-2),
                        )
                        if ba_stats["rejected"]:
                            print(
                                f"  local BA around kf{result.frame_id} REJECTED (would move a keyframe "
                                f"{ba_stats['max_pose_shift_m']:.1f}m - degenerate/underconstrained window)"
                            )

                if os.environ.get("DEBUG_PRE_MERGE_FRAME") and i >= int(os.environ["DEBUG_PRE_MERGE_FRAME"]):
                    save_map(mapper.world_map, os.environ["DEBUG_PRE_MERGE_MAP"])
                    del os.environ["DEBUG_PRE_MERGE_FRAME"]

                if loop_enabled and n_keyframes % loop_cfg.every_n_kf == 0:
                    n_loops += try_close_loops(
                        mapper, rectifier, retrieval_index, loop_cfg, loop_feats_cache,
                        loop_consistency, result.frame_id,
                    )

                if cfg.mapping.keyframe_culling_enabled and n_keyframes % cfg.mapping.keyframe_culling_every_n_kf == 0:
                    # Own window size/min_shared, decoupled from local_ba_window_size - the
                    # BA window's size is a compute-cost/local-accuracy tradeoff for
                    # optimization, unrelated to how many candidates culling should even
                    # consider. Reusing it (previously, default 10) meant a densely-inserted
                    # region (this port needs 2-3x ORB-SLAM3's own keyframe count on KITTI -
                    # see tracker.py's _need_new_keyframe docstring) had far more genuinely
                    # redundant keyframes nearby than the top-10-by-covisibility BA window
                    # could ever surface as candidates in one pass, capping how much culling
                    # could actually remove regardless of how permissive redundancy_ratio was.
                    # Default 30, validated on both KITTI (2596kf/1.048m -> a range of
                    # options between 1721-2928kf depending on the paired redundancy_ratio)
                    # and EuRoC (297kf/0.0610m -> 230kf/0.0576m at ratio=0.7, BOTH better).
                    window = mapper.world_map.covisible_window(
                        result.frame_id,
                        window_size=getattr(cfg.mapping, "keyframe_culling_window_size", 30),
                        min_shared=getattr(cfg.mapping, "keyframe_culling_min_shared", cfg.mapping.local_ba_min_shared),
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
                    # Default 0.7 (was a required field, no code default, before this
                    # session's culling investigation): 0.75 under-culled on both datasets
                    # (this port needs 2-3x ORB-SLAM3's own KITTI keyframe count); 0.6 was a
                    # clear net win on KITTI (2596kf/1.048m -> 1721kf/1.0049m, beating even
                    # ORB-SLAM3's own 1.0235m) but cost EuRoC ~9% RMSE (IMU-coupled, more
                    # varied 6DOF motion - a "redundant" keyframe there more often still
                    # carries unique triangulation-relevant viewing angle than on KITTI's
                    # near-pure-forward highway motion). 0.7 is the validated middle ground:
                    # a real win on EuRoC (230kf/0.0576m, better than BOTH 0.75 and 0.6) and
                    # a still-reasonable choice on KITTI.
                    redundant = mapper.world_map.find_redundant_keyframes(
                        candidates,
                        min_observers=cfg.mapping.keyframe_culling_min_observers,
                        redundancy_ratio=getattr(cfg.mapping, "keyframe_culling_redundancy_ratio", 0.7),
                    )
                    for kf_id in redundant:
                        mapper.world_map.remove_keyframe(kf_id)
                    if redundant:
                        n_culled_keyframes += len(redundant)
                        if mapper.reloc_index is not None:
                            mapper.reloc_index.build(mapper.world_map)
                        print(f"  culled {len(redundant)} redundant keyframe(s): {redundant}")

        mapper.world_map.frame_processing_times_s.append(time.monotonic() - t_frame_start)

        if (i + 1) % 200 == 0 or (n_entries is not None and i == n_entries - 1):
            elapsed = time.monotonic() - t0
            fps = (i + 1) / elapsed
            total_str = f"/{n_entries}" if n_entries is not None else ""
            print(
                f"[{i + 1}{total_str}] keyframes={n_keyframes} (culled={n_culled_keyframes}) "
                f"map_points={len(mapper.world_map.map_points)} lost={n_lost} loops={n_loops} "
                f"reloc={mapper.n_relocalizations} ({fps:.1f} fps)"
            )

    if loop_enabled and n_keyframes > 0:
        # One extra check against the very last keyframe, bypassing loop_consistency's
        # "seen this candidate twice" requirement: a genuine, high-confidence closure
        # discovered only this late can never get a second, independent confirmation from
        # a later keyframe - there isn't one, the sequence just ended. Without this, that
        # closure just silently never fires (see the 07:30:12 and 10:03:21 datasets this
        # session - both had a real, high-inlier candidate sitting right at the end that
        # only ever got the "verified but awaiting re-confirmation" print).
        last_kf_id = mapper.world_map.keyframe_ids_sorted()[-1]
        n_final_loops = try_close_loops(
            mapper, rectifier, retrieval_index, loop_cfg, loop_feats_cache,
            loop_consistency, last_kf_id, require_confirmation=False,
        )
        n_loops += n_final_loops
        if n_final_loops:
            print(f"  final loop-closure pass: closed {n_final_loops} loop(s) against the last keyframe")

        # Orphan-Atlas-segment reconciliation: a re-init segment (see
        # OfflineMapper._start_new_map_segment) only ever gets welded to the rest of the
        # map if it happens to be the "current" (just-inserted) side of try_close_loops at
        # the moment a match is found - the per-frame loop above never revisits an OLDER
        # keyframe as the "current" side once processing has moved past it. A genuine
        # match against that segment's own keyframes can still exist in the final,
        # fully-built map (e.g. the matching territory wasn't covered by anything else yet
        # when this segment's keyframes were first inserted) without ever having been
        # checked. Retry every keyframe still in a non-main segment as the "current" side
        # against the fully-built retrieval index; repeat until a full pass makes no
        # further progress (one segment can merge onto another still-orphaned one before
        # THAT one finally reaches the main map, needing more than one pass to resolve
        # transitively). If a segment still can't find anything after this, it's presumably
        # a genuinely isolated excursion with no real overlap with the rest of the map, not
        # a detection failure.
        main_segment_id = mapper.world_map.keyframes[0].segment_id
        n_orphan_merges = 0
        while True:
            orphan_kf_ids = [
                kf_id for kf_id in mapper.world_map.keyframe_ids_sorted()
                if mapper.world_map.keyframes[kf_id].segment_id != main_segment_id
            ]
            if not orphan_kf_ids:
                break
            n_orphan_before = len(orphan_kf_ids)
            for kf_id in orphan_kf_ids:
                if mapper.world_map.keyframes[kf_id].segment_id == main_segment_id:
                    continue  # already welded by an earlier iteration of this same pass
                try_close_loops(
                    mapper, rectifier, retrieval_index, loop_cfg, loop_feats_cache,
                    loop_consistency, kf_id, require_confirmation=False, verbose_rejections=True,
                )
            n_orphan_after = sum(
                1 for kf_id in orphan_kf_ids if mapper.world_map.keyframes[kf_id].segment_id != main_segment_id
            )
            # Progress for THIS loop's purpose is "did any orphan actually join main_segment_id"
            # (a real cross-segment map merge or rigid-excursion weld), NOT try_close_loops'
            # raw return count - that also counts an ordinary same-segment loop edge formed
            # between two keyframes that are BOTH still in the same non-main orphan segment
            # (same_segment=True in try_close_loops, so no transform_segment/weld ever
            # happens). Nothing dedupes an already-added loop edge, so that case kept
            # re-verifying and re-adding the identical edge every single pass forever - a
            # genuine infinite loop (measured directly: one candidate pair alone printed
            # 582 times, 69400 total candidate-check lines from only 302 distinct pairs,
            # on a real dataset with several small mutually-isolated orphan segments) since
            # orphan_kf_ids never shrank and the raw progress sum never hit zero.
            n_orphan_merges += n_orphan_before - n_orphan_after
            if n_orphan_after == n_orphan_before:
                break
        n_loops += n_orphan_merges
        remaining_orphans = sum(
            1 for kf in mapper.world_map.keyframes.values() if kf.segment_id != main_segment_id
        )
        if n_orphan_merges or remaining_orphans:
            print(
                f"  orphan segment reconciliation: {n_orphan_merges} merge(s) found, "
                f"{remaining_orphans} keyframe(s) still isolated (no matching candidate "
                "anywhere in the final map)"
            )

        # Fallback stitch: tinynav_slam's own front-end has no multi-segment concept at
        # all - if tracking degrades badly it just keeps going on a degraded estimate
        # rather than splitting off and potentially losing continuity. Applied here as an
        # explicit LAST RESORT (after exhaustive reconciliation above already tried a
        # genuine visual re-identification and failed): weld any segment still orphaned
        # onto the most recent main-segment keyframe before it - not a verified closure,
        # just "don't leave a chunk of the trajectory permanently stranded in an unrelated
        # coordinate frame". Tagged separately (world_map.fallback_stitches, not
        # segment_merges) so trajectory-status visualization can flag it as distinctly
        # lower-confidence than a real, visually-verified merge.
        #
        # Tried and reverted: (1) also anchoring off a later, already-resolved segment and
        # blending both sides weighted by gap size, and (2) a tinynav-style lenient PnP
        # re-verification pass (much lower correspondence floor, stricter inlier ratio) to
        # use a real visual pose instead of extrapolating when one exists. Both are more
        # principled in theory, but on the one dataset that actually needed this path, the
        # lenient pass never found anything trustworthy (best real candidate was ~35-40%
        # inlier ratio - not self-consistent enough to accept) and loosening its ratio bar
        # far enough to accept something produced a confirmed wrong placement (a long
        # spurious "teleport" jump in the trajectory, checked visually). Simplicity won:
        # back to the single, before-only, constant-velocity guess below.
        #
        # Position: constant-velocity extrapolation from the main segment's own last few
        # keyframes before the anchor, projected forward to the orphan segment's first
        # keyframe's timestamp - NOT a zero-displacement guess (tried that first: measured
        # on one dataset, the camera was still doing ~0.9 m/s right up to the tracking-loss
        # point, with a ~0.9s gap before the orphan segment's first keyframe - assuming
        # zero motion during that gap silently discards ~0.86m of real, inferrable
        # displacement). This is the same idea the old `predicted`-parameter dead-
        # reckoning seed used, just reconstructed after the fact from saved keyframe
        # timestamps/poses instead of live tracker state.
        # Orientation: deliberately NOT extrapolated (kept equal to the anchor's own) -
        # this is the one thing that direct predecessor got wrong for a *different*
        # dataset (a fast rotation right at the loss point extrapolated into a ~33 deg
        # error) - translation errors from a bad velocity estimate degrade far more
        # gracefully than a compounding bad angular-velocity guess would.
        # IMU-bridge lookup: real accelerometer/gyro samples recorded across an Atlas
        # re-init transition (tight_fusion only - see tracker.py's _start_new_map_segment),
        # keyed by (old_kf_id, new_kf_id) i.e. exactly (anchor_kf_id, first_orphan_kf_id)
        # for the "before" case below. When present, this is a real physical measurement
        # (with a growing-but-known uncertainty) of how the two segments relate, not a
        # guess - strictly better-grounded than constant-velocity extrapolation whenever
        # it's available. Only covers the "before" anchor case for now (a two-sided
        # before+after blend, like the constant-velocity one this replaced, would need
        # composing two IMU-bridge predictions expressed in different segments' own frames
        # - deferred to avoid rushing that pose-composition derivation and risking a subtle
        # sign/order bug; single-sided is the same scope this fallback stitch already had).
        imu_transition_samples = {
            (old_kf_id, new_kf_id): samples
            for old_kf_id, new_kf_id, samples in mapper.world_map.segment_transition_imu_samples
        }

        def _imu_bridge_pose(anchor_kf_id: int, edge_kf_id: int) -> np.ndarray | None:
            samples = imu_transition_samples.get((anchor_kf_id, edge_kf_id))
            anchor_kf = mapper.world_map.keyframes[anchor_kf_id]
            if samples is None or anchor_kf.velocity is None:
                return None
            # Prefer this segment's last quality-gated (dynamic-init/periodic-reinit
            # accepted) bias over anchor_kf.imu_bias itself - the latter is whatever the
            # most recent *periodic local BA* pass wrote back (no acceptance gate at all),
            # which can drift to physically implausible values on a long, low-attitude-
            # diversity segment (see last_accepted_bias's own comment above). Falls back to
            # anchor_kf.imu_bias only if this segment never had an accepted solve at all.
            bias_vec = last_accepted_bias.get(anchor_kf.segment_id, anchor_kf.imu_bias)
            bias = bias_from_vector(bias_vec)
            preint = preintegrate(samples, bias, imu_params)
            body_pose = pose_cw_to_body_gtsam(anchor_kf.pose_cw, imu_calib.T_cam0_body)
            predicted = preint.predict(gtsam.NavState(body_pose, anchor_kf.velocity), bias)
            t_cam_body = matrix_to_gtsam_pose3(imu_calib.T_cam0_body)
            cam_wc = predicted.pose().compose(t_cam_body.inverse())
            return gtsam_pose_to_cw(cam_wc)

        velocity_window_s = 0.5
        remaining_orphan_segment_ids = sorted({
            kf.segment_id for kf in mapper.world_map.keyframes.values() if kf.segment_id != main_segment_id
        })
        for seg_id in remaining_orphan_segment_ids:
            seg_kf_ids = {kf_id for kf_id, kf in mapper.world_map.keyframes.items() if kf.segment_id == seg_id}
            first_orphan_kf_id = min(seg_kf_ids)
            anchor_candidates = [
                kf_id for kf_id, kf in mapper.world_map.keyframes.items()
                if kf.segment_id == main_segment_id and kf_id < first_orphan_kf_id
            ]
            if not anchor_candidates:
                continue  # no earlier main-segment keyframe to stitch onto - leave isolated
            anchor_kf_id = max(anchor_candidates)
            anchor_kf = mapper.world_map.keyframes[anchor_kf_id]
            orphan_first_kf = mapper.world_map.keyframes[first_orphan_kf_id]

            desired_pose = _imu_bridge_pose(anchor_kf_id, first_orphan_kf_id)
            if desired_pose is not None:
                source = "IMU bridge (real accel/gyro physics across the tracking-loss gap)"
                velocity = anchor_kf.velocity
                gap_s = (orphan_first_kf.timestamp_ns - anchor_kf.timestamp_ns) / 1e9
            else:
                source = "constant-velocity guess"
                velocity_ref_candidates = [
                    kf_id for kf_id in anchor_candidates
                    if anchor_kf.timestamp_ns - mapper.world_map.keyframes[kf_id].timestamp_ns
                    >= velocity_window_s * 1e9
                ]
                velocity = np.zeros(3)
                if velocity_ref_candidates:
                    velocity_ref_id = max(velocity_ref_candidates)
                    velocity_ref_kf = mapper.world_map.keyframes[velocity_ref_id]
                    dt = (anchor_kf.timestamp_ns - velocity_ref_kf.timestamp_ns) / 1e9
                    if dt > 0:
                        velocity = (camera_center(anchor_kf.pose_cw) - camera_center(velocity_ref_kf.pose_cw)) / dt
                gap_s = (orphan_first_kf.timestamp_ns - anchor_kf.timestamp_ns) / 1e9
                predicted_center = camera_center(anchor_kf.pose_cw) + velocity * gap_s
                desired_pose = np.eye(4)
                desired_pose[:3, :3] = anchor_kf.pose_cw[:3, :3]
                desired_pose[:3, 3] = -desired_pose[:3, :3] @ predicted_center

            t_correction = invert_pose(desired_pose) @ orphan_first_kf.pose_cw
            mapper.world_map.transform_segment(seg_id, t_correction, main_segment_id)
            mapper.world_map.add_fallback_stitch(anchor_kf_id, sorted(seg_kf_ids))
            # NOTE: the (anchor_kf_id, first_orphan_kf_id) transition's real IMU samples get
            # promoted into an actual CombinedImuFactor further below (see the generalized
            # pass after the POST-HOC CONSISTENCY CHECK), once every stitch/merge/correction
            # in this post-hoc pass has settled - not here, to avoid adding the same pair's
            # factor twice.
            print(
                f"  fallback stitch: segment {seg_id} ({len(seg_kf_ids)} kf, starting at "
                f"kf{first_orphan_kf_id}) welded onto kf{anchor_kf_id} via {source} "
                f"(v={np.linalg.norm(velocity):.2f}m/s over {gap_s:.2f}s gap) - NOT visually verified"
            )

        # POST-HOC CONSISTENCY CHECK: a fallback-stitched chain is anchored (via IMU
        # bridge or constant-velocity guess) only on its BEFORE side - if the very next
        # segment after it turns out to be a large, legitimate one that gets reconciled
        # completely independently (a real, much-later visual merge - see
        # tracker.py's _start_new_map_segment's chain-through-short-segments logic for the
        # case that DOES get handled directly), there was never any check that the two
        # sides actually agree where they meet. Measured directly: a ~2-3m visible seam at
        # exactly this kind of boundary even after the chain-through fix above removed the
        # redundant throwaway middleman. Do NOT try to blend/average both sides live (tried
        # and reverted earlier this session for a different reason - composing two
        # independently-uncertain guesses risked a confirmed-wrong "teleport" on another
        # dataset) - instead, wait until BOTH sides are already fully, independently
        # resolved, then treat whichever side has a real visual anchor as ground truth and
        # correct the IMU-bridge-only side to match it at the boundary (same "welding"
        # spirit as ORB-SLAM3's own map-merge: reconcile once both maps are already
        # internally consistent, don't fight over an uncertain in-progress estimate).
        #
        # Applies a single rigid transform to the whole chain so its far end lands exactly
        # on the independently-resolved target. KNOWN LIMITATION (not fixed): this ignores
        # the chain's own already-good near-end anchor (the fallback-stitch weld onto its
        # own earlier anchor before it), so the near-end jump can get measurably worse in
        # exchange for fixing the far end (measured on one dataset: kf932->kf979 went from
        # a reasonable 1.19m to 2.75m). Three more "principled" fixes were tried and
        # reverted after all measuring WORSE overall (not just at one end) than this plain
        # version: (1) feeding the IMU-bridge relative pose in as a pose-graph loop edge
        # and re-running optimize_pose_graph - this chain has no other constraint holding
        # its near end in place either (keyframe id 0 is the only hard prior in the whole
        # graph), so LM partially moved the "already fine" near end too, at both the
        # default sigma and a 10x-loosened one; (2) spreading this same correction across
        # the chain via linear interpolation weighted by each keyframe's position - closer,
        # but still left the far end under-corrected; (3) (2) plus moving each keyframe's
        # own map points by the same weight and recording the correction as a persistent
        # loop edge into final_global_ba - final BA has real reprojection constraints on
        # both sides pulling against this edge in ways that turned out hard to predict/
        # control, and produced the worst result of all four attempts. Given a full
        # architectural fix (e.g. a sliding-window/marginalization backend that never
        # creates a disconnected chain needing this kind of after-the-fact reconciliation
        # at all - see this session's SLAM-architecture research discussion) is a much
        # bigger undertaking, this plain version is kept as the best-measured option for
        # now rather than continuing to iterate blindly.
        # EXPERIMENTAL (env-gated via CONSISTENCY_MODE, default "rigid" = original behavior):
        #   "rigid"       - the original single rigid transform applied to the whole chain.
        #   "skip"        - ("Option C") no pre-snap at all; rely purely on the generalized
        #                   imu_factor pass below + final_global_ba's joint optimization.
        #                   Measured on one dataset: converges to within 0.001m of "rigid"'s
        #                   own final z_range - the pre-snap doesn't change where LM ends up,
        #                   just gives it a head start. No benefit measured; kept only for
        #                   reference.
        #   "interpolate" - ("Option A") spread the correction across the chain linearly
        #                   (SE(3) interpolated: slerp for rotation, lerp for translation) by
        #                   each keyframe's position in the chain, near-zero at the near end
        #                   (right after the anchor) growing to the full correction at the far
        #                   end (chain_end_id) - instead of applying the same full correction
        #                   to every keyframe uniformly.
        #   "accept_gap"  - ("Option B") don't touch the chain's poses at all AND don't
        #                   promote this specific boundary's transition samples into an
        #                   imu_factor either (see excluded_transition_pairs below) - accept a
        #                   visible, explicitly-known-low-confidence seam at this one boundary
        #                   rather than smearing its error into either neighbor.
        consistency_mode = os.environ.get("CONSISTENCY_MODE", "rigid")
        excluded_transition_pairs: set[tuple[int, int]] = set()
        trans_tol = getattr(cfg.mapping, "fallback_stitch_consistency_trans_tol_m", 0.3)
        rot_tol = getattr(cfg.mapping, "fallback_stitch_consistency_rot_tol_deg", 10.0)
        for anchor_kf_id, stitched_kf_ids in mapper.world_map.fallback_stitches:
            chain_end_id = max(stitched_kf_ids)
            candidates = [
                (old_id, new_id, samples)
                for old_id, new_id, samples in mapper.world_map.segment_transition_imu_samples
                if old_id == chain_end_id and new_id in mapper.world_map.keyframes
                and mapper.world_map.keyframes[new_id].segment_id == main_segment_id
            ]
            for _old_id, new_id, _samples in candidates:
                predicted_pose = _imu_bridge_pose(chain_end_id, new_id)
                if predicted_pose is None:
                    continue
                actual_pose = mapper.world_map.keyframes[new_id].pose_cw
                trans_diff, rot_diff = pose_delta(predicted_pose, actual_pose)
                if trans_diff <= trans_tol and rot_diff <= rot_tol:
                    continue  # already consistent - don't perturb a chain that's already fine
                t_correction = invert_pose(actual_pose) @ predicted_pose

                if consistency_mode == "skip":
                    print(
                        f"  fallback-stitch consistency check: chain ending at kf{chain_end_id} "
                        f"disagrees with the independently-resolved kf{new_id} by {trans_diff:.2f}m / "
                        f"{rot_diff:.1f}deg - NOT rigidly snapped (CONSISTENCY_MODE=skip), left for "
                        "final_global_ba's own imu_factor-based reconciliation"
                    )
                    continue

                if consistency_mode == "accept_gap":
                    excluded_transition_pairs.add((chain_end_id, new_id))
                    print(
                        f"  fallback-stitch consistency check: chain ending at kf{chain_end_id} "
                        f"disagrees with the independently-resolved kf{new_id} by {trans_diff:.2f}m / "
                        f"{rot_diff:.1f}deg - left AS-IS (CONSISTENCY_MODE=accept_gap): neither the "
                        "chain's poses nor this boundary's imu_factor are touched, so the disagreement "
                        "stays a visible, explicitly low-confidence seam here instead of being smeared "
                        "into either neighbor"
                    )
                    continue

                if consistency_mode == "interpolate":
                    from scipy.spatial.transform import Rotation, Slerp
                    chain_sorted = sorted(stitched_kf_ids)
                    n = len(chain_sorted)
                    key_rots = Rotation.from_matrix(np.stack([np.eye(3), t_correction[:3, :3]]))
                    slerp = Slerp([0.0, 1.0], key_rots)
                    for idx, kf_id in enumerate(chain_sorted):
                        w = idx / (n - 1) if n > 1 else 1.0
                        transform_i = np.eye(4)
                        transform_i[:3, :3] = slerp([w])[0].as_matrix()
                        transform_i[:3, 3] = w * t_correction[:3, 3]
                        kf = mapper.world_map.keyframes[kf_id]
                        kf.pose_cw = kf.pose_cw @ invert_pose(transform_i)
                    print(
                        f"  fallback-stitch consistency correction: chain ending at kf{chain_end_id} "
                        f"disagreed with the independently-resolved kf{new_id} by {trans_diff:.2f}m / "
                        f"{rot_diff:.1f}deg - spread the {len(stitched_kf_ids)}-keyframe chain's "
                        "correction linearly by position (CONSISTENCY_MODE=interpolate) instead of "
                        "applying it uniformly"
                    )
                    continue

                mapper.world_map.transform_keyframes(set(stitched_kf_ids), t_correction)
                print(
                    f"  fallback-stitch consistency correction: chain ending at kf{chain_end_id} "
                    f"disagreed with the independently-resolved kf{new_id} by {trans_diff:.2f}m / "
                    f"{rot_diff:.1f}deg - corrected the {len(stitched_kf_ids)}-keyframe chain "
                    f"(anchored on kf{anchor_kf_id}) to match kf{new_id}'s side"
                )

        # Promote EVERY Atlas-transition's real IMU samples into an actual CombinedImuFactor,
        # now that fallback stitching and the consistency correction above have settled all
        # segment_id reassignments and poses - not just the ones touched by this run's
        # fallback-stitch loop (see that loop's own narrower add_imu_factor for why THAT one
        # is scoped to just its own anchor/first-orphan pairs). segment_transition_imu_samples
        # on its own is only ever read by _imu_bridge_pose (a one-time rigid weld, computed
        # once and never revisited) and the consistency check above (also a one-time rigid
        # snap) - neither is seen by local_bundle_adjustment, which only iterates
        # world_map.imu_factors. Without this, a transition boundary resolved via a LIVE map
        # merge (try_close_loops' cross-segment branch, which adds no factor of its own
        # either) has no persistent constraint at all: final_global_ba below re-optimizes
        # every keyframe's pose from real vision+IMU evidence, and the two sides of an
        # unfactored boundary can drift apart even though they were only e.g. half a second
        # apart in real time.
        #
        # MEASURED TRADEOFF (kept anyway - see below): on one dataset this closed a 1.6m Z
        # gap between a fallback-stitched chain's end and a segment resolved much later via a
        # live map merge (kf1245->kf1262: -1.625m -> -0.194m) but reopened two near-end gaps
        # this same run's fallback-stitch imu_factor had just closed (kf1102->kf1121:
        # -0.009m -> -0.337m; kf1127->kf1145: -0.04m -> -0.370m) - net z_range still improved
        # (2.925m -> 2.614m, ~11%) but it's a redistribution of error across more boundaries,
        # not a clean elimination, matching the exact tradeoff the POST-HOC CONSISTENCY
        # CHECK's own docstring above already documents for a similar prior attempt. Judged
        # worth keeping on balance (smaller total error, no single boundary left as bad as
        # the original 1.6m cliff) rather than reverted - re-evaluate if a future dataset
        # shows this tradeoff going the other way.
        for old_id, new_id, samples in mapper.world_map.segment_transition_imu_samples:
            if (old_id, new_id) in excluded_transition_pairs:
                continue
            if (
                old_id in mapper.world_map.keyframes and new_id in mapper.world_map.keyframes
                and mapper.world_map.keyframes[old_id].segment_id == mapper.world_map.keyframes[new_id].segment_id
            ):
                mapper.world_map.add_imu_factor(old_id, new_id, samples)

    print(
        f"Done. tracked={n_tracked} lost={n_lost} keyframes_inserted={n_keyframes} "
        f"culled={n_culled_keyframes} keyframes_remaining={len(mapper.world_map.keyframes)} "
        f"map_points={len(mapper.world_map.map_points)} loops={n_loops} reloc={mapper.n_relocalizations} "
        f"map_segments={getattr(mapper, 'n_map_segments', 1)}"
    )
    if mapper.kf_insert_reason_counts:
        # Doesn't include bootstrap/Atlas-reinit seed keyframes (those are forced
        # structurally, not a _need_new_keyframe "need" decision) - see its docstring.
        print(f"  keyframe insertion reasons: {mapper.kf_insert_reason_counts}")

    if imu_enabled and mapper.imu_init_pending:
        print("  dynamic IMU init never completed (chain never reached the required window) - "
              "final BA will run vision-only, without IMU factors")

    if os.environ.get("DEBUG_PRE_FINAL_BA_MAP"):
        save_map(mapper.world_map, os.environ["DEBUG_PRE_FINAL_BA_MAP"])

    if getattr(cfg.mapping, "final_global_ba", False) and n_keyframes >= 3:
        t0 = time.monotonic()
        all_kf_ids = mapper.world_map.keyframe_ids_sorted()
        use_imu = mapper.imu_tight_fusion and not mapper.imu_init_pending
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

    # Vision-only, IMU-free vertical-axis fallback (see pose_graph.estimate_vertical_axis_pca's
    # own docstring for the full rationale/caveats) - only relevant when there's no IMU-based
    # gravity alignment to trust in the first place: with IMU enabled, the map's Z axis is
    # already a real, measured gravity estimate (however imperfect - see this project's own
    # extensive IMU-init tuning), and this near-planar-motion assumption is no substitute for
    # that. Default ON (only under imu.enabled=false) since it's a pure improvement when its
    # own self-check (max_planarity_ratio) passes, and a no-op (explicitly skipped, logged)
    # when it doesn't - never applied blindly.
    if not imu_enabled and getattr(cfg.mapping, "vertical_axis_pca_correction", True):
        max_ratio = getattr(cfg.mapping, "vertical_axis_pca_max_planarity_ratio", 0.02)
        up_axis, ratio = estimate_vertical_axis_pca(mapper.world_map, max_planarity_ratio=max_ratio)
        if up_axis is not None:
            angle_deg = float(np.degrees(np.arccos(np.clip(up_axis[2], -1.0, 1.0))))
            first_kf_id = mapper.world_map.keyframe_ids_sorted()[0]
            apply_vertical_axis_correction(mapper.world_map, up_axis, heading_ref_kf_id=first_kf_id)
            print(
                f"  vertical-axis PCA correction: applied ({angle_deg:.2f}deg rotation, "
                f"planarity ratio {ratio:.5f} <= {max_ratio}) - no IMU gravity alignment was "
                "available, so this vision-only near-planar-motion fallback was used instead"
            )
        else:
            print(
                f"  vertical-axis PCA correction: SKIPPED (planarity ratio {ratio:.5f} > "
                f"{max_ratio} - trajectory doesn't look near-planar enough to trust this "
                "vision-only fallback; map's Z axis is left as whatever the bootstrap "
                "keyframe's own camera orientation happened to be)"
            )

    out_path = Path(cfg.output.map_dir) / "map.pkl"
    save_map(mapper.world_map, out_path)
    print(f"Map saved to {out_path}")


if __name__ == "__main__":
    main()
