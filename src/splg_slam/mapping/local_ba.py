import gtsam
import numpy as np
from gtsam import (
    BetweenFactorPose3,
    Cal3_S2,
    Cal3_S2Stereo,
    GenericProjectionFactorCal3_S2,
    GenericStereoFactor3D,
    Point2,
    Point3,
    PriorFactorPose3,
    StereoPoint2,
    Values,
)
from gtsam.symbol_shorthand import L, X

from splg_slam.geometry.pose_utils import camera_center
from splg_slam.map.world_map import WorldMap
from splg_slam.mapping.gtsam_utils import (
    confidence_scaled_sigma,
    gtsam_pose_to_cw,
    matrix_to_gtsam_pose3,
    pose_cw_to_gtsam,
)


def local_bundle_adjustment(
    world_map: WorldMap,
    keyframe_ids: list[int],
    k_rect: np.ndarray,
    baseline: float | None = None,
    pixel_sigma: float = 1.0,
    min_obs_for_point: int = 2,
    loop_edges: list[tuple[int, int, np.ndarray, int]] | None = None,
    loop_trans_sigma: float = 0.02,
    loop_rot_sigma_deg: float = 1.0,
    loop_min_inliers: int = 60,
    min_depth: float = 0.05,
    max_pose_shift_m: float = 10.0,
    outlier_reproj_threshold_px: float = 5.0,
) -> dict:
    """Refines poses of `keyframe_ids` ("free") and positions of the map points they
    observe via GTSAM Levenberg-Marquardt. Updates world_map in place. Points seen by a
    single keyframe are left fixed (a single reprojection factor cannot constrain a 3D
    point on its own).

    Pass `baseline` (rectifier.baseline) to use a true stereo reprojection factor
    (GenericStereoFactor3D, constraining left AND right image pixels together) for any
    observation whose keyframe has a valid stereo depth at that keypoint - previously
    the stereo depth was only ever used once, to initialize the point's position, and
    every BA factor was monocular (left-image-only), throwing away the ongoing scale
    constraint stereo actually provides. Observations without a valid per-keypoint depth
    (e.g. added later via local-map projection matching, not from that keyframe's own
    stereo pass) fall back to the monocular factor, and passing `baseline=None` disables
    stereo factors entirely (monocular-only, the original behavior).

    Every OTHER keyframe that also observes one of these points ("boundary") is included
    too, as a hard-fixed Pose3 - mirroring ORB-SLAM's local BA, which keeps such
    observations as constraints instead of discarding them. A small window that only
    knows about its own keyframes' observations is weakly constrained and can converge to
    a degenerate solution (measured on MH01/MH04: enabling the old windows-only version
    mid-run made the map WORSE, not better); pulling in the boundary's real (but frozen)
    geometry fixes that without the cost of optimizing the whole map every time.

    (Tried gating boundary inclusion on the keyframe having survived >=1 prior BA pass,
    to fix a separate MH05 regression - it left MH05 completely unchanged and made MH04
    worse, so the real MH05 mechanism is something else; reverted.)

    Observations that would project a point behind the camera are dropped (a bad
    correspondence, e.g. from local-map radius+descriptor matching, which never checks
    chirality) - a single such factor makes every LM trial step evaluate to inf and the
    optimizer gives up entirely, so it must be filtered out rather than left for the
    optimizer to choke on. Any keyframe left with no factor at all after filtering keeps
    its current pose (it can't be part of the optimization - GTSAM requires every
    variable to be constrained).

    Pass `loop_edges` (e.g. world_map.loop_edges, for a final full-map pass) to fold the
    verified loop constraints into the SAME optimization as the reprojection factors,
    instead of only having pose-graph-only correction (optimize_pose_graph) followed by a
    separate reprojection-only BA that has no memory of those constraints.

    If a small/weakly-constrained window (few points, little parallax) lets LM converge to
    a degenerate solution, a single bad call can fling a keyframe cluster to a wildly wrong
    position that then propagates forward through everything tracked against it. As a
    circuit breaker, the whole update is discarded (world_map left untouched) if any free
    keyframe would move by more than `max_pose_shift_m` in one call.

    After a successful update, any individual observation whose (left-image) reprojection
    error is still above `outlier_reproj_threshold_px` is dropped from its point (the
    point itself survives if other observations still support it) - Huber down-weights
    outliers during the solve but never removes them, so a persistently bad correspondence
    keeps quietly dragging on every future optimization unless it's pruned here."""
    calib = Cal3_S2(k_rect[0, 0], k_rect[1, 1], 0.0, k_rect[0, 2], k_rect[1, 2])
    free_kf_ids = set(keyframe_ids)

    candidate_point_ids = set()
    for kf_id in keyframe_ids:
        for mp_id in world_map.keyframes[kf_id].map_point_ids:
            if mp_id >= 0:
                candidate_point_ids.add(int(mp_id))
    candidate_point_ids = {
        pid for pid in candidate_point_ids if world_map.map_points[pid].num_observations() >= min_obs_for_point
    }

    # Every keyframe observing a candidate point contributes a constraint - not just the
    # free ones - so boundary keyframes stay in as fixed geometry rather than being lost.
    valid_obs: dict[int, list[tuple[int, int]]] = {}
    n_skipped_cheirality = 0
    for pid in candidate_point_ids:
        position = world_map.map_points[pid].position
        kept = []
        for kf_id, kp_idx in world_map.map_points[pid].observations.items():
            if kf_id not in world_map.keyframes:
                continue
            pose_cw = world_map.keyframes[kf_id].pose_cw
            depth = (pose_cw[:3, :3] @ position + pose_cw[:3, 3])[2]
            if depth <= min_depth:
                n_skipped_cheirality += 1
                continue
            kept.append((kf_id, kp_idx))
        if len(kept) >= min_obs_for_point:
            valid_obs[pid] = kept

    stereo_calib = (
        Cal3_S2Stereo(k_rect[0, 0], k_rect[1, 1], 0.0, k_rect[0, 2], k_rect[1, 2], baseline)
        if baseline else None
    )

    graph = gtsam.NonlinearFactorGraph()
    initial = Values()
    free_touched: set[int] = set()
    fixed_touched: set[int] = set()

    noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)
    robust_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), noise)
    stereo_noise = gtsam.noiseModel.Isotropic.Sigma(3, pixel_sigma)
    robust_stereo_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345), stereo_noise)
    fixed_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))

    for pid, obs in valid_obs.items():
        initial.insert(L(pid), Point3(world_map.map_points[pid].position))
        for kf_id, kp_idx in obs:
            kf = world_map.keyframes[kf_id]
            if kf_id not in free_touched and kf_id not in fixed_touched:
                initial.insert(X(kf_id), pose_cw_to_gtsam(kf.pose_cw))
                if kf_id in free_kf_ids:
                    free_touched.add(kf_id)
                else:
                    fixed_touched.add(kf_id)
                    graph.add(PriorFactorPose3(X(kf_id), initial.atPose3(X(kf_id)), fixed_prior_noise))
            uv = kf.keypoints[kp_idx]
            depth = kf.depths[kp_idx]
            if stereo_calib is not None and np.isfinite(depth) and depth > min_depth:
                disparity = stereo_calib.fx() * baseline / depth
                measurement = StereoPoint2(float(uv[0]), float(uv[0] - disparity), float(uv[1]))
                graph.add(GenericStereoFactor3D(measurement, robust_stereo_noise, X(kf_id), L(pid), stereo_calib))
            else:
                graph.add(GenericProjectionFactorCal3_S2(Point2(uv[0], uv[1]), robust_noise, X(kf_id), L(pid), calib))

    if loop_edges:
        for a, b, rel, num_inliers in loop_edges:
            if a not in free_kf_ids or b not in free_kf_ids:
                continue
            for kf_id in (a, b):
                if kf_id not in free_touched and kf_id not in fixed_touched:
                    initial.insert(X(kf_id), pose_cw_to_gtsam(world_map.keyframes[kf_id].pose_cw))
                    free_touched.add(kf_id)
            trans_sigma = confidence_scaled_sigma(loop_trans_sigma, num_inliers, loop_min_inliers)
            rot_sigma = confidence_scaled_sigma(loop_rot_sigma_deg, num_inliers, loop_min_inliers)
            loop_base_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([np.radians(rot_sigma)] * 3 + [trans_sigma] * 3)
            )
            loop_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.0), loop_base_noise)
            graph.add(BetweenFactorPose3(X(a), X(b), matrix_to_gtsam_pose3(rel), loop_noise))

    if not free_touched:
        return {
            "num_keyframes": 0, "num_points": 0, "num_skipped_cheirality": n_skipped_cheirality,
            "num_outliers_removed": 0, "initial_error": 0.0, "final_error": 0.0, "rejected": False,
        }

    if not fixed_touched:
        # No frozen boundary geometry to anchor the gauge with - fall back to pinning the
        # oldest free keyframe (e.g. the very first local window of the whole sequence).
        anchor_kf_id = min(free_touched)
        graph.add(PriorFactorPose3(X(anchor_kf_id), initial.atPose3(X(anchor_kf_id)), fixed_prior_noise))

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, gtsam.LevenbergMarquardtParams())
    initial_error = graph.error(initial)
    result = optimizer.optimize()

    new_poses = {kf_id: gtsam_pose_to_cw(result.atPose3(X(kf_id))) for kf_id in free_touched}
    max_shift = max(
        (float(np.linalg.norm(camera_center(new_poses[kf_id]) - camera_center(world_map.keyframes[kf_id].pose_cw)))
         for kf_id in free_touched),
        default=0.0,
    )
    if max_shift > max_pose_shift_m:
        return {
            "num_keyframes": len(free_touched), "num_points": len(valid_obs),
            "num_skipped_cheirality": n_skipped_cheirality, "num_outliers_removed": 0,
            "initial_error": float(initial_error), "final_error": float(graph.error(result)),
            "rejected": True, "max_pose_shift_m": max_shift,
        }

    for kf_id, pose_cw in new_poses.items():
        world_map.keyframes[kf_id].pose_cw = pose_cw
    for pid in valid_obs:
        world_map.map_points[pid].position = np.array(result.atPoint3(L(pid)))

    all_poses = {kf_id: world_map.keyframes[kf_id].pose_cw for kf_id in (free_touched | fixed_touched)}
    n_outliers_removed = 0
    for pid, obs in valid_obs.items():
        mp = world_map.map_points.get(pid)
        if mp is None:
            continue
        position = mp.position
        for kf_id, kp_idx in obs:
            pose_cw = all_poses[kf_id]
            p_cam = pose_cw[:3, :3] @ position + pose_cw[:3, 3]
            if p_cam[2] <= min_depth:
                world_map.remove_observation(pid, kf_id)
                n_outliers_removed += 1
                continue
            uv_h = k_rect @ p_cam
            uv_proj = uv_h[:2] / uv_h[2]
            uv_obs = world_map.keyframes[kf_id].keypoints[kp_idx]
            if float(np.linalg.norm(uv_proj - uv_obs)) > outlier_reproj_threshold_px:
                world_map.remove_observation(pid, kf_id)
                n_outliers_removed += 1

    return {
        "num_keyframes": len(free_touched),
        "num_points": len(valid_obs),
        "num_skipped_cheirality": n_skipped_cheirality,
        "num_outliers_removed": n_outliers_removed,
        "initial_error": float(initial_error),
        "final_error": float(graph.error(result)),
        "rejected": False,
    }
