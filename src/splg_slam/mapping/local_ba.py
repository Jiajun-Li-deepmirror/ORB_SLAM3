import gtsam
import numpy as np
from gtsam import (
    BetweenFactorPose3,
    Cal3_S2,
    Cal3_S2Stereo,
    CombinedImuFactor,
    GenericProjectionFactorCal3_S2,
    GenericStereoFactor3D,
    Point2,
    Point3,
    PriorFactorConstantBias,
    PriorFactorPose3,
    PriorFactorVector,
    StereoPoint2,
    Values,
)
from gtsam.symbol_shorthand import B, L, V, X

from splg_slam.geometry.pose_utils import camera_center
from splg_slam.map.world_map import WorldMap
from splg_slam.mapping.gtsam_utils import (
    confidence_scaled_sigma,
    gtsam_pose_to_cw,
    matrix_to_gtsam_pose3,
    pose_cw_to_body_gtsam,
    pose_cw_to_gtsam,
)
from splg_slam.mapping.imu_preintegration import bias_from_vector, preintegrate

# gtsam.symbol_shorthand.Y isn't a builtin shorthand - build one for the IMU body pose,
# kept distinct from X() (camera pose) since GenericStereoFactor3D has no body_P_sensor
# support in this GTSAM build (GenericProjectionFactorCal3_S2 does), so X() must keep
# meaning "camera pose" everywhere the stereo factor is used.
from gtsam import Symbol as _Symbol


def Y(j: int) -> int:
    return _Symbol("y", j).key()


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
    imu_factors: list[tuple[int, int, np.ndarray]] | None = None,
    imu_calib=None,
    imu_params=None,
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
    keeps quietly dragging on every future optimization unless it's pruned here.

    Pass `imu_factors` (world_map.imu_factors), `imu_calib`, and `imu_params` (a
    gtsam.PreintegrationCombinedParams) to fold CombinedImuFactor constraints into the
    same graph for any (kf_a, kf_b) pair where both ends are already touched by a visual
    factor above. Since GenericStereoFactor3D has no body_P_sensor support in this GTSAM
    build, X(kf_id) keeps meaning camera pose everywhere (stereo/mono factors unchanged);
    IMU states live on a separate Y(kf_id) body-pose symbol, rigidly tied to X(kf_id) via
    a tight BetweenFactorPose3 using the known camera<-body extrinsic."""
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

    imu_touched: set[int] = set()
    if imu_factors and imu_params is not None and imu_calib is not None:
        tight_vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)
        tight_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
        extrinsic_pose = matrix_to_gtsam_pose3(imu_calib.T_cam0_body)

        def _ensure_imu_state(kf_id: int) -> None:
            if kf_id in imu_touched:
                return
            imu_touched.add(kf_id)
            kf = world_map.keyframes[kf_id]
            initial.insert(Y(kf_id), pose_cw_to_body_gtsam(kf.pose_cw, imu_calib.T_cam0_body))
            initial.insert(V(kf_id), kf.velocity if kf.velocity is not None else np.zeros(3))
            initial.insert(B(kf_id), bias_from_vector(kf.imu_bias))
            graph.add(BetweenFactorPose3(X(kf_id), Y(kf_id), extrinsic_pose, fixed_prior_noise))
            if kf_id in fixed_touched:
                graph.add(PriorFactorVector(V(kf_id), initial.atVector(V(kf_id)), tight_vel_noise))
                graph.add(PriorFactorConstantBias(B(kf_id), initial.atConstantBias(B(kf_id)), tight_bias_noise))

        for kf_a, kf_b, samples in imu_factors:
            if kf_a not in world_map.keyframes or kf_b not in world_map.keyframes:
                continue  # one end was since culled (redundant-keyframe removal)
            if kf_a not in (free_touched | fixed_touched) or kf_b not in (free_touched | fixed_touched):
                continue
            bias_ref = bias_from_vector(world_map.keyframes[kf_a].imu_bias)
            preint = preintegrate(samples, bias_ref, imu_params)
            _ensure_imu_state(kf_a)
            _ensure_imu_state(kf_b)
            graph.add(CombinedImuFactor(Y(kf_a), V(kf_a), Y(kf_b), V(kf_b), B(kf_a), B(kf_b), preint))

        if imu_touched:
            # Gauge-fix velocity/bias the same way the pose graph is gauge-fixed above -
            # pin one state's initial estimate, unless a boundary keyframe already pinned one.
            anchor_imu_id = min(imu_touched)
            if anchor_imu_id not in fixed_touched:
                graph.add(PriorFactorVector(V(anchor_imu_id), initial.atVector(V(anchor_imu_id)), tight_vel_noise))
                graph.add(
                    PriorFactorConstantBias(B(anchor_imu_id), initial.atConstantBias(B(anchor_imu_id)), tight_bias_noise)
                )

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
    for kf_id in imu_touched:
        world_map.keyframes[kf_id].velocity = np.array(result.atVector(V(kf_id)))
        world_map.keyframes[kf_id].imu_bias = np.array(result.atConstantBias(B(kf_id)).vector())

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
        "num_imu_states": len(imu_touched),
        "initial_error": float(initial_error),
        "final_error": float(graph.error(result)),
        "rejected": False,
    }
