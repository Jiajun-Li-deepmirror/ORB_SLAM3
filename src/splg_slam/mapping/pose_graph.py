import gtsam
import numpy as np
from gtsam import BetweenFactorPose3, PriorFactorPose3, Values
from gtsam.symbol_shorthand import X

from splg_slam.geometry.pose_utils import camera_center, invert_pose
from splg_slam.map.world_map import WorldMap
from splg_slam.mapping.gtsam_utils import (
    confidence_scaled_sigma,
    gtsam_pose_to_cw,
    matrix_to_gtsam_pose3,
    pose_cw_to_gtsam,
)


def relative_pose(pose_cw_a: np.ndarray, pose_cw_b: np.ndarray) -> np.ndarray:
    """T_a_from_b: maps points expressed in b's local camera frame into a's local frame."""
    return pose_cw_a @ invert_pose(pose_cw_b)


def relative_pose_discrepancy(rel_a: np.ndarray, rel_b: np.ndarray) -> tuple[float, float]:
    """How far apart two relative-transform estimates (same convention as `relative_pose`)
    are: translation (m) and rotation (deg) between them. Used to reject a loop-closure
    edge whose implied relative pose disagrees wildly with what the (already locally-
    accurate) odometry chain says - a hallmark of perceptual aliasing rather than a
    genuine loop, which Huber down-weighting alone won't fully suppress."""
    trans_diff = float(np.linalg.norm(rel_a[:3, 3] - rel_b[:3, 3]))
    r_rel = rel_a[:3, :3].T @ rel_b[:3, :3]
    cos_angle = np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0)
    rot_diff_deg = float(np.degrees(np.arccos(cos_angle)))
    return trans_diff, rot_diff_deg


def optimize_pose_graph(
    world_map: WorldMap,
    odom_trans_sigma: float = 0.05,
    odom_rot_sigma_deg: float = 2.0,
    loop_trans_sigma: float = 0.02,
    loop_rot_sigma_deg: float = 1.0,
    loop_min_inliers: int = 60,
    max_pose_shift_m: float = 20.0,
) -> dict:
    """Rebuilds a full pose graph over every keyframe: sequential Between-edges from the
    currently stored (drifted) poses, plus every verified loop edge in world_map.loop_edges.
    Updates keyframe poses in place; each map point is rigidly carried along with the pose
    change of one of its observing keyframes (a cheap correction, not a re-triangulation -
    a global BA pass afterward can refine further if needed).

    A single legitimate correction shouldn't need to move any one keyframe by tens of
    meters, so as a circuit breaker against a degenerate solve, the whole update is
    discarded (world_map left untouched) if any keyframe would move more than
    `max_pose_shift_m`."""
    kf_ids = world_map.keyframe_ids_sorted()
    old_poses = {kf_id: world_map.keyframes[kf_id].pose_cw.copy() for kf_id in kf_ids}

    graph = gtsam.NonlinearFactorGraph()
    initial = Values()
    for kf_id in kf_ids:
        initial.insert(X(kf_id), pose_cw_to_gtsam(old_poses[kf_id]))

    # (Tried a Huber kernel here too, reasoning that an odometry edge built from a
    # slightly-off BA solution shouldn't be trusted at full weight - it backfired: MH05
    # RMSE went 0.058m -> 0.157m, MH04 slightly worse too. A loop closure's whole point
    # is to redistribute a real, sometimes-large correction back across the odometry
    # chain; Huber-downweighting those edges fights that legitimate redistribution
    # instead of only catching genuine outliers, so it's a plain Gaussian here.)
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.radians(odom_rot_sigma_deg)] * 3 + [odom_trans_sigma] * 3)
    )
    for a, b in zip(kf_ids[:-1], kf_ids[1:]):
        rel = relative_pose(old_poses[a], old_poses[b])
        graph.add(BetweenFactorPose3(X(a), X(b), matrix_to_gtsam_pose3(rel), odom_noise))

    # Robust (Huber) kernel: a single mis-verified loop edge (perceptual aliasing, a
    # degenerate PnP solve, ...) gets down-weighted instead of dragging the whole graph.
    # On top of that, sigma itself scales down with verification confidence (num_inliers)
    # instead of every accepted edge - 61 inliers or 600 - getting the exact same trust.
    for a, b, rel, num_inliers in world_map.loop_edges:
        if a not in world_map.keyframes or b not in world_map.keyframes:
            continue
        trans_sigma = confidence_scaled_sigma(loop_trans_sigma, num_inliers, loop_min_inliers)
        rot_sigma = confidence_scaled_sigma(loop_rot_sigma_deg, num_inliers, loop_min_inliers)
        loop_base_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([np.radians(rot_sigma)] * 3 + [trans_sigma] * 3)
        )
        loop_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.0), loop_base_noise)
        graph.add(BetweenFactorPose3(X(a), X(b), matrix_to_gtsam_pose3(rel), loop_noise))

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
    graph.add(PriorFactorPose3(X(kf_ids[0]), initial.atPose3(X(kf_ids[0])), prior_noise))

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, gtsam.LevenbergMarquardtParams())
    initial_error = graph.error(initial)
    result = optimizer.optimize()

    new_poses = {kf_id: gtsam_pose_to_cw(result.atPose3(X(kf_id))) for kf_id in kf_ids}
    max_shift = max(
        (float(np.linalg.norm(camera_center(new_poses[kf_id]) - camera_center(old_poses[kf_id]))) for kf_id in kf_ids),
        default=0.0,
    )
    if max_shift > max_pose_shift_m:
        return {
            "num_keyframes": len(kf_ids), "num_loop_edges": len(world_map.loop_edges),
            "initial_error": float(initial_error), "final_error": float(graph.error(result)),
            "rejected": True, "max_pose_shift_m": max_shift,
        }

    for kf_id in kf_ids:
        world_map.keyframes[kf_id].pose_cw = new_poses[kf_id]

    for mp in world_map.map_points.values():
        if not mp.observations:
            continue
        anchor_kf = min(mp.observations.keys())
        if anchor_kf not in old_poses:
            continue
        p_cam = old_poses[anchor_kf][:3, :3] @ mp.position + old_poses[anchor_kf][:3, 3]
        pose_wc_new = invert_pose(new_poses[anchor_kf])
        mp.position = pose_wc_new[:3, :3] @ p_cam + pose_wc_new[:3, 3]

    return {
        "num_keyframes": len(kf_ids),
        "num_loop_edges": len(world_map.loop_edges),
        "initial_error": float(initial_error),
        "final_error": float(graph.error(result)),
        "rejected": False,
    }
