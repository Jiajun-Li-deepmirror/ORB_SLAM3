import gtsam
import numpy as np
from gtsam import BetweenFactorPose3, PriorFactorPose3, Values
from gtsam.symbol_shorthand import X

from splg_slam.geometry.pose_utils import camera_center, invert_pose, rotation_angle_deg
from splg_slam.map.world_map import WorldMap
from splg_slam.mapping.gtsam_utils import (
    gtsam_pose_to_cw,
    make_loop_edge_noise,
    matrix_to_gtsam_pose3,
    pose_cw_to_gtsam,
)
from splg_slam.mapping.imu_preintegration import rotation_aligning


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
    rot_diff_deg = rotation_angle_deg(rel_a[:3, :3].T @ rel_b[:3, :3])
    return trans_diff, rot_diff_deg


def odometry_arc_length_m(world_map: WorldMap, kf_id_a: int, kf_id_b: int) -> float:
    """Total translated distance along the id-sorted keyframe chain between two keyframes -
    a data-derived proxy for how much odometry drift could plausibly have accumulated
    between them. Stereo VO drift is conventionally budgeted as a percentage of distance
    traveled (this is exactly how KITTI's own official benchmark reports translational
    error), not a fixed number of meters - a constant tolerance sized right for a ~80m
    EuRoC room-scale trajectory is far too strict for a multi-km outdoor loop, and too loose
    for a tabletop-scale one. Used to scale the loop-closure consistency-check tolerance."""
    ids = world_map.keyframe_ids_sorted()
    lo, hi = min(kf_id_a, kf_id_b), max(kf_id_a, kf_id_b)
    chain = [k for k in ids if lo <= k <= hi]
    total = 0.0
    for a, b in zip(chain[:-1], chain[1:]):
        ca = camera_center(world_map.keyframes[a].pose_cw)
        cb = camera_center(world_map.keyframes[b].pose_cw)
        total += float(np.linalg.norm(cb - ca))
    return total


def _arc_scaled_odom_sigmas(
    world_map: WorldMap, kf_ids: list[int], odom_trans_sigma: float, odom_rot_sigma_deg: float,
    odom_arc_reference_m: float, imu_only_penalty_m: float = 0.0,
) -> dict[int, tuple[float, float]]:
    """EXPERIMENTAL (see optimize_pose_graph's odom_arc_scaling_enabled): per-keyframe
    (trans_sigma, rot_sigma) for the *outgoing* odometry edge starting at that keyframe,
    growing with distance traveled since the nearest earlier "anchored" keyframe - one that
    is an endpoint of some already-accepted loop edge, i.e. a place where the trajectory's
    own self-consistency has actually been checked. A keyframe deep into a long stretch of
    genuinely new territory (no anchor nearby) has had far more opportunity to accumulate
    real stereo-VO drift than one a few frames after the last loop closure, but the current
    per-edge sigma is a single fixed constant regardless - this lets a loop-closure
    correction preferentially bend the unanchored stretch it actually spans, instead of
    landing wherever the graph happens to be least otherwise constrained (see this
    function's caller for the empirical motivation).

    Growth is sqrt(1 + dist_since_anchor / odom_arc_reference_m), matching the standard
    random-walk assumption that positional uncertainty grows with the square root of
    distance traveled (same convention random-walk noise models use elsewhere in this
    codebase, e.g. IMU integration covariance) - not linear, which would over-punish long
    unanchored stretches relative to how much extra drift they actually likely accumulated.

    imu_only_penalty_m: one-time equivalent-distance penalty added to dist_since_anchor the
    moment the chain crosses a tracking.continuous_imu_tracking keyframe (either endpoint of
    that edge) - a plain KeyFrame.pose_cw for one of these came from dead-reckoned IMU
    integration, never cross-checked against a real image the way every other keyframe's
    pose was, so the whole stretch *after* it (until the next real anchor) deserves elevated
    uncertainty, not just that one edge. Modeled as "equivalent extra distance" (added into
    the same dist_since_anchor the sqrt-growth above already uses) rather than a flat
    multiplier on that single edge, since a flat per-edge multiplier was tried first and
    measured to do basically nothing: a same-segment loop closure's correction still landed
    almost entirely elsewhere in the graph, because the single bumped edge is a tiny
    fraction of the total edge count and every ordinary edge for the rest of that
    (long, otherwise "unanchored but confidently VO-tracked") stretch was still at normal
    sigma. Injecting the penalty into dist_since_anchor instead means EVERY edge from the
    imu_only jump onward - until the next loop-closure anchor - inherits elevated sigma via
    the existing sqrt(1 + dist/reference) growth, exactly like a real several-meter jump in
    accumulated drift would. Default 0.0 (no-op) keeps every existing
    odom_arc_scaling_enabled run's behavior identical."""
    anchored_kf_ids = set()
    for kf_a, kf_b, _rel, _n in world_map.loop_edges:
        anchored_kf_ids.add(kf_a)
        anchored_kf_ids.add(kf_b)

    sigmas = {}
    cum_dist = 0.0
    dist_since_anchor = 0.0 if kf_ids[0] in anchored_kf_ids else None
    prev_id = kf_ids[0]
    prev_center = camera_center(world_map.keyframes[prev_id].pose_cw)
    for kf_id in kf_ids[1:]:
        center = camera_center(world_map.keyframes[kf_id].pose_cw)
        step = float(np.linalg.norm(center - prev_center))
        cum_dist += step
        if kf_id in anchored_kf_ids:
            dist_since_anchor = 0.0
        else:
            step_cost = step
            if imu_only_penalty_m and (
                getattr(world_map.keyframes[kf_id], "imu_only", False)
                or getattr(world_map.keyframes[prev_id], "imu_only", False)
            ):
                step_cost += imu_only_penalty_m
            dist_since_anchor = step_cost if dist_since_anchor is None else dist_since_anchor + step_cost
        scale = np.sqrt(1.0 + (dist_since_anchor or 0.0) / odom_arc_reference_m)
        sigmas[kf_id] = (odom_trans_sigma * scale, odom_rot_sigma_deg * scale)
        prev_id, prev_center = kf_id, center
    return sigmas


def optimize_pose_graph(
    world_map: WorldMap,
    odom_trans_sigma: float = 0.05,
    odom_rot_sigma_deg: float = 2.0,
    loop_trans_sigma: float = 0.02,
    loop_rot_sigma_deg: float = 1.0,
    loop_min_inliers: int = 60,
    max_pose_shift_m: float = 20.0,
    odom_arc_scaling_enabled: bool = False,
    odom_arc_reference_m: float = 20.0,
    odom_imu_only_penalty_m: float = 0.0,
) -> dict:
    """Rebuilds a full pose graph over every keyframe: sequential Between-edges from the
    currently stored (drifted) poses, plus every verified loop edge in world_map.loop_edges.
    Updates keyframe poses in place; each map point is rigidly carried along with the pose
    change of one of its observing keyframes (a cheap correction, not a re-triangulation -
    a global BA pass afterward can refine further if needed).

    A single legitimate correction shouldn't need to move any one keyframe by tens of
    meters, so as a circuit breaker against a degenerate solve, the whole update is
    discarded (world_map left untouched) if any keyframe would move more than
    `max_pose_shift_m`.

    odom_arc_scaling_enabled: EXPERIMENTAL, off by default - see _arc_scaled_odom_sigmas.
    Motivated by a case where a long-arc loop closure's correction visibly distorted the
    trajectory at a point unrelated to the closure's own endpoints: every odometry edge
    trusts its relative pose equally regardless of how far into unexplored (unanchored)
    territory it sits, so LM was free to dump the correction wherever the graph happened to
    be least otherwise constrained. Scaling sequential-edge sigma up with distance-since-
    last-anchor is a more physically honest drift model (uncertainty should be higher deep
    in a long unrevisited stretch) and, unlike scaling the *loop* edge's own sigma (tried
    and reverted - see this file's other comment), doesn't touch the loop edge at all."""
    kf_ids = world_map.keyframe_ids_sorted()
    old_poses = {kf_id: world_map.keyframes[kf_id].pose_cw.copy() for kf_id in kf_ids}

    graph = gtsam.NonlinearFactorGraph()
    initial = Values()
    for kf_id in kf_ids:
        initial.insert(X(kf_id), pose_cw_to_gtsam(old_poses[kf_id]))

    # Sequential (odometry) edges only ever connect keyframes within the same Atlas
    # segment - an orphan segment's poses live in an unrelated, not-yet-reconciled
    # coordinate frame (see transform_segment's docstring), so a plain id-adjacency edge
    # across a segment boundary would assert a rigid constraint between two physically
    # unrelated poses and corrupt the graph. Grouping by segment_id (not id-adjacency
    # alone) also naturally reduces to the single global chain for the common
    # one-segment case.
    segment_groups: dict[int, list[int]] = {}
    for kf_id in kf_ids:
        segment_groups.setdefault(world_map.keyframes[kf_id].segment_id, []).append(kf_id)

    # (Tried a Huber kernel here too, reasoning that an odometry edge built from a
    # slightly-off BA solution shouldn't be trusted at full weight - it backfired: MH05
    # RMSE went 0.058m -> 0.157m, MH04 slightly worse too. A loop closure's whole point
    # is to redistribute a real, sometimes-large correction back across the odometry
    # chain; Huber-downweighting those edges fights that legitimate redistribution
    # instead of only catching genuine outliers, so it's a plain Gaussian here.)
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.radians(odom_rot_sigma_deg)] * 3 + [odom_trans_sigma] * 3)
    )
    for seg_kf_ids in segment_groups.values():
        arc_sigmas = (
            _arc_scaled_odom_sigmas(
                world_map, seg_kf_ids, odom_trans_sigma, odom_rot_sigma_deg, odom_arc_reference_m,
                imu_only_penalty_m=odom_imu_only_penalty_m,
            )
            if odom_arc_scaling_enabled else {}
        )
        for a, b in zip(seg_kf_ids[:-1], seg_kf_ids[1:]):
            rel = relative_pose(old_poses[a], old_poses[b])
            if b in arc_sigmas:
                trans_sigma_b, rot_sigma_b = arc_sigmas[b]
                edge_noise = gtsam.noiseModel.Diagonal.Sigmas(
                    np.array([np.radians(rot_sigma_b)] * 3 + [trans_sigma_b] * 3)
                )
            else:
                edge_noise = odom_noise
            graph.add(BetweenFactorPose3(X(a), X(b), matrix_to_gtsam_pose3(rel), edge_noise))

    # Robust (Huber) kernel: a single mis-verified loop edge (perceptual aliasing, a
    # degenerate PnP solve, ...) gets down-weighted instead of dragging the whole graph.
    # On top of that, sigma itself scales down with verification confidence (num_inliers)
    # instead of every accepted edge - 61 inliers or 600 - getting the exact same trust.
    for a, b, rel, num_inliers in world_map.loop_edges:
        if a not in world_map.keyframes or b not in world_map.keyframes:
            continue
        # (Tried additionally scaling sigma up with the arc length a loop edge spans, on
        # the theory that a long-arc closure was being trusted too tightly relative to
        # everything around it - backfired at both a mild (10x) and aggressive (100x)
        # scale: the milder one barely changed the resulting distortion, the aggressive
        # one made the whole graph badly under-constrained (relocalizations went
        # 69 -> 367, post-optimization residuals stopped converging). Reverted - a fixed
        # sigma per confidence tier, same as every other loop edge, is the more reliable
        # default for now.)
        loop_noise = make_loop_edge_noise(loop_trans_sigma, loop_rot_sigma_deg, num_inliers, loop_min_inliers)
        graph.add(BetweenFactorPose3(X(a), X(b), matrix_to_gtsam_pose3(rel), loop_noise))

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
    # Anchor every segment's first (lowest-id) keyframe at its own stored pose - not just
    # kf_ids[0] globally. Since odometry edges never cross a segment boundary (above), a
    # still-unreconciled orphan segment with no loop edge into it yet would otherwise be
    # completely unconstrained (gauge freedom) and make the LM solve singular for it. For
    # the common single-segment case this is exactly the original single global prior.
    for seg_kf_ids in segment_groups.values():
        graph.add(PriorFactorPose3(X(seg_kf_ids[0]), initial.atPose3(X(seg_kf_ids[0])), prior_noise))

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


def estimate_vertical_axis_pca(
    world_map: WorldMap, max_planarity_ratio: float = 0.02,
) -> tuple[np.ndarray | None, float]:
    """Vision-only, IMU-free estimate of "which way is up", for maps built with no IMU at
    all (or where IMU gravity alignment isn't available/trusted) - a fallback specifically
    for near-planar navigation (a handheld device walked around one floor, a ground robot,
    etc.), NOT a general substitute for real gravity alignment.

    Without IMU, a fresh map's Z axis is arbitrary (see tracker.py's bootstrap keyframe
    pose - it's whatever the first camera frame's own optical axes happened to be, not
    gravity), so the map can end up tilted by tens of degrees from true vertical - confirmed
    directly on a real handheld RealSense recording (~80deg with no correction at all).

    This assumes real motion stayed close to one horizontal plane and finds "up" as the
    direction the WHOLE keyframe trajectory varies LEAST along (PCA over camera centers,
    smallest-eigenvalue eigenvector) - confirmed directly on the same recording to match a
    completely independent method (a linear plane fit through the same positions) to within
    0.01deg, and to a residual height range of just 0.33m (33% of net displacement)
    afterward, versus 2.97m before - actually tighter than that same recording's own real
    IMU-based result (0.60m), because PCA draws on the trajectory's ENTIRE spatial extent
    (evidence accumulated over the whole recording) rather than one early, brief motion
    window's IMU solve.

    This only holds up when the "stayed near one horizontal plane" assumption is actually
    true - it would misfire on any genuinely 3D trajectory (a drone's real climb/descent,
    stairs, a ramp), mistaking real vertical motion for a correctable tilt. `max_planarity_ratio`
    is the self-check: the ratio of smallest-to-largest PCA eigenvalue (near-zero for a truly
    flat trajectory, order-0.1+ for one with real 3D structure) must fall under this before
    the estimate is trusted at all. Returns (None, ratio) - meaning "don't apply this" - if
    the check fails, ALONGSIDE the actual ratio either way so the caller can log why."""
    kf_ids = world_map.keyframe_ids_sorted()
    if len(kf_ids) < 3:
        return None, float("inf")
    centers = np.array([camera_center(world_map.keyframes[i].pose_cw) for i in kf_ids])
    centered = centers - centers.mean(axis=0)
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    ratio = float(eigvals[0] / eigvals[2]) if eigvals[2] > 1e-9 else float("inf")
    if ratio > max_planarity_ratio:
        return None, ratio
    up_axis = eigvecs[:, 0]
    if up_axis[2] < 0:  # orient roughly toward the pre-correction +Z, for a smaller residual rotation
        up_axis = -up_axis
    return up_axis, ratio


def apply_vertical_axis_correction(world_map: WorldMap, up_axis: np.ndarray, heading_ref_kf_id: int | None = None) -> None:
    """Rotates every keyframe pose and map point so `up_axis` (in the map's CURRENT frame)
    becomes +Z - same rigid-rotation application pattern as imu_init.py's
    _solve_and_realign (rotation only, no rescale: unlike a bad gravity solve, this never
    touches metric scale).

    `rotation_aligning` only constrains where "up" ends up - it has one full degree of
    freedom left (rotation about the new +Z, i.e. heading/yaw), which its minimal-rotation
    Rodrigues construction resolves to *whatever falls out of the cross-product*, not
    necessarily anything meaningful. Left alone, this silently drags the map's existing
    heading convention along with it by a small but real, uncontrolled amount (measured
    directly: a ~2.75deg incidental yaw from a ~10.8deg tilt correction) - harmless for the
    map's own internal consistency, but a real annoyance for anyone expecting the heading
    convention already established at kf0 (e.g. tracker.py's own fixed-axis bootstrap
    convention - see its docstring - which sets kf0's forward direction to exactly +X) to
    still hold true after this runs.

    `heading_ref_kf_id`, if given, cancels that incidental yaw: after the tilt-only
    rotation, an extra pure yaw (about the NEW +Z) is added so this keyframe's own forward
    direction (its camera-frame +Z expressed in world) lands back on the SAME horizontal
    heading it had before this function ran at all - restoring, not just preserving, "no
    heading change" as this function's actual behavior. None (default) skips this - only
    the tilt gets corrected, heading drifts by whatever the minimal rotation happens to do."""
    r_align = rotation_aligning(up_axis, np.array([0.0, 0.0, 1.0]))
    if heading_ref_kf_id is not None:
        kf_ref = world_map.keyframes[heading_ref_kf_id]
        forward_before = kf_ref.pose_cw[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        forward_mid = r_align @ forward_before
        horiz = forward_mid[:2]
        norm = float(np.linalg.norm(horiz))
        if norm > 1e-6:
            cos_a, sin_a = horiz[0] / norm, horiz[1] / norm
            r_yaw = np.array([
                [cos_a, sin_a, 0.0],
                [-sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0],
            ])
            r_align = r_yaw @ r_align
    for kf in world_map.keyframes.values():
        kf.pose_cw = kf.pose_cw.copy()
        kf.pose_cw[:3, :3] = kf.pose_cw[:3, :3] @ r_align.T
    for mp in world_map.map_points.values():
        mp.position = r_align @ mp.position
