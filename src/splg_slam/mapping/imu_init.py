import gtsam
import numpy as np
from gtsam import imuBias

from splg_slam.data.euroc import ImuCalibration
from splg_slam.geometry.pose_utils import rotation_angle_deg
from splg_slam.map.world_map import WorldMap
from splg_slam.mapping.gtsam_utils import pose_cw_to_body_gtsam
from splg_slam.mapping.imu_preintegration import (
    find_static_window,
    heading_fix_rotation,
    preintegrate,
    rotation_aligning,
)


def choose_imu_init_mode(
    imu_measurements: np.ndarray, init_static_samples: int, search_samples: int, gyro_static_threshold: float,
    gravity_norm: float | None = None, accel_mean_tolerance: float = 0.5, accel_wobble_tolerance: float = 0.4,
) -> tuple[str, np.ndarray | None]:
    """Inspects the leading `search_samples` raw IMU measurements and decides whether a
    static (device at rest) or dynamic (device already being handled/moved, e.g. picked up
    before takeoff - the common EuRoC case) gravity-alignment procedure should be used.
    Returns ("static", accel_window) if the most-still window found has mean gyro magnitude
    below `gyro_static_threshold` (and, if `gravity_norm` is given, passes the accelerometer
    check below), else ("dynamic", None).

    A low gyro magnitude alone only rules out *rotation* - a device undergoing pure
    translation (carried in a straight line, accelerating/braking, panned sideways with no
    twist) can hold a near-zero angular rate while its accelerometer reads gravity *plus*
    real linear acceleration, which would silently corrupt gravity_alignment_rotation's
    "average accel over this window = up" assumption. When `gravity_norm` is given, this
    additionally checks two things over the chosen window:
      1. the norm of the *averaged* accel vector stays within `accel_mean_tolerance` (m/s^2)
         of `gravity_norm` - catches a sustained net acceleration (most visible for
         up/down translation, which directly changes the measured magnitude);
      2. the per-sample accel *vector*'s RMS deviation from that same window-mean vector
         stays under `accel_wobble_tolerance` (m/s^2) - catches sideways/forward-back
         translation too. A magnitude-only check is nearly blind to that case (adding a
         lateral component barely moves the vector's norm since gravity already dominates
         it - sqrt(g^2+a^2) ~= g + a^2/2g for small lateral a), but real handheld/robot
         translation is essentially never a *perfectly constant* acceleration for a whole
         ~1-2s window - it accelerates and decelerates - so it shows up as vector wobble
         around the window's own mean even when the net average nets out close to g.
    Deliberately NOT a per-sample *magnitude* check against `gravity_norm` (tried that
    first): it false-rejected a genuinely static window (gyro 0.011 rad/s, per-sample
    accel-vector RMS deviation only 0.127 m/s^2) just because this particular IMU has an
    ordinary ~0.7% calibration bias (reads ~9.88 at rest, not 9.81007) plus a couple of
    noise samples up to 10.12 - the *pointwise* max-deviation-from-gravity_norm check
    flagged that as "dynamic" and silently skipped gravity alignment entirely, which is a
    worse failure than the bug this was meant to fix. `gravity_norm=None` (the default)
    skips this and preserves the old gyro-only behavior, for any caller not yet updated.

    Caveat shared with every accelerometer-only static check (ORB-SLAM3/VINS-Mono
    included): a perfectly constant acceleration sustained for the entire window is
    physically indistinguishable from a constant tilt - a specific-force sensor can't tell
    "1g down, tilted" from "1g down plus a steady 0.3g sideways push". Only genuinely
    jerky (non-constant) motion is caught here; that's the realistic case in practice."""
    accel_window, gyro_mag_mean = find_static_window(imu_measurements, init_static_samples, search_samples)
    if gyro_mag_mean >= gyro_static_threshold:
        return "dynamic", None
    if gravity_norm is not None:
        mean_accel = accel_window.mean(axis=0)
        mean_accel_mag = float(np.linalg.norm(mean_accel))
        residual = accel_window - mean_accel
        wobble_rms = float(np.sqrt((residual ** 2).sum(axis=1).mean()))
        if abs(mean_accel_mag - gravity_norm) > accel_mean_tolerance or wobble_rms > accel_wobble_tolerance:
            return "dynamic", None
    return "static", accel_window


def _estimate_gyro_bias(
    body_rotations: list[np.ndarray], pairs: list[tuple[int, int, np.ndarray]], params,
    n_iters: int = 3, eps: float = 1e-4, bg_init: np.ndarray | None = None,
) -> np.ndarray:
    """Gauss-Newton refinement of gyro bias so preintegrated rotations match vision-derived
    consecutive body rotations. Uses finite-difference bias Jacobians (this GTSAM Python
    build doesn't expose the internal analytic ones) - cheap since preintegration over a
    handful of keyframe intervals is itself cheap.

    `bg_init`, if given, seeds the Gauss-Newton iteration instead of zero - gyro bias is
    comparatively well-observed even with little motion (see this module's other
    docstrings), so this mostly just speeds convergence, but it also means a short/early
    window starts from the best available estimate (a pre-calibrated value, or whatever
    was last estimated for this sensor) instead of throwing that information away."""
    bg = np.zeros(3) if bg_init is None else np.array(bg_init, dtype=np.float64)
    for _ in range(n_iters):
        jac_rows, residuals = [], []
        for i, j, samples in pairs:
            target = body_rotations[i].T @ body_rotations[j]
            bias0 = imuBias.ConstantBias(np.zeros(3), bg)
            r0 = gtsam.Rot3.Logmap(gtsam.Rot3(preintegrate(samples, bias0, params).deltaRij().matrix().T @ target))
            jac = np.zeros((3, 3))
            for k in range(3):
                dbg = bg.copy()
                dbg[k] += eps
                biask = imuBias.ConstantBias(np.zeros(3), dbg)
                rk = gtsam.Rot3.Logmap(
                    gtsam.Rot3(preintegrate(samples, biask, params).deltaRij().matrix().T @ target)
                )
                jac[:, k] = (rk - r0) / eps
            residuals.append(r0)
            jac_rows.append(jac)
        a = np.vstack(jac_rows)
        b = np.concatenate(residuals)
        delta, *_ = np.linalg.lstsq(a, -b, rcond=None)
        bg = bg + delta
    return bg


def _estimate_gravity_bias_and_velocities(
    n: int, body_rotations: list[np.ndarray], body_positions: list[np.ndarray],
    pairs: list[tuple[int, int, np.ndarray]], params, bg: np.ndarray, eps: float = 1e-3,
    solve_scale: bool = False, ba_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], float] | None:
    """Closed-form linear solve for gravity (in the current, not-yet-aligned world frame),
    a single shared accelerometer bias, and per-keyframe velocities, from the standard IMU
    preintegration nav equations:
        v_j = v_i + g*dt + R_i @ dv_ij(ba)
        p_j = p_i + v_i*dt + 0.5*g*dt^2 + R_i @ dp_ij(ba)
    linearizing dv_ij/dp_ij around ba=0 via finite-difference Jacobians (same technique as
    the gyro-bias solve).

    solve_scale=False (stereo/RGBD - the default): `body_positions` are already metric, so
    (as in ORB-SLAM3/VINS-Mono's *stereo* initialization) there's no scale unknown - `scale`
    is returned as 1.0 (a no-op for the caller's rescale step).

    solve_scale=True (monocular): `body_positions` are only known up to an unknown scale
    (p_i = scale * body_positions[i]), so - exactly like ORB-SLAM3/VINS-Mono's *monocular*
    initialization - `scale` becomes one more unknown in the same linear system, appearing
    only in the position equation (velocity is solved in true metric units regardless, same
    as gravity): the constant `(body_positions[j] - body_positions[i])` term moves from the
    RHS into a new LHS column with coefficient `(body_positions[j] - body_positions[i])`
    multiplying the unknown scale.

    Returns (gravity, ba, {positional_index: velocity}, scale) - velocities dict is keyed by
    position (not every index need appear, if it wasn't touched by any surviving pair) - or
    None if the linear system is non-finite/ill-conditioned enough that LAPACK can't solve
    it, which the caller treats as "no usable solve this time".

    Accelerometer bias gets a staged Tikhonov prior pulling it toward `ba_prior` (a
    pre-calibrated value, or whatever was last estimated for this segment/sensor - zero
    if neither is available, the original behavior), added as extra rows to the normal
    equations (equivalent to minimizing ||Ax-b||^2 + lambda*||ba-ba_prior||^2) - borrowed
    from ORB-SLAM3's InitializeIMU, which regularizes accel bias with a very strong prior
    at bootstrap (priorA=1e5) and only fully releases it in its second inertial-BA stage
    15s in (see LocalMapping.cc's InitializeIMU/VIBA1/VIBA2 staging). Pulling toward zero
    specifically is only correct the very first time this ever runs for a sensor with no
    other information available; once a real estimate exists (from a prior window, a
    prior Atlas segment, or a user-supplied calibration), that's a strictly better center
    for this prior than an assumption of "no bias at all". Without this,
    this closed-form solve is a plain unregularized least-squares fit: accel bias and
    gravity direction are only weakly, jointly observable from limited/early motion
    diversity, so with too little data the fit can land anywhere in that near-degenerate
    direction - measured directly on one real dataset, accel_bias's y-component swung from
    -0.148 to +0.184 m/s^2 (a ~0.33 m/s^2 jump) across solves just 15-30 keyframes apart,
    each individually "accepted" by the gravity-magnitude check alone (which says nothing
    about whether ba itself is well-constrained). Gyro bias isn't given the same treatment -
    it's solved separately (_estimate_gyro_bias, Gauss-Newton against vision-derived
    rotations) and is comparatively well-observed even with little motion, since rotation is
    directly measurable - matching ORB-SLAM3's own choice to relax its gyro-bias prior (VIBA1,
    5s) well before its accel-bias one (VIBA2, 15s)."""
    bias0 = imuBias.ConstantBias(np.zeros(3), bg)
    dv0, dp0, dt, jv, jp = {}, {}, {}, {}, {}
    for i, j, samples in pairs:
        preint0 = preintegrate(samples, bias0, params)
        dv0[(i, j)] = preint0.deltaVij()
        dp0[(i, j)] = preint0.deltaPij()
        dt[(i, j)] = preint0.deltaTij()
        jv_ij, jp_ij = np.zeros((3, 3)), np.zeros((3, 3))
        for k in range(3):
            dba = np.zeros(3)
            dba[k] = eps
            biask = imuBias.ConstantBias(dba, bg)
            preintk = preintegrate(samples, biask, params)
            jv_ij[:, k] = (preintk.deltaVij() - dv0[(i, j)]) / eps
            jp_ij[:, k] = (preintk.deltaPij() - dp0[(i, j)]) / eps
        jv[(i, j)] = jv_ij
        jp[(i, j)] = jp_ij

    # unknowns: v_0..v_{n-1} (3 each, only those touched by a surviving pair matter), g, ba,
    # [scale] (mono only, 1 scalar)
    num_unknowns = 3 * n + 6 + (1 if solve_scale else 0)
    g_col, ba_col = 3 * n, 3 * n + 3
    scale_col = 3 * n + 6
    rows, rhs = [], []
    touched = set()
    for i, j, samples in pairs:
        r_i = body_rotations[i]
        touched.add(i)
        touched.add(j)

        row_v = np.zeros((3, num_unknowns))
        row_v[:, 3 * i:3 * i + 3] = -np.eye(3)
        row_v[:, 3 * j:3 * j + 3] = np.eye(3)
        row_v[:, g_col:g_col + 3] = -dt[(i, j)] * np.eye(3)
        row_v[:, ba_col:ba_col + 3] = -(r_i @ jv[(i, j)])
        rows.append(row_v)
        rhs.append(r_i @ dv0[(i, j)])

        row_p = np.zeros((3, num_unknowns))
        row_p[:, 3 * i:3 * i + 3] = -dt[(i, j)] * np.eye(3)
        row_p[:, g_col:g_col + 3] = -0.5 * dt[(i, j)] ** 2 * np.eye(3)
        row_p[:, ba_col:ba_col + 3] = -(r_i @ jp[(i, j)])
        rows.append(row_p)
        if solve_scale:
            row_p[:, scale_col] = body_positions[j] - body_positions[i]
            rhs.append(r_i @ dp0[(i, j)])
        else:
            rhs.append(r_i @ dp0[(i, j)] - (body_positions[j] - body_positions[i]))

    # Staged accel-bias prior (see docstring): strong early, relaxed as more of the window's
    # own elapsed time accumulates - same 5s/15s breakpoints ORB-SLAM3 uses for its own
    # VIBA1/VIBA2 relaxation. Implemented as extra rows minimizing lambda*||ba - ba_prior||^2
    # alongside the physical residual rows above (standard Tikhonov/ridge augmentation).
    elapsed_s = sum(dt.values())
    if elapsed_s < 5.0:
        accel_bias_prior_weight = 1.0e4   # bootstrap-equivalent: pin near ba_prior, matches ORB-SLAM3's priorA=1e5 stage
    elif elapsed_s < 15.0:
        accel_bias_prior_weight = 1.0e1   # VIBA1-equivalent: loosely regularized
    else:
        accel_bias_prior_weight = 0.0     # VIBA2-equivalent: fully data-driven, same as before this change
    if accel_bias_prior_weight > 0:
        prior_row = np.zeros((3, num_unknowns))
        prior_row[:, ba_col:ba_col + 3] = np.sqrt(accel_bias_prior_weight) * np.eye(3)
        rows.append(prior_row)
        rhs.append(np.sqrt(accel_bias_prior_weight) * (np.zeros(3) if ba_prior is None else np.asarray(ba_prior, dtype=np.float64)))

    a = np.vstack(rows)
    b = np.concatenate(rhs)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        # A non-finite entry means the vision poses feeding this solve are already corrupt
        # (e.g. a diverged segment) - there is nothing to recover here, and LAPACK would
        # either throw or return garbage.
        return None
    try:
        x, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        # Ill-conditioned window (seen in practice once Atlas re-initialization mixes
        # segments with very different scales): skip this attempt rather than killing the
        # whole run - the caller treats None as "no usable solve, try again later".
        return None
    velocities = {i: x[3 * i:3 * i + 3] for i in touched}
    gravity = x[g_col:g_col + 3]
    ba = x[ba_col:ba_col + 3]
    scale = float(x[scale_col]) if solve_scale else 1.0
    return gravity, ba, velocities, scale


def _rotation_axis_diversity(body_rotations: list[np.ndarray], pairs: list[tuple[int, int, np.ndarray]]) -> float:
    """0.0 (worst) - 1.0 (best) measure of how much the window's consecutive-keyframe
    rotations actually re-oriented the body relative to gravity, vs. just re-labeling the
    same rotation over and over (e.g. pure yaw, or no rotation at all). Gravity is fixed
    in world frame and accelerometer bias is (assumed) fixed in body frame - the ONLY way
    a solve can tell them apart is if the body's attitude actually changes enough that
    gravity's OWN apparent direction in body frame visibly shifts between samples; a
    window whose rotations all happen to share (near enough) one single axis produces
    exactly the same body-frame view of gravity throughout, which is indistinguishable
    from a constant bias no matter how much such data is collected.

    Computed as the smallest eigenvalue of the (normalized) second-moment matrix of the
    unit rotation axes between consecutive pairs, relative to the largest: axes clustered
    on a single line (pure single-axis rotation) or a single plane (e.g. yaw-only, whose
    axes are all near-parallel to gravity) collapse this ratio toward 0; axes spread
    across all 3 dimensions (genuine roll+pitch+yaw diversity) push it toward its maximum
    of 1/3, reported rescaled to a 0-1 range for readability."""
    axes = []
    for i, j, _ in pairs:
        d_r = body_rotations[i].T @ body_rotations[j]
        # Antisymmetric-part axis formula: ill-conditioned near 0 or 180 degrees of
        # rotation (sin(theta)~=0), where the axis itself is either undefined or
        # numerically unstable - skip both rather than inject noise into the diversity
        # estimate from a pair that isn't actually informative about the axis anyway.
        angle_deg = rotation_angle_deg(d_r)
        if angle_deg < 1.0 or angle_deg > 179.0:
            continue
        v = np.array([d_r[2, 1] - d_r[1, 2], d_r[0, 2] - d_r[2, 0], d_r[1, 0] - d_r[0, 1]])
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            continue
        axes.append(v / norm)
    if len(axes) < 2:
        return 0.0
    axes = np.array(axes)
    m = (axes.T @ axes) / len(axes)
    eigvals = np.clip(np.linalg.eigvalsh(m), 0.0, None)
    return float(eigvals[0] / (1.0 / 3.0))


def _solve_and_realign(
    world_map: WorldMap, kf_ids_ordered: list[int], imu_calib: ImuCalibration, imu_params, gravity_norm: float,
    gravity_error_tolerance: float | None = None, best_gravity_error_so_far: float | None = None,
    best_axis_diversity_so_far: float | None = None,
    solve_scale: bool = False, apply_kf_ids: list[int] | None = None, max_rotation_deg: float | None = None,
    min_rotation_axis_diversity: float | None = None,
) -> dict | None:
    """Solves gyro bias + accel bias + gravity direction + per-keyframe velocities (+ a
    metric scale factor, if solve_scale - monocular tracking's poses/map are otherwise only
    known up to an arbitrary scale) from `kf_ids_ordered`'s vision poses and the IMU data
    accumulated between them (world_map.imu_factors). Returns diagnostics, or None if no
    consecutive pair in the window has surviving IMU data at all.

    If `gravity_error_tolerance` is given, the solve is only *applied* (retroactive
    rotation/rescale + velocity/bias write-back) when the estimated gravity magnitude is
    within that relative tolerance of `gravity_norm` (a physical constant we know
    independent of any ground-truth trajectory - a basic sanity check that catches a
    degenerate/corrupted solve) AND (if `best_gravity_error_so_far` is given) it's a
    Pareto-frontier improvement over history: no worse on magnitude, OR - when
    `best_axis_diversity_so_far` is also given - strictly better on motion diversity
    (axis_diversity below) than whatever solve set that incumbent. Comparing magnitude
    alone (the original, `best_axis_diversity_so_far=None` behavior) lets an early,
    low-diversity window that got lucky on magnitude permanently block every later, better-
    conditioned solve from ever superseding it - confirmed directly on a real RealSense
    recording: a 20-keyframe/axis_diversity=0.27 window's 0.3% error became an unbeatable
    ratchet floor for the rest of a 110s segment, while later checks with 2-3x the motion
    diversity and comparable magnitude were rejected purely for not numerically undercutting
    that first lucky number, even as their own implied rotation grew from ~0.2deg to ~4deg
    over the segment - a real, progressively accumulating drift the magnitude-only ratchet
    had no mechanism to ever admit. `diag["accepted"]` reports which happened; the solve is
    always computed and returned either way so the caller can see how close it came.

    Gravity-MAGNITUDE agreement alone does not mean the estimated gravity DIRECTION is
    correct: a solve can land on |gravity|~=9.81 with the direction itself still tilted by a
    few degrees (measurement noise can preserve magnitude while direction drifts), and this
    check has no way to catch that on its own - confirmed as a real, non-hypothetical failure
    mode on one dataset: the ONLY periodic correction accepted the whole run passed the
    magnitude check (1.6% error) but baked in a ~2.4 degree tilt that a plane fit through the
    final trajectory's own (x, y, z) explained 90% of the Z variance with, and which nothing
    ever corrected afterward (every later periodic check failed the *magnitude* check and was
    rejected, for unrelated reasons). `max_rotation_deg`, if given, adds a second, independent
    check: the ANGLE of the rotation this correction would actually apply (computed from
    `r_align` below) must also stay under this cap. This is a meaningful check specifically
    BECAUSE by the time periodic reinit ever runs, the segment already has *some* gravity
    alignment (from its own bootstrap - see run_dynamic_imu_init, which correctly applies
    unconditionally since establishing that FIRST alignment is exactly the one time a large
    rotation is legitimate) - so a periodic re-solve finding it needs to rotate the world by
    much more than a few degrees to "fix" it is itself suspicious, not obviously more correct
    than what's already there, regardless of how good its gravity-magnitude number looks.

    `min_rotation_axis_diversity`, if given, adds a THIRD, independent check on top of the
    two above: reject a solve whose window's own motion didn't have enough attitude
    diversity to reliably separate gravity direction from accelerometer bias in the first
    place (see _rotation_axis_diversity), regardless of how good the gravity-magnitude and
    implied-rotation numbers look - unlike those two, this checks the WINDOW's own data
    quality directly instead of comparing the answer against an assumed-trustworthy prior,
    so it doesn't depend on this segment already having a decent alignment to compare
    against (also reported in diag either way, so a caller with this check disabled can
    still see the number and decide on a threshold later).

    `apply_kf_ids` (defaults to `kf_ids_ordered` if not given): the accepted rotation/scale
    correction is a SINGLE rigid transform, valid for re-orienting the whole segment (it only
    corrects one shared, segment-wide unknown - "which way is up" - not per-keyframe drift),
    so it can safely be computed from just a recent SLIDING WINDOW (kf_ids_ordered, for a
    locally-relevant, not-yet-drift-corrupted solve - see run_periodic_imu_reinit) while still
    being *applied* to the segment's full keyframe/map-point set (apply_kf_ids). Applying it
    only to the solved window instead would tear the trajectory: the window's keyframes would
    rotate together while the rest of the same, already-continuous segment stayed exactly
    where it was, opening a kink at the window boundary instead of a uniform re-orientation."""
    imu_factor_by_pair = {(a, b): samples for a, b, samples in world_map.imu_factors}

    body_rotations, body_positions = [], []
    for kf_id in kf_ids_ordered:
        kf = world_map.keyframes[kf_id]
        body_pose = pose_cw_to_body_gtsam(kf.pose_cw, imu_calib.T_cam0_body)
        body_rotations.append(body_pose.rotation().matrix())
        body_positions.append(np.array(body_pose.translation()))

    pairs = []
    for idx in range(len(kf_ids_ordered) - 1):
        a, b = kf_ids_ordered[idx], kf_ids_ordered[idx + 1]
        # A tracking.continuous_imu_tracking (imu_only) keyframe's pose came from dead-
        # reckoned IMU integration in the first place, not an independent vision
        # observation - feeding it into body_positions/body_rotations above as if it were
        # a known "ground truth" endpoint makes this solve circular (it ends up fitting
        # the IMU model against its own earlier prediction instead of against a real
        # measurement), which can corrupt gravity/bias for the whole window. Skip any pair
        # touching one. Measured directly: including these pairs left a persistent,
        # uncorrected Z-offset band that no amount of downstream loop-closure pose-graph
        # sigma reweighting could undo (realsense_230436: z_range stuck at ~0.75-0.78m vs
        # 0.60m for the equivalent Atlas+tight_fusion run, across a wide sweep of
        # odom_imu_only_penalty_m from 0 to 300) - the damage was already done here, well
        # upstream of any pose-graph step.
        if getattr(world_map.keyframes[a], "imu_only", False) or getattr(world_map.keyframes[b], "imu_only", False):
            continue
        samples = imu_factor_by_pair.get((a, b))
        if samples is not None:
            pairs.append((idx, idx + 1, samples))
    if not pairs:
        return None

    # Seed both solves from whatever bias this segment's own first keyframe already
    # carries - either a user-supplied calibration (OfflineMapper._imu_bias_prior) or
    # the last value actually estimated for this sensor (KeyFrame.imu_bias is carried
    # forward across Atlas segment restarts rather than reset to zero - see
    # tracker.py's _start_new_map_segment) - instead of discarding that information and
    # re-deriving everything from an assumed-zero start every single time this runs.
    seed_bias = world_map.keyframes[kf_ids_ordered[0]].imu_bias
    seed_ba = None if seed_bias is None else np.asarray(seed_bias[:3], dtype=np.float64)
    seed_bg = None if seed_bias is None else np.asarray(seed_bias[3:6], dtype=np.float64)

    bg = _estimate_gyro_bias(body_rotations, pairs, imu_params, bg_init=seed_bg)
    solved = _estimate_gravity_bias_and_velocities(
        len(kf_ids_ordered), body_rotations, body_positions, pairs, imu_params, bg,
        solve_scale=solve_scale, ba_prior=seed_ba,
    )
    if solved is None:
        return None
    gravity, ba, velocities, scale = solved
    axis_diversity = _rotation_axis_diversity(body_rotations, pairs)

    gravity_mag = float(np.linalg.norm(gravity))
    gravity_error = abs(gravity_mag - gravity_norm) / gravity_norm

    # Computed regardless of acceptance (needed for the rotation-magnitude check below, and
    # reported in diag either way for visibility) - see this function's docstring on why
    # gravity-magnitude agreement alone can't be trusted to mean the DIRECTION is correct too.
    up_dir = -gravity / max(gravity_mag, 1e-6)
    r_align = rotation_aligning(up_dir, np.array([0.0, 0.0, 1.0]))

    # rotation_aligning's minimal-rotation construction only constrains 2 of 3 rotational
    # DOF (which way is "up") - the residual yaw is an arbitrary side effect of whatever
    # raw orientation this segment's reference keyframe happened to start with, and a FRESH
    # arbitrary yaw gets introduced on every call site's own minimal rotation (bootstrap,
    # kf20-refine, VIBA1, VIBA2, periodic reinit) independently, with nothing tying them
    # together - confirmed directly: two visually-identical straight-line recordings ended
    # up ~91 degrees apart in world heading purely from this. Pin the segment's own
    # reference keyframe's camera-forward direction to world +X (the same convention the
    # no-IMU bootstrap already uses in tracker.py) by composing an extra world-Z-axis
    # rotation on top - this only touches heading, never the vertical alignment r_align
    # just established (a Z-axis rotation composed on the left leaves world Z fixed).
    forward_before = world_map.keyframes[kf_ids_ordered[0]].pose_cw[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    forward_mid = r_align @ forward_before
    r_align = heading_fix_rotation(forward_mid[:2]) @ r_align
    r_align_angle_deg = rotation_angle_deg(r_align)

    accepted = True
    if gravity_error_tolerance is not None:
        accepted = gravity_error < gravity_error_tolerance
        if accepted and best_gravity_error_so_far is not None:
            # Pareto-frontier ratchet, not a magnitude-only one: a later solve supersedes
            # the incumbent if it beats it on EITHER axis - gravity-magnitude accuracy, OR
            # motion-diversity (axis_diversity - see _rotation_axis_diversity). Comparing
            # magnitude alone (the original behavior) lets a small, low-diversity early
            # window "get lucky" on magnitude and then permanently block every later,
            # better-conditioned solve from ever superseding it, even once real accumulated
            # drift shows up - confirmed directly on a real RealSense recording: a 20-
            # keyframe/axis_diversity=0.27 window's 0.3% error became an unbeatable ratchet
            # floor for the REST of a 110s segment, while later checks with 2-3x the motion
            # diversity (0.5-0.65) and comparable magnitude were rejected purely for not
            # numerically undercutting that first lucky number, even as their own implied
            # rotation grew from ~0.2deg to ~4deg over the segment - a real, progressively
            # accumulating drift the ratchet had no mechanism to ever admit.
            beats_on_magnitude = gravity_error <= best_gravity_error_so_far
            beats_on_diversity = (
                best_axis_diversity_so_far is not None and axis_diversity > best_axis_diversity_so_far
            )
            accepted = beats_on_magnitude or beats_on_diversity
    if max_rotation_deg is not None and r_align_angle_deg > max_rotation_deg:
        accepted = False
    if min_rotation_axis_diversity is not None and axis_diversity < min_rotation_axis_diversity:
        # Independent of whether the answer LOOKS good (small gravity error, small
        # implied rotation): if this window's own motion couldn't have told gravity
        # direction and accelerometer bias apart in the first place (see
        # _rotation_axis_diversity), the answer isn't trustworthy regardless of how
        # confident it looks - reject and let the caller retry with a later, hopefully
        # more diverse window instead of committing to a possibly-arbitrary point along
        # an under-constrained direction.
        accepted = False
    if solve_scale and scale <= 0:
        # A degenerate/ill-conditioned solve (near-planar motion, too little data) can
        # return a nonsense non-positive scale - never apply that, regardless of how good
        # the gravity-magnitude check alone looks.
        accepted = False

    diag = {
        "gravity_norm_estimated": gravity_mag,
        "gravity_norm_expected": gravity_norm,
        "gravity_error": gravity_error,
        "rotation_angle_deg": r_align_angle_deg,
        "rotation_axis_diversity": axis_diversity,
        "gyro_bias": bg,
        "accel_bias": ba,
        "scale": scale,
        "num_keyframes": len(kf_ids_ordered),
        "num_pairs": len(pairs),
        "accepted": accepted,
    }
    if not accepted:
        return diag

    # Scoped to apply_kf_ids (defaults to kf_ids_ordered) - NOT
    # world_map.keyframes.values()/map_points.values() wholesale. This solve's gravity/bias/
    # scale are only meaningful in the coordinate frame shared by kf_ids_ordered/apply_kf_ids
    # (the caller is expected to pass one Atlas segment's own keyframes, per ORB-SLAM3's IMU
    # init likewise being scoped to mpAtlas->GetCurrentMap() alone, never the whole Atlas).
    # Applying this correction to every keyframe/map point in the entire world map would also
    # rigidly rotate/rescale any OTHER segment's already-correct poses using a correction that
    # has nothing to do with their own coordinate frame - confirmed as a real, active bug:
    # once a second Atlas segment existed, this unconditionally corrupted its poses on every
    # accepted re-solve from the first segment's own periodic reinit.
    solve_kf_ids = set(apply_kf_ids) if apply_kf_ids is not None else set(kf_ids_ordered)
    for kf_id in solve_kf_ids:
        kf = world_map.keyframes[kf_id]
        kf.pose_cw = kf.pose_cw.copy()
        kf.pose_cw[:3, :3] = kf.pose_cw[:3, :3] @ r_align.T
        kf.pose_cw[:3, 3] = kf.pose_cw[:3, 3] * scale  # no-op when solve_scale=False (scale=1.0)
    for mp in world_map.map_points.values():
        observers = set(mp.observations.keys())
        if observers and observers <= solve_kf_ids:
            mp.position = r_align @ (mp.position * scale)

    bias_vec = np.concatenate([ba, bg])
    for idx, v in velocities.items():
        kf = world_map.keyframes[kf_ids_ordered[idx]]
        kf.velocity = r_align @ v
        kf.imu_bias = bias_vec

    return diag


def run_dynamic_imu_init(
    world_map: WorldMap, kf_ids_ordered: list[int], imu_calib: ImuCalibration, imu_params, gravity_norm: float,
    solve_scale: bool = False, gravity_error_tolerance: float | None = None,
    min_rotation_axis_diversity: float | None = None, max_rotation_deg: float | None = None,
) -> dict | None:
    """ORB-SLAM-style motion-based IMU bootstrap: instead of assuming the sequence starts
    static, run ordinary vision-only tracking for the first `len(kf_ids_ordered)` keyframes
    in an arbitrary (gravity-unaware, and for monocular - see solve_scale - arbitrary-scale)
    world frame, then solve for gravity/bias/velocities(/scale) and retroactively re-align -
    see _solve_and_realign.

    gravity_error_tolerance=None (the stereo/RGBD default): applies unconditionally - there's
    no earlier estimate to compare against yet, *some* alignment is needed to even start
    using IMU factors, and a bad *rotation-only* correction here is survivable (the periodic
    re-init path, which does get to be picky, corrects it later).

    gravity_error_tolerance=<a value> (used for solve_scale=True / monocular): unlike stereo,
    a bad bootstrap solve here also corrupts the map's *scale* (every position/depth gets
    multiplied by a wrong factor) - verified in practice to be able to blow up the whole map
    on a single ill-conditioned early window (e.g. |g| solved to ~3x the true 9.81 alongside
    a wild scale) and never recover. Gating on the same physical gravity-magnitude sanity
    check periodic reinit uses (diag['accepted']) keeps init pending for a later attempt with
    more/better motion, instead of applying an unconditionally-accepted bad correction.

    Returns None (init left pending for a later attempt) if the IMU-factor chain across this
    window is entirely missing, or if the solve was computed but not accepted by any of
    _solve_and_realign's checks - gravity_error_tolerance (mono only, by default),
    min_rotation_axis_diversity, or max_rotation_deg (either mode, if given). Checking
    diag["accepted"] unconditionally (not just when gravity_error_tolerance is set) matters
    once min_rotation_axis_diversity/max_rotation_deg are in play: for stereo
    (gravity_error_tolerance=None), _solve_and_realign still returns a non-None diag on a
    diversity- or rotation-rejected solve (it just skips applying it) - treating that as
    "done" here instead of "still pending" would leave the segment permanently stuck at its
    provisional pose, believing (incorrectly) that it had already been corrected.

    max_rotation_deg=None is the right default HERE specifically for the segment's very
    first bootstrap call (kf_ids_ordered starting at the segment's own first keyframe, no
    prior alignment yet) - see _solve_and_realign's own docstring on why a large rotation is
    legitimate exactly once, establishing that first alignment. A caller re-using this same
    function to REFINE an already-aligned segment later (e.g. build_map.py's kf20-refine/
    VIBA1/VIBA2 under orbslam3_style_init) should pass this explicitly, mirroring
    run_periodic_imu_reinit's own use of the same check - measured directly as a real,
    non-hypothetical gap: a kf20-refine/VIBA1 pair that both passed a 0.3% gravity-magnitude
    check nonetheless baked in a ~3.2 degree tilt (confirmed via a plane fit through the
    whole segment's own keyframe positions explaining 78% of its Z variance) - magnitude
    agreement alone doesn't catch a bad DIRECTION, exactly per this function's and
    _solve_and_realign's own long-standing warning, but until this parameter existed here
    there was no way for this refine path to use the same rotation-magnitude check
    run_periodic_imu_reinit already had."""
    diag = _solve_and_realign(
        world_map, kf_ids_ordered, imu_calib, imu_params, gravity_norm,
        gravity_error_tolerance=gravity_error_tolerance, solve_scale=solve_scale,
        min_rotation_axis_diversity=min_rotation_axis_diversity, max_rotation_deg=max_rotation_deg,
    )
    if diag is not None and not diag["accepted"]:
        return None
    return diag


def run_periodic_imu_reinit(
    world_map: WorldMap, kf_ids_ordered: list[int], imu_calib: ImuCalibration, imu_params, gravity_norm: float,
    gravity_error_tolerance: float, best_gravity_error_so_far: float | None, solve_scale: bool = False,
    apply_kf_ids: list[int] | None = None, max_rotation_deg: float | None = None,
    min_rotation_axis_diversity: float | None = None, best_axis_diversity_so_far: float | None = None,
) -> dict | None:
    """ORB-SLAM3 VIBA-style staged re-optimization: redo the same gravity/bias/velocity
    solve later, with much more accumulated motion diversity available by then than the
    initial bootstrap window had - this is also the only place accelerometer bias ever gets
    a proper estimate instead of sitting at 0. Runs regardless of whether the original
    bootstrap was static or dynamic.

    `kf_ids_ordered` should be a bounded, recent SLIDING window (e.g. the segment's last
    ~100 keyframes), not its entire history - measured directly on a real run: an unbounded,
    ever-growing window's own gravity-magnitude error grew from 0.4% (early, short window)
    past 80%+ once the window had grown to span several hundred keyframes, because the
    solve trusts the vision-derived trajectory *over the window* as ground truth, and that
    trajectory's own accumulated drift grows right along with an unbounded window - a
    bounded recent window keeps that assumption valid regardless of how long the segment has
    been running in total. `apply_kf_ids` lets the resulting rotation/scale correction still
    be applied to the segment's full keyframe set even though it was solved from just the
    recent window (see _solve_and_realign's docstring on why that's safe and necessary to
    avoid a kink at the window boundary) - pass the segment's complete keyframe list there.

    Gated by gravity_error_tolerance alone now (see _solve_and_realign) - since each window
    covers a different, independent stretch rather than progressively refining one
    persistent global estimate, `best_gravity_error_so_far` no longer has a consistent
    meaning across calls; pass None. The caller is expected to call this periodically (e.g.
    every N keyframes).

    `max_rotation_deg`, if given, adds the independent direction-sanity check described in
    _solve_and_realign's docstring - reject a correction whose gravity MAGNITUDE passes but
    whose implied rotation is itself larger than this, since periodic reinit only ever runs
    once the segment already has some (bootstrap-established) gravity alignment, so a
    solve claiming it needs to rotate the world by more than a few degrees to "fix" that is
    itself suspicious rather than obviously an improvement.

    `best_axis_diversity_so_far`, if given alongside `best_gravity_error_so_far`, makes the
    ratchet a Pareto frontier over BOTH gravity-magnitude accuracy and motion diversity
    instead of magnitude alone - see _solve_and_realign's own docstring on why a magnitude-
    only ratchet lets an early, low-diversity, "lucky" window permanently block every later,
    better-conditioned solve from ever superseding it, even as real drift accumulates."""
    return _solve_and_realign(
        world_map, kf_ids_ordered, imu_calib, imu_params, gravity_norm,
        gravity_error_tolerance=gravity_error_tolerance, best_gravity_error_so_far=best_gravity_error_so_far,
        best_axis_diversity_so_far=best_axis_diversity_so_far,
        solve_scale=solve_scale, apply_kf_ids=apply_kf_ids, max_rotation_deg=max_rotation_deg,
        min_rotation_axis_diversity=min_rotation_axis_diversity,
    )
