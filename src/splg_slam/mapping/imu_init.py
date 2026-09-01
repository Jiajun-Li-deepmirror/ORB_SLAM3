import gtsam
import numpy as np
from gtsam import imuBias

from splg_slam.data.euroc import ImuCalibration
from splg_slam.map.world_map import WorldMap
from splg_slam.mapping.gtsam_utils import pose_cw_to_body_gtsam
from splg_slam.mapping.imu_preintegration import find_static_window, preintegrate, rotation_aligning


def choose_imu_init_mode(
    imu_measurements: np.ndarray, init_static_samples: int, search_samples: int, gyro_static_threshold: float
) -> tuple[str, np.ndarray | None]:
    """Inspects the leading `search_samples` raw IMU measurements and decides whether a
    static (device at rest) or dynamic (device already being handled/moved, e.g. picked up
    before takeoff - the common EuRoC case) gravity-alignment procedure should be used.
    Returns ("static", accel_window) if the most-still window found has mean gyro magnitude
    below `gyro_static_threshold`, else ("dynamic", None)."""
    accel_window, gyro_mag_mean = find_static_window(imu_measurements, init_static_samples, search_samples)
    if gyro_mag_mean < gyro_static_threshold:
        return "static", accel_window
    return "dynamic", None


def _estimate_gyro_bias(
    body_rotations: list[np.ndarray], pairs: list[tuple[int, int, np.ndarray]], params,
    n_iters: int = 3, eps: float = 1e-4,
) -> np.ndarray:
    """Gauss-Newton refinement of gyro bias so preintegrated rotations match vision-derived
    consecutive body rotations. Uses finite-difference bias Jacobians (this GTSAM Python
    build doesn't expose the internal analytic ones) - cheap since preintegration over a
    handful of keyframe intervals is itself cheap."""
    bg = np.zeros(3)
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
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Closed-form linear solve for gravity (in the current, not-yet-aligned world frame),
    a single shared accelerometer bias, and per-keyframe velocities, from the standard IMU
    preintegration nav equations:
        v_j = v_i + g*dt + R_i @ dv_ij(ba)
        p_j = p_i + v_i*dt + 0.5*g*dt^2 + R_i @ dp_ij(ba)
    linearizing dv_ij/dp_ij around ba=0 via finite-difference Jacobians (same technique as
    the gyro-bias solve). Translations are already metric (stereo, not monocular), so unlike
    ORB-SLAM3/VINS-Mono's initialization there is no scale unknown to solve for.

    Returns (gravity, ba, {positional_index: velocity}) - velocities dict is keyed by
    position (not every index need appear, if it wasn't touched by any surviving pair)."""
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

    # unknowns: v_0..v_{n-1} (3 each, only those touched by a surviving pair matter), g, ba
    num_unknowns = 3 * n + 6
    g_col, ba_col = 3 * n, 3 * n + 3
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
        rhs.append(r_i @ dp0[(i, j)] - (body_positions[j] - body_positions[i]))

    a = np.vstack(rows)
    b = np.concatenate(rhs)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    velocities = {i: x[3 * i:3 * i + 3] for i in touched}
    gravity = x[g_col:g_col + 3]
    ba = x[ba_col:ba_col + 3]
    return gravity, ba, velocities


def _solve_and_realign(
    world_map: WorldMap, kf_ids_ordered: list[int], imu_calib: ImuCalibration, imu_params, gravity_norm: float,
    gravity_error_tolerance: float | None = None, best_gravity_error_so_far: float | None = None,
) -> dict | None:
    """Solves gyro bias + accel bias + gravity direction + per-keyframe velocities from
    `kf_ids_ordered`'s vision poses and the IMU data accumulated between them
    (world_map.imu_factors). Returns diagnostics, or None if no consecutive pair in the
    window has surviving IMU data at all.

    If `gravity_error_tolerance` is given, the solve is only *applied* (retroactive
    whole-map rotation + velocity/bias write-back) when the estimated gravity magnitude is
    both within that relative tolerance of `gravity_norm` (a physical constant we know
    independent of any ground-truth trajectory - a basic sanity check that catches a
    degenerate/corrupted solve) AND no worse than `best_gravity_error_so_far` (if given) -
    letting the caller keep checking periodically without committing a correction that's
    worse than one already applied. `diag["accepted"]` reports which happened; the solve is
    always computed and returned either way so the caller can see how close it came."""
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
        samples = imu_factor_by_pair.get((a, b))
        if samples is not None:
            pairs.append((idx, idx + 1, samples))
    if not pairs:
        return None

    bg = _estimate_gyro_bias(body_rotations, pairs, imu_params)
    gravity, ba, velocities = _estimate_gravity_bias_and_velocities(
        len(kf_ids_ordered), body_rotations, body_positions, pairs, imu_params, bg,
    )

    gravity_mag = float(np.linalg.norm(gravity))
    gravity_error = abs(gravity_mag - gravity_norm) / gravity_norm

    accepted = True
    if gravity_error_tolerance is not None:
        accepted = gravity_error < gravity_error_tolerance and (
            best_gravity_error_so_far is None or gravity_error <= best_gravity_error_so_far
        )

    diag = {
        "gravity_norm_estimated": gravity_mag,
        "gravity_norm_expected": gravity_norm,
        "gravity_error": gravity_error,
        "gyro_bias": bg,
        "accel_bias": ba,
        "num_keyframes": len(kf_ids_ordered),
        "num_pairs": len(pairs),
        "accepted": accepted,
    }
    if not accepted:
        return diag

    up_dir = -gravity / max(gravity_mag, 1e-6)
    r_align = rotation_aligning(up_dir, np.array([0.0, 0.0, 1.0]))

    for kf in world_map.keyframes.values():
        kf.pose_cw = kf.pose_cw.copy()
        kf.pose_cw[:3, :3] = kf.pose_cw[:3, :3] @ r_align.T
    for mp in world_map.map_points.values():
        mp.position = r_align @ mp.position

    bias_vec = np.concatenate([ba, bg])
    for idx, v in velocities.items():
        kf = world_map.keyframes[kf_ids_ordered[idx]]
        kf.velocity = r_align @ v
        kf.imu_bias = bias_vec

    return diag


def run_dynamic_imu_init(
    world_map: WorldMap, kf_ids_ordered: list[int], imu_calib: ImuCalibration, imu_params, gravity_norm: float,
) -> dict | None:
    """ORB-SLAM-style motion-based IMU bootstrap: instead of assuming the sequence starts
    static, run ordinary vision-only tracking for the first `len(kf_ids_ordered)` keyframes
    in an arbitrary (gravity-unaware) world frame, then solve for gravity/bias/velocities
    and retroactively re-align - see _solve_and_realign. Always applies unconditionally
    (there's no earlier estimate to compare against yet, and *some* alignment is needed to
    even start using IMU factors) - the periodic re-init path is what gets to be picky.
    Returns None (init left pending for a later attempt) if the IMU-factor chain across
    this window is entirely missing."""
    return _solve_and_realign(world_map, kf_ids_ordered, imu_calib, imu_params, gravity_norm)


def run_periodic_imu_reinit(
    world_map: WorldMap, kf_ids_ordered: list[int], imu_calib: ImuCalibration, imu_params, gravity_norm: float,
    gravity_error_tolerance: float, best_gravity_error_so_far: float | None,
) -> dict | None:
    """ORB-SLAM3 VIBA-style staged re-optimization: redo the same gravity/bias/velocity
    solve later, over ALL keyframes accumulated so far rather than just the initial
    bootstrap window - much more motion diversity is available by then, which is what
    actually makes bias and gravity direction observable, and (unlike the bootstrap path)
    this is also the only place accelerometer bias ever gets a proper estimate instead of
    sitting at 0. Runs regardless of whether the original bootstrap was static or dynamic.

    Gated by gravity_error_tolerance/best_gravity_error_so_far (see _solve_and_realign) -
    the caller is expected to call this periodically (e.g. every N keyframes) rather than
    at one or two hand-picked keyframe counts: a longer window isn't reliably better (the
    solve trusts the vision-derived trajectory as ground truth, and that trajectory's own
    drift grows with the window too), so instead of guessing a magic window size, keep
    checking and only keep a correction that's at least as self-consistent as the last one
    applied."""
    return _solve_and_realign(
        world_map, kf_ids_ordered, imu_calib, imu_params, gravity_norm,
        gravity_error_tolerance=gravity_error_tolerance, best_gravity_error_so_far=best_gravity_error_so_far,
    )
