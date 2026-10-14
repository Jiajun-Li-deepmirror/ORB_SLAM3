import gtsam
import numpy as np
from gtsam import imuBias

from splg_slam.data.euroc import ImuCalibration


def make_preintegration_params(
    imu_calib: ImuCalibration, gravity_norm: float, integration_sigma: float,
    bias_init_covariance: float | None = None,
) -> gtsam.PreintegrationCombinedParams:
    params = gtsam.PreintegrationCombinedParams(np.array([0.0, 0.0, -gravity_norm]))
    params.setAccelerometerCovariance(np.eye(3) * imu_calib.accel_noise_density ** 2)
    params.setGyroscopeCovariance(np.eye(3) * imu_calib.gyro_noise_density ** 2)
    params.setIntegrationCovariance(np.eye(3) * integration_sigma ** 2)
    params.setBiasAccCovariance(np.eye(3) * imu_calib.accel_random_walk ** 2)
    params.setBiasOmegaCovariance(np.eye(3) * imu_calib.gyro_random_walk ** 2)
    # GTSAM 4.2's legacy default adds I6 at every integration step and greatly
    # weakens the IMU factors. Zero uses the sensor-noise model with graph bias
    # priors/random walk. Keep that change explicit: full-sequence regression
    # exposed large RealSense loop deformation when enabled unconditionally.
    if bias_init_covariance is not None:
        if not np.isfinite(bias_init_covariance) or bias_init_covariance < 0:
            raise ValueError("bias_init_covariance must be finite and nonnegative")
        if hasattr(params, "setBiasAccOmegaInit"):
            params.setBiasAccOmegaInit(np.eye(6) * bias_init_covariance)
    return params


def bias_from_vector(vec: np.ndarray | None) -> imuBias.ConstantBias:
    if vec is None:
        return imuBias.ConstantBias()
    vec = np.asarray(vec)
    return imuBias.ConstantBias(vec[:3], vec[3:6])


def imu_samples_between(measurements: np.ndarray, start_ns: int, end_ns: int) -> np.ndarray | None:
    """Clip sorted Nx7 measurements to the complete camera-time interval.

    Include synthetic rows at both endpoints, using the same zero-order hold as
    preintegrate(). Never extrapolate beyond the recorded IMU coverage. Adjacent
    intervals may share an endpoint; its zero dt is ignored by preintegrate().
    """
    if end_ns <= start_ns or len(measurements) < 2:
        return None
    ts = measurements[:, 0]
    if start_ns < ts[0] or end_ns > ts[-1]:
        return None
    start = int(np.searchsorted(ts, start_ns, side="right"))
    end = int(np.searchsorted(ts, end_ns, side="left"))
    first = measurements[start - 1].copy()
    last = measurements[int(np.searchsorted(ts, end_ns, side="right")) - 1].copy()
    first[0], last[0] = start_ns, end_ns
    return np.concatenate([first[None], measurements[start:end], last[None]], axis=0)


def preintegrate(
    samples: np.ndarray, bias: imuBias.ConstantBias, params: gtsam.PreintegrationCombinedParams
) -> gtsam.PreintegratedCombinedMeasurements:
    """samples: Nx7 [timestamp_ns, wx,wy,wz, ax,ay,az], sorted by time. Each row's gyro/accel
    is held constant (zero-order hold) over the interval to the next row, matching what
    `integrateMeasurement`'s `deltaT` expects."""
    preint = gtsam.PreintegratedCombinedMeasurements(params, bias)
    for i in range(len(samples) - 1):
        dt = (samples[i + 1, 0] - samples[i, 0]) / 1e9
        if dt <= 0:
            continue
        gyro = samples[i, 1:4]
        accel = samples[i, 4:7]
        preint.integrateMeasurement(accel, gyro, dt)
    return preint


def find_static_window(
    imu_measurements: np.ndarray, window_samples: int, search_samples: int
) -> tuple[np.ndarray, float]:
    """Returns (accelerometer rows [window_samples x 3], mean gyro magnitude over that
    window) for the most-static window among the leading `search_samples` measurements,
    ranked by lowest mean gyro magnitude. EuRoC's machine_hall sequences are handled/moved
    before takeoff, so the very first samples are NOT reliably static (checked empirically:
    MH01's first 200 samples average ~0.2 rad/s gyro magnitude and an accel magnitude with
    std ~1.6 m/s^2, nowhere near a stable 9.81) - picking the window with the least rotation
    is a simple, robust proxy for "actually at rest". The returned gyro magnitude is what
    the caller uses to decide whether this window is *actually* still enough to trust."""
    search_samples = min(search_samples, len(imu_measurements))
    window_samples = min(window_samples, search_samples)
    gyro_mag = np.linalg.norm(imu_measurements[:search_samples, 1:4], axis=1)
    cumsum = np.concatenate([[0.0], np.cumsum(gyro_mag)])
    window_mean = (cumsum[window_samples:] - cumsum[:-window_samples]) / window_samples
    best_start = int(np.argmin(window_mean))
    accel = imu_measurements[best_start:best_start + window_samples, 4:7]
    return accel, float(window_mean[best_start])


def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns R (3x3) such that R @ (a/|a|) ~= (b/|b|), via Rodrigues' rotation formula."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # a anti-parallel to b: 180deg rotation about any axis perpendicular to a.
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(axis, a)
        axis = axis / np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def heading_fix_rotation(forward_xy: np.ndarray) -> np.ndarray:
    """Returns R (3x3, a pure rotation about world +Z) such that R @ v rotates a world
    vector whose horizontal (XY) component is `forward_xy` to point along world +X,
    leaving the Z component untouched - used to cancel the arbitrary residual yaw that
    rotation_aligning's minimal-rotation construction leaves undetermined (gravity
    alignment only constrains 2 of 3 rotational DOF; nothing here has a compass to fix
    the third). Returns identity if forward_xy is ~zero (forward is ~vertical, no
    meaningful heading to fix)."""
    norm = float(np.linalg.norm(forward_xy))
    if norm < 1e-6:
        return np.eye(3)
    cos_a, sin_a = forward_xy[0] / norm, forward_xy[1] / norm
    return np.array([
        [cos_a, sin_a, 0.0],
        [-sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0],
    ])


def gravity_alignment_rotation(accel_samples: np.ndarray, accel_bias: np.ndarray | None = None) -> np.ndarray:
    """Returns R_world_body (3x3): rotates a body-frame vector into a world frame whose +Z
    axis points opposite gravity, estimated from a (near-)static window's average
    accelerometer reading - a stationary accelerometer reads the reaction to gravity, i.e.
    "up" in the body frame. Use find_static_window() to locate that window first.

    `accel_bias`, if given (a pre-calibrated or previously-estimated accelerometer bias,
    same 3-vector convention as KeyFrame.imu_bias[:3]), is subtracted from every sample
    before averaging. Without it, this method has no way to separate "true gravity
    direction" from "sensor bias" at all - unlike the dynamic/motion-based solve
    (_estimate_gravity_bias_and_velocities), a single static window has no rotational
    diversity to tell the two apart, so it silently bakes whatever bias the sensor has
    straight into the resulting orientation. A known bias removes that specific error
    source even though this method still can't detect a bias it doesn't already know
    about."""
    samples = accel_samples if accel_bias is None else accel_samples - accel_bias
    return rotation_aligning(samples.mean(axis=0), np.array([0.0, 0.0, 1.0]))
