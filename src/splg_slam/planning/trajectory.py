import numpy as np


def to_xyz(points: np.ndarray) -> np.ndarray:
    """Pads Nx2 world points to Nx3 by appending z=0; Nx3 input passes through unchanged.
    Lets the curvature/arc-length math below stay dimension-agnostic."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[1] == 2:
        return np.concatenate([points, np.zeros((len(points), 1))], axis=1)
    return points


def shortcut_path(waypoints: np.ndarray, is_segment_free, max_iters: int = 3) -> np.ndarray:
    """Greedy visibility-based path simplification ("string pulling"): repeatedly connects
    the current waypoint directly to the farthest later waypoint whose straight-line segment
    stays entirely in inflated-free space, skipping every zig-zag grid step in between.
    `is_segment_free(p0, p1) -> bool` does the actual collision/clearance sampling, so the
    same shortcutting logic serves both the 2D elevation-cost-grid and the 3D octomap without
    duplicating the lookup code. A few passes (not just one) let a later pass straighten
    whatever corner the first pass's greedy choice happened to leave behind."""
    waypoints = np.asarray(waypoints, dtype=np.float64)
    for _ in range(max_iters):
        if len(waypoints) <= 2:
            break
        simplified = [waypoints[0]]
        i = 0
        n = len(waypoints)
        while i < n - 1:
            j = n - 1
            while j > i + 1 and not is_segment_free(waypoints[i], waypoints[j]):
                j -= 1
            simplified.append(waypoints[j])
            i = j
        new_waypoints = np.array(simplified)
        if len(new_waypoints) == len(waypoints):
            break
        waypoints = new_waypoints
    return waypoints


def chaikin_smooth(waypoints: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Chaikin corner-cutting subdivision: repeatedly replaces each segment with two points
    at 1/4 and 3/4 along it. Unlike a spline fit, the result never leaves the convex hull of
    each consecutive pair of input points, so it can't swing wide of a corner into space the
    shortcutted path's collision check never verified as clear - important since the input
    here is a *safety-inflated* free-space path, not an arbitrary curve-fitting problem.
    Endpoints are kept exact (start/goal must not move)."""
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) < 3:
        return waypoints
    for _ in range(iterations):
        pts = [waypoints[0]]
        for i in range(len(waypoints) - 1):
            p0, p1 = waypoints[i], waypoints[i + 1]
            pts.append(0.75 * p0 + 0.25 * p1)
            pts.append(0.25 * p0 + 0.75 * p1)
        pts.append(waypoints[-1])
        waypoints = np.array(pts)
    return waypoints


def resample_by_arclength(waypoints: np.ndarray, ds: float) -> np.ndarray:
    """Resamples a polyline at uniform arc-length spacing `ds` via per-axis linear
    interpolation over cumulative arc length, so the curvature and velocity-profile math
    downstream sees evenly spaced samples instead of Chaikin's geometrically uneven ones."""
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) < 2:
        return waypoints
    seg_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = arc[-1]
    if total < 1e-9:
        return waypoints[:1].copy()
    n_samples = max(2, int(np.ceil(total / ds)) + 1)
    sample_arc = np.linspace(0.0, total, n_samples)
    return np.stack(
        [np.interp(sample_arc, arc, waypoints[:, dim]) for dim in range(waypoints.shape[1])],
        axis=1,
    )


def compute_curvature(waypoints: np.ndarray) -> np.ndarray:
    """Per-sample curvature (1/turn-radius, in 1/m) via the standard parametric-curve
    formula |v x a| / |v|^3 on central-finite-difference velocity/acceleration. Dimension-
    agnostic: a 2D path padded to Nx3 (z=0) has velocity/acceleration vectors with z=0 too, so
    their cross product's x/y components vanish and only the z component - the familiar 2D
    scalar curvature - survives, giving the same answer a dedicated 2D formula would."""
    waypoints = to_xyz(waypoints)
    n = len(waypoints)
    if n < 3:
        return np.zeros(n)
    velocity = np.gradient(waypoints, axis=0)
    accel = np.gradient(velocity, axis=0)
    cross_norm = np.linalg.norm(np.cross(velocity, accel), axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.where(speed > 1e-9, cross_norm / np.power(speed, 3), 0.0)
    return np.nan_to_num(curvature, nan=0.0, posinf=0.0)


def velocity_profile(
    waypoints: np.ndarray, v_max: float, a_max: float, a_lat_max: float,
    v_start: float = 0.0, v_end: float = 0.0,
) -> dict:
    """Forward-backward velocity planning over uniformly arc-length-spaced waypoints: first
    caps speed at each sample by lateral-acceleration-vs-curvature (v <= sqrt(a_lat_max /
    kappa), so the profile slows for tight turns instead of assuming the robot corners at
    v_max), then a forward pass enforces the acceleration limit and a backward pass enforces
    the deceleration limit (braking for a tight turn a few meters ahead has to start now, not
    at the turn itself). This is the standard two-pass velocity-profile approximation used by
    most mobile-robot local planners - not a globally time-optimal solve, but a simple,
    robust, and correctly *feasible* one. Returns arrays aligned with `waypoints`: cumulative
    arc-length distance, velocity, and integrated time."""
    waypoints = np.asarray(waypoints, dtype=np.float64)
    n = len(waypoints)
    if n < 2:
        return {"distance_m": np.zeros(n), "velocity_mps": np.zeros(n), "time_s": np.zeros(n)}

    seg_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    curvature = compute_curvature(waypoints)
    v = np.minimum(v_max, np.sqrt(a_lat_max / np.maximum(curvature, 1e-6)))
    v[0] = min(v[0], v_start)
    v[-1] = min(v[-1], v_end)

    for i in range(1, n):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a_max * seg_lengths[i - 1]))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_max * seg_lengths[i]))

    time_s = np.zeros(n)
    for i in range(1, n):
        avg_v = max(0.5 * (v[i - 1] + v[i]), 1e-3)
        time_s[i] = time_s[i - 1] + seg_lengths[i - 1] / avg_v

    return {"distance_m": arc, "velocity_mps": v, "time_s": time_s}


def build_trajectory(
    waypoints: np.ndarray, is_segment_free, *, shortcut_iters: int = 3, chaikin_iters: int = 3,
    resample_ds: float = 0.2, v_max: float = 1.0, a_max: float = 0.5, a_lat_max: float = 1.0,
) -> dict:
    """Full pipeline from a raw grid-cell A* path to an executable trajectory: shortcut ->
    smooth -> resample -> velocity-profile. Shared by the 2D elevation-map planner and the 3D
    octomap planner; only `is_segment_free` differs between them."""
    shortcut = shortcut_path(waypoints, is_segment_free, max_iters=shortcut_iters)
    smoothed = chaikin_smooth(shortcut, iterations=chaikin_iters)
    resampled = resample_by_arclength(smoothed, ds=resample_ds)
    profile = velocity_profile(resampled, v_max=v_max, a_max=a_max, a_lat_max=a_lat_max)
    return {
        "shortcut": shortcut,
        "positions": resampled,
        "distance_m": profile["distance_m"],
        "velocity_mps": profile["velocity_mps"],
        "time_s": profile["time_s"],
    }
