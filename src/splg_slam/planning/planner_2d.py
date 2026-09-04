import heapq

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt

NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def world_to_grid(x: float, y: float, x_min: float, y_min: float, resolution: float) -> tuple[int, int]:
    return int(round((x - x_min) / resolution)), int(round((y - y_min) / resolution))


def grid_to_world(ix: int, iy: int, x_min: float, y_min: float, resolution: float) -> tuple[float, float]:
    return x_min + ix * resolution, y_min + iy * resolution


def close_free_space_gaps(blocked: np.ndarray, resolution: float, close_radius_m: float) -> np.ndarray:
    """Morphological closing of the FREE mask (not the blocked mask - closing `blocked`
    would instead eat into free area at concavities, the wrong direction): dilates free space
    to bridge small gaps, then erodes back, so a thin seam of falsely-blocked cells (dense-
    stereo noise, a single missed frame, a parked-car shadow) spanning less than
    `close_radius_m` no longer splits an otherwise-contiguous road into disconnected islands.
    Verified on the KITTI 00 elevation map: 6051 disconnected traversable components with the
    largest covering 90.3% of free space collapse to 340 components at 0.9m closing radius,
    largest now 97.7% - most of the remaining fragments are genuinely isolated (a patch behind
    an obstacle, never reachable from the road), not noise. Only WIDENS free space (a cell
    that was free stays free), so this never turns a real, sizable unexplored/blocked area
    passable the way `allow_unknown` does - only gaps narrower than the radius get bridged."""
    close_cells = max(1, int(round(close_radius_m / resolution)))
    structure = np.ones((2 * close_cells + 1, 2 * close_cells + 1))
    # border_value=1 (not scipy's default 0): binary_closing's final erosion step otherwise
    # treats space just outside the array as blocked, artificially eroding genuinely free
    # cells near the grid's edge - violates closing's "only ever widens free space" guarantee.
    # Not visible in the KITTI validation (a large map, edge band negligible) but a real bug.
    free_closed = binary_closing(~blocked, structure=structure, border_value=1)
    return ~free_closed


def inflate_obstacles(
    blocked: np.ndarray, cost: np.ndarray, resolution: float, robot_radius_m: float, inflate_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """ROS-costmap-style obstacle inflation: `distance_transform_edt` gives every cell its
    distance (in cells) to the nearest cell that was ALREADY blocked (steep/rough/stepped/
    unobserved), before any inflation. A cell within `robot_radius_m` of one of those becomes
    hard-blocked too (the robot's own footprint can't fit there even if the cell itself looks
    flat) - lethal, not just costly. Between `robot_radius_m` and `robot_radius_m +
    inflate_radius_m`, cost ramps linearly from 1 down to 0 and is blended into the existing
    terrain cost via a max (never lowers a cell's cost, only raises it) - a soft push to keep
    extra clearance from obstacles when a flatter-but-tighter alternative exists, without
    forbidding it outright the way a hard block would."""
    dist_m = distance_transform_edt(~blocked) * resolution
    if inflate_radius_m > 0:
        ramp = np.clip(1.0 - (dist_m - robot_radius_m) / inflate_radius_m, 0.0, 1.0)
        cost = np.clip(np.maximum(cost, ramp), 0.0, 1.0)
    blocked = blocked | (dist_m < robot_radius_m)
    return blocked, cost


def make_segment_free_check(blocked: np.ndarray, x_min: float, y_min: float, resolution: float):
    ny, nx = blocked.shape

    def is_segment_free(p0: np.ndarray, p1: np.ndarray) -> bool:
        dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        n_samples = max(2, int(np.ceil(dist / (0.5 * resolution))) + 1)
        for t in np.linspace(0.0, 1.0, n_samples):
            x, y = p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])
            gx, gy = world_to_grid(x, y, x_min, y_min, resolution)
            if not (0 <= gx < nx and 0 <= gy < ny) or blocked[gy, gx]:
                return False
        return True

    return is_segment_free


def sample_height_at(xy_points: np.ndarray, height: np.ndarray, x_min: float, y_min: float, resolution: float) -> np.ndarray:
    ny, nx = height.shape
    z = np.zeros(len(xy_points))
    for i, (x, y) in enumerate(xy_points):
        gx, gy = world_to_grid(x, y, x_min, y_min, resolution)
        gx, gy = int(np.clip(gx, 0, nx - 1)), int(np.clip(gy, 0, ny - 1))
        h = height[gy, gx]
        z[i] = 0.0 if np.isnan(h) else h
    return z


def astar(
    cost_grid: np.ndarray, blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int], cost_weight: float,
    max_expansions: int | None = None,
):
    """Grid A* over an 8-connected neighborhood. cost_grid[y, x] in [0, 1] (traversability
    cost from build_elevation_map.py); blocked[y, x] True = impassable (not traversable, or
    unobserved if the caller chose not to allow that). Edge cost blends step distance with the
    average of the two endpoints' terrain cost, so the planner prefers a longer flat/smooth
    route over a shorter one through rough/steep cells, not just shortest-path-by-distance.

    `max_expansions` bounds how many nodes get popped before giving up, so a repeatedly-called
    online replanning loop fails fast on a genuinely disconnected goal instead of exhausting
    the whole known-free region on every single call."""
    ny, nx = cost_grid.shape

    def heuristic(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    if blocked[start[1], start[0]] or blocked[goal[1], goal[0]]:
        return None, float("inf")

    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from: dict = {}
    g_score = {start: 0.0}
    visited = set()

    expansions = 0
    while open_heap:
        if max_expansions is not None and expansions >= max_expansions:
            return None, float("inf")
        _, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        expansions += 1
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g

        cx, cy = current
        for dx, dy in NEIGHBORS_8:
            nx_, ny_ = cx + dx, cy + dy
            if not (0 <= nx_ < nx and 0 <= ny_ < ny) or blocked[ny_, nx_]:
                continue
            step_dist = float(np.hypot(dx, dy))
            avg_cost = 0.5 * (float(cost_grid[cy, cx]) + float(cost_grid[ny_, nx_]))
            edge_cost = step_dist * (1.0 + cost_weight * avg_cost)
            tentative_g = g + edge_cost
            neighbor = (nx_, ny_)
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbor, goal), tentative_g, neighbor))

    return None, float("inf")


def build_cost_and_blocked(
    elevation: dict, *, allow_unknown: bool = False, use_strict_traversable: bool = False,
    block_cost_threshold: float = 0.85, robot_radius_m: float = 0.3, inflate_radius_m: float = 0.5,
    close_radius_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the final (blocked, cost) grids a planning call will search over - factored
    out of plan_2d so a caller building its own segment-free check (for shortcutting a path
    plan_2d already found) uses the *exact same* mask the search did, not a re-derived one
    that forgot inflation or gap-closing and could pass a shortcut through a cell A* would
    never have allowed."""
    height, cost, traversable = elevation["height"], elevation["cost"], elevation["traversable"]
    resolution = elevation["resolution"]
    observed = ~np.isnan(height)

    if use_strict_traversable:
        blocked = ~traversable
    else:
        blocked = np.nan_to_num(cost, nan=1.0) > block_cost_threshold
    if not allow_unknown:
        blocked = blocked | ~observed

    if close_radius_m > 0:
        blocked = close_free_space_gaps(blocked, resolution, close_radius_m)

    return inflate_obstacles(blocked, cost, resolution, robot_radius_m, inflate_radius_m)


def plan_2d(
    elevation: dict, start_xy: np.ndarray, goal_xy: np.ndarray, *, cost_weight: float = 5.0,
    allow_unknown: bool = False, use_strict_traversable: bool = False, block_cost_threshold: float = 0.85,
    robot_radius_m: float = 0.3, inflate_radius_m: float = 0.5, close_radius_m: float = 0.0,
    max_expansions: int | None = 400_000,
):
    """End-to-end single-call planner over a pre-built elevation map dict (height/cost/
    traversable/x_min/y_min/resolution, as saved by build_elevation_map.py): inflates
    obstacles, runs 2D A*, and returns (path_world Nx2 array or None, path_grid list or None,
    blocked mask actually searched over, error string or None). Returning `blocked` lets a
    caller build a segment-free check for shortcutting that matches exactly what the search
    saw - important once plan_2d_with_fallback may have relaxed settings the caller didn't
    pass in directly. Mirrors planner_3d.plan_3d's role for the octomap planner - a single
    call handles "given a static map and two points, find a path", shared by the CLI
    (plan_path.py) and any online replanning loop."""
    x_min, y_min, resolution = elevation["x_min"], elevation["y_min"], elevation["resolution"]
    blocked, cost = build_cost_and_blocked(
        elevation, allow_unknown=allow_unknown, use_strict_traversable=use_strict_traversable,
        block_cost_threshold=block_cost_threshold, robot_radius_m=robot_radius_m,
        inflate_radius_m=inflate_radius_m, close_radius_m=close_radius_m,
    )
    height = elevation["height"]

    start_grid = world_to_grid(start_xy[0], start_xy[1], x_min, y_min, resolution)
    goal_grid = world_to_grid(goal_xy[0], goal_xy[1], x_min, y_min, resolution)

    ny, nx = height.shape
    for name, (gx, gy) in [("start", start_grid), ("goal", goal_grid)]:
        if not (0 <= gx < nx and 0 <= gy < ny):
            return None, None, blocked, f"{name} {(gx, gy)} is outside the map grid ({nx}x{ny})"
        if blocked[gy, gx]:
            return None, None, blocked, f"{name} cell is blocked (not traversable, unobserved, or within robot_radius_m of an obstacle)"

    path_grid, _ = astar(cost, blocked, start_grid, goal_grid, cost_weight, max_expansions=max_expansions)
    if path_grid is None:
        return None, None, blocked, "no path found"

    path_world = np.array([grid_to_world(ix, iy, x_min, y_min, resolution) for ix, iy in path_grid])
    return path_world, path_grid, blocked, None


DEFAULT_FALLBACK_STAGES = [
    {"close_radius_m": 1.8},
    {"close_radius_m": 1.8, "block_cost_threshold": 0.95},
    {"close_radius_m": 1.8, "block_cost_threshold": 0.95, "robot_radius_m": 0.5, "inflate_radius_m": 0.75},
    {"close_radius_m": 1.8, "block_cost_threshold": 0.95, "robot_radius_m": 0.5, "inflate_radius_m": 0.75, "allow_unknown": True},
]


def plan_2d_with_fallback(
    elevation: dict, start_xy: np.ndarray, goal_xy: np.ndarray, *,
    fallback_stages: list[dict] | None = None, **base_kwargs,
):
    """Tries `plan_2d` at the caller's normal (safest) settings first, then - only if that
    finds no path - progressively relaxes the settings through `fallback_stages` (each a dict
    of plan_2d kwarg overrides, applied cumulatively on top of `base_kwargs`) until one
    succeeds or all stages are exhausted. The dense-stereo elevation maps this planner runs
    on are noisy enough that a real, driveable route can come up 1-2m short of connecting at
    strict settings (this codebase's own KITTI 00 map: 6051 disconnected traversable
    fragments before any relaxation) - reporting "no path" outright when a cheap, still-safe
    relaxation would have found one is a worse failure mode than momentarily being more
    permissive. `DEFAULT_FALLBACK_STAGES` escalates: wider gap-closing -> looser cost
    threshold -> smaller safety margins -> (last resort) allow planning through
    never-observed cells. Returns (path_world, path_grid, blocked, error, stage) - `stage` is
    0 for the unrelaxed attempt, or the 1-based index into the stage list that succeeded, so
    the caller can log/flag whenever a fallback stage was actually needed instead of silently
    masking it."""
    stages = fallback_stages if fallback_stages is not None else DEFAULT_FALLBACK_STAGES

    path_world, path_grid, blocked, err = plan_2d(elevation, start_xy, goal_xy, **base_kwargs)
    if path_world is not None:
        return path_world, path_grid, blocked, None, 0

    for stage_idx, overrides in enumerate(stages, start=1):
        kwargs = {**base_kwargs, **overrides}
        path_world, path_grid, blocked, err = plan_2d(elevation, start_xy, goal_xy, **kwargs)
        if path_world is not None:
            return path_world, path_grid, blocked, None, stage_idx

    return None, None, blocked, err, len(stages)
