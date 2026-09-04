import heapq

import numpy as np
import octomap
from scipy.ndimage import binary_closing, distance_transform_edt

NEIGHBORS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def extract_dense_grid(
    tree: octomap.OcTree, resolution: float, bbx_min: np.ndarray | None = None, bbx_max: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Queries the sparse octree ONCE, over the whole tree (or a given fixed box), and
    returns dense boolean `occupied`/`unknown` arrays (indexed [x, y, z], `unknown` used only
    when a caller later chooses NOT to allow planning through it - kept separate from
    `occupied` rather than pre-merged into one `blocked` array, so that choice stays a
    plan-time parameter, same as build_elevation_map.py stores height and lets plan_2d decide
    `allow_unknown` from it rather than baking the decision in at build time) plus the grid's
    world-space origin.

    This mirrors build_elevation_map.py's role exactly: a one-time, offline conversion from
    the map's native representation into a fixed dense grid that the planner then loads and
    reuses across every replan - NOT re-extracted per query. Re-extracting a fresh local box
    around a moving start (an earlier version of this module did this) defeats D* Lite's
    entire incremental benefit, since a new box means a new grid means a new persistent
    search built from scratch on every single replan. Only practical for a room/building-
    scale map (a KITTI-scale outdoor octomap would need a bounded `bbx_min`/`bbx_max`, same
    scale trade-off build_elevation_map.py's resolution choice makes for a 2D grid)."""
    bbx_min = bbx_min if bbx_min is not None else np.array(tree.getMetricMin())
    bbx_max = bbx_max if bbx_max is not None else np.array(tree.getMetricMax())
    shape = tuple(int(round(v)) for v in (bbx_max - bbx_min) / resolution)
    occupied = np.zeros(shape, dtype=bool)
    unknown = np.zeros(shape, dtype=bool)
    for ix in range(shape[0]):
        x = bbx_min[0] + ix * resolution
        for iy in range(shape[1]):
            y = bbx_min[1] + iy * resolution
            for iz in range(shape[2]):
                z = bbx_min[2] + iz * resolution
                node = tree.search(np.array([x, y, z], dtype=np.float64))
                try:
                    occupied[ix, iy, iz] = tree.isNodeOccupied(node)
                except octomap.NullPointerException:
                    unknown[ix, iy, iz] = True
    return occupied, unknown, bbx_min.copy()


def close_gaps_3d(blocked: np.ndarray, resolution: float, close_radius_m: float) -> np.ndarray:
    """3D analog of planner_2d.close_free_space_gaps: closes the FREE mask (dilate then
    erode), bridging thin noise-driven occupied/unknown seams narrower than
    `close_radius_m` that would otherwise fragment an actually-contiguous free volume into
    disconnected islands - the same dense-stereo-noise problem the 2D elevation-map planner
    had, just in 3D."""
    if close_radius_m <= 0:
        return blocked
    close_cells = max(1, int(round(close_radius_m / resolution)))
    # Spherical structuring element, not a cube - a cube would bridge gaps up to
    # close_radius_m*sqrt(3) along diagonals, silently more permissive than the stated radius.
    r = close_cells
    zz, yy, xx = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
    structure = (xx ** 2 + yy ** 2 + zz ** 2) <= r ** 2
    # border_value=1 (not scipy's default 0): binary_closing's final erosion step otherwise
    # treats space just outside the array as blocked, artificially eroding genuinely free
    # cells near the grid's edge - violates closing's "only ever widens free space" guarantee.
    free_closed = binary_closing(~blocked, structure=structure, border_value=1)
    return ~free_closed


def inflate_obstacles_3d(
    blocked: np.ndarray, resolution: float, robot_radius_m: float, inflate_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """3D analog of planner_2d.inflate_obstacles: distance_transform_edt gives every free
    voxel its distance to the nearest already-blocked voxel; within robot_radius_m that
    becomes hard-blocked (the robot's own body can't fit there), and between robot_radius_m
    and robot_radius_m+inflate_radius_m a soft cost ramps 1->0, nudging the planner toward
    extra clearance without forbidding a tight-but-passable gap outright."""
    dist_m = distance_transform_edt(~blocked) * resolution
    soft_cost = np.zeros_like(dist_m)
    if inflate_radius_m > 0:
        soft_cost = np.clip(1.0 - (dist_m - robot_radius_m) / inflate_radius_m, 0.0, 1.0)
    blocked = blocked | (dist_m < robot_radius_m)
    return blocked, soft_cost


def build_cost_and_blocked_3d(
    grid: dict, *, allow_unknown: bool = False, close_radius_m: float = 0.0, robot_radius_m: float = 0.2,
    inflate_radius_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies closing + safety inflation to a pre-extracted dense grid dict (occupied/
    resolution, as saved by build_octomap_grid.py). Mirrors planner_2d.build_cost_and_blocked
    exactly - both take a fixed, already-built map and a set of mask-shaping parameters, and
    both get called once per plan_3d_dense() the same way plan_2d() calls the 2D version."""
    occupied, unknown, resolution = grid["occupied"], grid["unknown"], grid["resolution"]
    blocked = occupied | (unknown if not allow_unknown else np.zeros_like(occupied))
    if close_radius_m > 0:
        blocked = close_gaps_3d(blocked, resolution, close_radius_m)
    return inflate_obstacles_3d(blocked, resolution, robot_radius_m, inflate_radius_m)


def world_to_grid_3d(p: np.ndarray, origin: np.ndarray, resolution: float) -> tuple[int, int, int]:
    idx = np.round((np.asarray(p) - origin) / resolution).astype(int)
    return tuple(idx.tolist())


def grid_to_world_3d(idx: tuple[int, int, int], origin: np.ndarray, resolution: float) -> np.ndarray:
    return origin + np.array(idx) * resolution


def astar_3d_dense(
    soft_cost: np.ndarray, blocked: np.ndarray, start: tuple[int, int, int], goal: tuple[int, int, int],
    cost_weight: float, max_expansions: int | None = None,
):
    """26-connected A* over a dense 3D grid - the octomap-planner analog of
    planner_2d.astar, operating on the fixed grid build_octomap_grid.py produced (not the
    octree - by the time this runs, the octree is out of the picture entirely, same as
    plan_2d never touches the original point cloud)."""
    nx, ny, nz = blocked.shape

    def in_bounds(s):
        return 0 <= s[0] < nx and 0 <= s[1] < ny and 0 <= s[2] < nz

    def heuristic(a, b):
        return float(np.linalg.norm(np.subtract(a, b)))

    if blocked[start] or blocked[goal]:
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

        for dx, dy, dz in NEIGHBORS_26:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if not in_bounds(neighbor) or neighbor in visited or blocked[neighbor]:
                continue
            step_dist = float(np.linalg.norm([dx, dy, dz]))
            avg_cost = 0.5 * (float(soft_cost[current]) + float(soft_cost[neighbor]))
            edge_cost = step_dist * (1.0 + cost_weight * avg_cost)
            tentative_g = g + edge_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbor, goal), tentative_g, neighbor))

    return None, float("inf")


def plan_3d_dense(
    grid: dict, start: np.ndarray, goal: np.ndarray, *, cost_weight: float = 0.0, allow_unknown: bool = False,
    close_radius_m: float = 0.0, robot_radius_m: float = 0.2, inflate_radius_m: float = 0.0,
    max_expansions: int | None = 400_000,
):
    """End-to-end single-call planner over a pre-built dense grid dict (as saved by
    build_octomap_grid.py): applies closing/inflation, runs 3D A*, and returns (path_world
    Nx3 or None, path_grid or None, blocked mask, origin, error or None). Mirrors
    planner_2d.plan_2d's signature and role - the fixed map is loaded once by the caller and
    passed in here every call, same as `elevation` is for the 2D planner."""
    resolution, origin = grid["resolution"], grid["origin"]
    blocked, soft_cost = build_cost_and_blocked_3d(
        grid, allow_unknown=allow_unknown, close_radius_m=close_radius_m, robot_radius_m=robot_radius_m,
        inflate_radius_m=inflate_radius_m,
    )
    start_idx = world_to_grid_3d(start, origin, resolution)
    goal_idx = world_to_grid_3d(goal, origin, resolution)
    nx, ny, nz = blocked.shape
    for name, idx in [("start", start_idx), ("goal", goal_idx)]:
        if not (0 <= idx[0] < nx and 0 <= idx[1] < ny and 0 <= idx[2] < nz):
            return None, None, blocked, origin, f"{name} {idx} is outside the map grid"
        if blocked[idx]:
            return None, None, blocked, origin, f"{name} cell is blocked (occupied, unobserved, or within robot_radius_m of an obstacle)"

    path_idx, _ = astar_3d_dense(soft_cost, blocked, start_idx, goal_idx, cost_weight, max_expansions=max_expansions)
    if path_idx is None:
        return None, None, blocked, origin, "no path found"

    path_world = np.array([grid_to_world_3d(idx, origin, resolution) for idx in path_idx])
    return path_world, path_idx, blocked, origin, None


def make_segment_free_check_3d(blocked: np.ndarray, origin: np.ndarray, resolution: float):
    """3D analog of planner_2d.make_segment_free_check, for build_trajectory's shortcutting
    step - samples along a straight-line segment and checks each sample against the same
    `blocked` grid the search used, so a shortcut can't cut through a cell A*/D* Lite would
    never have allowed."""
    nx, ny, nz = blocked.shape

    def is_segment_free(p0: np.ndarray, p1: np.ndarray) -> bool:
        dist = float(np.linalg.norm(np.subtract(p1, p0)))
        n_samples = max(2, int(np.ceil(dist / (0.5 * resolution))) + 1)
        for t in np.linspace(0.0, 1.0, n_samples):
            p = p0 + t * (p1 - p0)
            idx = world_to_grid_3d(p, origin, resolution)
            if not (0 <= idx[0] < nx and 0 <= idx[1] < ny and 0 <= idx[2] < nz) or blocked[idx]:
                return False
        return True

    return is_segment_free
