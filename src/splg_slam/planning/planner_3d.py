import heapq

import numpy as np
import octomap

# 26-connected 3D neighborhood (all offsets except (0,0,0)).
NEIGHBORS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


class OccupancyChecker:
    """Wraps OcTree.search()/isNodeOccupied() into a clean 3-state query, plus (once
    dynamicEDT_generate() has been called) a Euclidean-distance-to-nearest-obstacle query
    used for safety-margin inflation. This binding doesn't return None for never-observed
    space - search() still hands back a node wrapper, and isNodeOccupied() on it raises
    octomap.NullPointerException, which is the actual signal for "unknown" (verified
    empirically: a point 1000m outside the map raises this, a genuinely free voxel returns
    False, an occupied one returns True)."""

    def __init__(self, tree: octomap.OcTree):
        self.tree = tree

    def state(self, xyz: np.ndarray) -> str:
        node = self.tree.search(np.asarray(xyz, dtype=np.float64))
        try:
            return "occupied" if self.tree.isNodeOccupied(node) else "free"
        except octomap.NullPointerException:
            return "unknown"

    def clearance(self, xyz: np.ndarray) -> float:
        """Distance (m) to the nearest obstacle, from the dynamicEDT computed over a bounded
        region around the plan (see build_edt below) - NOT valid outside that region."""
        return float(self.tree.dynamicEDT_getDistance(np.asarray(xyz, dtype=np.float64)))


def build_edt(tree: octomap.OcTree, bbx_min: np.ndarray, bbx_max: np.ndarray, max_dist: float, treat_unknown_as_occupied: bool) -> None:
    """Computes the Euclidean distance transform (DynamicEDTOctomap, wrapped by the octomap
    python bindings) only over a padded box around the planned region, not the whole octree.
    A real outdoor octomap can span hundreds of meters at a few-centimeter resolution -
    running the EDT over the full extent is billions of voxels and does not finish in any
    reasonable time (verified: killed after minutes on a ~580x520x11m map). Bounding it to
    just the region a given start/goal pair could plausibly need keeps this a sub-second call."""
    tree.dynamicEDT_generate(max_dist, bbx_min.astype(np.float64), bbx_max.astype(np.float64), treat_unknown_as_occupied)


def make_is_blocked(checker: OccupancyChecker, robot_radius_m: float, allow_unknown: bool):
    def is_blocked(xyz: np.ndarray) -> bool:
        state = checker.state(xyz)
        if state == "occupied":
            return True
        if state == "unknown":
            return not allow_unknown
        return checker.clearance(xyz) < robot_radius_m

    return is_blocked


def make_segment_free_check(checker: OccupancyChecker, resolution: float, robot_radius_m: float, allow_unknown: bool):
    is_blocked = make_is_blocked(checker, robot_radius_m, allow_unknown)

    def is_segment_free(p0: np.ndarray, p1: np.ndarray) -> bool:
        dist = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
        n_samples = max(2, int(np.ceil(dist / (0.5 * resolution))) + 1)
        for t in np.linspace(0.0, 1.0, n_samples):
            p = p0 + t * (p1 - p0)
            if is_blocked(p):
                return False
        return True

    return is_segment_free


def astar_3d(
    checker: OccupancyChecker, resolution: float, start_idx: tuple[int, int, int], goal_idx: tuple[int, int, int],
    origin: np.ndarray, allow_unknown: bool, robot_radius_m: float, inflate_radius_m: float, cost_weight: float,
    max_expansions: int | None = None,
):
    """3D A* over a 26-connected voxel grid. With `cost_weight=0` (the default) this is pure
    shortest-Euclidean-distance search - a drone flying through open 3D space doesn't care
    about "slope" or "roughness", only what's physically in the way, and the heuristic
    (straight-line distance) exactly matches the edge-cost metric, so the result is
    provably the true shortest path in the safety-inflated free space. Setting `cost_weight`
    > 0 additionally softly discourages hugging the inflation boundary (ROS-costmap-style),
    trading that optimality guarantee for extra clearance preference when a flatter
    alternative exists.

    `max_expansions` bounds how many nodes get popped before giving up - without it, a
    genuinely disconnected start/goal pair (e.g. across an unmapped gap) makes the search
    exhaust the *entire* reachable free/unknown component before reporting failure, which for
    an online replanning loop called every few seconds is a real latency problem, not just an
    edge case."""

    def idx_to_world(idx):
        return origin + np.array(idx) * resolution

    def heuristic(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    is_blocked = make_is_blocked(checker, robot_radius_m, allow_unknown)

    def soft_cost(idx):
        if inflate_radius_m <= 0:
            return 0.0
        d = checker.clearance(idx_to_world(idx))
        return float(np.clip(1.0 - (d - robot_radius_m) / inflate_radius_m, 0.0, 1.0))

    if is_blocked(idx_to_world(start_idx)) or is_blocked(idx_to_world(goal_idx)):
        return None, float("inf")

    open_heap = [(heuristic(start_idx, goal_idx), 0.0, start_idx)]
    came_from = {}
    g_score = {start_idx: 0.0}
    visited = set()
    blocked_cache = {}
    cost_cache = {}

    def blocked_cached(idx):
        if idx not in blocked_cache:
            blocked_cache[idx] = is_blocked(idx_to_world(idx))
        return blocked_cache[idx]

    def cost_cached(idx):
        if idx not in cost_cache:
            cost_cache[idx] = soft_cost(idx)
        return cost_cache[idx]

    expansions = 0
    while open_heap:
        if max_expansions is not None and expansions >= max_expansions:
            return None, float("inf")
        _, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        expansions += 1
        if current == goal_idx:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g

        cx, cy, cz = current
        for dx, dy, dz in NEIGHBORS_26:
            neighbor = (cx + dx, cy + dy, cz + dz)
            if neighbor in visited or blocked_cached(neighbor):
                continue
            step_dist = float(np.linalg.norm([dx, dy, dz]))
            avg_cost = 0.5 * (cost_cached(current) + cost_cached(neighbor))
            edge_cost = step_dist * (1.0 + cost_weight * avg_cost)
            tentative_g = g + edge_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbor, goal_idx), tentative_g, neighbor))

    return None, float("inf")


def plan_3d(
    tree: octomap.OcTree, start: np.ndarray, goal: np.ndarray, *, robot_radius_m: float = 0.3,
    inflate_radius_m: float = 0.0, cost_weight: float = 0.0, allow_unknown: bool = False,
    edt_padding_m: float | None = None, max_expansions: int | None = 400_000,
):
    """End-to-end single-call planner: bounds+builds the EDT around start/goal, runs astar_3d,
    and returns (path_world Nx3 array or None, path_idx list or None). Used by both the CLI
    (plan_path_3d.py) and the online replanning loop (online_plan_loop.py) so a single call
    handles the whole "given a live octree and two points, find a path" job."""
    checker = OccupancyChecker(tree)
    resolution = tree.getResolution()
    tree_min, tree_max = np.array(tree.getMetricMin()), np.array(tree.getMetricMax())

    for name, p in [("start", start), ("goal", goal)]:
        state = checker.state(p)
        if state != "free":
            return None, None, f"{name} is {state}, not free space"

    padding = edt_padding_m if edt_padding_m is not None else max(10.0, 1.5 * float(np.linalg.norm(goal - start)))
    bbx_min = np.maximum(np.minimum(start, goal) - padding, tree_min)
    bbx_max = np.minimum(np.maximum(start, goal) + padding, tree_max)
    edt_max_dist = max(0.5, robot_radius_m + inflate_radius_m + resolution)
    build_edt(tree, bbx_min, bbx_max, edt_max_dist, treat_unknown_as_occupied=not allow_unknown)

    if checker.clearance(start) < robot_radius_m:
        return None, None, f"start is only {checker.clearance(start):.2f}m from an obstacle, less than robot_radius_m={robot_radius_m}"
    if checker.clearance(goal) < robot_radius_m:
        return None, None, f"goal is only {checker.clearance(goal):.2f}m from an obstacle, less than robot_radius_m={robot_radius_m}"

    origin = tree_min

    def world_to_idx(p):
        return tuple(int(round(v)) for v in (np.array(p) - origin) / resolution)

    path_idx, _ = astar_3d(
        checker, resolution, world_to_idx(start), world_to_idx(goal), origin, allow_unknown,
        robot_radius_m, inflate_radius_m, cost_weight, max_expansions=max_expansions,
    )
    if path_idx is None:
        return None, None, "no path found"

    path_world = np.array([origin + np.array(idx) * resolution for idx in path_idx])
    return path_world, path_idx, None
