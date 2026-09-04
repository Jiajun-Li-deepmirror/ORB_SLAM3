import heapq

import numpy as np

from splg_slam.planning.planner_2d import build_cost_and_blocked, grid_to_world, world_to_grid

NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
INF = float("inf")


class DijkstraField2D:
    """The non-incremental alternative to DStarLite2D for this codebase's actual access
    pattern (fixed map, fixed goal, only the start moves every replan): computes ONE full
    Dijkstra distance-to-goal field over the whole reachable free space, once, up front.
    D* Lite's entire complexity (the km trick, lazy re-keying, an ever-growing pending-node
    queue that has to be repaired on every start move - see the SPIKE diagnostics in
    localize_plan_loop.py) exists to cheaply repair a search when the START moves or EDGES
    change, without recomputing the whole thing. Here the start is the only thing that ever
    moves and the map/goal never change within a run, so that machinery buys nothing: once
    this field exists, every future query is a pure O(path length) greedy descent along
    already-known distances - no search, no queue, no spikes. The trade-off is the other
    direction: if the GOAL moved or the map changed, this whole field must be recomputed
    from scratch (D* Lite could repair it incrementally) - the right choice specifically
    because this codebase's goal is fixed for the life of one planner instance."""

    def __init__(self, cost_grid: np.ndarray, blocked: np.ndarray, goal: tuple[int, int], cost_weight: float):
        self.cost_grid = cost_grid
        self.blocked = blocked
        self.goal = goal
        self.cost_weight = cost_weight
        self.ny, self.nx = cost_grid.shape
        self.g = self._compute_field()

    def _neighbors(self, s: tuple[int, int]):
        x, y = s
        for dx, dy in NEIGHBORS_8:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny:
                yield (nx_, ny_)

    def _edge_cost(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        ax, ay = a
        bx, by = b
        if self.blocked[ay, ax] or self.blocked[by, bx]:
            return INF
        step = float(np.hypot(bx - ax, by - ay))
        avg_cost = 0.5 * (float(self.cost_grid[ay, ax]) + float(self.cost_grid[by, bx]))
        return step * (1.0 + self.cost_weight * avg_cost)

    def _compute_field(self) -> np.ndarray:
        g = np.full((self.ny, self.nx), INF, dtype=np.float64)
        g[self.goal[1], self.goal[0]] = 0.0
        heap: list[tuple[float, tuple[int, int]]] = [(0.0, self.goal)]
        visited: set[tuple[int, int]] = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            for v in self._neighbors(u):
                c = self._edge_cost(u, v)
                if c == INF:
                    continue
                nd = d + c
                if nd < g[v[1], v[0]]:
                    g[v[1], v[0]] = nd
                    heapq.heappush(heap, (nd, v))
        return g

    def plan(self, start: tuple[int, int]) -> list[tuple[int, int]] | None:
        if self.blocked[start[1], start[0]] or self.g[start[1], start[0]] == INF:
            return None
        path = [start]
        current = start
        guard = 0
        max_guard = self.nx * self.ny
        while current != self.goal:
            best_next, best_val = None, INF
            for v in self._neighbors(current):
                c = self._edge_cost(current, v)
                if c < INF:
                    val = c + self.g[v[1], v[0]]
                    if val < best_val:
                        best_val, best_next = val, v
            if best_next is None:
                return None
            current = best_next
            path.append(current)
            guard += 1
            if guard > max_guard:
                return None
        return path


class DijkstraField2DPlanner:
    """Drop-in alternative to DStarLiteFallbackPlanner (same fixed-mask design, no fallback
    ladder) matching plan_2d_with_fallback's calling convention: construct once per
    (elevation, goal) pair, call `.plan(pos_xy)` on every replan."""

    _MASK_KWARGS = (
        "allow_unknown", "use_strict_traversable", "block_cost_threshold", "robot_radius_m",
        "inflate_radius_m", "close_radius_m",
    )

    def __init__(self, elevation: dict, goal_xy: np.ndarray, **base_kwargs):
        self.elevation = elevation
        self.x_min, self.y_min, self.resolution = elevation["x_min"], elevation["y_min"], elevation["resolution"]
        self.goal_grid = world_to_grid(goal_xy[0], goal_xy[1], self.x_min, self.y_min, self.resolution)
        self.cost_weight = base_kwargs.get("cost_weight", 5.0)
        mask_kwargs = {k: base_kwargs[k] for k in self._MASK_KWARGS if k in base_kwargs}
        self.blocked, self.cost = build_cost_and_blocked(elevation, **mask_kwargs)
        self.field = DijkstraField2D(self.cost, self.blocked, self.goal_grid, self.cost_weight)

    def plan(self, pos_xy: np.ndarray, max_expansions: int | None = None):
        start_grid = world_to_grid(pos_xy[0], pos_xy[1], self.x_min, self.y_min, self.resolution)
        ny, nx = self.blocked.shape
        gx, gy = start_grid
        if not (0 <= gx < nx and 0 <= gy < ny):
            return None, None, self.blocked, "start is outside the map grid", 0
        if self.blocked[gy, gx]:
            return None, None, self.blocked, "start cell is blocked", 0
        path_grid = self.field.plan(start_grid)
        if path_grid is None:
            return None, None, self.blocked, "no path found", 0
        path_world = np.array([grid_to_world(ix, iy, self.x_min, self.y_min, self.resolution) for ix, iy in path_grid])
        return path_world, path_grid, self.blocked, None, 0
