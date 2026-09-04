import heapq

import numpy as np

from splg_slam.planning.dense_grid_3d import NEIGHBORS_26, build_cost_and_blocked_3d, grid_to_world_3d, world_to_grid_3d

INF = float("inf")


class DijkstraField3D:
    """3D sibling of dijkstra_field_2d.DijkstraField2D - see its docstring for why this
    replaces DStarLite3D for this codebase's fixed-map/fixed-goal/moving-start access
    pattern: one full Dijkstra sweep from the goal, computed once, then every future query
    is a pure greedy descent along already-known distances with no search and no queue to
    maintain (so no equivalent of the D* Lite spikes diagnosed in
    localize_plan_loop_3d.py)."""

    def __init__(self, cost_grid: np.ndarray, blocked: np.ndarray, goal: tuple[int, int, int], cost_weight: float):
        self.cost_grid = cost_grid
        self.blocked = blocked
        self.goal = goal
        self.cost_weight = cost_weight
        self.nx, self.ny, self.nz = blocked.shape
        self.g = self._compute_field()

    def _neighbors(self, s: tuple[int, int, int]):
        x, y, z = s
        for dx, dy, dz in NEIGHBORS_26:
            nx_, ny_, nz_ = x + dx, y + dy, z + dz
            if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny and 0 <= nz_ < self.nz:
                yield (nx_, ny_, nz_)

    def _edge_cost(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        if self.blocked[a] or self.blocked[b]:
            return INF
        step = float(np.linalg.norm(np.subtract(a, b)))
        avg_cost = 0.5 * (float(self.cost_grid[a]) + float(self.cost_grid[b]))
        return step * (1.0 + self.cost_weight * avg_cost)

    def _compute_field(self) -> np.ndarray:
        g = np.full(self.blocked.shape, INF, dtype=np.float64)
        g[self.goal] = 0.0
        heap: list[tuple[float, tuple[int, int, int]]] = [(0.0, self.goal)]
        visited: set[tuple[int, int, int]] = set()
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
                if nd < g[v]:
                    g[v] = nd
                    heapq.heappush(heap, (nd, v))
        return g

    def plan(self, start: tuple[int, int, int]) -> list[tuple[int, int, int]] | None:
        if self.blocked[start] or self.g[start] == INF:
            return None
        path = [start]
        current = start
        guard = 0
        max_guard = self.nx * self.ny * self.nz
        while current != self.goal:
            best_next, best_val = None, INF
            for v in self._neighbors(current):
                c = self._edge_cost(current, v)
                if c < INF:
                    val = c + self.g[v]
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


class DijkstraField3DPlanner:
    """Drop-in alternative to DStarLite3DPlanner, matching plan_3d_dense's calling
    convention: construct once per (grid, goal) pair, call `.plan(pos)` on every replan."""

    def __init__(self, grid: dict, goal_xy: np.ndarray, cost_weight: float = 0.0, **mask_kwargs):
        self.resolution, self.origin = grid["resolution"], grid["origin"]
        self.blocked, self.soft_cost = build_cost_and_blocked_3d(grid, **mask_kwargs)
        self.goal_idx = world_to_grid_3d(goal_xy, self.origin, self.resolution)
        self.field = DijkstraField3D(self.soft_cost, self.blocked, self.goal_idx, cost_weight)

    def plan(self, pos: np.ndarray, max_expansions: int | None = None):
        start_idx = world_to_grid_3d(pos, self.origin, self.resolution)
        nx, ny, nz = self.blocked.shape
        if not (0 <= start_idx[0] < nx and 0 <= start_idx[1] < ny and 0 <= start_idx[2] < nz):
            return None, None, self.blocked, self.origin, "start is outside the map grid"
        if self.blocked[start_idx]:
            return None, None, self.blocked, self.origin, "start cell is blocked (occupied, unobserved, or within robot_radius_m of an obstacle)"
        path_idx = self.field.plan(start_idx)
        if path_idx is None:
            return None, None, self.blocked, self.origin, "no path found"
        path_world = np.array([grid_to_world_3d(idx, self.origin, self.resolution) for idx in path_idx])
        return path_world, path_idx, self.blocked, self.origin, None
