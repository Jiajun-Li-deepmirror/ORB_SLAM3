import heapq

import numpy as np

from splg_slam.planning.dense_grid_3d import NEIGHBORS_26, build_cost_and_blocked_3d, grid_to_world_3d, world_to_grid_3d

INF = float("inf")


def _heuristic(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.subtract(a, b)))


class DStarLite3D:
    """3D sibling of dstar_lite_2d.DStarLite2D, over the dense grid produced by
    dense_grid_3d.build_dense_planning_grid: incremental replanner specialized for a fixed
    goal and a moving start against a static local grid - the octomap-planner analog of the
    same "goal fixed, replan every new localized position" loop the 2D elevation-map planner
    was built for. See DStarLite2D's docstring for the algorithm; this is the same
    Koenig & Likhachev (2002) construction with 26-connected 3D neighbors instead of
    8-connected 2D ones."""

    def __init__(
        self, cost_grid: np.ndarray, blocked: np.ndarray, goal: tuple[int, int, int], cost_weight: float,
        rebuild_interval: int | None = None,
    ):
        """`rebuild_interval`: see DStarLite2D's docstring - every this many `update_start`
        calls that actually move the start, discard the accumulated queue/km state and do
        one fresh full sweep from the current start instead of another incremental repair,
        bounding the otherwise ever-growing pending-node queue and the per-call re-keying
        tax that scales with it. None disables this (the original unbounded behavior)."""
        self.cost_grid = cost_grid
        self.blocked = blocked
        self.goal = goal
        self.cost_weight = cost_weight
        self.rebuild_interval = rebuild_interval
        self.nx, self.ny, self.nz = blocked.shape
        self._n_start_moves = 0
        self.last_rebuilt = False  # set by update_start, for external monitoring

        self._reset(goal)

    def _reset(self, start: tuple[int, int, int]) -> None:
        self.g = np.full(self.blocked.shape, INF, dtype=np.float64)
        self.rhs = np.full(self.blocked.shape, INF, dtype=np.float64)
        self.rhs[self.goal] = 0.0

        self.km = 0.0
        self.start = start
        self._heap: list[tuple[float, float, tuple[int, int, int]]] = []
        self._in_queue: dict[tuple[int, int, int], tuple[float, float]] = {}
        self._push(self.goal, self._calculate_key(self.goal))
        self.last_num_expansions = 0  # set by _compute_shortest_path, for external monitoring
        self.last_num_requeues = 0

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

    def _calculate_key(self, s: tuple[int, int, int]) -> tuple[float, float]:
        g_rhs = min(self.g[s], self.rhs[s])
        return (g_rhs + _heuristic(self.start, s) + self.km, g_rhs)

    def _push(self, s: tuple[int, int, int], key: tuple[float, float]) -> None:
        heapq.heappush(self._heap, (key[0], key[1], s))
        self._in_queue[s] = key

    def _remove(self, s: tuple[int, int, int]) -> None:
        self._in_queue.pop(s, None)

    def _peek_top_key(self) -> tuple[float, float]:
        while self._heap:
            k1, k2, s = self._heap[0]
            if self._in_queue.get(s) == (k1, k2):
                return (k1, k2)
            heapq.heappop(self._heap)
        return (INF, INF)

    def _pop_top(self):
        while self._heap:
            k1, k2, s = heapq.heappop(self._heap)
            if self._in_queue.get(s) == (k1, k2):
                del self._in_queue[s]
                return s, (k1, k2)
        return None, None

    def _update_vertex(self, u: tuple[int, int, int]) -> None:
        if u != self.goal:
            min_rhs = INF
            for s2 in self._neighbors(u):
                c = self._edge_cost(u, s2)
                if c < INF:
                    val = c + self.g[s2]
                    if val < min_rhs:
                        min_rhs = val
            self.rhs[u] = min_rhs
        self._remove(u)
        if self.g[u] != self.rhs[u]:
            self._push(u, self._calculate_key(u))

    def _compute_shortest_path(self, max_expansions: int | None = None) -> bool:
        expansions = 0
        requeues = 0  # popped only to discover a stale key and get re-pushed - no g/rhs change
        while True:
            start_key = self._calculate_key(self.start)
            start_consistent = self.g[self.start] == self.rhs[self.start]
            if self._peek_top_key() >= start_key and start_consistent:
                self.last_num_expansions, self.last_num_requeues = expansions, requeues
                return True
            if max_expansions is not None and expansions >= max_expansions:
                self.last_num_expansions, self.last_num_requeues = expansions, requeues
                return False
            u, k_old = self._pop_top()
            if u is None:
                self.last_num_expansions, self.last_num_requeues = expansions, requeues
                return True
            expansions += 1
            k_new = self._calculate_key(u)
            if k_old < k_new:
                requeues += 1
                self._push(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s2 in self._neighbors(u):
                    self._update_vertex(s2)
            else:
                self.g[u] = INF
                self._update_vertex(u)
                for s2 in self._neighbors(u):
                    self._update_vertex(s2)

    def update_start(self, new_start: tuple[int, int, int], max_expansions: int | None = None):
        self.last_rebuilt = False
        if new_start != self.start:
            self._n_start_moves += 1
            if self.rebuild_interval is not None and self._n_start_moves % self.rebuild_interval == 0:
                self._reset(new_start)
                self.last_rebuilt = True
            else:
                self.km += _heuristic(self.start, new_start)
                self.start = new_start

        if self.blocked[new_start]:
            return None

        ok = self._compute_shortest_path(max_expansions)
        if not ok or self.g[new_start] == INF:
            return None

        path = [new_start]
        current = new_start
        guard = 0
        max_guard = self.nx * self.ny * self.nz
        while current != self.goal:
            best_next, best_val = None, INF
            for s2 in self._neighbors(current):
                c = self._edge_cost(current, s2)
                if c < INF:
                    val = c + self.g[s2]
                    if val < best_val:
                        best_val, best_next = val, s2
            if best_next is None:
                return None
            current = best_next
            path.append(current)
            guard += 1
            if guard > max_guard:
                return None
        return path


class DStarLite3DPlanner:
    """Drop-in wrapper matching dense_grid_3d.plan_3d_dense's calling convention, but backed
    by one persistent DStarLite3D instance built once (against a FIXED goal and a FIXED,
    already-built dense grid - see build_octomap_grid.py) and reused across every
    `.plan(pos)` call. This is the single-fixed-mask design (no runtime fallback ladder),
    matching localize_plan_loop.py's default for the 2D planner - the mask (robot_radius_m/
    inflate_radius_m/close_radius_m/allow_unknown) is decided once up front, not re-decided
    per query."""

    def __init__(
        self, grid: dict, goal_xy: np.ndarray, cost_weight: float = 0.0, rebuild_interval: int | None = None,
        **mask_kwargs,
    ):
        self.resolution, self.origin = grid["resolution"], grid["origin"]
        self.blocked, self.soft_cost = build_cost_and_blocked_3d(grid, **mask_kwargs)
        self.goal_idx = world_to_grid_3d(goal_xy, self.origin, self.resolution)
        self.dsl = DStarLite3D(self.soft_cost, self.blocked, self.goal_idx, cost_weight, rebuild_interval=rebuild_interval)

    def plan(self, pos: np.ndarray, max_expansions: int | None = None):
        start_idx = world_to_grid_3d(pos, self.origin, self.resolution)
        nx, ny, nz = self.blocked.shape
        if not (0 <= start_idx[0] < nx and 0 <= start_idx[1] < ny and 0 <= start_idx[2] < nz):
            return None, None, self.blocked, self.origin, "start is outside the map grid"
        if self.blocked[start_idx]:
            return None, None, self.blocked, self.origin, "start cell is blocked (occupied, unobserved, or within robot_radius_m of an obstacle)"
        path_idx = self.dsl.update_start(start_idx, max_expansions=max_expansions)
        if path_idx is None:
            return None, None, self.blocked, self.origin, "no path found"
        path_world = np.array([grid_to_world_3d(idx, self.origin, self.resolution) for idx in path_idx])
        return path_world, path_idx, self.blocked, self.origin, None
