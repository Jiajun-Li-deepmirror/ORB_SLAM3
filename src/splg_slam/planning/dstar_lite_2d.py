import heapq

import numpy as np

from splg_slam.planning.planner_2d import DEFAULT_FALLBACK_STAGES, build_cost_and_blocked, grid_to_world, world_to_grid

NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
INF = float("inf")


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class DStarLite2D:
    """Incremental replanner (Koenig & Likhachev, 2002) for the same 8-connected, cost-
    weighted grid planner_2d.astar solves from scratch every call. Specialized for this
    codebase's actual access pattern in an online replanning loop: the GOAL is fixed across
    many consecutive replans, only the START (the robot's current localized position) moves
    each call. D* Lite searches outward from the goal once; each later start move only needs
    to repair the part of that search the new start actually depends on, instead of
    re-exploring the whole grid - the expensive part of every plan_2d call when replanning
    repeatedly. The cost/blocked grids are fixed for the life of one instance (this static-
    elevation-map planner never edits them after loading) - construct a new instance if the
    map itself changes.

    Usage: construct once per (map, goal) pair, then call `update_start(new_start)` on every
    replan - the first call does a full search (comparable cost to one plan_2d call); later
    calls are typically far cheaper, the whole point of using this over plain A* in a loop."""

    def __init__(
        self, cost_grid: np.ndarray, blocked: np.ndarray, goal: tuple[int, int], cost_weight: float,
        rebuild_interval: int | None = None,
    ):
        """`rebuild_interval`: every this many `update_start` calls that actually move the
        start, throw away the accumulated queue/km state and do one fresh full sweep from
        the current start instead of another incremental repair. The queue of still-
        inconsistent nodes left over from earlier starts never shrinks on its own under this
        goal-fixed/start-moving usage pattern - it only grows - and every km increase forces
        re-keying (pop, discover stale, re-push) a share of it proportional to its size, so
        the per-call overhead climbs over a long run even though no real new exploration is
        needed. A periodic full rebuild bounds that queue size instead of letting it grow
        for the life of the instance. None disables this (the original unbounded behavior)."""
        self.cost_grid = cost_grid
        self.blocked = blocked
        self.goal = goal
        self.cost_weight = cost_weight
        self.rebuild_interval = rebuild_interval
        self.ny, self.nx = cost_grid.shape
        self._n_start_moves = 0
        self.last_rebuilt = False  # set by update_start, for external monitoring

        self._reset(goal)

    def _reset(self, start: tuple[int, int]) -> None:
        self.g = np.full((self.ny, self.nx), INF, dtype=np.float64)
        self.rhs = np.full((self.ny, self.nx), INF, dtype=np.float64)
        self.rhs[self.goal[1], self.goal[0]] = 0.0

        self.km = 0.0
        self.start = start
        self._heap: list[tuple[float, float, tuple[int, int]]] = []
        self._in_queue: dict[tuple[int, int], tuple[float, float]] = {}
        self._push(self.goal, self._calculate_key(self.goal))
        self.last_num_expansions = 0
        self.last_num_requeues = 0

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

    def _calculate_key(self, s: tuple[int, int]) -> tuple[float, float]:
        g_rhs = min(self.g[s[1], s[0]], self.rhs[s[1], s[0]])
        return (g_rhs + _heuristic(self.start, s) + self.km, g_rhs)

    def _push(self, s: tuple[int, int], key: tuple[float, float]) -> None:
        heapq.heappush(self._heap, (key[0], key[1], s))
        self._in_queue[s] = key

    def _remove(self, s: tuple[int, int]) -> None:
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

    def _update_vertex(self, u: tuple[int, int]) -> None:
        if u != self.goal:
            min_rhs = INF
            for s2 in self._neighbors(u):
                c = self._edge_cost(u, s2)
                if c < INF:
                    val = c + self.g[s2[1], s2[0]]
                    if val < min_rhs:
                        min_rhs = val
            self.rhs[u[1], u[0]] = min_rhs
        self._remove(u)
        if self.g[u[1], u[0]] != self.rhs[u[1], u[0]]:
            self._push(u, self._calculate_key(u))

    def _compute_shortest_path(self, max_expansions: int | None = None) -> bool:
        """Returns False if `max_expansions` ran out before convergence (caller should treat
        the current g[start] as not-yet-reliable), True once start is locally consistent."""
        expansions = 0
        requeues = 0  # popped only to discover a stale key and get re-pushed - no g/rhs change
        while True:
            start_key = self._calculate_key(self.start)
            start_consistent = self.g[self.start[1], self.start[0]] == self.rhs[self.start[1], self.start[0]]
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
            elif self.g[u[1], u[0]] > self.rhs[u[1], u[0]]:
                self.g[u[1], u[0]] = self.rhs[u[1], u[0]]
                for s2 in self._neighbors(u):
                    self._update_vertex(s2)
            else:
                self.g[u[1], u[0]] = INF
                self._update_vertex(u)
                for s2 in self._neighbors(u):
                    self._update_vertex(s2)

    def update_cost(self, changed_cells: list[tuple[int, int]]) -> None:
        """Call after mutating self.cost_grid/self.blocked at `changed_cells` (grid (x,y)
        coords) to repair the search incrementally instead of rebuilding - not used by the
        static-elevation-map replanning loop today, but this is the other half of what makes
        D* Lite worth its complexity over plain A* (edges changing, not just the start
        moving), included so a future live-updating map doesn't need a second planner."""
        for cell in changed_cells:
            self._update_vertex(cell)
            for s2 in self._neighbors(cell):
                self._update_vertex(s2)

    def update_start(self, new_start: tuple[int, int], max_expansions: int | None = None):
        """Moves the search's start to `new_start` and returns the shortest path from there
        to the fixed goal as a list of (x,y) grid cells, or None if no path exists (or the
        expansion budget ran out before determining one). This is the one call an online
        replanning loop makes per new localized position."""
        self.last_rebuilt = False
        if new_start != self.start:
            self._n_start_moves += 1
            if self.rebuild_interval is not None and self._n_start_moves % self.rebuild_interval == 0:
                self._reset(new_start)
                self.last_rebuilt = True
            else:
                self.km += _heuristic(self.start, new_start)
                self.start = new_start

        if self.blocked[new_start[1], new_start[0]]:
            return None

        ok = self._compute_shortest_path(max_expansions)
        if not ok or self.g[new_start[1], new_start[0]] == INF:
            return None

        path = [new_start]
        current = new_start
        guard = 0
        max_guard = self.nx * self.ny
        while current != self.goal:
            best_next, best_val = None, INF
            for s2 in self._neighbors(current):
                c = self._edge_cost(current, s2)
                if c < INF:
                    val = c + self.g[s2[1], s2[0]]
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


_MASK_KWARGS = ("allow_unknown", "use_strict_traversable", "block_cost_threshold", "robot_radius_m", "inflate_radius_m", "close_radius_m")


class DStarLiteFallbackPlanner:
    """Drop-in incremental alternative to planner_2d.plan_2d_with_fallback for a replanning
    LOOP (fixed elevation map + fixed goal, called repeatedly with a new current position):
    keeps one persistent DStarLite2D instance per fallback stage (built lazily, only if that
    stage is ever actually needed), so the very first call pays roughly the cost of one
    plan_2d call but every later call at an already-built stage is typically 1-2 orders of
    magnitude cheaper (verified on the KITTI 00 map: ~1-2s for a fresh plan_2d call vs
    4-14ms for a warm DStarLite2D.update_start once the base-stage instance exists) - the
    replanning frequency this then supports is bounded by localization, not planning.

    Construct once per (elevation, goal) pair; call `.plan(pos_xy)` on every replan. Interface
    mirrors plan_2d_with_fallback's return shape (path_world, path_grid, blocked, error,
    stage) so it can be swapped in without changing the caller's downstream handling."""

    def __init__(
        self, elevation: dict, goal_xy: np.ndarray, fallback_stages: list[dict] | None = None,
        rebuild_interval: int | None = None, **base_kwargs,
    ):
        self.elevation = elevation
        self.x_min, self.y_min, self.resolution = elevation["x_min"], elevation["y_min"], elevation["resolution"]
        self.goal_grid = world_to_grid(goal_xy[0], goal_xy[1], self.x_min, self.y_min, self.resolution)
        self.cost_weight = base_kwargs.get("cost_weight", 5.0)
        self.rebuild_interval = rebuild_interval

        stages = fallback_stages if fallback_stages is not None else DEFAULT_FALLBACK_STAGES
        self._stage_mask_kwargs: list[dict] = [{k: base_kwargs.get(k) for k in _MASK_KWARGS if k in base_kwargs}]
        cumulative = dict(self._stage_mask_kwargs[0])
        for overrides in stages:
            cumulative = {**cumulative, **{k: v for k, v in overrides.items() if k in _MASK_KWARGS}}
            self._stage_mask_kwargs.append(dict(cumulative))

        self._planners: dict[int, tuple[DStarLite2D, np.ndarray]] = {}

    def _get_stage(self, stage_idx: int) -> tuple[DStarLite2D, np.ndarray]:
        if stage_idx not in self._planners:
            mask_kwargs = self._stage_mask_kwargs[stage_idx]
            blocked, cost = build_cost_and_blocked(self.elevation, **mask_kwargs)
            self._planners[stage_idx] = (
                DStarLite2D(cost, blocked, self.goal_grid, self.cost_weight, rebuild_interval=self.rebuild_interval),
                blocked,
            )
        return self._planners[stage_idx]

    def plan(self, pos_xy: np.ndarray, max_expansions: int | None = None):
        start_grid = world_to_grid(pos_xy[0], pos_xy[1], self.x_min, self.y_min, self.resolution)
        blocked = None
        for stage_idx in range(len(self._stage_mask_kwargs)):
            dsl, blocked = self._get_stage(stage_idx)
            ny, nx = blocked.shape
            gx, gy = start_grid
            if not (0 <= gx < nx and 0 <= gy < ny) or blocked[gy, gx]:
                continue
            path_grid = dsl.update_start(start_grid, max_expansions=max_expansions)
            if path_grid is not None:
                path_world = np.array([grid_to_world(ix, iy, self.x_min, self.y_min, self.resolution) for ix, iy in path_grid])
                return path_world, path_grid, blocked, None, stage_idx
        return None, None, blocked, "no path found", len(self._stage_mask_kwargs) - 1
