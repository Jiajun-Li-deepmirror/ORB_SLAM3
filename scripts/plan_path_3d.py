import argparse
import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import octomap

# 26-connected 3D neighborhood (all offsets except (0,0,0)).
NEIGHBORS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


class OccupancyChecker:
    """Wraps OcTree.search()/isNodeOccupied() into a clean 3-state query. This binding
    doesn't return None for never-observed space - search() still hands back a node
    wrapper, and isNodeOccupied() on it raises octomap.NullPointerException, which is
    the actual signal for "unknown" (verified empirically: a point 1000m outside the map
    raises this, a genuinely free voxel returns False, an occupied one returns True)."""

    def __init__(self, tree: octomap.OcTree):
        self.tree = tree

    def state(self, xyz: np.ndarray) -> str:
        node = self.tree.search(np.asarray(xyz, dtype=np.float64))
        try:
            return "occupied" if self.tree.isNodeOccupied(node) else "free"
        except octomap.NullPointerException:
            return "unknown"


def astar_3d(
    checker: OccupancyChecker, resolution: float, start_idx: tuple[int, int, int], goal_idx: tuple[int, int, int],
    origin: np.ndarray, allow_unknown: bool = False,
):
    """Pure shortest-Euclidean-distance A* over a 3D voxel grid derived from the octomap -
    no terrain cost weighting (unlike plan_path.py's ground-robot planner): a drone flying
    through open 3D space doesn't care about "slope" or "roughness", only what's physically
    in the way. Grid indices are (ix, iy, iz); world position = origin + idx * resolution."""

    def idx_to_world(idx):
        return origin + np.array(idx) * resolution

    def is_blocked(idx):
        state = checker.state(idx_to_world(idx))
        if state == "occupied":
            return True
        if state == "unknown":
            return not allow_unknown
        return False

    def heuristic(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    if is_blocked(start_idx) or is_blocked(goal_idx):
        return None, float("inf")

    open_heap = [(heuristic(start_idx, goal_idx), 0.0, start_idx)]
    came_from = {}
    g_score = {start_idx: 0.0}
    visited = set()
    blocked_cache = {}

    def blocked_cached(idx):
        if idx not in blocked_cache:
            blocked_cache[idx] = is_blocked(idx)
        return blocked_cache[idx]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
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
            tentative_g = g + step_dist
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbor, goal_idx), tentative_g, neighbor))

    return None, float("inf")


def main():
    parser = argparse.ArgumentParser(
        description="3D shortest-path planning directly on an Octomap, for drone-style "
        "navigation: pure Euclidean distance (26-connected), no terrain cost weighting - a "
        "drone in free 3D space only cares about what's physically blocking it, unlike the "
        "ground-robot elevation-map planner (plan_path.py) which also penalizes rough/steep "
        "cells a legged robot would rather avoid."
    )
    parser.add_argument("octomap_path", type=str, help=".bt or .ot octomap file")
    parser.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"), help="world meters")
    parser.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--allow_unknown", action="store_true", help="allow flying through never-observed space (default: blocked, safer)")
    parser.add_argument("--out", type=str, default=None, help="output path viewer HTML (reuses view_octomap.py's renderer, with the path overlaid)")
    args = parser.parse_args()

    tree = octomap.OcTree(str(args.octomap_path).encode())
    resolution = tree.getResolution()
    checker = OccupancyChecker(tree)

    metric_min = np.array(tree.getMetricMin())
    origin = metric_min

    def world_to_idx(p):
        return tuple(int(round(v)) for v in (np.array(p) - origin) / resolution)

    start_idx = world_to_idx(args.start)
    goal_idx = world_to_idx(args.goal)
    print(f"start world={tuple(args.start)} -> voxel {start_idx}; goal world={tuple(args.goal)} -> voxel {goal_idx}")
    print(f"octomap resolution={resolution}m, bounds min={tree.getMetricMin()} max={tree.getMetricMax()}")

    for name, p in [("start", args.start), ("goal", args.goal)]:
        state = checker.state(np.array(p))
        if state != "free":
            print(f"ERROR: {name} point is {state}, not free space - pick a different point")
            return

    path_idx, total_dist_cells = astar_3d(checker, resolution, start_idx, goal_idx, origin, args.allow_unknown)
    if path_idx is None:
        print("NO PATH FOUND")
        return

    world_path = [origin + np.array(idx) * resolution for idx in path_idx]
    path_len_m = sum(
        float(np.linalg.norm(world_path[i + 1] - world_path[i])) for i in range(len(world_path) - 1)
    )
    straight_line_m = float(np.linalg.norm(np.array(args.goal) - np.array(args.start)))
    print(
        f"Path found: {len(path_idx)} voxels, {path_len_m:.2f}m flight path "
        f"(straight-line distance {straight_line_m:.2f}m, detour factor {path_len_m / max(straight_line_m, 1e-6):.2f}x)"
    )

    out_path = args.out or str(Path(args.octomap_path).with_suffix("")) + "_3dpath.npz"
    np.savez(out_path, path_world=np.array(world_path), path_voxel=np.array(path_idx))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
