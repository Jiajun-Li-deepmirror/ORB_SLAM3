import argparse
import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def world_to_grid(x: float, y: float, x_min: float, y_min: float, resolution: float) -> tuple[int, int]:
    return int(round((x - x_min) / resolution)), int(round((y - y_min) / resolution))


def grid_to_world(ix: int, iy: int, x_min: float, y_min: float, resolution: float) -> tuple[float, float]:
    return x_min + ix * resolution, y_min + iy * resolution


def astar(
    cost_grid: np.ndarray, blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int], cost_weight: float,
):
    """Grid A* over an 8-connected neighborhood. cost_grid[y, x] in [0, 1] (traversability
    cost from build_elevation_map.py); blocked[y, x] True = impassable (not traversable, or
    unobserved if the caller chose not to allow that). Edge cost blends step distance with the
    average of the two endpoints' terrain cost, so the planner prefers a longer flat/smooth
    route over a shorter one through rough/steep cells, not just shortest-path-by-distance."""
    ny, nx = cost_grid.shape

    def heuristic(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    if blocked[start[1], start[0]] or blocked[goal[1], goal[0]]:
        return None, float("inf")

    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from: dict = {}
    g_score = {start: 0.0}
    visited = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
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


def main():
    parser = argparse.ArgumentParser(
        description="A* global path planning on a traversability cost map produced by "
        "build_elevation_map.py. Blocks non-traversable cells (steep/rough/stepped) and, by "
        "default, unobserved cells too (can't safely plan through space nobody ever saw)."
    )
    parser.add_argument("elevation_npz", type=str)
    parser.add_argument("--start", type=float, nargs=2, required=True, metavar=("X", "Y"), help="world (map-frame) meters")
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--cost_weight", type=float, default=5.0, help="how strongly to penalize rough/steep cells vs pure distance")
    parser.add_argument("--allow_unknown", action="store_true", help="allow planning through never-observed cells (default: blocked)")
    parser.add_argument(
        "--block_cost_threshold", type=float, default=0.85,
        help="block a cell only once its continuous cost exceeds this, instead of the strict "
        "binary `traversable` flag (which requires slope AND step AND roughness to ALL pass "
        "at once - noisy dense-stereo reconstruction fails that AND every so often even on "
        "genuinely flat floor, fragmenting it into many disconnected traversable islands with "
        "no path between them). Set to >=1.0 or pass --use_strict_traversable to fall back to "
        "the binary flag.",
    )
    parser.add_argument("--use_strict_traversable", action="store_true", help="block on the binary `traversable` flag instead of --block_cost_threshold")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    data = np.load(args.elevation_npz)
    height, cost, traversable = data["height"], data["cost"], data["traversable"]
    x_min, y_min, resolution = float(data["x_min"]), float(data["y_min"]), float(data["resolution"])
    observed = ~np.isnan(height)

    if args.use_strict_traversable:
        blocked = ~traversable
    else:
        blocked = np.nan_to_num(cost, nan=1.0) > args.block_cost_threshold
    if not args.allow_unknown:
        blocked = blocked | ~observed

    start_grid = world_to_grid(args.start[0], args.start[1], x_min, y_min, resolution)
    goal_grid = world_to_grid(args.goal[0], args.goal[1], x_min, y_min, resolution)
    print(f"start world={tuple(args.start)} -> grid={start_grid}; goal world={tuple(args.goal)} -> grid={goal_grid}")

    ny, nx = height.shape
    for name, (gx, gy) in [("start", start_grid), ("goal", goal_grid)]:
        if not (0 <= gx < nx and 0 <= gy < ny):
            print(f"ERROR: {name} {(gx, gy)} is outside the map grid ({nx}x{ny})")
            return
        if blocked[gy, gx]:
            print(f"ERROR: {name} cell is blocked (not traversable{'or unobserved' if not args.allow_unknown else ''}) - pick a different point")
            return

    path, total_cost = astar(cost, blocked, start_grid, goal_grid, args.cost_weight)
    if path is None:
        print("NO PATH FOUND")
        return

    world_path = [grid_to_world(ix, iy, x_min, y_min, resolution) for ix, iy in path]
    path_len_m = sum(
        float(np.hypot(world_path[i + 1][0] - world_path[i][0], world_path[i + 1][1] - world_path[i][1]))
        for i in range(len(world_path) - 1)
    )
    print(f"Path found: {len(path)} cells, {path_len_m:.2f}m path length, total weighted cost={total_cost:.2f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    display_cost = np.where(observed, cost, np.nan)
    im = ax.imshow(np.ma.masked_invalid(display_cost), origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, label="traversability cost")
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, "b-", linewidth=2, label="planned path")
    ax.plot(start_grid[0], start_grid[1], "g^", markersize=12, label="start")
    ax.plot(goal_grid[0], goal_grid[1], "r*", markersize=14, label="goal")
    ax.legend()
    ax.set_title(f"A* path: {path_len_m:.2f}m, {len(path)} cells")
    plt.tight_layout()

    out_path = args.out or str(Path(args.elevation_npz).with_suffix("")) + "_path.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")

    npz_out = str(Path(args.elevation_npz).with_suffix("")) + "_path.npz"
    np.savez(npz_out, grid_path=np.array(path), world_path=np.array(world_path))
    print(f"Saved {npz_out}")


if __name__ == "__main__":
    main()
