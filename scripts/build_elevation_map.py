import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.ndimage import generic_filter


def build_elevation_map(
    points: np.ndarray, resolution: float, ground_percentile: float = 5.0,
) -> dict:
    """Projects a gravity-aligned (Z=up) point cloud onto an (x,y) grid. Each cell's height
    is a robust low-percentile of the points landing in it (not the min, which a single noise
    point below the true floor would corrupt; not the mean, which furniture/structure above
    the floor would pull upward) - a simple, standard proxy for "the ground surface here"
    that doesn't require a full raycasting ground-segmentation pass.

    Returns a dict: height (2D array, NaN where unobserved), roughness (per-cell z std,
    NaN where <2 points), count (per-cell point count), x_min/y_min/resolution (grid origin)."""
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    ix = np.floor((points[:, 0] - x_min) / resolution).astype(np.int64)
    iy = np.floor((points[:, 1] - y_min) / resolution).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1

    height = np.full((ny, nx), np.nan)
    roughness = np.full((ny, nx), np.nan)
    count = np.zeros((ny, nx), dtype=np.int64)

    order = np.lexsort((points[:, 2], iy, ix))
    flat_idx = ix[order] * ny + iy[order]
    z_sorted = points[order, 2]
    boundaries = np.flatnonzero(np.diff(flat_idx)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(flat_idx)]])

    for start, end in zip(starts, ends):
        cell_idx = flat_idx[start]
        cx, cy = cell_idx // ny, cell_idx % ny
        z_cell = z_sorted[start:end]
        count[cy, cx] = len(z_cell)
        height[cy, cx] = np.percentile(z_cell, ground_percentile)
        if len(z_cell) >= 2:
            roughness[cy, cx] = float(np.std(z_cell))

    return {
        "height": height, "roughness": roughness, "count": count,
        "x_min": x_min, "y_min": y_min, "resolution": resolution,
    }


def smooth_height(height: np.ndarray, size: int) -> np.ndarray:
    """NaN-aware median filter over the height grid (unobserved cells don't get treated as
    z=0 and dragged into neighboring medians). Dense stereo reconstruction from a moving/
    oblique viewpoint is considerably noisier per-cell than a real depth sensor, and unlike a
    real elevation-mapping system (which fuses many observations of the same cell over time,
    averaging noise out), this batch one-shot percentile-per-column estimate has no such
    averaging - a median filter is a cheap stand-in that suppresses single-cell noise before
    slope/step/roughness differentiate it into spurious "obstacles"."""
    def nanmedian_or_nan(window):
        valid = window[~np.isnan(window)]
        return np.median(valid) if len(valid) > 0 else np.nan

    return generic_filter(height, nanmedian_or_nan, size=size, mode="constant", cval=np.nan)


def compute_traversability(
    elev: dict, max_slope_deg: float = 25.0, max_step_m: float = 0.15, max_roughness_m: float = 0.05,
) -> dict:
    """Slope (from height differences to the 4-neighborhood), step height (max abs height
    jump to any of the 8 neighbors - catches a discrete step a smoothed slope estimate could
    average away), and per-cell roughness (already computed) combine into a traversable mask
    plus a continuous cost in [0, 1] (0=flat/easy, 1=untraversable), for a legged robot's
    locomotion controller to handle small variation itself but flag genuine obstacles/steps."""
    h = elev["height"]
    res = elev["resolution"]

    dzdx = np.full_like(h, np.nan)
    dzdy = np.full_like(h, np.nan)
    dzdx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2 * res)
    dzdy[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2 * res)
    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    step = np.full_like(h, np.nan)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(h, dy, axis=0), dx, axis=1)
            diff = np.abs(h - shifted)
            step = np.fmax(step, diff)  # fmax ignores NaN unless both sides are NaN

    roughness = np.nan_to_num(elev["roughness"], nan=0.0)
    observed = ~np.isnan(h)

    cost = np.zeros_like(h)
    cost += np.clip(np.nan_to_num(slope_deg, nan=0.0) / max_slope_deg, 0, 1) / 3
    cost += np.clip(np.nan_to_num(step, nan=0.0) / max_step_m, 0, 1) / 3
    cost += np.clip(roughness / max_roughness_m, 0, 1) / 3
    cost = np.clip(cost, 0, 1)

    traversable = observed & (slope_deg < max_slope_deg) & (np.nan_to_num(step, nan=0.0) < max_step_m) & (roughness < max_roughness_m)

    return {"slope_deg": slope_deg, "step_m": step, "cost": cost, "traversable": traversable, "observed": observed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud_path", type=str, help="gravity-aligned (Z=up) point cloud, e.g. dense_cloud_gravity_imu_registration.ply")
    parser.add_argument("--resolution", type=float, default=0.05, help="grid cell size in meters")
    parser.add_argument("--ground_percentile", type=float, default=5.0)
    parser.add_argument("--max_slope_deg", type=float, default=25.0)
    parser.add_argument("--max_step_m", type=float, default=0.15)
    parser.add_argument("--max_roughness_m", type=float, default=0.05)
    parser.add_argument("--smooth_size", type=int, default=1, help="NaN-aware median filter kernel size on the height grid before computing slope/step (1 = no smoothing)")
    parser.add_argument("--out_prefix", type=str, default=None)
    parser.add_argument(
        "--height_band", type=float, nargs=2, default=None, metavar=("Z_MIN", "Z_MAX"),
        help="keep only points with z in [Z_MIN, Z_MAX] before gridding - a real ground robot's "
        "depth sensor only ever sees a band around its own height, not a whole room's ceiling/"
        "rigging; without this, a column's low-percentile 'ground' estimate can latch onto "
        "overhead structure instead of the actual floor whenever the floor itself wasn't "
        "densely observed in that column (this is a real issue for EuRoC's flying-drone data, "
        "which looked at the whole room from many altitudes, not a floor-level walking view).",
    )
    args = parser.parse_args()

    out_prefix = args.out_prefix or str(Path(args.cloud_path).with_suffix(""))

    cloud = o3d.io.read_point_cloud(args.cloud_path)
    points = np.asarray(cloud.points)
    print(f"Loaded {len(points)} points, bounds {points.min(axis=0)} .. {points.max(axis=0)}")

    if args.height_band is not None:
        z_min, z_max = args.height_band
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        print(f"Height band [{z_min}, {z_max}]: kept {mask.sum()}/{len(points)} points")
        points = points[mask]

    elev = build_elevation_map(points, args.resolution, args.ground_percentile)
    if args.smooth_size > 1:
        elev["height"] = smooth_height(elev["height"], args.smooth_size)
    trav = compute_traversability(elev, args.max_slope_deg, args.max_step_m, args.max_roughness_m)

    n_observed = int(trav["observed"].sum())
    n_traversable = int(trav["traversable"].sum())
    print(
        f"Grid: {elev['height'].shape[1]}x{elev['height'].shape[0]} cells @ {args.resolution}m, "
        f"{n_observed} observed ({100 * n_observed / elev['height'].size:.1f}%), "
        f"{n_traversable} traversable ({100 * n_traversable / max(n_observed, 1):.1f}% of observed)"
    )

    np.savez(
        f"{out_prefix}_elevation.npz",
        height=elev["height"], roughness=elev["roughness"], count=elev["count"],
        slope_deg=trav["slope_deg"], step_m=trav["step_m"], cost=trav["cost"], traversable=trav["traversable"],
        x_min=elev["x_min"], y_min=elev["y_min"], resolution=elev["resolution"],
    )
    print(f"Saved {out_prefix}_elevation.npz")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    h_masked = np.ma.masked_invalid(elev["height"])
    im0 = axes[0].imshow(h_masked, origin="lower", cmap="terrain")
    axes[0].set_title("Ground height (m)")
    plt.colorbar(im0, ax=axes[0], fraction=0.03)

    cost_display = np.where(trav["observed"], trav["cost"], np.nan)
    im1 = axes[1].imshow(np.ma.masked_invalid(cost_display), origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_title("Traversability cost (0=easy, 1=blocked)")
    plt.colorbar(im1, ax=axes[1], fraction=0.03)

    trav_display = np.where(trav["observed"], trav["traversable"].astype(float), np.nan)
    im2 = axes[2].imshow(np.ma.masked_invalid(trav_display), origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    axes[2].set_title(f"Traversable mask (green) - {n_traversable}/{n_observed} cells")
    plt.colorbar(im2, ax=axes[2], fraction=0.03)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_elevation.png", dpi=130)
    print(f"Saved {out_prefix}_elevation.png")


if __name__ == "__main__":
    main()
