import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from splg_slam.data import kitti as kitti_data
from splg_slam.geometry.alignment import robust_umeyama, umeyama
from splg_slam.geometry.pose_utils import camera_center
from splg_slam.map.io import load_map


def load_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts, xyz = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            ts.append(int(row[0]))
            xyz.append([float(row[1]), float(row[2]), float(row[3])])
    return np.array(ts), np.array(xyz)


def print_stats(label: str, values: np.ndarray, unit: str) -> None:
    print(
        f"{label}: min={values.min():.4f}{unit} max={values.max():.4f}{unit} "
        f"median={np.median(values):.4f}{unit} mean={values.mean():.4f}{unit}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map_path", type=str)
    parser.add_argument(
        "gt_path", type=str,
        help="EuRoC state_groundtruth_estimate0/data.csv, OR a KITTI sequence directory "
        "(e.g. .../sequences/00, containing times.txt - ground truth is read from the "
        "sibling data_odometry_poses/dataset/poses/<seq>.txt) - detected by whether the "
        "path is a directory or a file",
    )
    parser.add_argument("--out", type=str, default=None, help="output plot path (PNG)")
    parser.add_argument(
        "--robust_trim", type=float, default=0.2,
        help="also report a robust-Sim3-aligned RMSE that iteratively excludes this fraction "
        "of worst-fitting keyframes before refitting (see robust_umeyama) - a supplementary "
        "diagnostic for when part of the trajectory was corrected late (e.g. by a loop closure "
        "well into the sequence) and drags the single-fit RMSE up without reflecting the "
        "converged accuracy; the plain RMSE above remains the primary, comparable-everywhere "
        "number. Set to 0 to skip.",
    )
    args = parser.parse_args()

    world_map = load_map(args.map_path)
    kf_ids = world_map.keyframe_ids_sorted()
    centers = np.array([camera_center(world_map.keyframes[i].pose_cw) for i in kf_ids])
    timestamps = np.array([world_map.keyframes[i].timestamp_ns for i in kf_ids])
    print(f"num keyframes: {len(kf_ids)}, num map points: {len(world_map.map_points)}")

    gt_path = Path(args.gt_path)
    gt_ts, gt_xyz = kitti_data.load_gt_as_xyz(gt_path) if gt_path.is_dir() else load_gt(gt_path)
    gt_idx = np.clip(np.searchsorted(gt_ts, timestamps), 0, len(gt_ts) - 1)
    gt_matched = gt_xyz[gt_idx]

    r, s, t = umeyama(centers, gt_matched)
    aligned = s * (r @ centers.T).T + t

    err = np.linalg.norm(aligned - gt_matched, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    time_s = (timestamps - timestamps[0]) / 1e9

    print(f"scale={s:.4f}")
    print_stats("ATE (m)", err, "m")
    print(f"ATE (m): rmse={rmse:.4f}m")

    robust_mask = None
    if args.robust_trim > 0:
        r_rb, s_rb, t_rb, robust_mask = robust_umeyama(centers, gt_matched, trim_ratio=args.robust_trim)
        aligned_rb = s_rb * (r_rb @ centers.T).T + t_rb
        err_rb_all = np.linalg.norm(aligned_rb - gt_matched, axis=1)
        rmse_rb = float(np.sqrt((err_rb_all[robust_mask] ** 2).mean()))
        n_excluded = int((~robust_mask).sum())
        print(
            f"ATE (m), robust alignment (excluded {n_excluded}/{len(centers)} worst-fitting "
            f"keyframes): rmse={rmse_rb:.4f}m [supplementary - see --robust_trim help]"
        )

    frame_times = np.asarray(getattr(world_map, "frame_processing_times_s", []), dtype=float)
    if frame_times.size > 0:
        print_stats("Per-frame processing time", frame_times * 1000.0, "ms")
    else:
        print("Per-frame processing time: no timing data in this map (rebuild to record it)")

    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(14, 7))

    # Pick the two axes with the most spread as the "top-down" view, instead of hardcoding
    # (x, y): that's correct for EuRoC's gravity/IMU-frame ground truth (Z already up), but
    # KITTI's ground truth is in the raw cam0 frame (X=right, Y=down, Z=forward) - a fixed
    # (x, y) plot there is a sideways slice through the road, not a bird's-eye view.
    axis_names = ["x", "y", "z"]
    plot_axes = np.argsort(gt_matched.std(axis=0))[-2:]
    plot_axes = plot_axes[np.argsort(-gt_matched[:, plot_axes].std(axis=0))]  # wider spread first -> horizontal
    ax_a, ax_b = int(plot_axes[0]), int(plot_axes[1])

    ax_traj.plot(gt_matched[:, ax_a], gt_matched[:, ax_b], "r-", label="ground truth", linewidth=2)
    ax_traj.plot(aligned[:, ax_a], aligned[:, ax_b], "b--", label="SPLG estimate (Sim3-aligned)", linewidth=1.5)
    ax_traj.set_xlabel(f"{axis_names[ax_a]} (m)")
    ax_traj.set_ylabel(f"{axis_names[ax_b]} (m)")
    ax_traj.axis("equal")
    ax_traj.legend()
    ax_traj.set_title(f"{Path(args.map_path).parent.name}, RMSE={rmse:.3f} m")

    ax_err.plot(time_s, err, "b-", linewidth=1.0)
    ax_err.axhline(rmse, color="gray", linestyle="--", linewidth=1.0, label=f"RMSE={rmse:.3f}m")
    if robust_mask is not None and not robust_mask.all():
        ax_err.scatter(
            time_s[~robust_mask], err[~robust_mask], color="red", s=12, zorder=3,
            label=f"excluded by robust fit ({int((~robust_mask).sum())})",
        )
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("ATE error (m)")
    ax_err.legend()
    ax_err.set_title("Per-keyframe ATE error over time")

    plt.tight_layout()

    out_path = args.out or str(Path(args.map_path).parent / "trajectory_eval.png")
    plt.savefig(out_path, dpi=120)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
