import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from splg_slam.geometry.pose_utils import camera_center
from splg_slam.map.io import load_map


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity (rotation + scale + translation) mapping src onto dst."""
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = (dst_c.T @ src_c) / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1
    r = u @ d @ vt
    var_src = (src_c ** 2).sum() / len(src)
    scale = np.trace(np.diag(s) @ d) / var_src
    t = mu_dst - scale * r @ mu_src
    return r, scale, t


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
    parser.add_argument("gt_csv", type=str, help="EuRoC state_groundtruth_estimate0/data.csv")
    parser.add_argument("--out", type=str, default=None, help="output plot path (PNG)")
    args = parser.parse_args()

    world_map = load_map(args.map_path)
    kf_ids = world_map.keyframe_ids_sorted()
    centers = np.array([camera_center(world_map.keyframes[i].pose_cw) for i in kf_ids])
    timestamps = np.array([world_map.keyframes[i].timestamp_ns for i in kf_ids])
    print(f"num keyframes: {len(kf_ids)}, num map points: {len(world_map.map_points)}")

    gt_ts, gt_xyz = load_gt(args.gt_csv)
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

    frame_times = np.asarray(getattr(world_map, "frame_processing_times_s", []), dtype=float)
    if frame_times.size > 0:
        print_stats("Per-frame processing time", frame_times * 1000.0, "ms")
    else:
        print("Per-frame processing time: no timing data in this map (rebuild to record it)")

    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(14, 7))

    ax_traj.plot(gt_matched[:, 0], gt_matched[:, 1], "r-", label="ground truth", linewidth=2)
    ax_traj.plot(aligned[:, 0], aligned[:, 1], "b--", label="SPLG estimate (Sim3-aligned)", linewidth=1.5)
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.axis("equal")
    ax_traj.legend()
    ax_traj.set_title(f"{Path(args.map_path).parent.name}, RMSE={rmse:.3f} m")

    ax_err.plot(time_s, err, "b-", linewidth=1.0)
    ax_err.axhline(rmse, color="gray", linestyle="--", linewidth=1.0, label=f"RMSE={rmse:.3f}m")
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
