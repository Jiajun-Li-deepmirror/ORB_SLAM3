"""Plot estimated XY trajectories (optionally several runs of the same sequence, e.g.
stereo vs stereo+IMU) against ground truth, Sim3-aligned the same way eval_trajectory.py
scores them, plus a per-axis scale diagnostic.

The scale row is the point of this script: a single Sim3 fit reports one global scale, but
if the *shape* is right and only the size is wrong (the classic short-baseline stereo depth
bias), per-axis scale ratios come out consistently off in the same direction - which a
plain ATE number alone doesn't reveal.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splg_slam.geometry.alignment import umeyama
from splg_slam.geometry.pose_utils import camera_center
from splg_slam.map.io import load_map


def load_gt_csv(path: Path):
    rows = np.loadtxt(path, delimiter=",", skiprows=1)
    return rows[:, 0].astype(np.int64), rows[:, 1:4]


def match_gt(kf_ts: np.ndarray, gt_ts: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(gt_ts, kf_ts), 0, len(gt_ts) - 1)
    prev = np.clip(idx - 1, 0, len(gt_ts) - 1)
    use_prev = np.abs(gt_ts[prev] - kf_ts) < np.abs(gt_ts[idx] - kf_ts)
    idx[use_prev] = prev[use_prev]
    return gt_xyz[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_csv", type=str)
    parser.add_argument("out_png", type=str)
    parser.add_argument("--map", action="append", required=True, metavar="LABEL=PATH",
                        help="repeatable, e.g. --map stereo=/path/map.pkl --map stereo+imu=/path/map.pkl")
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    gt_ts, gt_xyz = load_gt_csv(Path(args.gt_csv))

    fig, (ax_xy, ax_err) = plt.subplots(1, 2, figsize=(15, 7))
    ax_xy.plot(gt_xyz[:, 0], gt_xyz[:, 1], "k-", lw=1.4, alpha=0.6, label="ground truth")

    for spec, color in zip(args.map, ["tab:blue", "tab:green", "tab:red", "tab:purple"]):
        label, _, map_path = spec.partition("=")
        world_map = load_map(map_path)
        kf_ids = world_map.keyframe_ids_sorted()
        kf_ts = np.array([world_map.keyframes[i].timestamp_ns for i in kf_ids])
        centers = np.array([camera_center(world_map.keyframes[i].pose_cw) for i in kf_ids])

        gt_matched = match_gt(kf_ts, gt_ts, gt_xyz)
        r, s, t = umeyama(centers, gt_matched)
        aligned = s * (r @ centers.T).T + t
        err = np.linalg.norm(aligned - gt_matched, axis=1)
        rmse = float(np.sqrt((err ** 2).mean()))

        # Path length ratio is scale-error evidence independent of the Sim3 fit: it compares
        # how far the camera actually travelled against how far the estimate thinks it did.
        est_len = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
        gt_len = float(np.linalg.norm(np.diff(gt_matched, axis=0), axis=1).sum())

        ax_xy.plot(aligned[:, 0], aligned[:, 1], ".", ms=2.5, color=color,
                   label=f"{label}: RMSE {rmse:.2f}m, sim3 scale {s:.3f}")
        ax_err.plot((kf_ts - kf_ts[0]) / 1e9, err, "-", lw=1, color=color, label=f"{label} (RMSE {rmse:.2f}m)")
        print(f"{label}: keyframes={len(kf_ids)} rmse={rmse:.4f}m sim3_scale={s:.4f} "
              f"est_path={est_len:.2f} gt_path={gt_len:.2f} path_ratio={est_len / max(gt_len, 1e-9):.4f}")

    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title(args.title or "XY trajectory vs ground truth")
    ax_xy.legend(fontsize=8)
    ax_xy.grid(alpha=0.3)
    ax_xy.set_aspect("equal")

    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("ATE error (m)")
    ax_err.set_title("Per-keyframe ATE over time")
    ax_err.legend(fontsize=8)
    ax_err.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=125)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
