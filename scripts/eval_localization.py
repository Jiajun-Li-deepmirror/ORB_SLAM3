import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_trajectory import load_gt, umeyama

from splg_slam.config import load_config
from splg_slam.data.euroc import load_stereo_frames, load_stereo_rig
from splg_slam.geometry.pose_utils import camera_center
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.relocalizer import Relocalizer
from splg_slam.map.io import load_map


def print_stats(label: str, values: np.ndarray, unit: str) -> None:
    print(
        f"{label}: min={values.min():.4f}{unit} max={values.max():.4f}{unit} "
        f"median={np.median(values):.4f}{unit} mean={values.mean():.4f}{unit}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("map_path", type=str)
    parser.add_argument("gt_csv", type=str, help="EuRoC state_groundtruth_estimate0/data.csv")
    parser.add_argument("--stride", type=int, default=3, help="query every Nth stereo pair across the whole sequence")
    args = parser.parse_args()

    cfg = load_config(args.config)
    world_map = load_map(args.map_path)
    print(f"Loaded map: {len(world_map.keyframes)} keyframes, {len(world_map.map_points)} map points")

    rig = load_stereo_rig(cfg.dataset.mav0_dir)
    rectifier = StereoRectifier(rig)

    # Fit the map-frame -> ground-truth-frame similarity transform from the map's own
    # keyframes (same approach as eval_trajectory.py), so query poses can be compared
    # against GT without needing a second, separate alignment procedure.
    kf_ids = world_map.keyframe_ids_sorted()
    kf_centers = np.array([camera_center(world_map.keyframes[i].pose_cw) for i in kf_ids])
    kf_timestamps = np.array([world_map.keyframes[i].timestamp_ns for i in kf_ids])
    gt_ts, gt_xyz = load_gt(args.gt_csv)
    gt_idx = np.clip(np.searchsorted(gt_ts, kf_timestamps), 0, len(gt_ts) - 1)
    r, s, t = umeyama(kf_centers, gt_xyz[gt_idx])

    relocalizer = Relocalizer(world_map, rectifier, cfg)

    entries = load_stereo_frames(cfg.dataset.mav0_dir)[:: args.stride]
    print(f"Querying {len(entries)} frames (stride={args.stride}) in pure localization mode...")

    query_times_s = []
    errors_m = []
    n_success = 0
    n_fail = 0
    for i, e in enumerate(entries):
        img_l = cv2.imread(str(e.left_path), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(e.right_path), cv2.IMREAD_GRAYSCALE)
        rect_l, _ = rectifier.rectify(img_l, img_r)

        t_start = time.monotonic()
        ok, pose_cw, info = relocalizer.localize(rect_l)
        query_times_s.append(time.monotonic() - t_start)

        if not ok:
            n_fail += 1
            continue
        n_success += 1

        center = camera_center(pose_cw)
        aligned = s * (r @ center) + t
        gi = int(np.clip(np.searchsorted(gt_ts, e.timestamp_ns), 0, len(gt_ts) - 1))
        errors_m.append(float(np.linalg.norm(aligned - gt_xyz[gi])))

        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(entries)}] success={n_success} fail={n_fail}")

    print(f"\nsuccess: {n_success}/{len(entries)} ({100 * n_success / len(entries):.1f}%)")

    query_times_s = np.asarray(query_times_s)
    print_stats("Query time", query_times_s * 1000.0, "ms")

    if errors_m:
        errors_m = np.asarray(errors_m)
        rmse = float(np.sqrt((errors_m ** 2).mean()))
        print_stats("Localization error", errors_m, "m")
        print(f"Localization error: rmse={rmse:.4f}m")
    else:
        print("No successful localizations to compute error stats")


if __name__ == "__main__":
    main()
