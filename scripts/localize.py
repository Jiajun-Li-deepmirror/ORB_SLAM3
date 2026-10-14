import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splg_slam.config import load_config
from splg_slam.data.loader import dataset_dir, dataset_module, load_dataset_rig, read_frame_images, set_dataset_dir
from splg_slam.geometry.alignment import umeyama
from splg_slam.geometry.pose_utils import camera_center
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.relocalizer import Relocalizer
from splg_slam.map.io import load_map
from splg_slam.utils import nearest_indices


def load_gt_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts, xyz = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            ts.append(int(row[0]))
            xyz.append([float(row[1]), float(row[2]), float(row[3])])
    return np.array(ts), np.array(xyz)


def fit_map_to_gt(world_map, gt_ts: np.ndarray, gt_xyz: np.ndarray):
    """Same Umeyama fit eval_trajectory.py uses to score ATE - the map's own keyframe
    poses are in an arbitrary (gravity-aligned but not GT-registered) frame, so a
    relocalized query pose needs this same similarity transform before it's comparable to
    GT: `s*(r@p)+t`."""
    kf_ids = world_map.keyframe_ids_sorted()
    centers = np.array([camera_center(world_map.keyframes[i].pose_cw) for i in kf_ids])
    timestamps = np.array([world_map.keyframes[i].timestamp_ns for i in kf_ids])
    gt_idx = nearest_indices(gt_ts, timestamps)
    r, s, t = umeyama(centers, gt_xyz[gt_idx])
    return r, s, t


def main():
    parser = argparse.ArgumentParser(
        description="Pure relocalization (no tracking state) fps/accuracy probe: every "
        "query frame independently pays for the full Relocalizer.localize() path (global "
        "descriptor + retrieval + LightGlue x top_k + PnP+RANSAC). Useful for isolating the "
        "cost/accuracy of individual relocalization knobs (fp16, top_k, early-exit "
        "threshold, max_keypoints) without ContinuousLocalizer's cheap tracking tier "
        "masking most queries."
    )
    parser.add_argument("config", type=str)
    parser.add_argument("--dataset-dir", help="override the input dataset directory")
    parser.add_argument("--gt-path", help="optional EuRoC-format ground-truth CSV")
    parser.add_argument("map_path", type=str)
    parser.add_argument("--start", type=int, default=0, help="query frame index start")
    parser.add_argument("--count", type=int, default=20, help="number of query frames")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=5, help="override Relocalizer.localize()'s retrieval top_k")
    parser.add_argument(
        "--early_exit_inliers", type=int, default=None,
        help="override cfg.tracking.relocalization_early_exit_inliers (None: use the config's own value)",
    )
    parser.add_argument(
        "--max_keypoints", type=int, default=None,
        help="override cfg.tracking.relocalization_max_keypoints (None: use the config's own value)",
    )
    parser.add_argument("--fp16", action="store_true", help="override cfg.tracking.relocalization_fp16 to True")
    parser.add_argument(
        "--load_unused_right_image", action="store_true",
        help="reproduce the old (wasteful) behavior of reading+rectifying the right image "
        "even though Relocalizer never uses it - only exists for an A/B fps comparison "
        "against the fixed (default) path below.",
    )
    args = parser.parse_args()

    if args.start < 0 or args.count <= 0 or args.stride <= 0:
        parser.error("--start must be nonnegative; --count and --stride must be positive")
    cfg = load_config(args.config)
    if args.dataset_dir:
        set_dataset_dir(cfg, args.dataset_dir)
    if args.early_exit_inliers is not None:
        cfg.tracking.relocalization_early_exit_inliers = args.early_exit_inliers
    if args.max_keypoints is not None:
        cfg.tracking.relocalization_max_keypoints = args.max_keypoints
    if args.fp16:
        cfg.tracking.relocalization_fp16 = True

    world_map = load_map(args.map_path)
    print(f"Loaded map: {len(world_map.keyframes)} keyframes, {len(world_map.map_points)} map points")

    rig = load_dataset_rig(cfg)
    rectifier = StereoRectifier(rig)
    relocalizer = Relocalizer(world_map, rectifier, cfg)
    print(
        f"Relocalizer config: top_k={args.top_k} early_exit_inliers={cfg.tracking.relocalization_early_exit_inliers} "
        f"max_keypoints={cfg.tracking.relocalization_max_keypoints} fp16={getattr(cfg.tracking, 'relocalization_fp16', False)}"
    )

    gt_ts = gt_xyz = None
    gt_path = Path(args.gt_path) if args.gt_path else dataset_dir(cfg) / "state_groundtruth_estimate0" / "data.csv"
    if args.gt_path or gt_path.is_file():
        gt_ts, gt_xyz = load_gt_csv(gt_path)
    if gt_ts is not None:
        r_fit, s_fit, t_fit = fit_map_to_gt(world_map, gt_ts, gt_xyz)
    else:
        print("No ground truth supplied; reporting localization success and timing only.")

    entries = dataset_module(cfg).load_stereo_frames(dataset_dir(cfg))
    query_entries = itertools.islice(entries, args.start, args.start + args.count * args.stride, args.stride)

    n_success = 0
    errors = []
    query_times_s = []
    for e in query_entries:
        t0 = time.monotonic()
        img_l, img_r = read_frame_images(e, stereo=args.load_unused_right_image)
        if args.load_unused_right_image:
            rect_l, _ = rectifier.rectify(img_l, img_r)
        else:
            rect_l = rectifier.rectify_left(img_l)

        ok, pose_cw, info = relocalizer.localize(rect_l, top_k=args.top_k)
        query_times_s.append(time.monotonic() - t0)

        if ok:
            n_success += 1
            accuracy = ""
            if gt_ts is not None:
                center = camera_center(pose_cw)
                est_gt = s_fit * (r_fit @ center) + t_fit
                gt_i = int(nearest_indices(gt_ts, e.timestamp_ns))
                err_m = float(np.linalg.norm(est_gt - gt_xyz[gt_i]))
                errors.append(err_m)
                accuracy = f"gt_err={err_m:.3f}m "
            print(
                f"frame {e.index} ts={e.timestamp_ns}: OK  inliers={info['num_inliers']} "
                f"candidate_kf={info['candidate_kf']} sim={info['retrieval_sim']:.3f} "
                f"{accuracy}({query_times_s[-1]*1000:.0f}ms)"
            )
        else:
            print(f"frame {e.index} ts={e.timestamp_ns}: FAILED ({info.get('reason')}) ({query_times_s[-1]*1000:.0f}ms)")

    close = getattr(entries, "close", None)
    if close is not None:
        close()
    if not query_times_s:
        raise ValueError("No query frames in the requested range")
    times_ms = np.array(query_times_s) * 1000.0
    print(f"\n{n_success}/{len(query_times_s)} localized")
    print(
        f"Per-query time (ms): mean={times_ms.mean():.1f} median={np.median(times_ms):.1f} "
        f"min={times_ms.min():.1f} max={times_ms.max():.1f} -> {1000.0/times_ms.mean():.2f} fps (mean)"
    )
    if errors:
        errs = np.array(errors)
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        print(
            f"Accuracy vs GT (m): mean={errs.mean():.3f} median={np.median(errs):.3f} "
            f"rmse={rmse:.3f} min={errs.min():.3f} max={errs.max():.3f}"
        )


if __name__ == "__main__":
    main()
