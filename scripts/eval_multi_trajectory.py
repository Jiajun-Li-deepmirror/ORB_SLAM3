import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from eval_trajectory import load_gt, print_stats, umeyama

from splg_slam.geometry.pose_utils import camera_center
from splg_slam.map.io import load_map


def main():
    parser = argparse.ArgumentParser(
        description="ATE for a map whose keyframes were pooled from multiple EuRoC sequences "
        "(build_map_multi.py) - identifies each keyframe's source dataset via its image_path, "
        "pools ground truth from all datasets' own GT files, and fits ONE combined Sim3 "
        "alignment so the reported RMSE reflects whether the single fused map is geometrically "
        "consistent with the real trajectory across all sequences, not just one."
    )
    parser.add_argument("map_path", type=str)
    parser.add_argument(
        "dataset_gt", type=str, nargs="+",
        help="pairs of '<mav0_dir_substring>:<gt_csv_path>', e.g. MH_01_easy:datasets/.../MH_01_easy/.../data.csv",
    )
    args = parser.parse_args()

    dataset_gts = []
    for spec in args.dataset_gt:
        marker, gt_path = spec.split(":", 1)
        gt_ts, gt_xyz = load_gt(Path(gt_path))
        dataset_gts.append((marker, gt_ts, gt_xyz))

    world_map = load_map(args.map_path)
    kf_ids = world_map.keyframe_ids_sorted()
    print(f"num keyframes: {len(kf_ids)}, num map points: {len(world_map.map_points)}")

    centers, gt_matched, per_dataset_count = [], [], {}
    unmatched = 0
    for kf_id in kf_ids:
        kf = world_map.keyframes[kf_id]
        if kf.image_path is None:
            unmatched += 1
            continue
        marker_match = next((m for m, _, _ in dataset_gts if m in kf.image_path), None)
        if marker_match is None:
            unmatched += 1
            continue
        _, gt_ts, gt_xyz = next((m, t, x) for m, t, x in dataset_gts if m == marker_match)
        gi = int(np.clip(np.searchsorted(gt_ts, kf.timestamp_ns), 0, len(gt_ts) - 1))
        centers.append(camera_center(kf.pose_cw))
        gt_matched.append(gt_xyz[gi])
        per_dataset_count[marker_match] = per_dataset_count.get(marker_match, 0) + 1

    print(f"matched {len(centers)} keyframes to ground truth ({unmatched} unmatched), by dataset: {per_dataset_count}")
    centers = np.array(centers)
    gt_matched = np.array(gt_matched)

    r, s, t = umeyama(centers, gt_matched)
    aligned = s * (r @ centers.T).T + t
    err = np.linalg.norm(aligned - gt_matched, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))

    print(f"scale={s:.4f}")
    print_stats("ATE (m), whole fused map, one combined alignment", err, "m")
    print(f"ATE (m): rmse={rmse:.4f}m")


if __name__ == "__main__":
    main()
