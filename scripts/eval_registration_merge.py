import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from eval_trajectory import load_gt, print_stats, umeyama

from splg_slam.geometry.pose_utils import camera_center
from splg_slam.map.io import load_map

DATASETS = [
    ("mh01", "MH_01_easy"),
    ("mh02", "MH_02_easy"),
    ("mh03", "MH_03_medium"),
    ("mh04", "MH_04_difficult"),
    ("mh05", "MH_05_difficult"),
]


def main():
    transforms = np.load("results/merged_registration_transforms.npz")

    centers, gt_matched = [], []
    per_dataset_count = {}
    for short_name, dir_name in DATASETS:
        if short_name not in transforms:
            print(f"{short_name}: no registration transform, skipping")
            continue
        t_wa_wb = transforms[short_name]
        world_map = load_map(f"results/{short_name}_map/map.pkl")
        gt_ts, gt_xyz = load_gt(Path(f"datasets/machine_hall/{dir_name}/mav0/state_groundtruth_estimate0/data.csv"))

        for kf_id in world_map.keyframe_ids_sorted():
            kf = world_map.keyframes[kf_id]
            center_b = camera_center(kf.pose_cw)
            center_a = (t_wa_wb[:3, :3] @ center_b) + t_wa_wb[:3, 3]
            gi = int(np.clip(np.searchsorted(gt_ts, kf.timestamp_ns), 0, len(gt_ts) - 1))
            centers.append(center_a)
            gt_matched.append(gt_xyz[gi])
        per_dataset_count[short_name] = len(world_map.keyframes)

    print(f"pooled {len(centers)} keyframes across {len(per_dataset_count)} datasets: {per_dataset_count}")
    centers = np.array(centers)
    gt_matched = np.array(gt_matched)

    r, s, t = umeyama(centers, gt_matched)
    aligned = s * (r @ centers.T).T + t
    err = np.linalg.norm(aligned - gt_matched, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))

    print(f"scale={s:.4f}")
    print_stats("ATE (m), registration-merged map, one combined alignment", err, "m")
    print(f"ATE (m): rmse={rmse:.4f}m")


if __name__ == "__main__":
    main()
