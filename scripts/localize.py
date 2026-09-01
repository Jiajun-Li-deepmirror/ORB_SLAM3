import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splg_slam.config import load_config
from splg_slam.data.euroc import load_stereo_frames, load_stereo_rig
from splg_slam.geometry.pose_utils import camera_center
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.relocalizer import Relocalizer
from splg_slam.map.io import load_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("map_path", type=str)
    parser.add_argument("--start", type=int, default=0, help="query frame index start")
    parser.add_argument("--count", type=int, default=20, help="number of query frames")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    world_map = load_map(args.map_path)
    print(f"Loaded map: {len(world_map.keyframes)} keyframes, {len(world_map.map_points)} map points")

    rig = load_stereo_rig(cfg.dataset.mav0_dir)
    rectifier = StereoRectifier(rig)
    relocalizer = Relocalizer(world_map, rectifier, cfg)

    entries = load_stereo_frames(cfg.dataset.mav0_dir)
    query_entries = entries[args.start: args.start + args.count * args.stride: args.stride]

    n_success = 0
    t0 = time.time()
    for e in query_entries:
        img_l = cv2.imread(str(e.left_path), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(e.right_path), cv2.IMREAD_GRAYSCALE)
        rect_l, _ = rectifier.rectify(img_l, img_r)

        ok, pose_cw, info = relocalizer.localize(rect_l)
        if ok:
            n_success += 1
            center = camera_center(pose_cw)
            print(
                f"frame {e.index} ts={e.timestamp_ns}: OK  inliers={info['num_inliers']} "
                f"candidate_kf={info['candidate_kf']} sim={info['retrieval_sim']:.3f} "
                f"pos=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
            )
        else:
            print(f"frame {e.index} ts={e.timestamp_ns}: FAILED ({info.get('reason')})")

    elapsed = time.time() - t0
    print(f"\n{n_success}/{len(query_entries)} localized, {elapsed / len(query_entries):.3f} s/query")


if __name__ == "__main__":
    main()
