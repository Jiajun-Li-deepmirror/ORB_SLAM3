"""Compare baseline vs Kalibr-extrinsic maps (no ground truth available).

Per map, from the final keyframe poses:
  z_range / z_end      camera height excursion and start->end change
  tilt                 best-fit linear z slope along the trajectory (straight runs: along-track)
  floor_vs_up pitch/roll  median angle between the depth-measured floor normal (camera frame,
                       re-detected offline at each keyframe) and the map's world +Z; this is
                       the tilt the ground-plane factor could not see
  floor_z              camera z minus measured camera height above the floor (p5-p95 range)
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "/home/dm/slam_ws/src/splg_slam/src")
from splg_slam.data import rosbag2
from splg_slam.data.ground_depth import GroundDepthReader
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.map.io import load_map
from splg_slam.mapping.ground_plane import GroundPlaneOptions, detect_ground

R = Path(__file__).resolve().parent
NAMES = ["line_for", "line_for_2", "line_lat", "cali_line_for", "turn_for",
         "loop_for", "loop_for_2", "loop_lat", "cali_loop_for"]
OPTIONS = GroundPlaneOptions()
VARIANTS = ("baseline", "kalibr", "kalibr_ground", "kalibr_ground_n05", "kalibr_ground_n05_d02", "kalibr_ground_n05_bias", "kalibr_tol015", "kalibr_ground_n05_bias_tol015")


def p595(v):
    return float(np.percentile(v, 95) - np.percentile(v, 5))


def evaluate(map_path, bag):
    world = load_map(map_path)
    kfs = sorted(world.keyframes.values(), key=lambda k: k.timestamp_ns)
    pos = np.array([k.pose_wc()[:3, 3] for k in kfs])
    p = pos - pos[0]
    xy0 = p[:, :2] - p[:, :2].mean(axis=0)
    _, sv, axes = np.linalg.svd(xy0, full_matrices=False)
    s = xy0 @ axes.T
    if sv[1] < 0.1 * sv[0]:
        s = s[:, :1]
    A = np.c_[s, np.ones(len(p))]
    c, *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
    out = dict(keyframes=len(kfs), segments=len({k.segment_id for k in kfs}),
               z_range=float(np.ptp(p[:, 2])), z_end=float(p[-1, 2]),
               tilt_deg=float(np.degrees(np.arctan(np.linalg.norm(c[:-1])))),
               z_resid_range=float(np.ptp(p[:, 2] - A @ c)))
    reader = GroundDepthReader(bag, StereoRectifier(rosbag2.load_stereo_rig(bag)), OPTIONS)
    rows = []
    try:
        for k, cam in zip(kfs, pos):
            points, _ = reader.points_at(k.timestamp_ns)
            if points is None:
                continue
            up = k.pose_cw[:3, :3] @ np.array([0.0, 0.0, 1.0])
            obs, _ = detect_ground(points, up, OPTIONS, seed=int(k.frame_id))
            if obs is None:
                continue
            v = np.degrees(np.cross(up, obs["normal"]))
            if np.linalg.norm(v) < 5:  # drop wrong planes (ramps, tabletops)
                rows.append((v[0], v[2], cam[2] - obs["height_m"]))
    finally:
        reader.close()
    rows = np.array(rows)
    out.update(floor_obs=len(rows), floor_vs_up_pitch=float(np.median(rows[:, 0])),
               floor_vs_up_roll=float(np.median(rows[:, 1])), floor_z_range=p595(rows[:, 2]))
    return out


def main():
    results = {}
    path = R / "metrics.json"
    if path.exists():
        results = json.loads(path.read_text())
    for name in NAMES:
        for variant in VARIANTS:
            key = f"{variant}/{name}"
            m = R / variant / name / "map.pkl"
            if key in results or not m.exists():
                continue
            results[key] = evaluate(m, f"/home/dm/slam_ws/datasets/realsense/{name}")
            path.write_text(json.dumps(results, indent=1))
            print(key, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in results[key].items()}),
                  flush=True)
    print(f"\n{'dataset':14s} {'variant':13s} {'z_range':>7s} {'z_end':>7s} {'tilt':>6s} "
          f"{'floor-up pitch/roll':>20s} {'floor_z':>7s} {'segs':>4s}")
    for name in NAMES:
        for variant in VARIANTS:
            r = results.get(f"{variant}/{name}")
            if r:
                print(f"{name:14s} {variant:13s} {r['z_range']:7.3f} {r['z_end']:+7.3f} {r['tilt_deg']:5.2f}° "
                      f"{r['floor_vs_up_pitch']:+9.2f}° {r['floor_vs_up_roll']:+7.2f}° "
                      f"{r['floor_z_range']:7.3f} {r['segments']:4d}")


if __name__ == "__main__":
    main()
