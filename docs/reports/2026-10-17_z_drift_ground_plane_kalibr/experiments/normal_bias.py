"""Separate a camera-fixed bias of the depth floor normal from a world-fixed tilt.

Per keyframe (offline floor detection, as in evaluate.py):
  v_i = cross(up_c, n_c) in camera coords (deg)        up_c = map +Z in camera frame
Model: v_i = B + R_cw,i @ u,  B camera-fixed (depth/plane bias), u = horizontal world tilt
of the map's +Z relative to the floor (as a rotation vector, z component fixed to 0).
Only v's x (pitch) and z (roll) components carry information. Separable when the heading
changes (loops); for straight runs B is taken from the level loop and u = R^T (v - B).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/dm/slam_ws/src/splg_slam/src")
from splg_slam.data import rosbag2
from splg_slam.data.ground_depth import GroundDepthReader
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.map.io import load_map
from splg_slam.mapping.ground_plane import GroundPlaneOptions, detect_ground

R = Path(__file__).resolve().parent
OPTIONS = GroundPlaneOptions()


def observations(variant, name):
    bag = f"/home/dm/slam_ws/datasets/realsense/{name}"
    world = load_map(R / variant / name / "map.pkl")
    reader = GroundDepthReader(bag, StereoRectifier(rosbag2.load_stereo_rig(bag)), OPTIONS)
    rows = []
    try:
        for k in sorted(world.keyframes.values(), key=lambda k: k.timestamp_ns):
            pts, _ = reader.points_at(k.timestamp_ns)
            if pts is None:
                continue
            Rcw = k.pose_cw[:3, :3]
            up = Rcw[:, 2]
            obs, _ = detect_ground(pts, up, OPTIONS, seed=int(k.frame_id))
            if obs is None:
                continue
            v = np.degrees(np.cross(up, obs["normal"]))
            if np.linalg.norm(v) < 5:
                rows.append((v, Rcw))
    finally:
        reader.close()
    return rows


def fit(rows, B=None):
    """Least squares on the x/z rows; returns B (3, y set 0), u (world x, y), residual rms."""
    A, y = [], []
    for v, Rcw in rows:
        for comp in (0, 2):
            row = np.zeros(5)
            row[:2] = Rcw[comp, :2]          # world horizontal tilt u -> camera component
            if B is None:
                row[2 + (0 if comp == 0 else 1)] = 1.0
            A.append(row)
            y.append(v[comp] - (0 if B is None else B[comp]))
    A, y = np.array(A), np.array(y)
    cols = [0, 1, 2, 3] if B is None else [0, 1]
    sol, *_ = np.linalg.lstsq(A[:, cols], y, rcond=None)
    resid = y - A[:, cols] @ sol
    u = sol[:2]
    Bout = np.array([sol[2], 0.0, sol[3]]) if B is None else B
    return Bout, u, float(np.sqrt(np.mean(resid ** 2)))


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "kalibr"
    loops = ["loop_for_2", "loop_for", "loop_lat", "cali_loop_for", "turn_for"]
    lines = ["line_for_2", "line_for", "cali_line_for", "line_lat"]
    data = {n: observations(variant, n) for n in loops + lines}
    print(f"variant {variant}: joint fit per rotating sequence (B = camera-fixed normal bias, "
          f"u = world tilt of map +Z vs floor), degrees")
    biases = {}
    for n in loops:
        B, u, rms = fit(data[n])
        biases[n] = B
        print(f"  {n:14s} obs {len(data[n]):3d}  B pitch {B[0]:+.2f} roll {B[2]:+.2f} | "
              f"world tilt |u| {np.linalg.norm(u):.2f} | resid rms {rms:.2f}")
    B_level = biases["loop_for_2"]
    print(f"\nstraight runs, B fixed to loop_for_2's (level floor) {np.round(B_level[[0, 2]], 2)}:")
    for n in lines:
        _, u, rms = fit(data[n], B_level)
        raw = np.median([v for v, _ in data[n]], axis=0)
        print(f"  {n:14s} obs {len(data[n]):3d}  raw median pitch {raw[0]:+.2f} roll {raw[2]:+.2f} | "
              f"world tilt after removing B |u| {np.linalg.norm(u):.2f} | resid rms {rms:.2f}")


if __name__ == "__main__":
    main()
