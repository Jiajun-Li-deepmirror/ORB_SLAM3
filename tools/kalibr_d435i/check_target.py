"""Check a D435i calibration recording before running Kalibr.

1. Target config: tag IDs, row/column layout and tag spacing ratio vs aprilgrid.yaml.
2. Recording quality: per 3 s block, how many tags are detected and how fast the camera
   rotates. Blocks that rotate but detect nothing are motion-blurred: Kalibr cannot use
   them, and a long gap breaks its continuous-time trajectory.

Detection uses the `aprilgrid` package (a detector for Kalibr-style AprilGrids; OpenCV's
aruco detector fails on the black corner squares) inside a docker image
(APRILGRID_IMAGE, default calibr-looper:humble).
Absolute tagSize cannot be checked from images; kalibr_to_splg.py compares the stereo
baseline afterwards.

Usage: .venv/bin/python check_target.py <rosbag2_dir> [aprilgrid.yaml] [--work DIR]
Exit code: 0 ok, 1 target config mismatch, 2 recording quality insufficient.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from splg_slam.data import rosbag2

# Any image with python3 + opencv + the `aprilgrid` pip package.
DETECTOR_IMAGE = os.environ.get("APRILGRID_IMAGE", "calibr-looper:humble")
DETECT_SCRIPT = """
import glob, json, sys
import cv2, numpy as np
from aprilgrid import Detector
det = Detector("t36h11")
out = {}
for f in sorted(glob.glob(sys.argv[1] + "/*.png")):
    tags = det.detect(cv2.imread(f, 0))
    out[f.split("/")[-1]] = {int(d.tag_id): np.asarray(d.corners).reshape(4, 2).tolist() for d in tags}
json.dump(out, open(sys.argv[2], "w"))
"""
BLOCK_S = 3.0


def grid_fit(tags, cols):
    """Homography from grid coordinates (units of tag pitch) to tag centres; returns
    (median residual px, tag edge / pitch ratio) for this column count."""
    ids = np.array(sorted(tags))
    grid = np.column_stack([ids % cols, ids // cols]).astype(np.float64)
    centres = np.array([tags[i].mean(axis=0) for i in ids])
    H, _ = cv2.findHomography(grid, centres, 0)
    if H is None or abs(np.linalg.det(H)) < 1e-12:  # wrong column counts can be degenerate
        return np.inf, np.nan
    proj = cv2.perspectiveTransform(grid[None], H)[0]
    resid = float(np.median(np.linalg.norm(proj - centres, axis=1)))
    back = cv2.perspectiveTransform(np.concatenate([tags[i] for i in ids])[None], np.linalg.inv(H))[0]
    back = back.reshape(-1, 4, 2)
    edge = np.linalg.norm(np.diff(np.concatenate([back, back[:, :1]], axis=1), axis=1), axis=2)
    return resid, float(np.median(edge))


def detect_frames(bag_dir, work, every):
    frame_dir = work / "check_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    stamps = {}
    for k, e in enumerate(rosbag2.load_stereo_frames(bag_dir)):
        if k % every == 0:
            name = f"{k:05d}.png"
            cv2.imwrite(str(frame_dir / name), e.left_image)
            stamps[name] = e.timestamp_ns
    (work / "detect.py").write_text(DETECT_SCRIPT)
    subprocess.run(["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
                    "-v", f"{work}:/w", "--entrypoint", "python3", DETECTOR_IMAGE,
                    "/w/detect.py", "/w/check_frames", "/w/detections.json"], check=True)
    raw = json.loads((work / "detections.json").read_text())
    return [(stamps[n], {int(i): np.array(c) for i, c in raw[n].items()}) for n in sorted(raw)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_dir")
    ap.add_argument("target", nargs="?", default=str(Path(__file__).with_name("aprilgrid.yaml")))
    ap.add_argument("--work", help="scratch dir for sampled frames (default: a temp dir)")
    ap.add_argument("--every", type=int, default=3, help="use every Nth stereo frame")
    args = ap.parse_args()
    target = yaml.safe_load(Path(args.target).read_text())
    rows, cols = int(target["tagRows"]), int(target["tagCols"])
    work = Path(args.work or tempfile.mkdtemp(prefix="check_target_")).resolve()

    frames = detect_frames(args.bag_dir, work, args.every)
    t0 = frames[0][0]
    rich = [d for _, d in frames if len(d) >= 8]
    seen = sorted({i for _, d in frames for i in d})
    print(f"{len(frames)} sampled frames, {sum(bool(d) for _, d in frames)} with tags, "
          f"{len(rich)} with >=8 tags; IDs {seen[0] if seen else '-'}..{seen[-1] if seen else '-'} "
          f"({len(seen)} distinct)")
    if not rich:
        print("FAIL: board (almost) never detected")
        return 2

    # --- target configuration ---------------------------------------------------------
    config_ok = True
    if seen[-1] >= rows * cols:
        print(f"FAIL: tag ID {seen[-1]} does not fit the configured {rows}x{cols} grid")
        config_ok = False
    scores = {c: np.median([grid_fit(d, c)[0] for d in rich]) for c in range(2, 13) if seen[-1] < c * 12}
    best = min(scores, key=scores.get)
    print("grid residual by column count: " + ", ".join(f"{c}:{scores[c]:.1f}px" for c in sorted(scores)))
    if best != cols:
        print(f"FAIL: ID layout fits {best} columns, aprilgrid.yaml says tagCols={cols}")
        config_ok = False
    px = [np.median([np.linalg.norm(np.diff(c, axis=0, append=c[:1]), axis=1).mean() for c in d.values()])
          for d in rich]
    big = [d for d, e in zip(rich, px) if e >= 40] or rich  # corner bias ~1 px: prefer large tags
    spacing = 1.0 / float(np.median([grid_fit(d, best)[1] for d in big])) - 1.0
    print(f"tagSpacing measured {spacing:.3f}, configured {target['tagSpacing']}")
    if abs(spacing - float(target["tagSpacing"])) > 0.06:
        print("FAIL: measured tagSpacing differs from aprilgrid.yaml")
        config_ok = False

    # --- recording quality --------------------------------------------------------------
    imu = rosbag2.load_imu_measurements(args.bag_dir)
    t_imu = (imu[:, 0] - t0) * 1e-9
    t_img = np.array([(ts - t0) * 1e-9 for ts, _ in frames])
    n_tags = np.array([len(d) for _, d in frames])
    print(f"\n{'block s':>9s} {'tags/frame':>10s} {'frames>=8 tags':>14s} {'gyro rms x/y/z deg/s':>22s}")
    usable_s, blind_s, longest_gap, gap = 0.0, 0.0, 0.0, 0.0
    for start in np.arange(0.0, t_img[-1] + 1e-9, BLOCK_S):
        m = (t_img >= start) & (t_img < start + BLOCK_S)
        g = (t_imu >= start) & (t_imu < start + BLOCK_S)
        if not m.any():
            continue
        rate = np.degrees(np.sqrt((imu[g, 1:4] ** 2).mean(axis=0))) if g.any() else np.zeros(3)
        good = float((n_tags[m] >= 8).mean())
        moving = rate.max() > 5.0
        note = ""
        if good >= 0.5:
            gap = 0.0
            if moving:
                usable_s += BLOCK_S
        else:
            gap += BLOCK_S
            longest_gap = max(longest_gap, gap)
            if moving:
                blind_s += BLOCK_S
                note = "  <- moving but board not detected (blur / too fast / out of view)"
        print(f"{start:4.0f}-{start + BLOCK_S:<4.0f} {np.median(n_tags[m]):10.0f} {good:14.0%} "
              f"{np.array2string(rate, precision=0, floatmode='fixed'):>22s}{note}")
    rate_all = np.degrees(np.sqrt((imu[:, 1:4] ** 2).mean(axis=0)))
    print(f"\nmoving time with board visible: {usable_s:.0f} s; moving but blind: {blind_s:.0f} s; "
          f"longest gap without detections: {longest_gap:.0f} s; overall gyro rms {np.round(rate_all, 0)} deg/s")
    quality_ok = usable_s >= 30 and longest_gap <= 3
    if not quality_ok:
        print("FAIL: need >= 30 s of motion with the board detected and no detection gap > 3 s. "
              "Move slower (about 30 deg/s), shorten exposure / add light, keep the board in view.")
    if config_ok and quality_ok:
        print("OK: target config consistent and recording usable")
    return 0 if config_ok and quality_ok else (1 if not config_ok else 2)


if __name__ == "__main__":
    sys.exit(main())
