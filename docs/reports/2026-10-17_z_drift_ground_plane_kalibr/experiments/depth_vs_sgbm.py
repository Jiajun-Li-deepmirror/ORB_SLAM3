"""Floor normal from the D435's own depth vs from SGBM on the same rectified IR pair, at the
keyframes of a Kalibr-extrinsic map, plus the disparity offset that would level each one.
Angles: cross(map +Z in camera, floor normal), camera x = pitch, z = roll, degrees."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/dm/slam_ws/src/splg_slam/src")
from splg_slam.data import rosbag2
from splg_slam.data.ground_depth import GroundDepthReader
from splg_slam.geometry.stereo import StereoDepthEstimator, StereoRectifier
from splg_slam.map.io import load_map
from splg_slam.mapping.ground_plane import GroundPlaneOptions, detect_ground

R = Path(__file__).resolve().parent
OPT = GroundPlaneOptions()
OFFSETS = np.round(np.arange(-0.40, 0.401, 0.05), 3)


def with_disparity_offset(points, fb, dd):
    z = points[:, 2]
    z2 = fb / (fb / z + dd)
    return points * (z2 / z)[:, None]


def angle(up, pts, seed):
    obs, _ = detect_ground(pts, up, OPT, seed=seed)
    if obs is None:
        return None
    v = np.degrees(np.cross(up, obs["normal"]))
    return v if np.linalg.norm(v) < 5 else None


def main(name, variant="kalibr", every=2):
    bag = f"/home/dm/slam_ws/datasets/realsense/{name}"
    rect = StereoRectifier(rosbag2.load_stereo_rig(bag))
    fb = rect.fx_rect * rect.baseline
    sgbm = StereoDepthEstimator(rect)
    kfs = sorted(load_map(R / variant / name / "map.pkl").keyframes.values(), key=lambda k: k.timestamp_ns)
    by_ts = {k.timestamp_ns: k for k in kfs[::every]}
    reader = GroundDepthReader(bag, rect, OPT)
    rs, sg = {d: [] for d in OFFSETS}, {d: [] for d in OFFSETS}
    pairs = []
    h, w = None, None
    try:
        for e in rosbag2.load_stereo_frames(bag):
            k = by_ts.get(e.timestamp_ns)
            if k is None:
                continue
            up = k.pose_cw[:3, :3][:, 2]
            seed = int(k.frame_id)
            pts_rs, _ = reader.points_at(e.timestamp_ns)
            # SGBM points on the same ROI / stride / depth limits as GroundDepthReader
            disp = sgbm.compute_disparity(*rect.rectify(e.left_image, e.right_image))
            h, w = disp.shape
            yy, xx = np.mgrid[int(h * OPT.roi_top):h:OPT.pixel_stride, 0:w:OPT.pixel_stride]
            d = disp[yy, xx]
            ok = d > 0.5
            z = fb / d[ok]
            keep = (z >= OPT.min_depth_m) & (z <= OPT.max_depth_m)
            xs, ys, z = xx[ok][keep], yy[ok][keep], z[keep]
            pts_sg = np.column_stack(((xs - rect.cx_rect) * z / rect.fx_rect, (ys - rect.cy_rect) * z / rect.fy_rect, z))
            a_rs = angle(up, pts_rs, seed) if pts_rs is not None else None
            a_sg = angle(up, pts_sg, seed) if len(pts_sg) > OPT.min_inliers else None
            if a_rs is not None and a_sg is not None:
                pairs.append((a_rs, a_sg))
            for dd in OFFSETS:
                if pts_rs is not None:
                    a = angle(up, with_disparity_offset(pts_rs, fb, dd), seed)
                    if a is not None:
                        rs[dd].append(a)
                a = angle(up, with_disparity_offset(pts_sg, fb, dd), seed) if len(pts_sg) > OPT.min_inliers else None
                if a is not None:
                    sg[dd].append(a)
    finally:
        reader.close()
    p = np.array(pairs)
    print(f"{name} ({variant}), {len(p)} keyframes with both floors:")
    print(f"  D435 depth  floor-vs-up median pitch {np.median(p[:, 0, 0]):+.2f} roll {np.median(p[:, 0, 2]):+.2f}")
    print(f"  SGBM        floor-vs-up median pitch {np.median(p[:, 1, 0]):+.2f} roll {np.median(p[:, 1, 2]):+.2f}")
    print(f"  per-frame D435 - SGBM pitch: median {np.median(p[:, 0, 0] - p[:, 1, 0]):+.2f}, "
          f"std {np.std(p[:, 0, 0] - p[:, 1, 0]):.2f}")
    print("  disparity offset dd [px] -> median pitch (D435 | SGBM):")
    print("   " + "  ".join(f"{dd:+.2f}:{np.median([a[0] for a in rs[dd]]):+.2f}|{np.median([a[0] for a in sg[dd]]):+.2f}"
                           for dd in OFFSETS[::2]))


if __name__ == "__main__":
    main(sys.argv[1])
