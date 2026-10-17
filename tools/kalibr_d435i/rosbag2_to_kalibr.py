"""Convert a RealSense D435i rosbag2 recording into a ROS1 bag that Kalibr can read.

Output topics:
  /cam0/image_raw, /cam1/image_raw  infra1/infra2 (mono8), only exact-timestamp stereo pairs
  /imu0                             combined sensor_msgs/Imu on the gyro clock; accel is
                                    linearly interpolated (not nearest-neighbour) onto it

The IMU stream goes through the same per-unit intrinsics splg_slam applies at load time
(apply_imu_intrinsics) plus the configured gyro scale, so Kalibr calibrates the extrinsic
for the IMU data SLAM actually consumes. No time offset is applied: Kalibr estimates it.

Also writes <out>-camchain-factory.yaml from the bag's own CameraInfo/tf, a fallback for
kalibr_calibrate_imu_camera if the camera-only calibration step is skipped.

Usage:
  .venv/bin/python rosbag2_to_kalibr.py <rosbag2_dir> <out.bag> [--gyro-scale 1.01]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from splg_slam.data import rosbag2
from splg_slam.data.rosbag_common import ACCEL_TOPIC, GYRO_TOPIC, apply_imu_intrinsics, stamp_to_ns

STORE = get_typestore(Stores.ROS1_NOETIC)
T = STORE.types
IMAGE, IMU = "sensor_msgs/msg/Image", "sensor_msgs/msg/Imu"


def header(seq, ns, frame_id):
    stamp = T["builtin_interfaces/msg/Time"](sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))
    return T["std_msgs/msg/Header"](seq=seq, stamp=stamp, frame_id=frame_id)


def image_msg(seq, ns, img, frame_id):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return T[IMAGE](header=header(seq, ns, frame_id), height=img.shape[0], width=img.shape[1],
                    encoding="mono8", is_bigendian=0, step=img.shape[1], data=img.reshape(-1))


def read_imu(bag_dir, gyro_scale):
    with rosbag2._open(bag_dir) as reader:
        streams = {}
        for topic in (GYRO_TOPIC, ACCEL_TOPIC):
            conns = [c for c in reader.connections if c.topic == topic]
            if not conns:
                raise ValueError(f"{bag_dir} has no {topic}; record with unite_imu_method:=0")
            rows, frame_id = [], None
            for conn, _, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                v = msg.angular_velocity if topic == GYRO_TOPIC else msg.linear_acceleration
                rows.append((stamp_to_ns(msg.header.stamp), v.x, v.y, v.z))
                frame_id = msg.header.frame_id
            rows.sort(key=lambda r: r[0])
            ts = np.array([r[0] for r in rows], dtype=np.int64)
            keep = np.r_[True, np.diff(ts) > 0]  # drop duplicate stamps before interpolating
            streams[topic] = (ts[keep], np.array([r[1:] for r in rows])[keep], frame_id)
    g_ts, g_val, frame_id = streams[GYRO_TOPIC]
    a_ts, a_val, _ = streams[ACCEL_TOPIC]
    inside = (g_ts >= a_ts[0]) & (g_ts <= a_ts[-1])
    # Gyro samples falling in an accel dropout would get a fabricated accel value.
    gap = np.diff(a_ts)
    k = np.clip(np.searchsorted(a_ts, g_ts) - 1, 0, len(gap) - 1)
    inside &= gap[k] <= 5 * np.median(gap)
    g_ts, g_val = g_ts[inside], g_val[inside]
    t_rel, a_rel = (g_ts - a_ts[0]).astype(np.float64), (a_ts - a_ts[0]).astype(np.float64)
    accel = np.column_stack([np.interp(t_rel, a_rel, a_val[:, i]) for i in range(3)])
    values = np.concatenate([g_val, accel], axis=1)
    apply_imu_intrinsics(values)
    values[:, 0:3] *= gyro_scale
    dropped = int((~inside).sum())
    return g_ts, values, frame_id, dropped


def factory_camchain(bag_dir, path):
    rig = rosbag2.load_stereo_rig(bag_dir)
    cams = {}
    for name, cam in (("cam0", rig.cam0), ("cam1", rig.cam1)):
        k = cam.K
        cams[name] = {
            "camera_model": "pinhole",
            "intrinsics": [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])],
            "distortion_model": "radtan",
            "distortion_coeffs": [float(x) for x in np.r_[np.ravel(cam.dist_coeffs), np.zeros(4)][:4]],
            "resolution": [int(cam.width), int(cam.height)],
            "rostopic": f"/{name}/image_raw",
        }
    cams["cam1"]["T_cn_cnm1"] = np.asarray(rig.T_cam1_cam0, dtype=float).tolist()
    cams["cam1"]["cam_overlaps"] = [0]
    cams["cam0"]["cam_overlaps"] = [1]
    Path(path).write_text(yaml.safe_dump(cams, sort_keys=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_dir")
    ap.add_argument("out_bag")
    ap.add_argument("--gyro-scale", type=float, default=1.01,
                    help="same factor as configs' imu.gyro_scale (default 1.01)")
    ap.add_argument("--max-seconds", type=float, default=None, help="truncate (for smoke tests)")
    args = ap.parse_args()
    out = Path(args.out_bag)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")

    imu_ts, imu_val, imu_frame, dropped = read_imu(args.bag_dir, args.gyro_scale)
    t0 = t_end = None
    n_img = n_imu = 0
    zero9 = np.zeros(9)
    cov_unknown = np.r_[-1.0, np.zeros(8)]  # orientation not provided
    with Writer(out) as writer:
        c0 = writer.add_connection("/cam0/image_raw", IMAGE, typestore=STORE)
        c1 = writer.add_connection("/cam1/image_raw", IMAGE, typestore=STORE)
        ci = writer.add_connection("/imu0", IMU, typestore=STORE)
        i = 0

        def write_imu_until(limit_ns):
            nonlocal i, n_imu
            while i < len(imu_ts) and imu_ts[i] <= limit_ns:
                ns, v = int(imu_ts[i]), imu_val[i]
                if ns >= t0 - 1_000_000_000:  # keep 1 s of IMU before the first image
                    msg = T[IMU](header=header(n_imu, ns, imu_frame),
                                 orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
                                 orientation_covariance=cov_unknown,
                                 angular_velocity=T["geometry_msgs/msg/Vector3"](x=v[0], y=v[1], z=v[2]),
                                 angular_velocity_covariance=zero9,
                                 linear_acceleration=T["geometry_msgs/msg/Vector3"](x=v[3], y=v[4], z=v[5]),
                                 linear_acceleration_covariance=zero9)
                    writer.write(ci, ns, STORE.serialize_ros1(msg, IMU))
                    n_imu += 1
                i += 1

        # Messages are written in global timestamp order (images and IMU interleaved).
        for entry in rosbag2.load_stereo_frames(args.bag_dir):
            ns = entry.timestamp_ns
            t0 = ns if t0 is None else t0
            if args.max_seconds is not None and ns - t0 > args.max_seconds * 1e9:
                break
            write_imu_until(ns)
            for conn, img, frame in ((c0, entry.left_image, "cam0"), (c1, entry.right_image, "cam1")):
                writer.write(conn, ns, STORE.serialize_ros1(image_msg(n_img, ns, img, frame), IMAGE))
            n_img += 1
            t_end = ns
        if t0 is None:
            raise SystemExit("no stereo image pairs found")
        write_imu_until(t_end + 1_000_000_000)
    factory_camchain(args.bag_dir, out.with_name(out.stem + "-camchain-factory.yaml"))
    span = (min(imu_ts[-1], t_end + 1e9) - max(imu_ts[0], t0 - 1e9)) * 1e-9
    rate = (n_imu - 1) / span if n_imu > 1 else 0
    print(f"wrote {out}: {n_img} stereo pairs over {(t_end - t0) * 1e-9:.1f}s, {n_imu} IMU samples "
          f"(~{rate:.0f} Hz, frame '{imu_frame}', {dropped} gyro samples outside accel coverage dropped), "
          f"gyro_scale={args.gyro_scale}")


if __name__ == "__main__":
    main()
