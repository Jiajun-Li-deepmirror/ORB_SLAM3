"""Convert a ROS2 rosbag2 recording (mcap or db3) of a stereo + IMU rig into the EuRoC
mav0/ directory layout that splg_slam.data.euroc already knows how to load.

Handles two rig flavors seen so far, auto-detected per camera/IMU:
 - raw fisheye ('equidistant' distortion model) infra1/infra2 + a combined
   /.../imu topic (custom rig) - each camera is undistorted here (in its own
   original frame, R=identity, so T_BS stays correct) into a zero-distortion
   pinhole image; stereo epipolar rectification itself is still done downstream
   by StereoRectifier.
 - already-rectified infra1/infra2 image_rect_raw (D=0, standard RealSense
   driver output) + separate gyro/sample and accel/sample topics - images are
   copied through as-is and the two IMU streams are merged by nearest-neighbor
   timestamp matching.

Camera/IMU extrinsics (T_BS, sensor -> body) are derived by BFS-composing
/tf_static edges from each sensor's optical frame to a chosen body frame (the
first IMU-like topic's frame_id, so EuRoC's "IMU defines body" convention
holds) - this works for both direct edges and multi-hop chains (e.g. through
an intermediate depth-aligned frame).

Also converts a TUM ground-truth trajectory (t x y z qx qy qz qw, t in
fractional seconds) into the EuRoC-style GT csv (timestamp_ns,x,y,z)
eval_trajectory.py expects.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splg_slam.data.rosbag_common import (  # noqa: E402
    DEPTH_TOPIC,
    IMU_ACCEL_NOISE_DENSITY,
    IMU_ACCEL_RANDOM_WALK,
    IMU_COMBINED,
    IMU_GYRO_NOISE_DENSITY,
    IMU_GYRO_RANDOM_WALK,
    INFRA1_INFO,
    INFRA1_RAW,
    INFRA1_RECT,
    INFRA2_INFO,
    INFRA2_RAW,
    INFRA2_RECT,
    bfs_transform,
    decode_mono8,
    first_message,
    imu_rate_hz,
    read_imu_combined_samples,
    read_imu_split_samples,
    read_tf_static,
    resolve_camera,
    stamp_to_ns,
)


def write_cam_sensor_yaml(path: Path, t_bs: np.ndarray, k_new: np.ndarray, size: tuple[int, int], rate_hz: float, name: str, comment: str):
    doc = {
        "sensor_type": "camera",
        "comment": comment,
        "T_BS": {"cols": 4, "rows": 4, "data": [float(x) for x in t_bs.flatten()]},
        "rate_hz": rate_hz,
        "resolution": list(size),
        "camera_model": "pinhole",
        "intrinsics": [float(k_new[0, 0]), float(k_new[1, 1]), float(k_new[0, 2]), float(k_new[1, 2])],
        "distortion_model": "radial-tangential",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
    }
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def convert_camera(reader, mav0_dir: Path, cam_name: str, raw_topic: str, rect_topic: str, info_topic: str, tf: dict, body_frame: str, balance: float):
    src = resolve_camera(reader, raw_topic, rect_topic, info_topic, balance)
    t_bs = bfs_transform(tf, body_frame, src.frame_id)
    print(f"{cam_name}: topic={src.image_topic} frame={src.frame_id} {src.width}x{src.height} distortion_model={src.dist_model} D={src.dist_coeffs.ravel().tolist()}")

    if src.map1 is None:
        comment = f"real rig {cam_name}, already rectified by the camera driver (image_rect_raw)"
    elif src.dist_model == "equidistant":
        comment = f"real rig {cam_name}, fisheye-undistorted to pinhole at conversion time"
    else:
        comment = f"real rig {cam_name}, radial-tangential-undistorted at conversion time"

    cam_dir = mav0_dir / cam_name
    data_dir = cam_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    conns = [c for c in reader.connections if c.topic == src.image_topic]
    rows = []
    for conn, _, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        img = decode_mono8(msg)
        out = img if src.map1 is None else cv2.remap(img, src.map1, src.map2, cv2.INTER_LINEAR)
        ts_ns = stamp_to_ns(msg.header.stamp)
        fname = f"{ts_ns}.png"
        cv2.imwrite(str(data_dir / fname), out)
        rows.append((ts_ns, fname))
    rows.sort(key=lambda r: r[0])

    with open(cam_dir / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#timestamp [ns]", "filename"])
        w.writerows(rows)

    rate_hz = (len(rows) - 1) / ((rows[-1][0] - rows[0][0]) * 1e-9) if len(rows) > 1 else 20.0
    write_cam_sensor_yaml(cam_dir / "sensor.yaml", t_bs, src.k_new, (src.width, src.height), round(rate_hz, 3), cam_name, comment)
    print(f"{cam_name}: wrote {len(rows)} frames, rate~{rate_hz:.2f} Hz, K=\n{src.k_new}")


def convert_imu_combined(reader, mav0_dir: Path) -> str:
    """Returns the IMU frame_id used, so callers can pick it as the body frame."""
    frame_id, ts, values = read_imu_combined_samples(reader)
    rows = [(int(ts[i]), *values[i]) for i in range(len(ts))]
    _write_imu(mav0_dir, rows, ts, "noise-density fields are typical D4xx-class MEMS placeholders, not calibrated from this rig")
    return frame_id


def convert_imu_split(reader, mav0_dir: Path) -> str:
    """Merges separate gyro/sample + accel/sample topics (RealSense default when
    unite_imu_method is off) by nearest-neighbor timestamp matching onto the gyro
    (higher-rate) stream. Returns the gyro frame_id, used as the body frame."""
    frame_id, ts, values = read_imu_split_samples(reader)
    rows = [(int(ts[i]), *values[i]) for i in range(len(ts))]
    _write_imu(mav0_dir, rows, ts, "gyro+accel merged by nearest-neighbor timestamp match; noise-density fields are typical D4xx-class MEMS placeholders, not calibrated from this rig")
    return frame_id


def _write_imu(mav0_dir: Path, rows: list[tuple], ts: np.ndarray, comment: str):
    imu_dir = mav0_dir / "imu0"
    imu_dir.mkdir(parents=True, exist_ok=True)
    with open(imu_dir / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "#timestamp [ns]",
                "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]",
                "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]",
            ]
        )
        w.writerows(rows)

    rate_hz = imu_rate_hz(ts)
    doc = {
        "sensor_type": "imu",
        "comment": comment,
        "T_BS": {"cols": 4, "rows": 4, "data": [float(x) for x in np.eye(4).flatten()]},
        "rate_hz": round(rate_hz, 3),
        "gyroscope_noise_density": IMU_GYRO_NOISE_DENSITY,
        "gyroscope_random_walk": IMU_GYRO_RANDOM_WALK,
        "accelerometer_noise_density": IMU_ACCEL_NOISE_DENSITY,
        "accelerometer_random_walk": IMU_ACCEL_RANDOM_WALK,
    }
    with open(imu_dir / "sensor.yaml", "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    print(f"imu0: wrote {len(rows)} samples, rate~{rate_hz:.2f} Hz")


def convert_depth(reader, mav0_dir: Path):
    """Optional: an onboard depth stream (e.g. a RealSense's own depth ASIC output),
    already registered to cam0's raw/unrectified pixel grid (same K, same resolution -
    verified against the driver's own camera_info) - written as 16-bit mm PNGs so
    splg_slam.data.euroc.load_depth_lookup / OfflineMapper's sensor-depth path can sample
    it directly, sidestepping our own stereo-baseline calibration for depth entirely."""
    msg0 = first_message(reader, DEPTH_TOPIC)
    if msg0 is None:
        return
    depth_dir = mav0_dir / "depth0"
    data_dir = depth_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    conns = [c for c in reader.connections if c.topic == DEPTH_TOPIC]
    rows = []
    for conn, _, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        ts_ns = stamp_to_ns(msg.header.stamp)
        fname = f"{ts_ns}.png"
        cv2.imwrite(str(data_dir / fname), depth_mm)
        rows.append((ts_ns, fname))
    rows.sort(key=lambda r: r[0])

    with open(depth_dir / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#timestamp [ns]", "filename"])
        w.writerows(rows)
    print(f"depth0: wrote {len(rows)} frames (16-bit mm, on cam0's raw/unrectified pixel grid, frame={msg0.header.frame_id})")


def convert_gt_tum(tum_path: Path, out_csv: Path):
    rows = []
    with open(tum_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            t = float(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            rows.append((int(round(t * 1e9)), x, y, z))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#timestamp [ns]", "x", "y", "z"])
        w.writerows(rows)
    print(f"gt: wrote {len(rows)} poses -> {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_bag", type=str, help="path to the rosbag2_* directory (contains metadata.yaml + .mcap/.db3)")
    parser.add_argument("output", type=str, help="output mav0/ directory")
    parser.add_argument("--gt_tum", type=str, default=None, help="optional TUM ground-truth file to convert alongside")
    parser.add_argument("--gt_out", type=str, default=None, help="output path for the converted GT csv (default: alongside output)")
    parser.add_argument("--balance", type=float, default=0.0, help="cv2.fisheye undistort balance: 0=crop to valid pixels, 1=keep full FOV with black borders")
    args = parser.parse_args()

    mav0_dir = Path(args.output)
    mav0_dir.mkdir(parents=True, exist_ok=True)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([Path(args.camera_bag)], default_typestore=typestore) as reader:
        tf = read_tf_static(reader)

        if first_message(reader, IMU_COMBINED) is not None:
            body_frame = convert_imu_combined(reader, mav0_dir)
        else:
            body_frame = convert_imu_split(reader, mav0_dir)
        print(f"body frame (from IMU): {body_frame}")

        convert_camera(reader, mav0_dir, "cam0", INFRA1_RAW, INFRA1_RECT, INFRA1_INFO, tf, body_frame, args.balance)
        convert_camera(reader, mav0_dir, "cam1", INFRA2_RAW, INFRA2_RECT, INFRA2_INFO, tf, body_frame, args.balance)
        convert_depth(reader, mav0_dir)

    if args.gt_tum:
        gt_out = Path(args.gt_out) if args.gt_out else mav0_dir.parent / "gt.csv"
        convert_gt_tum(Path(args.gt_tum), gt_out)


if __name__ == "__main__":
    main()
