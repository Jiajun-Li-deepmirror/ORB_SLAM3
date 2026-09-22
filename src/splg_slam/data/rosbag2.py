"""Reads a rosbag2_* recording (mcap or db3) directly at build_map.py runtime - same
load_stereo_rig/load_stereo_frames/load_mono_frames/load_imu_calibration/
load_imu_measurements/load_depth_lookup interface as splg_slam.data.euroc, so
data.loader's dataset_module(cfg) dispatch treats it as just another dataset kind. No
EuRoC-layout copy is ever written to disk.

load_stereo_frames/load_mono_frames return **generators**, not lists: only one decoded
frame (plus a small L/R timestamp-matching buffer) is ever held in memory at a time,
unlike materializing every frame as a PNG (or in-memory array) up front. All the
rig-detection/undistortion/IMU-merge logic is shared with scripts/rosbag_to_euroc.py via
splg_slam.data.rosbag_common - this module never re-derives it, so the copy path and this
direct-read path can't silently diverge.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from splg_slam.data.euroc import ImuCalibration, StereoRig
from splg_slam.data.rosbag_common import (
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
    GYRO_TOPIC,
    bfs_transform,
    decode_mono8,
    first_message,
    read_imu_samples,
    read_tf_static,
    resolve_camera,
    stamp_to_ns,
)
from splg_slam.geometry.camera import PinholeCamera


def _open(bag_dir):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    return AnyReader([Path(bag_dir)], default_typestore=typestore)


def _imu_frame_id(reader) -> str:
    msg = first_message(reader, IMU_COMBINED)
    if msg is not None:
        return msg.header.frame_id
    return first_message(reader, GYRO_TOPIC).header.frame_id


def load_stereo_rig(bag_dir, tf_static_from=None) -> StereoRig:
    """tf_static_from: optional path to another rosbag2 recording to read /tf_static from
    instead of bag_dir - for a bag that was recorded without /tf_static (e.g. a topic-list
    slip at record time), reuse a known-good recording of the *same physical rig*, since
    camera/IMU extrinsics are a fixed hardware property, not something that changes
    per-recording. Verify same-rig-ness first (matching frame_ids/intrinsics) before
    relying on this."""
    if tf_static_from:
        with _open(tf_static_from) as tf_reader:
            tf = read_tf_static(tf_reader)
    with _open(bag_dir) as reader:
        if not tf_static_from:
            tf = read_tf_static(reader)
        body_frame = _imu_frame_id(reader)
        src0 = resolve_camera(reader, INFRA1_RAW, INFRA1_RECT, INFRA1_INFO, balance=0.0)
        src1 = resolve_camera(reader, INFRA2_RAW, INFRA2_RECT, INFRA2_INFO, balance=0.0)
        t_bs0 = bfs_transform(tf, body_frame, src0.frame_id)
        t_bs1 = bfs_transform(tf, body_frame, src1.frame_id)

    cam0 = PinholeCamera(
        fx=src0.k_new[0, 0], fy=src0.k_new[1, 1], cx=src0.k_new[0, 2], cy=src0.k_new[1, 2],
        dist_coeffs=np.zeros(4), width=src0.width, height=src0.height,
    )
    cam1 = PinholeCamera(
        fx=src1.k_new[0, 0], fy=src1.k_new[1, 1], cx=src1.k_new[0, 2], cy=src1.k_new[1, 2],
        dist_coeffs=np.zeros(4), width=src1.width, height=src1.height,
    )
    t_cam1_body = np.linalg.inv(t_bs1)
    t_cam1_cam0 = t_cam1_body @ t_bs0
    return StereoRig(cam0=cam0, cam1=cam1, T_cam1_cam0=t_cam1_cam0)


@dataclass
class RosbagMonoFrameEntry:
    index: int
    timestamp_ns: int
    left_image: np.ndarray


@dataclass
class RosbagStereoFrameEntry:
    index: int
    timestamp_ns: int
    left_image: np.ndarray
    right_image: np.ndarray


def load_mono_frames(bag_dir):
    """Generator over cam0 only - mirrors euroc.load_mono_frames's no-cam1-dependency
    behavior. bag_dir stays open for the lifetime of the generator; the caller (or
    itertools.islice, for stride/max_frames) determines how much of it actually gets
    consumed."""
    with _open(bag_dir) as reader:
        src0 = resolve_camera(reader, INFRA1_RAW, INFRA1_RECT, INFRA1_INFO, balance=0.0)
        conns = [c for c in reader.connections if c.topic == src0.image_topic]
        idx = 0
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            img = decode_mono8(msg)
            out = img.copy() if src0.map1 is None else cv2.remap(img, src0.map1, src0.map2, cv2.INTER_LINEAR)
            yield RosbagMonoFrameEntry(index=idx, timestamp_ns=stamp_to_ns(msg.header.stamp), left_image=out)
            idx += 1


def load_stereo_frames(bag_dir):
    """Generator merging infra1+infra2 by exact timestamp match (the same guarantee
    euroc.load_stereo_frames relies on for hardware-synced EuRoC data, and what's been
    observed for both rig flavors tested so far) via a small pending-frame buffer keyed by
    timestamp - unmatched frames (e.g. a stream's leading/trailing edge) are dropped, same
    as euroc.load_stereo_frames does for a cam1 entry missing at a given timestamp."""
    with _open(bag_dir) as reader:
        src0 = resolve_camera(reader, INFRA1_RAW, INFRA1_RECT, INFRA1_INFO, balance=0.0)
        src1 = resolve_camera(reader, INFRA2_RAW, INFRA2_RECT, INFRA2_INFO, balance=0.0)
        conns = [c for c in reader.connections if c.topic in (src0.image_topic, src1.image_topic)]
        pending_l: dict[int, np.ndarray] = {}
        pending_r: dict[int, np.ndarray] = {}
        idx = 0
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            ts_ns = stamp_to_ns(msg.header.stamp)
            img = decode_mono8(msg)
            if conn.topic == src0.image_topic:
                out = img.copy() if src0.map1 is None else cv2.remap(img, src0.map1, src0.map2, cv2.INTER_LINEAR)
                other = pending_r.pop(ts_ns, None)
                if other is None:
                    pending_l[ts_ns] = out
                    continue
                left_img, right_img = out, other
            else:
                out = img.copy() if src1.map1 is None else cv2.remap(img, src1.map1, src1.map2, cv2.INTER_LINEAR)
                other = pending_l.pop(ts_ns, None)
                if other is None:
                    pending_r[ts_ns] = out
                    continue
                left_img, right_img = other, out
            yield RosbagStereoFrameEntry(index=idx, timestamp_ns=ts_ns, left_image=left_img, right_image=right_img)
            idx += 1


def load_imu_calibration(bag_dir, tf_static_from=None) -> ImuCalibration:
    """See load_stereo_rig's tf_static_from docstring."""
    if tf_static_from:
        with _open(tf_static_from) as tf_reader:
            tf = read_tf_static(tf_reader)
    with _open(bag_dir) as reader:
        if not tf_static_from:
            tf = read_tf_static(reader)
        body_frame = _imu_frame_id(reader)
        cam0_frame_id = resolve_camera(reader, INFRA1_RAW, INFRA1_RECT, INFRA1_INFO, balance=0.0).frame_id
        t_cam0_body = np.linalg.inv(bfs_transform(tf, body_frame, cam0_frame_id))
    return ImuCalibration(
        gyro_noise_density=IMU_GYRO_NOISE_DENSITY,
        gyro_random_walk=IMU_GYRO_RANDOM_WALK,
        accel_noise_density=IMU_ACCEL_NOISE_DENSITY,
        accel_random_walk=IMU_ACCEL_RANDOM_WALK,
        T_cam0_body=t_cam0_body,
    )


def load_imu_measurements(bag_dir) -> np.ndarray:
    """Returns an Nx7 float64 array [timestamp_ns, wx, wy, wz, ax, ay, az], sorted by
    time - same contract as euroc.load_imu_measurements (including that same source's
    float64 timestamp precision floor: fine for the dt/rate computations this feeds, see
    rosbag_common.read_imu_combined_samples)."""
    with _open(bag_dir) as reader:
        _, ts, values = read_imu_samples(reader)
    return np.concatenate([ts.astype(np.float64)[:, None], values], axis=1)


def expected_frame_count(bag_dir, mono_mode: bool) -> int | None:
    """Best-effort frame count from the bag's own metadata (for build_map.py's progress
    print - a generator has no len()); None if it can't be determined, e.g. an unusual
    connection layout the caller should just print "unknown" for instead of failing."""
    try:
        with _open(bag_dir) as reader:
            src0 = resolve_camera(reader, INFRA1_RAW, INFRA1_RECT, INFRA1_INFO, balance=0.0)
            count0 = sum(c.msgcount for c in reader.connections if c.topic == src0.image_topic)
            if mono_mode:
                return count0
            src1 = resolve_camera(reader, INFRA2_RAW, INFRA2_RECT, INFRA2_INFO, balance=0.0)
            count1 = sum(c.msgcount for c in reader.connections if c.topic == src1.image_topic)
            return min(count0, count1)
    except Exception:
        return None


def load_depth_lookup(bag_dir) -> dict[int, np.ndarray] | None:
    """Optional onboard depth stream (16-bit mm, on cam0's raw/unrectified pixel grid,
    same as euroc.load_depth_lookup's depth0/) -> {timestamp_ns: decoded array}, or None
    if this recording has no depth topic. Unlike the stereo/mono frame generators above,
    this reads the whole depth stream into memory up front (matching depth0/ being fully
    on disk already in the file-based path) - only used when dataset.use_sensor_depth is
    set, a less common mode where the extra memory is an accepted tradeoff."""
    with _open(bag_dir) as reader:
        msg0 = first_message(reader, DEPTH_TOPIC)
        if msg0 is None:
            return None
        conns = [c for c in reader.connections if c.topic == DEPTH_TOPIC]
        lookup: dict[int, np.ndarray] = {}
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            lookup[stamp_to_ns(msg.header.stamp)] = depth_mm.copy()
    return lookup
