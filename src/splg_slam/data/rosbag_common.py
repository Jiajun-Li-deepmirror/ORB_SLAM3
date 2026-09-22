"""Shared rosbag2 (mcap/db3) reading helpers for the two things that read these bags:
scripts/rosbag_to_euroc.py (writes an EuRoC mav0/ copy to disk) and
splg_slam.data.rosbag2 (reads frames directly at build_map.py runtime, no copy). Both
rig flavors seen so far are handled here - raw fisheye ('equidistant' distortion model)
+ a combined IMU topic (custom rig), and already-rectified image_rect_raw (D=0) + split
gyro/accel topics (RealSense driver default) - auto-detected per camera/IMU exactly the
same way in both callers, so behavior never diverges between the copy path and the
direct-read path.

Extrinsics (T_BS, sensor -> body) are derived by BFS-composing /tf_static edges from a
sensor's optical frame to a chosen body frame - works for direct edges and multi-hop
chains alike (e.g. through an intermediate depth-aligned frame).
"""
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from splg_slam.utils import nearest_indices

INFRA1_RAW = "/camera/camera/infra1/image_raw"
INFRA1_RECT = "/camera/camera/infra1/image_rect_raw"
INFRA1_INFO = "/camera/camera/infra1/camera_info"
INFRA2_RAW = "/camera/camera/infra2/image_raw"
INFRA2_RECT = "/camera/camera/infra2/image_rect_raw"
INFRA2_INFO = "/camera/camera/infra2/camera_info"
IMU_COMBINED = "/camera/camera/imu"
GYRO_TOPIC = "/camera/camera/gyro/sample"
ACCEL_TOPIC = "/camera/camera/accel/sample"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"

# Typical Bosch BMI055-class MEMS IMU values (RealSense D4xx-family datasheets) - neither
# bag carries an IMU calibration report, so these are approximate placeholders.
IMU_GYRO_NOISE_DENSITY = 1.6e-4
IMU_GYRO_RANDOM_WALK = 1.9e-5
IMU_ACCEL_NOISE_DENSITY = 2.8e-3
IMU_ACCEL_RANDOM_WALK = 8.6e-4


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def quat_translation_to_matrix(t, q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = [t.x, t.y, t.z]
    return m


def read_tf_static(reader) -> dict[tuple[str, str], np.ndarray]:
    conns = [c for c in reader.connections if c.topic == "/tf_static"]
    edges: dict[tuple[str, str], np.ndarray] = {}
    if not conns:
        # AnyReader.messages(connections=[]) disables connection filtering entirely
        # (documented behavior) rather than matching nothing - without this guard, a bag
        # missing /tf_static would silently iterate every OTHER topic's messages instead
        # and (best case) crash on the first mismatched message type, or (worst case)
        # silently misparse one that happens to share a field name.
        return edges
    for conn, _, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        for tr in msg.transforms:
            m = quat_translation_to_matrix(tr.transform.translation, tr.transform.rotation)
            edges[(tr.header.frame_id, tr.child_frame_id)] = m
            edges[(tr.child_frame_id, tr.header.frame_id)] = np.linalg.inv(m)
    return edges


def bfs_transform(edges: dict[tuple[str, str], np.ndarray], src: str, dst: str) -> np.ndarray:
    """Composed transform mapping points from dst's frame into src's frame (T_src_dst),
    found by BFS over the (possibly multi-hop) /tf_static graph."""
    if src == dst:
        return np.eye(4)
    adjacency: dict[str, list[str]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
    visited = {src}
    queue = deque([(src, np.eye(4))])
    while queue:
        node, t_src_node = queue.popleft()
        for nxt in adjacency.get(node, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            t_src_next = t_src_node @ edges[(node, nxt)]
            if nxt == dst:
                return t_src_next
            queue.append((nxt, t_src_next))
    raise RuntimeError(f"no tf_static path from {src!r} to {dst!r}")


def first_message(reader, topic: str):
    conns = [c for c in reader.connections if c.topic == topic]
    if not conns:
        return None
    for conn, _, raw in reader.messages(connections=conns):
        return reader.deserialize(raw, conn.msgtype)
    return None


def read_camera_info(reader, topic: str):
    msg = first_message(reader, topic)
    if msg is None:
        raise RuntimeError(f"no messages on {topic}")
    k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
    d = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
    p = np.array(msg.p, dtype=np.float64).reshape(3, 4)
    return k, d, p, int(msg.width), int(msg.height), msg.distortion_model


def build_fisheye_undistort_maps(k: np.ndarray, d: np.ndarray, size: tuple[int, int], balance: float):
    identity = np.eye(3)
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, size, identity, balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, identity, new_k, size, cv2.CV_32FC1)
    return map1, map2, new_k


def build_pinhole_undistort_maps(k: np.ndarray, d: np.ndarray, size: tuple[int, int]):
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, d, size, alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(k, d, None, new_k, size, cv2.CV_32FC1)
    return map1, map2, new_k


def decode_mono8(msg) -> np.ndarray:
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)


@dataclass
class CameraSource:
    image_topic: str
    frame_id: str
    width: int
    height: int
    dist_model: str
    dist_coeffs: np.ndarray
    map1: np.ndarray | None  # None if the driver already delivers a rectified/undistorted image
    map2: np.ndarray | None
    k_new: np.ndarray  # post-undistortion (or already-rectified) intrinsics


def resolve_camera(reader, raw_topic: str, rect_topic: str, info_topic: str, balance: float = 0.0) -> CameraSource:
    """Auto-detects already-rectified (image_rect_raw, D=0) vs. raw-fisheye vs. raw
    radial-tangential, and returns everything needed to either undistort a frame
    (map1/map2, or None if no undistortion is needed) or just declare the resulting
    intrinsics (k_new)."""
    rect_first = first_message(reader, rect_topic)
    use_rect = rect_first is not None
    image_topic = rect_topic if use_rect else raw_topic
    frame_id = (rect_first if use_rect else first_message(reader, raw_topic)).header.frame_id

    k, d, p, width, height, dist_model = read_camera_info(reader, info_topic)

    if use_rect or np.allclose(d, 0.0):
        map1 = map2 = None
        k_new = p[:3, :3]
    elif dist_model == "equidistant":
        map1, map2, k_new = build_fisheye_undistort_maps(k, d, (width, height), balance)
    else:
        map1, map2, k_new = build_pinhole_undistort_maps(k, d, (width, height))

    return CameraSource(
        image_topic=image_topic, frame_id=frame_id, width=width, height=height,
        dist_model=dist_model, dist_coeffs=d, map1=map1, map2=map2, k_new=k_new,
    )


def read_imu_combined_samples(reader) -> tuple[str, np.ndarray, np.ndarray]:
    """Returns (frame_id, timestamps_ns [int64, sorted], values [float64 Nx6:
    wx,wy,wz,ax,ay,az]). Timestamps are kept as a separate int64 array (not folded into
    one float64 Nx7 array here) so callers writing them back out - e.g. as EuRoC CSV text -
    don't lose nanosecond precision to a premature float64 cast; int64 timestamps this
    large (~1e18) only have ~9.2e15 of exact float64 mantissa range to work with."""
    msg0 = first_message(reader, IMU_COMBINED)
    conns = [c for c in reader.connections if c.topic == IMU_COMBINED]
    rows = []
    for conn, _, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        ts_ns = stamp_to_ns(msg.header.stamp)
        av, la = msg.angular_velocity, msg.linear_acceleration
        rows.append((ts_ns, av.x, av.y, av.z, la.x, la.y, la.z))
    rows.sort(key=lambda r: r[0])
    ts = np.array([r[0] for r in rows], dtype=np.int64)
    values = np.array([r[1:] for r in rows], dtype=np.float64)
    return msg0.header.frame_id, ts, values


def read_imu_split_samples(reader) -> tuple[str, np.ndarray, np.ndarray]:
    """Merges separate gyro/sample + accel/sample topics (RealSense default when
    unite_imu_method is off) by nearest-neighbor timestamp matching onto the gyro
    (higher-rate) stream. Returns (frame_id, timestamps_ns, values) like
    read_imu_combined_samples."""
    gyro_msg0 = first_message(reader, GYRO_TOPIC)
    gyro_conns = [c for c in reader.connections if c.topic == GYRO_TOPIC]
    accel_conns = [c for c in reader.connections if c.topic == ACCEL_TOPIC]

    gyro = []
    for conn, _, raw in reader.messages(connections=gyro_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        av = msg.angular_velocity
        gyro.append((stamp_to_ns(msg.header.stamp), av.x, av.y, av.z))
    accel = []
    for conn, _, raw in reader.messages(connections=accel_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        la = msg.linear_acceleration
        accel.append((stamp_to_ns(msg.header.stamp), la.x, la.y, la.z))
    gyro.sort(key=lambda r: r[0])
    accel.sort(key=lambda r: r[0])
    gyro_ts = np.array([r[0] for r in gyro], dtype=np.int64)
    gyro_vals = np.array([r[1:] for r in gyro], dtype=np.float64)
    accel_ts = np.array([r[0] for r in accel], dtype=np.int64)
    accel_vals = np.array([r[1:] for r in accel], dtype=np.float64)

    j = nearest_indices(accel_ts, gyro_ts)
    match_dt_ns = np.abs(accel_ts[j] - gyro_ts)
    # Both streams normally run continuously for the whole bag, so a gyro sample should
    # always have a genuinely nearby accel sample; a match several accel sample periods
    # away means the two streams don't actually overlap there (a mid-stream gap, or one
    # starting/ending well before the other) - silently pairing gyro/accel across that
    # gap would feed corrupted IMU preintegration input with no warning.
    max_dt_ns = 5 * int(np.median(np.diff(accel_ts))) if len(accel_ts) > 1 else 0
    bad = match_dt_ns > max_dt_ns if max_dt_ns > 0 else np.zeros_like(match_dt_ns, dtype=bool)
    if bad.any():
        print(
            f"  [rosbag] read_imu_split_samples: dropping {int(bad.sum())}/{len(gyro)} gyro "
            f"sample(s) with no accel sample within {max_dt_ns / 1e6:.1f}ms"
        )
    keep = ~bad
    ts = gyro_ts[keep]
    values = np.concatenate([gyro_vals[keep], accel_vals[j[keep]]], axis=1)
    return gyro_msg0.header.frame_id, ts, values


def read_imu_samples(reader) -> tuple[str, np.ndarray, np.ndarray]:
    """Auto-detects combined vs. split IMU topics and returns (frame_id, timestamps_ns,
    values)."""
    if first_message(reader, IMU_COMBINED) is not None:
        return read_imu_combined_samples(reader)
    return read_imu_split_samples(reader)


def imu_rate_hz(timestamps_ns: np.ndarray) -> float:
    if len(timestamps_ns) <= 1:
        return 200.0
    return (len(timestamps_ns) - 1) / ((int(timestamps_ns[-1]) - int(timestamps_ns[0])) * 1e-9)
