"""Bounded-memory, timestamp-checked sensor depth for ground observations."""
import numpy as np

from splg_slam.data.rosbag2 import _open
from splg_slam.data.rosbag_common import (
    INFRA1_INFO, bfs_transform, first_message, read_tf_static, stamp_to_ns,
)


def decode_depth(msg, scale):
    """Respect ROS image encoding, endianness and padded row stride."""
    if msg.encoding in ("16UC1", "mono16"):
        dtype, factor = np.dtype(">u2" if msg.is_bigendian else "<u2"), scale
    elif msg.encoding == "32FC1":
        dtype, factor = np.dtype(">f4" if msg.is_bigendian else "<f4"), 1.0
    else:
        raise ValueError(f"Unsupported ground depth encoding: {msg.encoding}")
    depth = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                       strides=(msg.step, dtype.itemsize)).astype(np.float64) * factor
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


class GroundDepthReader:
    """Read the rectified depth stream, keeping only two neighboring frames.

    Depth is backprojected with its own CameraInfo, then transformed into the
    rectified left camera used by the optimizer. No RGB/IR pixel-grid equivalence
    is assumed. Requests must be in nondecreasing timestamp order.
    """

    def __init__(self, bag_dir, rectifier, options, tf_static_from=None):
        self.options = options
        self.reader = _open(bag_dir)
        self.reader.open()
        try:
            info = first_message(self.reader, options.depth_info_topic)
            left = first_message(self.reader, INFRA1_INFO)
            connections = [c for c in self.reader.connections if c.topic == options.depth_topic]
            if info is None or left is None or not connections:
                raise ValueError("Ground constraints require depth images and depth/infra1 CameraInfo")
            self.size = (info.height, info.width)
            self.k = np.asarray(info.p).reshape(3, 4)[:, :3].copy()
            if not np.isfinite(self.k).all() or min(self.k[0, 0], self.k[1, 1]) <= 0:
                raise ValueError("Invalid rectified depth CameraInfo projection matrix")
            if tf_static_from:
                with _open(tf_static_from) as reader:
                    tf = read_tf_static(reader)
            else:
                tf = read_tf_static(self.reader)
            depth_rect_to_raw = np.eye(4)
            depth_rect_to_raw[:3, :3] = np.asarray(info.r).reshape(3, 3).T
            self.transform = (rectifier.T_rect_cam0
                              @ bfs_transform(tf, left.header.frame_id, info.header.frame_id)
                              @ depth_rect_to_raw)
            self.messages = iter(self.reader.messages(connections=connections))
            self.previous = None
            self.current = self._next()
            self.last_request = -1
        except Exception:
            self.reader.close()
            raise

    def _next(self):
        item = next(self.messages, None)
        if item is None:
            return None
        connection, _, raw = item
        msg = self.reader.deserialize(raw, connection.msgtype)
        return stamp_to_ns(msg.header.stamp), msg

    def points_at(self, timestamp_ns):
        if timestamp_ns < self.last_request:
            raise ValueError("Ground depth requests must be chronological")
        self.last_request = timestamp_ns
        while self.current is not None and self.current[0] < timestamp_ns:
            self.previous, self.current = self.current, self._next()
        candidates = [entry for entry in (self.previous, self.current) if entry is not None]
        if not candidates:
            return None, {"reason": "missing_depth"}
        ts, msg = min(candidates, key=lambda entry: abs(entry[0] - timestamp_ns))
        dt_ms = (ts - timestamp_ns) / 1e6
        if abs(dt_ms) > self.options.max_depth_dt_ms:
            return None, {"reason": "depth_time_gap", "depth_dt_ms": dt_ms}
        if (msg.height, msg.width) != self.size:
            raise ValueError("Ground depth image and CameraInfo sizes differ")
        depth = decode_depth(msg, self.options.depth_scale)
        step = self.options.pixel_stride
        yy, xx = np.mgrid[int(msg.height * self.options.roi_top):msg.height:step,
                          0:msg.width:step]
        zz = depth[yy, xx]
        valid = (zz >= self.options.min_depth_m) & (zz <= self.options.max_depth_m)
        z = zz[valid]
        points = np.column_stack(((xx[valid] - self.k[0, 2]) * z / self.k[0, 0],
                                  (yy[valid] - self.k[1, 2]) * z / self.k[1, 1], z))
        points = points @ self.transform[:3, :3].T + self.transform[:3, 3]
        return points, {"depth_dt_ms": dt_ms, "depth_timestamp_ns": int(ts),
                        "sampled_points": len(points)}

    def close(self):
        self.reader.close()
