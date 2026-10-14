"""Scale-preserving trajectory evaluation at actual image timestamps."""
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml


def rigid_alignment(source, target):
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("At least three finite paired positions are required")
    a, b = source-source.mean(0), target-target.mean(0)
    u, _, vt = np.linalg.svd(b.T @ a)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(sign) @ vt
    translation = target.mean(0) - rotation @ source.mean(0)
    return source @ rotation.T + translation, rotation, translation


def euroc_camera_gt(mav0_dir, timestamp_ns, max_gap_s=.05):
    """Interpolate only GT (never estimated frames); account for body/camera lever arm."""
    directory = Path(mav0_dir)
    timestamps, rows = [], []
    with (directory / "state_groundtruth_estimate0/data.csv").open() as f:
        for row in csv.reader(f):
            if row and not row[0].startswith("#"):
                timestamps.append(int(row[0]))
                rows.append([float(v) for v in row[1:8]])
    ts = np.array(timestamps, dtype=np.int64)
    if len(ts) < 2 or np.any(np.diff(ts) <= 0):
        raise ValueError("GT timestamps must be ordered and unique")
    rows = np.asarray(rows)
    query = np.asarray(timestamp_ns, dtype=np.int64)
    index = np.searchsorted(ts, query).clip(0, len(ts)-1)
    previous = np.maximum(index-1, 0)
    gap = np.minimum(abs(query-ts[index]), abs(query-ts[previous])) / 1e9
    # Large holes in GT must not be crossed simply because one endpoint is close.
    exact = query == ts[index]
    bracket = (ts[index]-ts[previous]) / 1e9
    valid = ((query >= ts[0]) & (query <= ts[-1]) & (gap <= max_gap_s)
             & (exact | (bracket <= 2*max_gap_s)))
    output = np.full((len(query), 3), np.nan)
    if valid.any():
        time = (ts-ts[0])/1e9
        qt = (query[valid]-ts[0])/1e9
        body = np.column_stack([np.interp(qt, time, rows[:,i]) for i in range(3)])
        rotations = Rotation.from_quat(rows[:,[4,5,6,3]])
        extrinsic = np.asarray(yaml.safe_load((directory/'cam0/sensor.yaml').read_text())['T_BS']['data']).reshape(4,4)
        output[valid] = body + Slerp(time, rotations)(qt).apply(extrinsic[:3,3])
    return output, valid


def camera_centers(poses_cw):
    return -np.einsum("nji,nj->ni", poses_cw[:,:3,:3], poses_cw[:,:3,3])


def error_stats(error):
    return dict(rmse_m=float(np.sqrt(np.mean(error**2))), mean_m=float(error.mean()),
                median_m=float(np.median(error)), p95_m=float(np.percentile(error,95)),
                max_m=float(error.max()))


def valid_estimates(data):
    finite = np.isfinite(data["pose_final_cw"]).all(axis=(1,2))
    # IMU-only predictions do not count as recovered visual tracking.
    visual = np.isin(data["status"], ["initialized", "tracked", "relocalized", "new_segment"])
    valid = finite & visual
    segment = data["final_segment_id"]
    if valid.any():
        valid &= segment == segment[np.flatnonzero(valid)[0]]
    return valid


def paired_frame_indices(first, second):
    """Exact integer timestamps only. Missing estimates are excluded and counted by callers."""
    for data in (first, second):
        if np.any(np.diff(data["timestamp_ns"]) <= 0):
            raise ValueError("Estimated frame timestamps must increase")
    common, a, b = np.intersect1d(first["timestamp_ns"], second["timestamp_ns"], return_indices=True)
    keep = valid_estimates(first)[a] & valid_estimates(second)[b]
    return common[keep], a[keep], b[keep]
