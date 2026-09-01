import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from splg_slam.geometry.camera import PinholeCamera


def _load_sensor_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _t_bs_to_matrix(sensor_yaml: dict) -> np.ndarray:
    return np.array(sensor_yaml["T_BS"]["data"], dtype=np.float64).reshape(4, 4)


@dataclass
class StereoRig:
    cam0: PinholeCamera
    cam1: PinholeCamera
    T_cam1_cam0: np.ndarray  # 4x4, maps points from cam0 frame into cam1 frame


def load_stereo_rig(mav0_dir: Path) -> StereoRig:
    mav0_dir = Path(mav0_dir)
    y0 = _load_sensor_yaml(mav0_dir / "cam0" / "sensor.yaml")
    y1 = _load_sensor_yaml(mav0_dir / "cam1" / "sensor.yaml")

    def to_camera(y: dict) -> PinholeCamera:
        fx, fy, cx, cy = y["intrinsics"]
        dist = np.array(y["distortion_coefficients"], dtype=np.float64)
        w, h = y["resolution"]
        return PinholeCamera(fx=fx, fy=fy, cx=cx, cy=cy, dist_coeffs=dist, width=w, height=h)

    cam0 = to_camera(y0)
    cam1 = to_camera(y1)

    # Body-frame extrinsics -> relative pose between the two cameras.
    t_b_c0 = _t_bs_to_matrix(y0)
    t_b_c1 = _t_bs_to_matrix(y1)
    t_c1_b = np.linalg.inv(t_b_c1)
    t_cam1_cam0 = t_c1_b @ t_b_c0

    return StereoRig(cam0=cam0, cam1=cam1, T_cam1_cam0=t_cam1_cam0)


@dataclass
class StereoFrameEntry:
    index: int
    timestamp_ns: int
    left_path: Path
    right_path: Path


def _read_data_csv(path: Path) -> list[tuple[int, str]]:
    entries = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            entries.append((int(row[0]), row[1].strip()))
    return entries


def load_stereo_frames(mav0_dir: Path) -> list[StereoFrameEntry]:
    """EuRoC guarantees cam0/cam1 are hardware-synced with identical timestamps."""
    mav0_dir = Path(mav0_dir)
    left_entries = _read_data_csv(mav0_dir / "cam0" / "data.csv")
    right_by_ts = dict(_read_data_csv(mav0_dir / "cam1" / "data.csv"))

    frames = []
    for idx, (ts_ns, lname) in enumerate(left_entries):
        rname = right_by_ts.get(ts_ns)
        if rname is None:
            continue
        frames.append(
            StereoFrameEntry(
                index=idx,
                timestamp_ns=ts_ns,
                left_path=mav0_dir / "cam0" / "data" / lname,
                right_path=mav0_dir / "cam1" / "data" / rname,
            )
        )
    return frames
