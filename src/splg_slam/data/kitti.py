from pathlib import Path

import cv2
import numpy as np

from splg_slam.data.euroc import StereoFrameEntry, StereoRig
from splg_slam.geometry.camera import PinholeCamera


def _read_calib(calib_path: Path) -> dict:
    projections = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, values = line.split(":", 1)
            projections[name.strip()] = np.array([float(v) for v in values.split()], dtype=np.float64).reshape(3, 4)
    return projections


def load_stereo_rig(sequence_dir: Path) -> StereoRig:
    """KITTI odometry image_0/image_1 are already stereo-rectified: P0/P1 in calib.txt give
    the post-rectification intrinsics directly, and cam1 is purely x-translated relative to
    cam0 (no relative rotation) - the standard KITTI rectified-pair convention."""
    sequence_dir = Path(sequence_dir)
    proj = _read_calib(sequence_dir / "calib.txt")
    p0, p1 = proj["P0"], proj["P1"]

    first_img = sorted((sequence_dir / "image_0").iterdir())[0]
    h, w = cv2.imread(str(first_img), cv2.IMREAD_GRAYSCALE).shape

    zero_dist = np.zeros(5, dtype=np.float64)
    cam0 = PinholeCamera(fx=p0[0, 0], fy=p0[1, 1], cx=p0[0, 2], cy=p0[1, 2], dist_coeffs=zero_dist, width=w, height=h)
    cam1 = PinholeCamera(fx=p1[0, 0], fy=p1[1, 1], cx=p1[0, 2], cy=p1[1, 2], dist_coeffs=zero_dist, width=w, height=h)

    baseline = float(-p1[0, 3] / p1[0, 0])  # meters, positive: cam1 sits to the right of cam0
    t_cam1_cam0 = np.eye(4, dtype=np.float64)
    t_cam1_cam0[0, 3] = -baseline

    return StereoRig(cam0=cam0, cam1=cam1, T_cam1_cam0=t_cam1_cam0)


def load_stereo_frames(sequence_dir: Path) -> list[StereoFrameEntry]:
    sequence_dir = Path(sequence_dir)
    times = np.loadtxt(sequence_dir / "times.txt")
    right_dir = sequence_dir / "image_1"
    left_files = sorted((sequence_dir / "image_0").iterdir())

    frames = []
    for idx, left_path in enumerate(left_files):
        right_path = right_dir / left_path.name
        if not right_path.exists():
            continue
        timestamp_ns = int(round(float(times[idx]) * 1e9))
        frames.append(StereoFrameEntry(index=idx, timestamp_ns=timestamp_ns, left_path=left_path, right_path=right_path))
    return frames


def load_gt_poses(sequence_dir: Path) -> np.ndarray | None:
    """KITTI odometry ground truth (sequences 00-10 only): Nx12 flattened 3x4 poses (cam0
    frame, first pose = identity) in datasets/data_odometry_poses/dataset/poses/<seq>.txt,
    a sibling directory to the images archive, not inside sequence_dir itself. Returns
    Nx4x4 world_from_cam0 matrices, or None if no poses file exists for this sequence."""
    sequence_dir = Path(sequence_dir)
    seq_id = sequence_dir.name
    # sequence_dir = .../datasets/data_odometry_gray/dataset/sequences/<seq> ->
    # up 4 levels to datasets/, then into the sibling data_odometry_poses release.
    poses_path = sequence_dir.parents[3] / "data_odometry_poses" / "dataset" / "poses" / f"{seq_id}.txt"
    if not poses_path.exists():
        return None
    raw = np.loadtxt(poses_path)
    n = raw.shape[0]
    poses = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    poses[:, :3, :4] = raw.reshape(n, 3, 4)
    return poses


def load_gt_as_xyz(sequence_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (timestamp_ns, xyz) - the same shape eval_trajectory.py's EuRoC load_gt()
    produces from state_groundtruth_estimate0/data.csv - built instead from times.txt +
    poses/<seq>.txt, so the rest of that script's Sim3-alignment/RMSE/plotting logic is
    fully dataset-agnostic and needs no KITTI-specific branching beyond this loader."""
    sequence_dir = Path(sequence_dir)
    times = np.loadtxt(sequence_dir / "times.txt")
    poses = load_gt_poses(sequence_dir)
    if poses is None:
        raise FileNotFoundError(
            f"No ground-truth poses for KITTI sequence {sequence_dir.name} "
            "(only sequences 00-10 have them)"
        )
    n = min(len(times), len(poses))
    ts_ns = np.round(times[:n] * 1e9).astype(np.int64)
    xyz = poses[:n, :3, 3]
    return ts_ns, xyz
