import random

import cv2
import numpy as np
import torch


def nearest_indices(reference_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """For each timestamp in `query_ts`, the index into sorted `reference_ts` of the
    closest value (not `searchsorted`'s "first reference >= query", which is a
    systematic forward-time bias whenever the reference is coarser than the query -
    e.g. matching keyframe timestamps against sparser ground-truth samples for ATE).
    Same nearest-neighbor rule as `rosbag_common.read_imu_split_samples`, vectorized."""
    j = np.searchsorted(reference_ts, query_ts)
    j = np.clip(j, 0, len(reference_ts) - 1)
    j_prev = np.clip(j - 1, 0, len(reference_ts) - 1)
    use_prev = (j > 0) & (np.abs(reference_ts[j_prev] - query_ts) < np.abs(reference_ts[j] - query_ts))
    return np.where(use_prev, j_prev, j)


def set_global_seed(seed: int) -> None:
    """Seeds every source of randomness this pipeline touches (Python's random, numpy,
    OpenCV's RANSAC RNG, torch CPU/CUDA, cuDNN's algorithm selection) so repeated runs - same
    machine or a different one - are reproducible instead of drifting via
    cv2.solvePnPRansac's unseeded RNG and GPU-kernel nondeterminism into a different keyframe
    count and ATE each time. Needed to isolate a real config/model change from this run-to-run
    noise, e.g. when comparing two feature extractors on otherwise-identical settings."""
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
