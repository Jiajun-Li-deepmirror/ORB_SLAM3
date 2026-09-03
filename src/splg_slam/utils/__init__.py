import random

import cv2
import numpy as np
import torch


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
