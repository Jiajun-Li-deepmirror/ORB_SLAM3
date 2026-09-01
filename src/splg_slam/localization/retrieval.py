from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

from splg_slam.map.world_map import WorldMap

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_dinov2_vits14():
    """torch.hub.load('facebookresearch/dinov2', ...) always pings GitHub to resolve the
    ref, even with a warm cache - flaky in restricted/proxied network setups. Prefer the
    already-downloaded local repo and only hit the network if it isn't there yet."""
    local_repo = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
    if local_repo.exists():
        return torch.hub.load(str(local_repo), "dinov2_vits14", source="local")
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")


class GlobalDescriptorExtractor:
    """DINOv2 (ViT-S/14) CLS-token embedding, used as a compact whole-image descriptor
    for loop-closure / relocalization candidate retrieval in medium-to-large scenes."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load_dinov2_vits14().eval().to(self.device)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(224, antialias=True),
            T.CenterCrop(224),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    @torch.no_grad()
    def extract(self, img: np.ndarray) -> np.ndarray:
        img_rgb = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        feat = self.model(tensor).cpu().numpy()[0]
        return feat / (np.linalg.norm(feat) + 1e-9)


class GlobalDescriptorIndex:
    """Brute-force cosine-similarity search over keyframe global descriptors."""

    def __init__(self):
        self.keyframe_ids: list[int] = []
        self.descriptors: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def build(self, world_map: WorldMap) -> None:
        ids, descs = [], []
        for kf_id in world_map.keyframe_ids_sorted():
            gd = world_map.keyframes[kf_id].global_descriptor
            if gd is not None:
                ids.append(kf_id)
                descs.append(gd)
        self.keyframe_ids = ids
        self.descriptors = np.stack(descs, axis=0).astype(np.float32) if descs else np.zeros((0, 0), dtype=np.float32)

    def query(self, descriptor: np.ndarray, top_k: int = 5, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        if self.descriptors.shape[0] == 0:
            return []
        sims = self.descriptors @ descriptor.astype(np.float32)
        order = np.argsort(-sims)
        results = []
        for i in order:
            kf_id = self.keyframe_ids[i]
            if exclude and kf_id in exclude:
                continue
            results.append((kf_id, float(sims[i])))
            if len(results) >= top_k:
                break
        return results
