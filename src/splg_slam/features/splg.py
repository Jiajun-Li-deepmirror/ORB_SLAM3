import numpy as np
import torch


def image_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """img: HxW (gray) or HxWx3 (BGR, from cv2), uint8. Returns 1xCxHxW float tensor in [0,1]."""
    if img.ndim == 2:
        tensor = torch.from_numpy(img)[None].float() / 255.0
    else:
        img_rgb = np.ascontiguousarray(img[..., ::-1])
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return tensor[None].to(device)


class SPLG:
    """SuperPoint feature extraction + LightGlue matching, wrapped for the SLAM pipeline."""

    def __init__(self, max_keypoints: int = 1024, device: torch.device | None = None, use_fp16: bool = False):
        from lightglue import LightGlue, SuperPoint

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        # Autocast, not .half() weights: keeps mapping-time callers (which never pass this)
        # at full fp32 precision/behavior untouched, and autocast lets ops that are numerically
        # sensitive (e.g. softmax in attention) stay in fp32 internally even under the context.
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        # Tried torch.compile(dynamic=True) on extractor/matcher forward here for localization
        # latency: steady-state median was no better than plain fp16 autocast, but the number
        # of detected keypoints varies per call, and dynamic shape tracing doesn't guarantee no
        # recompiles - occasionally hit a fresh shape mid-run and stalled a single query for
        # 4-11 SECONDS. Unacceptable tail latency for a real-time relocalizer, reverted.

    @torch.no_grad()
    def extract(self, img: np.ndarray) -> dict:
        """Returns the batched (dim-0 size 1) feature dict LightGlue expects as input."""
        tensor = image_to_tensor(img, self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
            feats = self.extractor.extract(tensor)
        return {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in feats.items()}

    @torch.no_grad()
    def match(self, feats0: dict, feats1: dict) -> dict:
        from lightglue.utils import rbd

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
            matches01 = self.matcher({"image0": feats0, "image1": feats1})
        f0, f1, m01 = rbd(feats0), rbd(feats1), rbd(matches01)

        matches = m01["matches"].cpu().numpy()  # Mx2 indices into kpts0/kpts1
        scores = m01["scores"].cpu().numpy() if "scores" in m01 else None
        return {
            "kpts0": f0["keypoints"].cpu().numpy(),
            "kpts1": f1["keypoints"].cpu().numpy(),
            "desc0": f0["descriptors"].cpu().numpy(),
            "desc1": f1["descriptors"].cpu().numpy(),
            "matches": matches,
            "scores": scores,
        }

    @staticmethod
    def to_frame_arrays(feats: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract (keypoints Nx2, descriptors Nx256) numpy arrays from a raw batched feats dict."""
        from lightglue.utils import rbd

        f = rbd(feats)
        return f["keypoints"].cpu().numpy(), f["descriptors"].cpu().numpy()
