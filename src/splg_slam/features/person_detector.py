import numpy as np
import torch


class PersonDetector:
    """DynamicVINS-style dynamic-object detector, scoped to the "person" class: finds
    people in a frame so tracker.py can discard any SuperPoint keypoint that lands on one
    before it ever reaches PnP/triangulation/matching (a moving person violates the
    static-scene assumption everything downstream relies on).

    Uses torchvision's fasterrcnn_mobilenet_v3_large_320_fpn (COCO-pretrained, 320px
    input - the lightest detector torchvision ships) rather than YOLO: neither
    ultralytics nor yolov5 are installed in this project's environment, and torchvision
    is already a dependency here, so this avoids adding a new one."""

    def __init__(self, score_threshold: float = 0.6, device: torch.device | None = None):
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        # box_score_thresh prunes low-confidence boxes inside the model itself, so
        # detect_person_boxes never needs a separate score filter on its output.
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights, box_score_thresh=score_threshold
        ).eval().to(self.device)
        self.person_label = weights.meta["categories"].index("person")

    @torch.no_grad()
    def detect_person_boxes(self, img: np.ndarray) -> np.ndarray:
        """img: HxW (gray) or HxWx3 (BGR, from cv2), uint8 - same convention as
        splg.image_to_tensor. Returns Nx4 [x1,y1,x2,y2] boxes (image pixel coords) for
        every detected person already above this detector's score_threshold."""
        if img.ndim == 2:
            img_rgb = np.repeat(img[..., None], 3, axis=2)
        else:
            img_rgb = np.ascontiguousarray(img[..., ::-1])
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(self.device) / 255.0
        out = self.model([tensor])[0]
        keep = out["labels"] == self.person_label
        return out["boxes"][keep].cpu().numpy()
