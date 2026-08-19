# yolov5n.onnx

Classic (pre-anchor-free) YOLOv5n head, exported so that OpenCV 4.5.4's `cv::dnn` ONNX
importer can load it -- neither YOLOv8's anchor-free head nor a stock recent-PyTorch YOLOv5
export work on this OpenCV version (see below).

Output: `[1, 25200, 85]`, already sigmoid-activated and grid/anchor-decoded to
`[cx, cy, w, h, objectness, class0..class79]` in pixels of the 640x640 input. COCO classes.

## Why a plain `yolo export` doesn't work here

- `ultralytics` package (v8+) always exports the anchor-free head (`[1,84,8400]`, no
  objectness). OpenCV 4.5.4 fails on an `Add` node in the detection head
  (`model.22`): `blob_0.size == blob_1.size` assertion in `parseBias`.
- The classic `yolov5` package (fcakyon's PyPI wrapper around the pre-anchor-free
  architecture) exports fine architecturally, but a plain export with recent PyTorch (2.3.1)
  fails on two more things OpenCV 4.5.4 can't parse:
  1. `nn.Upsample(scale_factor=2)` traces to a `Resize` node via a dynamic `Floor` op under
     newer PyTorch -- "Can't create layer ... of type Floor".
  2. The `Detect` layer's cached `grid`/`anchor_grid` buffers get rebuilt *during tracing*
     (shape mismatch on a freshly loaded model triggers `_make_grid()` inside `forward()`),
     which traces to a 5-D `Expand` node OpenCV can't parse either.

## How this file was produced

```python
import torch, torch.nn as nn, torch.nn.functional as F
from yolov5.models.experimental import attempt_load  # pip install yolov5 (fcakyon)

model = attempt_load('yolov5n.pt', device='cpu', inplace=True, fuse=True)
model.eval()

# 1) Replace nn.Upsample(scale_factor=...) with an equivalent that computes an explicit
#    integer output size, avoiding the dynamic Floor op.
class StaticUpsample(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        h, w = int(x.shape[2] * self.scale_factor), int(x.shape[3] * self.scale_factor)
        return F.interpolate(x, size=(h, w), mode='nearest')

for name, module in model.named_modules():
    for child_name, child in module.named_children():
        if isinstance(child, nn.Upsample):
            new_mod = StaticUpsample(child.scale_factor)
            for attr in ('f', 'i', 'type', 'np'):  # yolov5's forward-routing bookkeeping
                if hasattr(child, attr):
                    setattr(new_mod, attr, getattr(child, attr))
            setattr(module, child_name, new_mod)

# 2) Warm up once at the export resolution BEFORE tracing, so Detect.grid[i]/anchor_grid[i]
#    are cached at the right shape and _make_grid() is never called during tracing.
dummy = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    model(dummy)

torch.onnx.export(model, dummy, 'yolov5n.onnx', opset_version=11,
                   input_names=['images'], output_names=['output'],
                   do_constant_folding=True)
```

Verified with a standalone `cv::dnn::readNetFromONNX` + `net.forward()` smoke test against
OpenCV 4.5.4 before wiring it into `DynamicDetector`.
