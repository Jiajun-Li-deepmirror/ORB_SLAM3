# cosplace_resnet18_512.onnx

CosPlace (Berton et al., CVPR 2022, "Rethinking Visual Geo-localization for Large-Scale
Applications") global-descriptor model, ResNet18 backbone, 512-dim output. Used by
`PlaceRecognizer` as an additional loop-closure/relocalization candidate source alongside DBoW2
-- see `include/PlaceRecognizer.h`.

Unlike the Dynamic-VINS branch's YOLOv5 model, this one only runs through ONNX Runtime, not
OpenCV's `cv::dnn`: OpenCV 4.5.4's ONNX importer can't parse the GeM-pooling layer's
`Reciprocal` op (`x^(1/p)`) either, so there's no CPU fallback path through OpenCV here.

## How this file was produced

```python
import torch
model = torch.hub.load('gmberton/cosplace', 'get_trained_model',
                        backbone='ResNet18', fc_output_dim=512, trust_repo=True)
model.eval()
dummy = torch.zeros(1, 3, 480, 640)
torch.onnx.export(model, dummy, 'cosplace_resnet18_512.onnx', opset_version=12,
                   input_names=['image'], output_names=['descriptor'],
                   do_constant_folding=True)
```

Output: `[1, 512]`, NOT L2-normalised by the model itself -- `PlaceRecognizer::computeDescriptor()`
normalises it so cosine similarity reduces to a plain dot product.

Verified against ONNX Runtime 1.18 (both CPU and CUDA execution providers) before wiring into the
main build. Measured per-image inference time on an RTX 3060 laptop GPU: ~54ms CPU, ~4.5ms CUDA
(steady-state, after warm-up -- CUDA's first call pays a ~1.6s one-time context/kernel-compile
cost).
