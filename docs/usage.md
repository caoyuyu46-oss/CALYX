# CALYX Usage Guide

This document covers practical use of the CALYX framework: data preparation, training loops, inference, and ONNX export. For module-level technical details refer to `architecture.md`; for figure reproduction refer to `reproduction.md`.

---

## 1. Installation

### 1.1 From source (recommended for development)

```bash
git clone https://github.com/caoyuyu46-oss/CALYX.git
cd calyx
pip install -e .
```

The editable install lets you modify CALYX modules without reinstalling.

### 1.2 With ultralytics integration

If you plan to train with Ultralytics YOLO11n as the backbone:

```bash
pip install -e ".[ultralytics]"
```

### 1.3 Full development environment

```bash
pip install -e ".[all]"
```

This includes `pytest`, `matplotlib`, and `ultralytics`.

### 1.4 Verify installation

```bash
python -c "import calyx; print(calyx.__version__)"
# Expected output: 1.0.0

pytest tests/ -v
```

---

## 2. Data Preparation

CALYX uses two datasets per paper Section 4.1:

### 2.1 KorlaPear (main dataset, 8 516 images)

- 4-class object detection labels in YOLO format (one `.txt` per image, each line `class_id x_center y_center width height`).
- Class IDs: `0` pear, `1` peduncle, `2` calyx, `3` defect. Geometry (concave/convex/saddle/flat) is an internal PSPK channel, not a detection class (paper Section 3.1.1).
- Splits per paper Section 4.1.1:
  - Train: 5 500 images
  - Val: 680 images
  - Test-IID: 680 images
  - OOD-illumination: 900
  - OOD-cultivar: 500
  - OOD-pose: 256

Expected directory layout:

```
data/KorlaPear/
├── images/
│   ├── train/      *.jpg
│   ├── val/        *.jpg
│   ├── test_iid/   *.jpg
│   ├── ood_illum/  *.jpg
│   ├── ood_cult/   *.jpg
│   └── ood_pose/   *.jpg
├── labels/
│   └── (mirror of images/)
└── KorlaPear.yaml
```

`KorlaPear.yaml` is a standard Ultralytics dataset YAML:

```yaml
path: ./data/KorlaPear
train: images/train
val: images/val
test: images/test_iid

nc: 4
names: [pear, peduncle, calyx, defect]
```

### 2.2 PearDepth-Pro (auxiliary subset, 80 pears, 480 RGB-Depth pairs)

Used only for:
- Section Section 4.5.6: W_geo Frobenius distance against ground-truth geometry.
- Section Section 4.7.1: Pearson correlation between specular ratio and CALYX gain.
- Section Section 4.9: RGB vs hyperspectral SPA-SVM comparison.

PearDepth-Pro is **not** used during training. It is independent of KorlaPear.

```
data/PearDepth-Pro/
├── pears/                     # 80 pear IDs: pear_001 .. pear_080
│   └── pear_XXX/
│       ├── view_01_rgb.png
│       ├── view_01_depth.tiff (16-bit, mm)
│       └── ...               (6 views per pear)
└── PearDepth-Pro.yaml
```

---

## 3. Training

### 3.1 Quick start with skeleton model

For prototyping and unit-test-level demos (no Ultralytics dependency):

```python
import torch
from calyx import CalyxYOLOSkeleton, CalyxTotalLoss

model = CalyxYOLOSkeleton(nc=4)
loss_fn = CalyxTotalLoss()  # default weights (0.10, 0.30, 0.50)

x = torch.randn(2, 3, 640, 640)
out = model(x)  # CalyxOutputs(detections, polarity_p2, polarity_p4, polarity_per_scale, depth_pred, sadr_intermediate)

# Build supervisory targets and call loss_fn(...)  # see Section 3.3 below
```

### 3.2 Production training with Ultralytics YOLO11n

```python
from calyx.models.register import register_calyx_modules
from ultralytics import YOLO

# 1. Patch ultralytics.nn.tasks so YAML can recognize CALYX layer types
register_calyx_modules()

# 2. Build model from CALYX-augmented YAML
model = YOLO("configs/calyx_yolo11n.yaml")

# 3. Train (Ultralytics-native training loop)
model.train(
    data="data/KorlaPear/KorlaPear.yaml",
    epochs=200,
    imgsz=640,
    batch=32,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
)
```

The Ultralytics training loop drives `L_det` natively. To enable the three CALYX auxiliary losses (`L_DAR`, `L_polar`, `L_probe`), you need a custom callback or a custom trainer. The simplest approach is the standalone training loop below.

### 3.3 Standalone training loop

```python
import torch
from torch.utils.data import DataLoader
from calyx import (
    CalyxYOLOSkeleton, CalyxTotalLoss, CalyxLossWeights,
    SiLogLoss, PolarConsistencyLoss, CalyxLinearProbe,
)

model = CalyxYOLOSkeleton(nc=4)
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)

loss_fn = CalyxTotalLoss(weights=CalyxLossWeights())
silog = SiLogLoss()
polar = PolarConsistencyLoss()
probe = CalyxLinearProbe()

# Pseudo-depth model (frozen, used only for L_DAR target)
depth_anything_v2 = load_depth_anything_v2()
depth_anything_v2.eval()

for epoch in range(200):
    for batch in train_loader:
        images, det_targets, calyx_rois, calyx_dcf_pcf_labels = batch  # see your DataLoader
        images = images.cuda()

        # 1. Forward pass — model returns all auxiliary outputs
        out = model(images)
        # out.detections        : YOLO head output for L_det
        # out.polarity_p2        : [B, 4, H/4, W/4]   for L_polar
        # out.polarity_p4        : [B, 4, H/16, W/16] for L_polar
        # out.polarity_per_scale : [P3, P4, P5] polarity maps (P3 used by the probe)
        # out.depth_pred         : [B, 1, H/8, W/8]   for L_DAR (training mode)

        # 2. Compute pseudo-depth target (no grad)
        with torch.no_grad():
            depth_target = depth_anything_v2(images)
            depth_target = torch.nn.functional.interpolate(
                depth_target, size=out.depth_pred.shape[-2:], mode="bilinear"
            )

        # 3. Compute the four component losses
        l_det = compute_yolo_loss(out.detections, det_targets)
        l_dar = silog(out.depth_pred, depth_target)
        l_polar = polar(out.polarity_p2, out.polarity_p4)
        roi_vecs = calyx_roi_polarity(out.polarity_per_scale[0], calyx_rois)  # [N, 4]
        l_probe = probe(roi_vecs, calyx_dcf_pcf_labels)

        # 4. Aggregate (Eq. 3.19), backward + step
        loss_out = loss_fn(
            loss_det=l_det,
            loss_dar=l_dar,
            loss_polar=l_polar,
            loss_probe=l_probe,
        )
        optim.zero_grad()
        loss_out.total.backward()
        optim.step()

    scheduler.step()
```

`compute_yolo_loss` is your standard YOLO detection loss (box + cls + dfl). Refer to Ultralytics' `v8DetectionLoss` for a reference implementation.

### 3.4 ROI alignment for the calyx probe (Eq. 3.18)

The calyx linear probe expects 4-D mean polarity vectors per detected calyx ROI. Sample alignment code:

```python
import torch
import torchvision.ops as ops

def calyx_roi_polarity(polarity_map, calyx_boxes, output_size=1):
    """
    polarity_map  : [B, 4, H, W] from PSPK at P3 stage
    calyx_boxes   : [N, 5] = (batch_idx, x1, y1, x2, y2) for N calyx detections
    Returns       : [N, 4] mean polarity per ROI
    """
    pooled = ops.roi_align(
        polarity_map,
        boxes=calyx_boxes,
        output_size=output_size,
        spatial_scale=1.0 / 8.0,  # P3 stride
        sampling_ratio=2,
    )  # [N, 4, output_size, output_size]
    return pooled.mean(dim=(2, 3))  # [N, 4]

# Usage in training (probe is a separate CalyxLinearProbe instance):
probe = CalyxLinearProbe()
calyx_polarity_4d = calyx_roi_polarity(out.polarity_per_scale[0], calyx_box_batch)
l_probe = probe(calyx_polarity_4d, calyx_dcf_pcf_labels)  # loss; probe.predict(...) for labels
```

The probe (5 trainable parameters: 4 weights + 1 bias) sits on top of these 4-D vectors and outputs binary DCF/PCF logits.

---

## 4. Inference

### 4.1 Standard inference

```python
import torch
from calyx import CalyxYOLOSkeleton

model = CalyxYOLOSkeleton(nc=4)
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()

with torch.no_grad():
    image = load_and_preprocess(...)         # [1, 3, 640, 640]
    out = model(image)
    boxes, scores, classes = decode_yolo(out.detections)
```

### 4.2 Strip auxiliary heads for deployment

For ONNX export or TensorRT engines, remove the DAR head and probe to ensure no auxiliary computation enters the inference graph:

```python
from calyx import CalyxYOLOSkeleton

model = CalyxYOLOSkeleton(nc=4)
model.load_state_dict(torch.load("checkpoint.pt"))
model.export_inference_only()  # in-place: removes the DAR head

# Now model.forward returns only the detection-relevant outputs.
torch.onnx.export(
    model.eval(),
    (torch.randn(1, 3, 640, 640),),
    "calyx_inference.onnx",
    opset_version=17,
    input_names=["image"],
    output_names=["detections"],
)
```

After `export_inference_only()`, the model contains only:
- `SADR` (no learnable params)
- backbone with `PSPKBlock` at P2 and P4
- neck (PAN-FPN) with `PGI` at P3/P4/P5
- detection head

The auxiliary `DARHead` is deleted, **not just frozen**, so it cannot leak into the inference graph. The calyx probe is a separate module used only during training and is never part of the skeleton, so it does not enter the export.

---

## 5. Hyperparameter Reference

All defaults match paper Chapter 3:

| Module | Hyperparameter | Default | Paper anchor |
| :----- | :------------- | :------ | :----------- |
| SADR | `s0` | 0.20 | Section 3.2.3 |
| SADR | `gamma` | 12.0 | Section 3.2.3 |
| SADR | `tau` | 0.85 | Section 3.2.3 |
| SADR | `eps` | 1e-6 | Section 3.2.2 |
| PSPK | kernel size | 4 | Section 3.3.2 |
| PSPK | num_kernels | 4 | Section 3.3.2 |
| PGI | `eta_init` | 0.5 | Section 3.4.3 |
| PGI | `W_geo_init` | I_4 | Section 3.4.3 |
| PGI | scales | P3, P4, P5 | Section 3.4.1 |
| DAR | `hidden_dim` | 32 | Section 3.5.1 |
| DAR | output activation | Softplus | Section 3.5.1 |
| SiLog | `lambda_d` | 1.0 | Section 3.5.2 |
| Total | `lambda_1` (L_DAR) | 0.10 | Section 3.5.4 |
| Total | `lambda_2` (L_polar) | 0.30 | Section 3.5.4 |
| Total | `lambda_3` (L_probe) | 0.50 | Section 3.5.4 |

To override defaults, pass arguments at construction time:

```python
sadr = SADR(s0=0.25, gamma=10.0, tau=0.80)
loss = CalyxTotalLoss(weights=CalyxLossWeights(lambda_1=0.20, lambda_2=0.50, lambda_3=0.30))
```

---

## 6. Common Issues

### 6.1 PSPK output shape `(C+4, H, W)` doesn't match downstream `(C, H, W)`

PSPK is designed to **append** 4 polarity channels to the input, increasing the channel count. Your downstream block (e.g., next C3k2 in YOLO11n) must accept `C+4` input channels. The provided YAML at `configs/calyx_yolo11n.yaml` already adjusts the `args` of the next layer accordingly. If you build a custom backbone, remember to add 4 to the input channel count of the layer following each PSPK insertion.

### 6.2 PGI eta is stuck at 0.5 throughout training

Two common causes:
1. **Learning rate too low for `eta_logit`**: try increasing it 10× via parameter groups.
2. **L_polar / L_probe weights too small**: `eta` only receives gradient through the polarity → injection → detection path. If auxiliary losses are heavily down-weighted, the gradient signal to `eta_logit` is weak.

### 6.3 SADR produces NaN

Almost always caused by zero-summed pixels (e.g., padded regions of letterboxed images). The implementation includes `eps` in `c_max = I_max / (I_sum + eps)`, which handles this. If NaN persists, check that your input image is correctly normalized to a non-negative range before SADR. SADR does not assume `[0, 1]`; any consistent non-negative range works.

### 6.4 KL divergence in PolarConsistencyLoss is very large at start

This is expected. P2 and P4 polarity distributions diverge initially because the kernels at the two scales see very different receptive fields. The loss decreases as `lambda_2 * L_polar` drives the kernels to learn consistent geometry classifications. Typical curve: starts at ~0.5–1.0, settles around 0.05–0.15 after convergence.

---

## 7. See Also

- `docs/architecture.md` — module-level implementation deep dive.
- `docs/reproduction.md` — replicating paper figures.
- `examples/quick_start.py` — minimal end-to-end example.
- `tests/` — unit tests for each module.
