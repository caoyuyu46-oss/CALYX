# CALYX

**Calyx-Aware Lighting-Yielded eXplainable Framework**
A physically-guided, single-stage detection framework for surface defect grading of Korla pears on production lines.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

---

## Overview

CALYX integrates physical optics priors, interpretable representation learning, and production-line detection within a single end-to-end framework. The framework consists of four innovation modules and one detection backbone:

| Module | Section | Purpose |
| :----- | :------ | :------ |
| **SADR** Specular-Aware Diffuse Recovery | Section 3.2 | Suppress specular highlights via Tan-Ikeuchi max-chromaticity decomposition + sigmoid soft gating |
| **PSPK** Photometric Shape-Prior Kernel block | Section 3.3 | Encode 4 local geometry-luminance signatures as 4 physically-initialized 4×4 convolution kernels |
| **PGI** Polarity-Guided Injection | Section 3.4 | Inject polarity priors into PAN-FPN P3/P4/P5 with independent W_geo + W_proj + η per scale |
| **DAR** Depth Alignment Regularization | Section 3.5.1 | Falsifiable verification of polarity map via scale-invariant log loss against pseudo-depth |
| **Backbone** | Section 4.2 | Ultralytics YOLO11n with PSPK embedded at P2 and P4 stages |

Auxiliary losses (Section 3.5.3): polar class consistency (KL divergence between P2 and P4 polarity distributions) and calyx linear probe (binary classification of DCF/PCF from 4-D mean polarity vectors).

---

## Repository Layout

```
calyx/
├── calyx/                         # Python package
│   ├── modules/                   # SADR, PSPK, PGI, DAR head
│   ├── losses/                    # SiLog, Polar consistency, Calyx probe, Total
│   ├── models/                    # End-to-end integration + Ultralytics registration
│   └── utils/                     # Visualization
├── configs/calyx_yolo11n.yaml     # Ultralytics YOLO11n + CALYX config
├── tests/                         # pytest unit tests for all modules
├── docs/                          # Architecture, usage, reproduction guides
├── examples/                      # Quickstart and visualization scripts
├── pyproject.toml                 # Package metadata
├── requirements.txt               # Runtime dependencies
└── README.md                      # This file
```

---

## Installation

### From source

```bash
git clone https://github.com/caoyuyu46-oss/CALYX.git
cd calyx
pip install -e .
```

### Dependencies

CALYX requires Python 3.9+ and PyTorch 2.0+. Optional dependencies:

- `ultralytics ≥ 8.3` for production deployment with YOLO11n backbone
- `matplotlib` for visualization utilities
- `pytest` for running unit tests

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Skeleton model (no Ultralytics dependency)

```python
import torch
from calyx import CalyxYOLOSkeleton, SiLogLoss, PolarConsistencyLoss

model = CalyxYOLOSkeleton(nc=4, p3_channels=128, p4_channels=256, p5_channels=512)
model.train()

img = torch.rand(2, 3, 640, 640)
out = model(img)

# out.detections, out.polarity_p2, out.polarity_p4, out.depth_pred
# out.polarity_per_scale (list of 3 tensors)
# out.sadr_intermediate (s_map, alpha)
```

### Full Ultralytics YOLO11n integration

```python
from calyx import register_calyx_modules
register_calyx_modules()

from ultralytics import YOLO
model = YOLO("configs/calyx_yolo11n.yaml")
model.train(data="<your_dataset>.yaml", epochs=300, imgsz=640)
```

### Compute total loss

```python
from calyx import CalyxTotalLoss, SiLogLoss, PolarConsistencyLoss, CalyxLinearProbe

silog = SiLogLoss(lambda_d=0.85)
polar = PolarConsistencyLoss(temperature=1.0)
probe = CalyxLinearProbe(n_features=4)
total = CalyxTotalLoss()

# In your training loop:
loss_det = ...                                               # standard YOLO loss
loss_dar = silog(out.depth_pred, depth_target)
loss_polar = polar(out.polarity_p2, out.polarity_p4)
loss_probe = probe(roi_polarity_vectors, calyx_subtype_labels)

result = total(loss_det, loss_dar, loss_polar, loss_probe)
result.total.backward()
```

---

## Reproducing Paper Figures

Each visualization in the paper is reproducible from `examples/`:

```bash
# SADR processing pipeline (Eqs. 3.5-3.7)
python examples/visualize_sadr.py --image path/to/pear.jpg --output sadr_pipeline.png

# PSPK kernels (no checkpoint required)
python examples/visualize_polarity.py --mode kernels --output pspk_kernels.png
```

---

## Tests

```bash
pytest tests/ -v
```

Coverage summary:
- `test_sadr.py`: 11 tests covering shape, no-params, hyperparameters, physical correctness of the corrected Eq. (3.5) (saturated color is diffuse; specular highlight detected; monotonicity; s ≤ I_min; non-negative recovery; achromatic-image guard; numerical stability)
- `test_pspk.py`: 10 tests covering K₁ zero-mean, K₂ = -K₁, K₃ block structure, K₄ near-uniform, forward shape, freezing
- `test_pgi.py`: 10 tests covering W_geo identity init, η in (0,1), Frobenius distance, multi-scale max-pool, additive injection, scale independence
- `test_losses.py`: 17 tests covering SiLog (λ_d range, scale invariance, masking), Polar (KL divergence properties), Probe (5 parameters, masking), Total (default weights, aggregation)
- `test_calyx_yolo.py`: 4 tests covering end-to-end shape, train/eval mode, DAR stripping

---

## Citation

If you use CALYX in your research, please cite the paper:

```bibtex
@article{calyx2026,
  title    = {Reflectance-aware and shape-prior-guided framework for
              production-line surface defect detection and interpretable
              grading of Korla pear},
  author   = {<authors>},
  journal  = {Computers and Electronics in Agriculture},
  year     = {2026},
  note     = {Under review.}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

---

## Acknowledgements

- The Korla pear dataset was annotated with Cohen's κ = 0.83 inter-rater agreement on a held-out subset.
- The SADR module follows the max-chromaticity decomposition framework of Tan and Ikeuchi (PAMI 2005).
- The DAR module uses Depth-Anything-V2 (Yang et al., NeurIPS 2024) as a frozen pseudo-depth source.
- The detection backbone is based on Ultralytics YOLO11n.
