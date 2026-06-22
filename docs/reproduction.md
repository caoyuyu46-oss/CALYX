# Figure Reproduction Protocol

This document explains how to reproduce the qualitative visualizations of the paper from the codebase. Each figure has its own example script under `examples/`. Reproductions require trained checkpoints; for unconditional reproductions (no training required, only the kernel banks), see Section 1 below.

---

## 1. Unconditional Reproductions (no training required)

### 1.1 PSPK kernels (K_1, K_2, K_3, K_4)

The four physically-initialized kernels are deterministic before any training. The example script visualizes their initialization values:

```bash
python examples/visualize_polarity.py --mode kernels --output figures/fig_3_2_kernels.png
```

**Expected outcome.** A 1×4 grid of heatmaps. Diverging colormap (RdBu_r recommended) with center 0.

- `K_1` (concave): center 2×2 dark blue (negative), surrounding ring light red (positive).
- `K_2` (convex): exact sign-inverted `K_1`.
- `K_3` (saddle): top-left and bottom-right 2×2 in red, off-diagonals in blue.
- `K_4` (planar): nearly uniform light gray with subtle perturbation visible.

### 1.2 SADR pipeline

The SADR module has no learnable parameters. Given any RGB image (with visible specular highlights), the pipeline produces a four-panel figure showing input → specular strength → soft gate → recovered:

```bash
python examples/visualize_sadr.py \
    --image path/to/input.jpg \
    --output figures/fig_3_3_sadr.png
```

**Expected outcome.** Four panels in a horizontal strip:

1. **Input RGB** — original image with specular highlights.
2. **Specular strength `s(x)`** — viridis-colored, bright = high specular component.
3. **Soft gate `α(x)`** — viridis-colored, smooth transition (no hard edges).
4. **Recovered `Ĩ(x)`** — RGB with specular suppressed; surface texture revealed.

If the input has no specular highlights, panels 2 and 3 will be near-zero everywhere and `Ĩ ≈ I`. The script accepts any RGB image; the `examples/sample_pear.jpg` (if provided) gives a representative pear example.

---

## 2. Conditional Reproductions (require trained checkpoint)

### 2.1 Polarity maps after training

Once you have a trained checkpoint, visualize the polarity map output of `PSPKBlock` on representative test images. The polarity map has 4 channels — one per geometry class — that should activate selectively over geometry-matching regions:

```bash
python examples/visualize_polarity.py \
    --mode polarity \
    --checkpoint path/to/calyx.pt \
    --image path/to/test_image.jpg \
    --output figures/fig_3_4_polarity.png
```

**Expected outcome.** A 1×5 grid: input image (panel 1) and four polarity channel responses (panels 2–5). Per channel:
- `P_1` (concave) lights up over genuine concave defects and calyx pits.
- `P_2` (convex) lights up over convex deformations or stems.
- `P_3` (saddle) shows weaker, less spatially clustered responses.
- `P_4` (planar) lights up over smooth pear surface regions away from defects.

### 2.2 Figure 4-X: W_geo Frobenius distance training dynamics

Reproducing the training-dynamics plot of `||W_geo - I_4||_F` over training steps requires running training with logging enabled. Add the following to your training callback:

```python
import torch

def log_pgi_dynamics(model, step, writer):
    """Log Frobenius distances of W_geo at all three scales."""
    for branch_name, dist in zip(["P3", "P4", "P5"], model.pgi.frobenius_distances()):
        writer.add_scalar(f"pgi/W_geo_distance_{branch_name}", dist.item(), step)
        writer.add_scalar(f"pgi/eta_{branch_name}",
                         torch.sigmoid(model.pgi.branches[branch_name].eta_logit).item(), step)
```

The expected pattern, per paper Section 4.5.6: distances start at 0 (initialization is `I_4`), grow rapidly during the first ~10% of training, then stabilize. The P3 distance typically grows largest (~0.4–0.7), reflecting the strong polarity reorganization at the smallest scale where geometric signals are most direct.

### 2.3 Figure 4-Y: Linear probe DCF/PCF accuracy on PearDepth-Pro

After training, run the calyx ROI extraction on PearDepth-Pro:

```bash
python examples/eval_calyx_probe.py \
    --checkpoint path/to/calyx.pt \
    --data data/PearDepth-Pro \
    --output results/probe_dcf_pcf.json
```

This script (you may need to adapt it to your evaluation pipeline) outputs:
- DCF/PCF accuracy with 95% confidence interval.
- Per-pear classification confusion matrix.
- Comparison against hyperspectral SPA-SVM baseline (paper Section 4.9), if you have run that baseline separately.

---

## 3. Reproducing Paper Numerics

### 3.1 PSPK kernel sum-zero verification (paper Section 3.3.2)

```bash
pytest tests/test_pspk.py::test_kernel_sums -v
```

Confirms `K_1.sum() = 0`, `K_2.sum() = 0`, `K_3.sum() = 0`, `K_4.sum() ≈ 1` to numerical precision.

### 3.2 PSPK kernel symmetry verification (paper Eq. 3.9)

```bash
pytest tests/test_pspk.py::test_K2_equals_negative_K1 -v
```

Confirms `K_2 = -K_1` exactly at initialization.

### 3.3 PGI scale independence verification

```bash
pytest tests/test_pgi.py::test_pgi_three_scales_independent -v
```

Confirms that `W_geo`, `W_proj`, and `eta` for the three scales are independently parameterized (modifying one does not affect the others).

### 3.4 Total loss weight sanity check

```bash
pytest tests/test_losses.py::test_total_loss_default_weights -v
```

Confirms `CalyxLossWeights()` returns `(0.10, 0.30, 0.50)`, matching paper Eq. (3.19) defaults.

### 3.5 SADR no-correction-on-pure-diffuse verification

```bash
pytest tests/test_sadr.py::test_pure_diffuse_zero_correction -v
```

Confirms that for a synthetic pure-diffuse image (where `c_max = 1/3` everywhere), the recovered image equals the input within numerical precision.

---

## 4. Tips for Faithful Reproduction

### 4.1 Determinism

For exactly reproducible results across runs, set:

```python
import torch
import random
import numpy as np

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 4.2 Numerical precision

- `SADR` uses `eps = 1e-6`. With FP16 training, this may need to be `1e-4` to avoid underflow. Pass `SADR(eps=1e-4)` or train in BF16 / FP32.
- `PolarConsistencyLoss` uses `eps = 1e-8` for the log clip; this is small enough to be safe in FP32 but borderline in FP16. Use BF16 if you need lower precision.

### 4.3 Hardware considerations

The standalone skeleton model runs on CPU for unit tests; expect ~5 s for a forward pass at `batch=1, imgsz=640`. For training and meaningful reproductions, use a GPU with at least 12 GB VRAM (we tested on RTX 3090 and A100; both work with `batch=32, imgsz=640`).

---

## 5. Troubleshooting

### 5.1 "PSPK initialization values look different from the paper"

The paper uses fractional values like `1/3` and `-1` for `K_1`. Floating-point representation will show `0.333...` in PyTorch tensors. The kernel sum-zero test passes within `1e-7` tolerance. Visualization with a coarser colormap may make the values look slightly different from the paper figure.

### 5.2 "Checkpoint loaded but polarity maps look noisy"

Common cause: the checkpoint was saved before training had converged the kernel weights. Check the training logs for `L_polar` curve — it should plateau before saving. Re-train for more epochs or use a later checkpoint.

### 5.3 "Frobenius distance starts non-zero"

The `frobenius_distances()` method returns `||W_geo - I_4||_F` as a `torch.Tensor`. If you compute it before any optimizer step, the value should be **zero** (since `W_geo` is initialized to `I_4`). If it's non-zero from the start, check whether you accidentally re-initialized `W_geo` somewhere (e.g., a custom weight init function applied uniformly).

---

## 6. See Also

- `examples/quick_start.py` — runs the complete forward pass on synthetic data.
- `examples/visualize_polarity.py` — kernel and polarity map visualization.
- `examples/visualize_sadr.py` — SADR pipeline visualization.
- `tests/` — comprehensive unit tests.
