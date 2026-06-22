# CALYX Architecture

This document explains the implementation of each CALYX module with strict alignment to **Paper Chapter 3 (v6.1)**. Each section maps a code component to its paper section and lists the equations it implements. Read this together with the paper; this document does not duplicate the physical motivation but focuses on **how** the math becomes code.

---

## 1. Module Map

| Paper section | Code path | Class / function |
| :------------ | :-------- | :--------------- |
| Section 3.1.3 framework structure | `calyx/models/calyx_yolo.py` | `CalyxYOLOSkeleton` |
| Section 3.2 SADR | `calyx/modules/sadr.py` | `SADR` |
| Section 3.3 PSPK | `calyx/modules/pspk.py` | `PSPKBlock`, `_build_K1`, `_build_K2`, `_build_K3`, `_build_K4` |
| Section 3.4 PGI | `calyx/modules/pgi.py` | `PGI`, `_SinglePGI` |
| Section 3.5.1 DAR head | `calyx/modules/dar.py` | `DARHead` |
| Section 3.5.2 silog loss (Eq. 3.16) | `calyx/losses/silog.py` | `SiLogLoss` |
| Section 3.5.3 polar consistency (Eq. 3.17) | `calyx/losses/polar_consistency.py` | `PolarConsistencyLoss` |
| Section 3.5.3 calyx probe (Eq. 3.18) | `calyx/losses/calyx_probe.py` | `CalyxLinearProbe` |
| Section 3.5.4 total loss (Eq. 3.19) | `calyx/losses/total.py` | `CalyxTotalLoss` |
| Section 4.2 backbone integration | `calyx/models/register.py` | `register_calyx_modules` |

---

## 2. SADR — Specular-Aware Diffuse Recovery

**Paper anchor:** Section 3.2, equations (3.5)–(3.7)
**Code anchor:** `calyx/modules/sadr.py`

### 2.1 Three sub-steps

The forward pass decomposes into three pure-tensor sub-steps with no learnable parameters. Each is a standalone method to support unit testing and visualization:

```
SADR.forward(I)
├── specular_strength(I)      → s        # Eq. (3.5)
├── soft_gate(s)              → α        # Eq. (3.6)
└── diffuse_recover(I, s, α)  → Ĩ        # Eq. (3.7)
return (Ĩ, s, α)
```

### 2.2 Equation (3.5) numerical handling

The paper defines

$$s(\mathbf{x}) = \max\!\left(0,\; I_{\max} - \frac{c_{\max}}{c_{\max} - 1/3 + \varepsilon}\cdot[\textstyle\sum_k I_k - I_{\max}]\right)$$

Two corner cases need attention:

1. **Pure achromatic pixel** (`c_max → 1/3`): denominator approaches `eps`, the main branch produces a large positive value, the outer `max(0, I_max - large)` clips to **0** — exactly matching the paper's "no correction needed for purely diffuse pixels" intent.
2. **Zero-sum pixel** (`I = 0`): the code computes `c_max = I_max / (I_sum + eps)` to avoid division by zero; subsequent operations remain bounded.

The implementation uses `torch.clamp(raw, min=0.0)` for the outer max. `eps = 1e-6` is sufficient because all RGB values are non-negative.

### 2.3 Hyperparameters

Default values are taken from Section 3.2.3:

```python
SADR(s0=0.20, gamma=12.0, tau=0.85, eps=1e-6)
```

Hyperparameters are registered as `buffer` (not `Parameter`), so they:
- serialize cleanly into `state_dict` for cross-device transfer;
- are not picked up by the optimizer;
- can be overridden at inference time without affecting checkpoints.

### 2.4 Why no learnable parameters?

The paper's design rationale (Section 3.2.3) is that SADR functions as a **deterministic preprocessing operator** providing geometry-dominated input to PSPK. A learnable SADR would entangle preprocessing with the geometry encoder and weaken the falsifiability of the geometry-luminance signature mapping (paper Φ in Eq. 3.4).

---

## 3. PSPK — Photometric Shape-Prior Kernels

**Paper anchor:** Section 3.3, equations (3.8)–(3.10)
**Code anchor:** `calyx/modules/pspk.py`

### 3.1 Four physically-initialized 4×4 kernels

Following Eq. (3.8) and Eq. (3.9), four kernels are constructed by helper functions:

| Kernel | Role | Construction (`_build_K*`) |
| :----- | :--- | :------------------------- |
| `K_1` (concave) | Center darker, ring brighter | `[1/3]*16` then center 2×2 set to `-1`. Sum = 0. |
| `K_2` (convex) | Sign-inverted `K_1` | `K_2 = -K_1`. First-order symmetric approximation; trainable thereafter. |
| `K_3` (saddle) | Diagonal block `+/−` | Top-left and bottom-right 2×2 blocks `+1/√8`; off-diagonal `-1/√8`. Sum = 0. |
| `K_4` (planar) | Near-uniform | `J/16 + ε·E` with small Frobenius perturbation `E`. Breaks symmetry to avoid frozen gradients. |

All four kernels stack into a 4-channel kernel bank `[4, 1, 4, 4]` returned by `build_pspk_kernel_bank()`.

### 3.2 Two-step forward pass

```
PSPKBlock.forward(F)
├── g = self.input_proj(F)              # 1×1 Conv: C → 1, "geometry-sensitive map"
├── P = F.conv2d(g, self.kernels, padding=...)   # 1 → 4 via K_1..K_4
└── return (concat(F, P), P)            # (C+4, H, W) for backbone, (4, H, W) for PGI
```

The `input_proj` (1×1 Conv from `C` to 1) is the only learnable component before the kernel bank. The paper (Section 3.3.2) describes this as "a learnable grayscale operator over the C-channel features at the embedding stage", letting the model decide which feature channels carry the geometry-relevant signal.

### 3.3 Padding choice for 4×4 kernels

A 4×4 kernel has **no canonical center**. The implementation uses `padding=2` (PyTorch convention) and accepts a 1-pixel asymmetric output. To preserve `(H, W)`, the forward pass crops the trailing row and column. This implementation choice is documented in `pspk.py` and matches the standard treatment of even-sized kernels.

### 3.4 Trainability

By default, all four kernels are `nn.Parameter` (trainable). The paper (Section 3.3.2) explicitly allows training to refine the first-order symmetric approximation `K_2 = -K_1` and the axis-aligned saddle assumption in `K_3`. To freeze them (for ablation experiments matching paper Section 4.5.1), call:

```python
pspk = PSPKBlock(in_channels=64, freeze_kernels=True)
```

This freezes the kernel bank but **not** the `input_proj` 1×1 Conv.

### 3.5 Embedding positions

Per Section 3.3.2 and the YAML config, PSPK is embedded at backbone P2 (stride 4) and P4 (stride 16). Each insertion is an independent `PSPKBlock` instance with its own learned parameters. The output polarity map at P2 is then used by:

1. **Backbone P2 onward** as the additional 4 channels in the concatenated feature map.
2. **PGI** as the input polarity for multi-scale injection (downsampled to P3/P4/P5).
3. **DAR head** as the input feature for pseudo-depth prediction (training only).

---

## 4. PGI — Polarity-Guided Injection

**Paper anchor:** Section 3.4, equations (3.11)–(3.14)
**Code anchor:** `calyx/modules/pgi.py`

### 4.1 Three independent scale branches

`PGI` holds three independent `_SinglePGI` modules — one each for P3, P4, P5. Each branch has its own:

- `W_geo`: 4×4 matrix, initialized to the identity `I_4` (Eq. 3.13 default).
- `W_proj`: 1×1 conv from 4 channels to `C_s` (target scale's channel count).
- `eta_logit`: scalar, sigmoid-transformed to obtain `eta ∈ (0, 1)`. Initialized to `logit(0.5) = 0`.

### 4.2 Forward pass per scale

```
_SinglePGI.forward(F_in, P_full)
├── P_s = channel_separated_max_pool(P_full, target_size)   # Eq. (3.11)
├── P_geo = einsum('ij,bjhw->bihw', W_geo, P_s)             # 4 → 4 mixing
├── P_proj = self.W_proj(P_geo)                              # 4 → C_s, 1×1 conv
├── eta = sigmoid(eta_logit)
└── return F_in + eta * P_proj                               # Eq. (3.13) additive injection
```

Note the **additive** (not multiplicative) injection, matching Eq. (3.13). Multiplicative gating would couple polarity strength to feature magnitude and break the linear interpretability of `W_geo`.

### 4.3 Channel-separated max-pool (Eq. 3.12)

The paper specifies that downsampling preserves the **strongest response per geometry class independently**. The implementation:

```python
P_s = F.adaptive_max_pool2d(P_full, output_size=(H_s, W_s))
```

`adaptive_max_pool2d` operates per-channel by default; this is exactly the channel-separated max-pool the paper requires. **Average pooling is not used**, since polarity values can be negative and averaging would cancel opposite signs (paper Section 3.4.2).

### 4.4 Frobenius distance (Eq. 3.14)

A method `frobenius_distances()` returns `[d_F^P3, d_F^P4, d_F^P5]` where each `d_F^(s) = ||W_geo^(s) - I_4||_F`. The paper Section 4.5.6 uses these three scalars to study the training dynamics of geometry-class mixing across scales.

### 4.5 Why two-stage projection?

The paper Section 3.4.3 acknowledges that `W_proj · W_geo` is mathematically equivalent to a single `C_s × 4` matrix. The decomposition is preserved for **interpretability**: the 4×4 `W_geo` directly reads as a "geometry class mixing matrix", which is opaque if the two are merged. This decomposition does not affect expressiveness or training compute; it only structures the parameters for post-hoc analysis (Frobenius distance plots in paper Fig. 4.5.6).

---

## 5. DAR — Depth Alignment Regularization

**Paper anchor:** Section 3.5.1, used by Eq. (3.16) (silog loss)
**Code anchor:** `calyx/modules/dar.py` and `calyx/losses/silog.py`

### 5.1 DARHead architecture

A lightweight 1×1 conv stack reading from the polarity map (4 channels) at the P3 stage and outputting a single-channel relative depth estimate `D̂(x)`. Output passes through `softplus` to enforce non-negativity (relative depth ≥ 0 in the units of `D*`).

```
DARHead = Sequential(
    Conv1x1(4, hidden_dim),
    SiLU,
    Conv1x1(hidden_dim, hidden_dim),
    SiLU,
    Conv1x1(hidden_dim, 1),
    Softplus,
)
```

`hidden_dim` defaults to 32. Total parameters: roughly `4*32 + 32*32 + 32*1 = 1184` (with biases, ~1248). This is intentionally tiny — DAR is a regularization signal, not a depth predictor competing with Depth-Anything-V2.

### 5.2 Training-only branch

`CalyxYOLOSkeleton.export_inference_only()` strips the DAR head before ONNX export. After export, the inference graph contains only:

- SADR
- PSPK
- PGI (three branches)
- YOLO detection head

The auxiliary linear probe (`CalyxLinearProbe`) is also stripped at export. The only inference-time computation is the detection head; **training-only artifacts produce zero inference overhead** (paper Section 3.1.1 design principle P3).

### 5.3 SiLog loss (Eq. 3.16)

```python
SiLogLoss(lambda_d=1.0).forward(D_hat, D_star)

# d = log(D_hat) - log(D_star)
# L = mean(d²) - lambda_d * (mean(d))²
```

`lambda_d` ∈ [0, 1]. `lambda_d = 1` reduces to the variance of the log residual (fully scale-invariant); `lambda_d = 0` reduces to MSE on log space. The default `1.0` matches Eigen et al. 2014 [10].

A configurable mask is supported: pixels outside the foreground (e.g., conveyor background) can be excluded by passing a 0/1 mask as `valid`.

---

## 6. Polar Consistency Loss

**Paper anchor:** Section 3.5.3, Eq. (3.17)
**Code anchor:** `calyx/losses/polar_consistency.py`

### 6.1 KL divergence between scale distributions

The loss measures KL divergence between the per-pixel softmax distributions of P2 and P4 polarity maps:

```
PolarConsistencyLoss.forward(P2, P4)

q^(P2) = softmax(P^(P2), dim=1)              # [B, 4, H_2, W_2]
q^(P4) = softmax(P^(P4), dim=1)              # [B, 4, H_4, W_4]
q^(P4)_up = upsample(q^(P4), size=(H_2, W_2))   # bilinear or nearest

L = sum_{x in Omega} sum_{i=1..4}
       q^(P2)_i(x) * log(q^(P2)_i(x) / q^(P4)_up_i(x))
L_polar = L / |Omega|
```

This implements paper Eq. (3.17). The use of softmax converts the raw polarity map into a probability distribution over the four geometry classes, making the cross-scale comparison invariant to the absolute polarity magnitude (which differs between P2 and P4 due to different receptive fields).

### 6.2 Numerical stability

KL divergence diverges if any `q^(P4)_up_i(x) = 0`. The implementation adds a small `eps = 1e-8` clip before taking logarithms. Both upsampling modes (`bilinear` and `nearest`) are supported via the `up_mode` argument.

### 6.3 Failure modes

The loss is appropriate when the dominant geometry class at P2 and P4 should be the same (typical for objects whose physical scale is between the P2 and P4 receptive field bounds, ~5 mm to ~20 mm). For extreme small targets below P2's receptive field, the assumption may break and the consistency loss can be down-weighted via `lambda_2` in the total loss (paper Section 3.5.3 caveat).

---

## 7. Calyx Linear Probe

**Paper anchor:** Section 3.5.3, Eq. (3.18)
**Code anchor:** `calyx/losses/calyx_probe.py`

### 7.1 5-parameter probe

```python
CalyxLinearProbe()  # 4 weights + 1 bias = 5 trainable parameters

probe.forward(p_calyx_mean: torch.Tensor) -> torch.Tensor
# Input: [B, 4]  - mean polarity vector pooled over each calyx ROI
# Output: [B]    - DCF/PCF logit (apply sigmoid + BCE for loss)
```

The probe is intentionally minimal (5 parameters) per paper Section 3.5.3. It does not solve the calyx orientation classification by itself; rather, its low capacity acts as a rigorous test of the polarity vector's direct discriminative power for DCF/PCF distinction.

### 7.2 Mean pooling over ROI

Before applying the probe, the upstream pipeline must:

1. Detect `calyx` boxes via the YOLO head.
2. Crop the polarity map to each box.
3. Mean-pool spatially within the box, producing a 4-D vector per box.

The `CalyxLinearProbe` consumes only this 4-D vector. ROI pooling logic is left to the training script (see `docs/usage.md`).

---

## 8. Total Loss (Eq. 3.19)

**Paper anchor:** Section 3.5.4
**Code anchor:** `calyx/losses/total.py`

```python
CalyxLossWeights(lambda_1=0.10, lambda_2=0.30, lambda_3=0.50)

L_total = L_det + lambda_1 * L_DAR + lambda_2 * L_polar + lambda_3 * L_probe
```

`CalyxTotalLoss.forward(...)` returns a `CalyxLossOutput` dataclass with all four components and the weighted total. This makes loss curves easy to monitor per component during training.

The default weights `(0.10, 0.30, 0.50)` come from Section 3.5.4 ablation. They reflect a deliberate ordering: detection ≥ probe (highest auxiliary, most reliable supervision) > polar consistency > DAR (lowest, since pseudo-depth is the least reliable signal).

---

## 9. End-to-End Skeleton

**Paper anchor:** Section 3.1.3 framework total architecture
**Code anchor:** `calyx/models/calyx_yolo.py`

`CalyxYOLOSkeleton` is a teaching reference implementation that wires SADR → backbone (with PSPK at P2 and P4) → neck (with PGI at P3, P4, P5) → detection head + DAR head + probe. It does **not** depend on Ultralytics; it is a pure-PyTorch demonstration so that researchers without the Ultralytics dependency can still inspect the data flow.

For production training with Ultralytics YOLO11n, use:

```python
from calyx.models.register import register_calyx_modules
from ultralytics import YOLO

register_calyx_modules()  # injects PSPKBlock, PGI, DARHead into ultralytics.nn.tasks
model = YOLO("configs/calyx_yolo11n.yaml")
model.train(data=..., epochs=...)
```

`register_calyx_modules()` patches `ultralytics.nn.tasks` so the YAML parser can recognize `PSPKBlock`, `PGI`, `DARHead` as layer types. The YAML at `configs/calyx_yolo11n.yaml` then specifies their positions in backbone and neck.

---

## 10. Mapping to Paper Equations

| Paper Eq. | Code call site |
| :-------- | :------------- |
| (3.1) frame composition | `CalyxYOLOSkeleton.forward` |
| (3.2) reflectance model | not in code (assumption) |
| (3.3) curvature taxonomy | not in code (motivation) |
| (3.4) Φ definition | conceptual; PSPK implements the inner-product score, not argmax |
| (3.5) specular strength | `SADR.specular_strength` |
| (3.6) soft gate α | `SADR.soft_gate` |
| (3.7) diffuse recover | `SADR.diffuse_recover` |
| (3.8) PSPK kernel role | `_build_K1`, `_build_K2`, `_build_K3`, `_build_K4` |
| (3.9) K matrix specs | inline numeric values in helpers |
| (3.10) zero-mean check | unit test `test_pspk_zero_mean` |
| (3.11) max-pool downsample | `_SinglePGI` via `adaptive_max_pool2d` |
| (3.12) channel-separated semantics | per-channel pooling |
| (3.13) additive injection | `_SinglePGI.forward` |
| (3.14) Frobenius distance | `PGI.frobenius_distances` |
| (3.15) DAR conv stack | `DARHead.__init__` |
| (3.16) silog loss | `SiLogLoss.forward` |
| (3.17) polar KL | `PolarConsistencyLoss.forward` |
| (3.18) probe BCE | `CalyxLinearProbe` + `BCEWithLogitsLoss` |
| (3.19) total loss | `CalyxTotalLoss.forward` |

---

## See Also

- `docs/usage.md` — training loops, data preparation, ROI alignment.
- `docs/reproduction.md` — figure-by-figure reproduction protocol.
- `examples/quick_start.py` — minimal end-to-end skeleton demo.
- `tests/` — unit tests verifying each module against paper specifications.
