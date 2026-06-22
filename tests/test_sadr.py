# -*- coding: utf-8 -*-
"""SADR unit tests (paper Section 3.2).

Covers the corrected Eq. (3.5) (achromatic specular strength) and Eqs. (3.6),
(3.7):
1. output shape consistency;
2. no learnable parameters / defaults matching Section 3.2.3;
3. physical correctness (regression guard):
   - highly saturated pure-colour pixels are diffuse, s ~ 0, alpha ~ 0
     (prevents the old "saturated colour mistaken for specular" behaviour);
   - a near-achromatic highlight on a coloured surface is detected as specular,
     s > 0, alpha -> 1, while the coloured surface has s ~ 0;
   - s increases monotonically with the achromatic increment of the highlight;
4. constraints: s <= I_min (per pixel); diffuse recovery is non-negative for
   inputs in [0, 1];
5. degeneracy and numerics: a near-achromatic (gray) image has no reliable
   chromaticity reference, so s = 0; no NaN/Inf.
"""

import pytest
import torch

from calyx.modules.sadr import SADR


@pytest.fixture
def sadr():
    return SADR()


def _fill(rgb, h=16, w=16):
    """Build an image whose every pixel is the given RGB, shape [1, 3, h, w]."""
    img = torch.zeros(1, 3, h, w)
    for c in range(3):
        img[:, c, :, :] = rgb[c]
    return img


# ---- basic properties ----

def test_output_shape(sadr):
    img = torch.rand(2, 3, 32, 32)
    i_tilde, s, alpha = sadr(img)
    assert i_tilde.shape == (2, 3, 32, 32)
    assert s.shape == (2, 1, 32, 32)
    assert alpha.shape == (2, 1, 32, 32)


def test_no_learnable_parameters(sadr):
    n = sum(p.numel() for p in sadr.parameters() if p.requires_grad)
    assert n == 0, f"SADR should have no learnable parameters, got {n}"


def test_hyperparameters_default():
    sadr = SADR()
    assert float(sadr.s0) == pytest.approx(0.20)
    assert float(sadr.gamma) == pytest.approx(12.0)
    assert float(sadr.tau) == pytest.approx(0.85)


# ---- physical correctness (regression guard) ----

def test_saturated_color_is_diffuse(sadr):
    """Saturated pure-colour pixels are diffuse: almost no correction.

    Regression guard against the old formula, which mistook saturated colour
    for strong specular.
    """
    img = _fill([0.9, 0.1, 0.1])  # saturated red
    _, s, alpha = sadr(img)
    assert s.max() < 1e-3, f"saturated-colour s should be ~0, got {float(s.max()):.4f}"
    assert alpha.max() < 0.05, f"saturated-colour alpha should be ~0, got {float(alpha.max()):.4f}"


def test_specular_highlight_detected(sadr):
    """A near-achromatic highlight on a coloured surface is specular (s>0,
    alpha->1), while the surface itself has s~0."""
    img = _fill([0.6, 0.4, 0.2])                      # yellow-green surface
    img[:, 0, 6:10, 6:10] = 0.95                       # near-achromatic highlight
    img[:, 1, 6:10, 6:10] = 0.92
    img[:, 2, 6:10, 6:10] = 0.88
    _, s, alpha = sadr(img)
    spot = s[0, 0, 6:10, 6:10]
    surf_mask = torch.ones(16, 16, dtype=torch.bool)
    surf_mask[6:10, 6:10] = False
    surf = s[0, 0][surf_mask]
    assert spot.mean() > 0.1, f"highlight s should be >0, got {float(spot.mean()):.4f}"
    assert alpha[0, 0, 6:10, 6:10].mean() > 0.5, "highlight alpha should be >0.5"
    assert surf.mean() < 1e-3, f"coloured-surface s should be ~0, got {float(surf.mean()):.4f}"


def test_specular_strength_monotonic_in_highlight(sadr):
    """Estimated specular strength increases with the achromatic increment."""
    base = [0.6, 0.4, 0.2]
    means = []
    for delta in [0.0, 0.1, 0.2, 0.35]:
        img = _fill(base)
        for c in range(3):
            img[:, c, 6:10, 6:10] = min(base[c] + delta, 1.0)
        _, s, _ = sadr(img)
        means.append(float(s[0, 0, 6:10, 6:10].mean()))
    assert all(means[i] < means[i + 1] for i in range(len(means) - 1)), (
        f"specular strength should increase with the achromatic increment, got {means}"
    )


# ---- constraints ----

def test_strength_not_exceed_min_channel(sadr):
    """s(x) must not exceed the minimum channel I_min(x), keeping recovery
    non-negative."""
    img = torch.rand(2, 3, 24, 24)
    _, s, _ = sadr(img)
    i_min = img.min(dim=1, keepdim=True).values
    assert torch.all(s <= i_min + 1e-6)


def test_recovered_image_nonnegative(sadr):
    """Diffuse recovery must be non-negative for inputs in [0, 1]."""
    img = torch.rand(2, 3, 24, 24)
    i_tilde, _, _ = sadr(img)
    assert i_tilde.min() >= -1e-6, f"I_tilde should be non-negative, got min={float(i_tilde.min()):.4f}"


# ---- degeneracy and numerical stability ----

def test_achromatic_image_no_correction(sadr):
    """A near-achromatic (gray) image lacks a reliable diffuse chromaticity
    reference, so s = 0 (no fabricated correction)."""
    img = _fill([0.5, 0.5, 0.5])
    i_tilde, s, alpha = sadr(img)
    assert torch.all(s < 1e-3), f"gray-image s should be ~0, got max={float(s.max()):.4f}"
    assert torch.isfinite(i_tilde).all()


def test_numerical_stability(sadr):
    """Neither a random image nor a near-achromatic perturbed image yields
    NaN/Inf."""
    for img in (torch.rand(1, 3, 16, 16), _fill([0.5, 0.5, 0.5]) + 1e-7):
        i_tilde, s, alpha = sadr(img)
        assert torch.isfinite(i_tilde).all()
        assert torch.isfinite(s).all()
        assert torch.isfinite(alpha).all()


def test_fixed_reference_chroma_runs():
    """An explicit Lambda_ref (line-calibration mode) runs and stays finite."""
    sadr = SADR(ref_chroma=0.5)
    img = torch.rand(1, 3, 16, 16)
    _, s, _ = sadr(img)
    assert torch.isfinite(s).all()
    i_min = img.min(dim=1, keepdim=True).values
    assert torch.all(s <= i_min + 1e-6)
