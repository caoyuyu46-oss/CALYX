# -*- coding: utf-8 -*-
"""PSPK unit tests.

Covers:
1. K_1 zero mean (4*(-1) + 12*(1/3) = 0);
2. K_2 = -K_1 first-order symmetry;
3. K_3 saddle zero mean (diagonal blocks cancel);
4. K_4 near-uniform (1/16 body + small perturbation);
5. PSPKBlock output [B, C+4, H, W] and polarity map [B, 4, H, W];
6. four kernels.
"""

import pytest
import torch

from calyx.modules.pspk import (
    build_pspk_kernel_bank,
    _build_K1,
    _build_K2,
    _build_K3,
    _build_K4,
    PSPKBlock,
)


def test_k1_zero_mean():
    """K_1 (centre -1, outer +1/3) is zero mean."""
    k1 = _build_K1()
    assert k1.shape == (4, 4)
    assert k1.sum().item() == pytest.approx(0.0, abs=1e-6)
    # centre 2x2 all -1
    assert torch.all(k1[1:3, 1:3] == -1.0)
    # outer 12 pixels all +1/3
    mask = torch.ones_like(k1, dtype=torch.bool)
    mask[1:3, 1:3] = False
    assert torch.allclose(k1[mask], torch.full((12,), 1.0 / 3.0))


def test_k2_equals_neg_k1():
    """K_2 = -K_1 first-order symmetric approximation (Eq. 3.9)."""
    k1 = _build_K1()
    k2 = _build_K2()
    assert torch.allclose(k2, -k1)


def test_k3_zero_mean_and_block_structure():
    """K_3 diagonal-block design, zero mean."""
    k3 = _build_K3()
    assert k3.shape == (4, 4)
    assert k3.sum().item() == pytest.approx(0.0, abs=1e-6)
    # top-left and bottom-right 2x2 same sign (positive)
    assert torch.all(k3[0:2, 0:2] > 0)
    assert torch.all(k3[2:4, 2:4] > 0)
    # top-right and bottom-left 2x2 same sign (negative)
    assert torch.all(k3[0:2, 2:4] < 0)
    assert torch.all(k3[2:4, 0:2] < 0)


def test_k4_near_uniform():
    """K_4 body ~ 1/16, perturbation << body."""
    k4 = _build_K4()
    assert k4.shape == (4, 4)
    # mean very close to 1/16
    assert k4.mean().item() == pytest.approx(1.0 / 16.0, abs=1e-3)
    # std far below the body value
    assert k4.std().item() < 1.0 / 16.0


def test_kernel_bank_shape():
    """Kernel bank shape [4, 1, 4, 4]."""
    bank = build_pspk_kernel_bank()
    assert bank.shape == (4, 1, 4, 4)


def test_pspk_forward_shape():
    """PSPK output (output, polarity) shapes are correct."""
    pspk = PSPKBlock(in_channels=64)
    feat = torch.randn(2, 64, 16, 16)
    output, polarity = pspk(feat)
    assert output.shape == (2, 64 + 4, 16, 16)
    assert polarity.shape == (2, 4, 16, 16)


def test_pspk_polarity_map_only():
    """The polarity_map interface works on its own."""
    pspk = PSPKBlock(in_channels=32)
    feat = torch.randn(1, 32, 8, 8)
    p = pspk.polarity_map(feat)
    assert p.shape == (1, 4, 8, 8)


def test_pspk_freeze_kernels():
    """With freeze_kernels=True the kernels do not update."""
    pspk = PSPKBlock(in_channels=16, freeze_kernels=True)
    assert not pspk.kernels.requires_grad


def test_pspk_default_trainable():
    """Default freeze_kernels=False, kernels are trainable."""
    pspk = PSPKBlock(in_channels=16)
    assert pspk.kernels.requires_grad


def test_pspk_invalid_input_raises():
    """A channel-count mismatch with construction raises."""
    pspk = PSPKBlock(in_channels=64)
    bad_feat = torch.randn(1, 32, 8, 8)
    with pytest.raises(ValueError):
        pspk(bad_feat)
