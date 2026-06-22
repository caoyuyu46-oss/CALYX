# -*- coding: utf-8 -*-
"""Unit tests for the DAR head and all losses."""

import pytest
import torch

from calyx.modules.dar import DARHead
from calyx.losses.silog import SiLogLoss
from calyx.losses.polar_consistency import PolarConsistencyLoss
from calyx.losses.calyx_probe import CalyxLinearProbe
from calyx.losses.total import (
    CalyxTotalLoss,
    CalyxLossWeights,
)


# -------- DAR head --------

def test_dar_head_output_shape():
    head = DARHead(in_channels=128)
    feat = torch.randn(2, 128, 32, 32)
    depth = head(feat)
    assert depth.shape == (2, 1, 32, 32)


def test_dar_head_output_nonneg():
    """softplus keeps depth non-negative."""
    head = DARHead(in_channels=64)
    feat = torch.randn(1, 64, 16, 16)
    depth = head(feat)
    assert torch.all(depth >= 0.0)


def test_dar_head_with_output_size():
    head = DARHead(in_channels=64, output_size=(64, 64))
    feat = torch.randn(1, 64, 16, 16)
    depth = head(feat)
    assert depth.shape == (1, 1, 64, 64)


# -------- SiLog loss --------

def test_silog_lambda_d_range():
    """lambda_d must be in [0, 1]."""
    SiLogLoss(lambda_d=0.0)   # OK
    SiLogLoss(lambda_d=1.0)   # OK
    SiLogLoss(lambda_d=0.85)  # default
    with pytest.raises(ValueError):
        SiLogLoss(lambda_d=-0.1)
    with pytest.raises(ValueError):
        SiLogLoss(lambda_d=1.1)


def test_silog_zero_when_pred_equals_target():
    loss_fn = SiLogLoss()
    x = torch.full((2, 1, 8, 8), 1.5)
    loss = loss_fn(x, x)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_silog_scale_invariance_at_lambda_1():
    """At lambda_d = 1 the loss is invariant to a global scaling of pred."""
    loss_fn = SiLogLoss(lambda_d=1.0)
    target = torch.rand(1, 1, 16, 16) + 1.0
    pred = torch.rand(1, 1, 16, 16) + 1.0
    loss_a = loss_fn(pred, target)
    loss_b = loss_fn(pred * 5.0, target)
    assert float(loss_a) == pytest.approx(float(loss_b), abs=1e-5)


def test_silog_with_mask():
    loss_fn = SiLogLoss()
    pred = torch.rand(1, 1, 8, 8) + 0.1
    target = torch.rand(1, 1, 8, 8) + 0.1
    mask = torch.zeros_like(pred)
    mask[..., 4:, 4:] = 1.0
    loss = loss_fn(pred, target, mask)
    assert torch.isfinite(loss)


# -------- polar consistency --------

def test_polar_consistency_zero_when_identical():
    """Identical P2 and upsampled P4 give zero KL divergence."""
    loss_fn = PolarConsistencyLoss(upsample_mode="nearest")
    p_p2 = torch.randn(1, 4, 16, 16)
    # P4 is a 4x downsample of P2; build it by 4x4 average pooling of P2.
    p_p4 = torch.nn.functional.avg_pool2d(p_p2, kernel_size=4)
    # After nearest upsampling, p_p4 is not strictly equal to p_p2, so use a
    # simple consistent case: both P2 and P4 are zero tensors.
    p_p2 = torch.zeros(1, 4, 16, 16)
    p_p4 = torch.zeros(1, 4, 4, 4)
    loss = loss_fn(p_p2, p_p4)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_polar_consistency_positive_general():
    """KL divergence is non-negative in the general case."""
    loss_fn = PolarConsistencyLoss()
    p_p2 = torch.randn(2, 4, 16, 16)
    p_p4 = torch.randn(2, 4, 4, 4)
    loss = loss_fn(p_p2, p_p4)
    assert float(loss) >= -1e-5  # numerical tolerance


def test_polar_consistency_invalid_channels():
    loss_fn = PolarConsistencyLoss()
    bad_p2 = torch.randn(1, 5, 16, 16)
    p_p4 = torch.randn(1, 4, 4, 4)
    with pytest.raises(ValueError):
        loss_fn(bad_p2, p_p4)


# -------- calyx linear probe --------

def test_probe_param_count():
    """The probe has only 4 + 1 = 5 learnable parameters."""
    probe = CalyxLinearProbe(n_features=4)
    n = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    assert n == 5


def test_probe_forward():
    probe = CalyxLinearProbe()
    polarity_vec = torch.randn(10, 4)
    target = torch.randint(0, 2, (10,))
    loss = probe(polarity_vec, target)
    assert torch.isfinite(loss)
    assert loss.dim() == 0


def test_probe_with_mask():
    probe = CalyxLinearProbe()
    polarity_vec = torch.randn(8, 4)
    target = torch.randint(0, 2, (8,))
    mask = torch.zeros(8)
    mask[:4] = 1.0
    loss = probe(polarity_vec, target, mask=mask)
    assert torch.isfinite(loss)


def test_probe_predict():
    probe = CalyxLinearProbe()
    polarity_vec = torch.randn(5, 4)
    pred = probe.predict(polarity_vec)
    assert pred.shape == (5,)
    assert torch.all((pred == 0) | (pred == 1))


# -------- total loss --------

def test_total_loss_default_weights():
    weights = CalyxLossWeights()
    assert weights.lambda_1 == 0.10
    assert weights.lambda_2 == 0.30
    assert weights.lambda_3 == 0.50


def test_total_loss_negative_weight_raises():
    with pytest.raises(ValueError):
        CalyxLossWeights(lambda_1=-0.1)


def test_total_loss_aggregation():
    total = CalyxTotalLoss()
    out = total(
        loss_det=torch.tensor(1.0),
        loss_dar=torch.tensor(2.0),
        loss_polar=torch.tensor(3.0),
        loss_probe=torch.tensor(4.0),
    )
    expected = 1.0 + 0.10 * 2.0 + 0.30 * 3.0 + 0.50 * 4.0
    assert float(out.total) == pytest.approx(expected)
    assert float(out.det) == 1.0
    assert float(out.dar) == 2.0
    assert float(out.polar) == 3.0
    assert float(out.probe) == 4.0
