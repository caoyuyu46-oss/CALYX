# -*- coding: utf-8 -*-
"""PGI unit tests.

Covers:
1. W_geo initialised to the identity;
2. eta initialised in (0, 1);
3. Frobenius distance is 0 at initialisation;
4. multi-scale max-pool downsampling shapes;
5. additive injection (not a multiplicative gate);
6. independent per-scale parameters.
"""

import pytest
import torch

from calyx.modules.pgi import PGI, _SinglePGI


def test_single_pgi_W_geo_init_is_identity():
    """W_geo initialises to the identity I_4 (Section 3.4.3)."""
    branch = _SinglePGI(target_channels=64)
    assert torch.allclose(branch.W_geo, torch.eye(4))


def test_single_pgi_eta_in_unit_interval():
    """eta after sigmoid lies in (0, 1)."""
    branch = _SinglePGI(target_channels=64, eta_init=0.5)
    eta = branch.eta
    assert 0.0 < float(eta) < 1.0


def test_single_pgi_eta_init_value():
    """eta_init=0.5 maps to logit 0, so the initial eta equals 0.5."""
    branch = _SinglePGI(target_channels=64, eta_init=0.5)
    assert float(branch.eta) == pytest.approx(0.5, abs=1e-5)


def test_single_pgi_frobenius_distance_initial_is_zero():
    """With W_geo = I the Frobenius distance is 0."""
    branch = _SinglePGI(target_channels=64)
    d = branch.frobenius_distance_to_identity()
    assert float(d) == pytest.approx(0.0, abs=1e-5)


def test_single_pgi_forward_shape():
    """Single-scale PGI output matches feat_in."""
    branch = _SinglePGI(target_channels=64)
    feat_in = torch.randn(2, 64, 16, 16)
    polarity = torch.randn(2, 4, 16, 16)
    out = branch(feat_in, polarity)
    assert out.shape == feat_in.shape


def test_pgi_three_scales():
    """PGI takes 3 scales and returns 3 outputs and 3 downsampled polarities."""
    pgi = PGI(target_channels=(64, 128, 256))
    polarity_p2 = torch.randn(2, 4, 64, 64)
    feats = [
        torch.randn(2, 64, 32, 32),    # P3
        torch.randn(2, 128, 16, 16),   # P4
        torch.randn(2, 256, 8, 8),     # P5
    ]
    outputs, polarities = pgi(feats, polarity_p2)
    assert len(outputs) == 3
    assert len(polarities) == 3
    for i, (out, p) in enumerate(zip(outputs, polarities)):
        assert out.shape == feats[i].shape
        assert p.shape == (2, 4, *feats[i].shape[-2:])


def test_pgi_independent_branches():
    """Per-scale W_geo and eta are fully independent."""
    pgi = PGI(target_channels=(64, 128, 256))
    # Editing the P3 branch must not affect P4/P5.
    with torch.no_grad():
        pgi.branches["P3"].W_geo.fill_(0.0)
    assert torch.allclose(pgi.branches["P4"].W_geo, torch.eye(4))
    assert torch.allclose(pgi.branches["P5"].W_geo, torch.eye(4))


def test_pgi_additive_injection_not_multiplicative():
    """Injection is additive, F_out = F_in + eta * inj, not multiplicative.

    Check: with zero injection (W_proj zeroed, bias already zero) and any
    feat_in, the output equals feat_in. A multiplicative gate would instead
    discount feat_in.
    """
    branch = _SinglePGI(target_channels=64)
    # Force W_proj output to zero (zero weights, bias already zero).
    with torch.no_grad():
        branch.W_proj.weight.zero_()
    feat_in = torch.randn(1, 64, 8, 8)
    polarity = torch.randn(1, 4, 8, 8)
    out = branch(feat_in, polarity)
    # Additive injection: eta * 0 = 0, so out equals feat_in exactly.
    assert torch.allclose(out, feat_in)


def test_pgi_downsample_polarity_max_pool():
    """Channel-wise max-pool keeps the per-channel maximum response."""
    p = torch.zeros(1, 4, 4, 4)
    p[0, 0, 2, 2] = 5.0   # channel 0 strong only at (2,2)
    p[0, 1, 0, 0] = 3.0   # channel 1 strong only at (0,0)
    down = PGI.downsample_polarity(p, target_size=(2, 2))
    assert down.shape == (1, 4, 2, 2)
    # channel 0 keeps 5.0 at (1,1) (the 2x2 pool covers (2,2))
    assert float(down[0, 0, 1, 1]) == pytest.approx(5.0)
    # channel 1 keeps 3.0 at (0,0)
    assert float(down[0, 1, 0, 0]) == pytest.approx(3.0)


def test_pgi_invalid_target_channels_length():
    """target_channels must have length 3."""
    with pytest.raises(ValueError):
        PGI(target_channels=(64, 128))
