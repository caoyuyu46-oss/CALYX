# -*- coding: utf-8 -*-
"""CalyxYOLO end-to-end integration tests.

Verifies the full data flow with SADR/PSPK/PGI/DAR working together.
"""

import pytest
import torch

from calyx.models.calyx_yolo import CalyxYOLOSkeleton


@pytest.fixture
def model():
    return CalyxYOLOSkeleton(nc=4, base_channels=16, p3_channels=64,
                              p4_channels=128, p5_channels=256)


def test_forward_shape(model):
    img = torch.rand(2, 3, 320, 320)
    model.eval()
    with torch.no_grad():
        out = model(img)
    assert out.detections.dim() == 3
    # P2 polarity shape [B, 4, H/4, W/4]
    assert out.polarity_p2.shape == (2, 4, 80, 80)
    # P4 polarity shape [B, 4, H/16, W/16]
    assert out.polarity_p4.shape == (2, 4, 20, 20)
    # PGI three-scale polarities
    assert len(out.polarity_per_scale) == 3
    # DAR inactive in eval mode
    assert out.depth_pred is None


def test_forward_train_mode_dar_active(model):
    img = torch.rand(1, 3, 320, 320)
    model.train()
    out = model(img)
    assert out.depth_pred is not None
    assert out.depth_pred.dim() == 4


def test_export_inference_only(model):
    """After stripping DAR, there is no depth output even in train mode."""
    model.export_inference_only()
    assert model.dar_head is None
    assert model.enable_dar is False
    img = torch.rand(1, 3, 320, 320)
    model.train()  # dar_head already removed, even in train mode
    out = model(img)
    assert out.depth_pred is None


def test_sadr_intermediate_returned(model):
    img = torch.rand(1, 3, 320, 320)
    model.eval()
    with torch.no_grad():
        out = model(img)
    s, alpha = out.sadr_intermediate
    assert s.shape == (1, 1, 320, 320)
    assert alpha.shape == (1, 1, 320, 320)
    assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0)
