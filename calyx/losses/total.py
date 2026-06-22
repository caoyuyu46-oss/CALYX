# -*- coding: utf-8 -*-
"""End-to-end total loss for CALYX (paper Eq. 3.19, Section 3.5.4).

Total loss, Eq. (3.19):

    L = L_det + lambda_1 * L_DAR + lambda_2 * L_polar + lambda_3 * L_probe

where
- L_det is the standard YOLO11 detection loss (box / cls / dfl, from the
  detection head),
- L_DAR is the scale-invariant logarithmic loss (silog.py),
- L_polar is the cross-scale polarity KL loss (polar_consistency.py),
- L_probe is the calyx-end linear-probe BCE (calyx_probe.py).

Default weights (Section 3.5.4):
    lambda_1 = 0.10  (DAR; weakest supervision, D_star from a zero-shot
                      monocular depth model)
    lambda_2 = 0.30  (polarity consistency; supervision from the model's own
                      two scales)
    lambda_3 = 0.50  (calyx probe; strongest supervision, from manual labels)

The ordering reflects an operational definition of increasing supervision
precision rather than a vague notion of trust.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CalyxLossWeights:
    """Loss-weight configuration.

    Defaults from Section 3.5.4; lambda_1 < lambda_2 < lambda_3 encodes
    increasing supervision precision.
    """
    lambda_1: float = 0.10  # DAR
    lambda_2: float = 0.30  # Polar consistency
    lambda_3: float = 0.50  # Calyx probe

    def __post_init__(self) -> None:
        for name, val in (
            ("lambda_1", self.lambda_1),
            ("lambda_2", self.lambda_2),
            ("lambda_3", self.lambda_3),
        ):
            if val < 0.0:
                raise ValueError(f"{name} must be non-negative, got {val}")


@dataclass
class CalyxLossOutput:
    """Per-term loss output for training logs and ablation analysis."""
    total: torch.Tensor
    det: torch.Tensor
    dar: torch.Tensor
    polar: torch.Tensor
    probe: torch.Tensor


class CalyxTotalLoss(nn.Module):
    """End-to-end total-loss weigher (paper Eq. 3.19).

    Parameters
    ----------
    weights : CalyxLossWeights or None, default None
        Loss-weight configuration; None uses the paper defaults.
    """

    def __init__(self, weights: Optional[CalyxLossWeights] = None) -> None:
        super().__init__()
        self.weights = weights or CalyxLossWeights()

    def forward(
        self,
        loss_det: torch.Tensor,
        loss_dar: torch.Tensor,
        loss_polar: torch.Tensor,
        loss_probe: torch.Tensor,
    ) -> CalyxLossOutput:
        """Weighted sum.

        Parameters
        ----------
        loss_det : torch.Tensor
            Standard YOLO11 detection loss; 0-dim or per-batch 1-D, reduced to
            a scalar.
        loss_dar : torch.Tensor
            Scale-invariant logarithmic loss.
        loss_polar : torch.Tensor
            Polarity-consistency KL divergence.
        loss_probe : torch.Tensor
            Calyx-end linear-probe BCE.

        Returns
        -------
        out : CalyxLossOutput
            Per-term and total losses.
        """
        # Accept 0-dim or 1-D per-batch losses; reduce to a mean scalar.
        def _scalar(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0:
                return t
            return t.mean()

        ld = _scalar(loss_det)
        ldar = _scalar(loss_dar)
        lpolar = _scalar(loss_polar)
        lprobe = _scalar(loss_probe)

        total = (
            ld
            + self.weights.lambda_1 * ldar
            + self.weights.lambda_2 * lpolar
            + self.weights.lambda_3 * lprobe
        )
        return CalyxLossOutput(
            total=total,
            det=ld,
            dar=ldar,
            polar=lpolar,
            probe=lprobe,
        )

    def extra_repr(self) -> str:
        w = self.weights
        return (
            f"lambda_1={w.lambda_1}, lambda_2={w.lambda_2}, "
            f"lambda_3={w.lambda_3}"
        )
