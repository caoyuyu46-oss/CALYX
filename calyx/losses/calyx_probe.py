# -*- coding: utf-8 -*-
"""Calyx-type linear-probe loss (paper Eq. 3.18, Section 3.5.3).

The calyx-end linear probe performs a DCF / PCF binary classification on each
box detected as calyx (DCF / PCF are the two calyx-end subclasses defined in
Section 3.5.3; refer to the paper for their exact definitions). The probe is a
linear binary classifier: the polarity map P inside the box is average-pooled
to a 4-D mean polarity vector p_bar, then a linear layer outputs the logit:

    logit = w^T * p_bar + b
    L_probe = BCE(sigmoid(logit), y_calyx)

with w in R^4 and b in R learnable.

Notes
-----
The polarity map comes from the PSPK output at the backbone P3 stage (the PGI
P3 downsampled version). Boxes are ROI-average-pooled at P3 resolution
(``roi_align`` with 1x1 output). Only ground-truth calyx samples contribute.
The label y_calyx is {0: PCF, 1: DCF} from the dataset annotation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalyxLinearProbe(nn.Module):
    """Calyx-end linear probe (paper Section 3.5.3).

    Parameters
    ----------
    n_features : int, default 4
        Mean-polarity-vector dimension, equal to the 4 PSPK output channels.

    Notes
    -----
    Only ``n_features + 1`` learnable parameters (default 5). This minimal
    capacity strongly constrains the separability of the underlying polarity
    representation; Section 3.5.3 uses the probe's DCF/PCF accuracy as a
    falsifiable check on the physical meaning of the PSPK polarity maps.
    """

    def __init__(self, n_features: int = 4) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.linear = nn.Linear(in_features=self.n_features, out_features=1)
        with torch.no_grad():
            self.linear.weight.normal_(mean=0.0, std=0.05)
            self.linear.bias.zero_()

    def forward(
        self,
        polarity_vec: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        polarity_vec : torch.Tensor
            [N, 4], the mean polarity vectors of N calyx ROIs (extracted by the
            caller, e.g. via ``torchvision.ops.roi_align`` on the polarity map).
        target : torch.Tensor
            [N], values in {0, 1}. 0 = PCF (pseudo-calyx-end facing inward),
            1 = DCF (calyx-end facing outward).
        mask : torch.Tensor or None
            [N] 0/1 mask selecting which samples contribute (e.g. ground-truth
            calyx). None means all contribute.

        Returns
        -------
        loss : torch.Tensor
            0-dim scalar; 0 if there are no valid samples.
        """
        if polarity_vec.dim() != 2 or polarity_vec.size(1) != self.n_features:
            raise ValueError(
                f"polarity_vec expects shape [N, {self.n_features}], "
                f"got {tuple(polarity_vec.shape)}"
            )
        if target.dim() != 1 or target.size(0) != polarity_vec.size(0):
            raise ValueError(
                "target must be [N] and match the first dim of polarity_vec"
            )

        logit = self.linear(polarity_vec).squeeze(-1)  # [N]

        if mask is not None:
            if mask.shape != target.shape:
                raise ValueError("mask must match the shape of target")
            valid = mask > 0.5
            if not torch.any(valid):
                return torch.zeros(
                    (), device=polarity_vec.device, dtype=polarity_vec.dtype
                )
            logit = logit[valid]
            target = target[valid]

        loss = F.binary_cross_entropy_with_logits(
            logit, target.to(logit.dtype), reduction="mean"
        )
        return loss

    def predict(self, polarity_vec: torch.Tensor) -> torch.Tensor:
        """DCF/PCF prediction at threshold 0.5.

        Parameters
        ----------
        polarity_vec : torch.Tensor
            [N, 4].

        Returns
        -------
        pred : torch.Tensor
            [N], values in {0, 1}.
        """
        with torch.no_grad():
            logit = self.linear(polarity_vec).squeeze(-1)
            return (torch.sigmoid(logit) >= 0.5).long()
