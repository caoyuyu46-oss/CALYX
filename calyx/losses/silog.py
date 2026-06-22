# -*- coding: utf-8 -*-
"""Scale-invariant logarithmic loss for DAR (paper Eqs. 3.15-3.16).

Log residual, Eq. (3.15):

    d(x) = log(D_hat(x) + epsilon) - log(D_star(x) + epsilon)

Scale-invariant log loss, Eq. (3.16):

    L_DAR = (1/N) * sum_x d(x)^2 - (lambda_d / N^2) * (sum_x d(x))^2

with lambda_d in [0, 1] (Eigen et al., 2014), default 0.85. At lambda_d = 1 the
loss is strictly scale-invariant; at lambda_d = 0 it reduces to a standard log
MSE. No global scaling factor beyond Eq. (3.16) is introduced.

References
----------
Eigen D, Puhrsch C, Fergus R. Depth Map Prediction from a Single Image using a
Multi-Scale Deep Network. NeurIPS 2014.
"""

from typing import Optional

import torch
import torch.nn as nn


class SiLogLoss(nn.Module):
    """Scale-invariant logarithmic loss (paper Eqs. 3.15/3.16).

    Parameters
    ----------
    lambda_d : float, default 0.85
        Scale-compensation weight, in [0, 1] (Section 3.5.2).
    eps : float, default 1e-6
        Numerical-stability term inside the log.
    """

    def __init__(self, lambda_d: float = 0.85, eps: float = 1e-6) -> None:
        super().__init__()
        if not (0.0 <= lambda_d <= 1.0):
            raise ValueError(
                f"lambda_d must be in [0, 1] (Eigen et al., 2014), got {lambda_d}"
            )
        self.lambda_d = float(lambda_d)
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted depth map D_hat, [B, 1, H, W], non-negative.
        target : torch.Tensor
            Soft target depth map D_star (e.g. from Depth-Anything-V2),
            [B, 1, H, W], non-negative.
        mask : torch.Tensor or None
            Optional valid-pixel mask, [B, 1, H, W], in {0, 1} or [0, 1]. None
            means all pixels are valid.

        Returns
        -------
        loss : torch.Tensor
            0-dim scalar.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, got "
                f"{tuple(pred.shape)} vs {tuple(target.shape)}"
            )

        # Resolution must be aligned by the caller; no implicit interpolation
        # here, to avoid silent errors.

        # Log residual d(x).
        log_pred = torch.log(pred + self.eps)
        log_target = torch.log(target + self.eps)
        diff = log_pred - log_target  # [B, 1, H, W]

        if mask is not None:
            if mask.shape != diff.shape:
                raise ValueError(
                    f"mask must match pred shape, got {tuple(mask.shape)}"
                )
            # Zero out invalid residuals and divide by the valid-pixel count.
            diff = diff * mask
            n_valid = mask.sum().clamp(min=1.0)
        else:
            n_valid = torch.tensor(
                float(diff.numel()), device=diff.device, dtype=diff.dtype
            )

        sum_d = diff.sum()
        sum_d_sq = (diff * diff).sum()

        # L = E[d^2] - lambda_d * (E[d])^2 = Var[d] + (1 - lambda_d) * (E[d])^2.
        loss = sum_d_sq / n_valid - self.lambda_d * (sum_d / n_valid) ** 2
        return loss

    def extra_repr(self) -> str:
        return f"lambda_d={self.lambda_d}, eps={self.eps:.0e}"
