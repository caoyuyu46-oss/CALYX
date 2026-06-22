# -*- coding: utf-8 -*-
"""Polar class-consistency loss (paper Eq. 3.17, Section 3.5.3).

For the P2 and P4 polarity maps P^(P2), P^(P4), a channel-wise softmax first
gives geometry-class probability distributions:

    q^(s)_i(x) = softmax_i(P^(s)(x) / T)

with softmax temperature T (default 1.0). A KL divergence then constrains the
P2 and P4 class distributions to agree at a common resolution (P4 upsampled to
P2):

    L_polar = (1/|Omega|) * sum_{x in Omega} sum_i q^(P2)_i(x)
              * log(q^(P2)_i(x) / U(q^(P4)_i)(x))

with U a nearest or bilinear upsampling operator. The KL asymmetry biases the
loss toward pulling P4 toward P2, taking the P2 local decision as reference
(Section 3.5.3).

Applicability
-------------
The constraint assumes that the dominant geometry class agrees across the P2 and
P4 receptive fields. For targets spanning several orders of scale (e.g. defects
far below the P2 receptive-field lower bound) this may fail; its influence is
then controlled by the weight lambda_2 in Eq. (3.19).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarConsistencyLoss(nn.Module):
    """Polar class-consistency loss (paper Eq. 3.17).

    Parameters
    ----------
    temperature : float, default 1.0
        Softmax temperature; < 1 sharpens, > 1 smooths.
    eps : float, default 1e-8
        Numerical-stability term in the KL divergence.
    upsample_mode : str, default 'bilinear'
        Interpolation for upsampling the P4 polarity map to P2 resolution,
        either ``'nearest'`` or ``'bilinear'``.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        eps: float = 1e-8,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if upsample_mode not in {"nearest", "bilinear"}:
            raise ValueError(
                f"upsample_mode must be 'nearest' or 'bilinear', "
                f"got {upsample_mode}"
            )
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.upsample_mode = upsample_mode

    def forward(
        self,
        polarity_p2: torch.Tensor,
        polarity_p4: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        polarity_p2 : torch.Tensor
            P2-stage polarity map, [B, 4, H_P2, W_P2].
        polarity_p4 : torch.Tensor
            P4-stage polarity map, [B, 4, H_P4, W_P4], typically 1/4 of the P2
            resolution.
        mask : torch.Tensor or None
            Optional valid-pixel mask, [B, 1, H_P2, W_P2].

        Returns
        -------
        loss : torch.Tensor
            0-dim scalar.
        """
        if polarity_p2.size(1) != 4 or polarity_p4.size(1) != 4:
            raise ValueError(
                "polarity_p2 and polarity_p4 must both have 4 channels, got "
                f"{polarity_p2.size(1)} and {polarity_p4.size(1)}"
            )
        target_size = polarity_p2.shape[-2:]

        # Upsample P4 to P2 resolution.
        align_corners = False if self.upsample_mode == "bilinear" else None
        if self.upsample_mode == "nearest":
            p4_up = F.interpolate(
                polarity_p4, size=target_size, mode="nearest"
            )
        else:
            p4_up = F.interpolate(
                polarity_p4,
                size=target_size,
                mode="bilinear",
                align_corners=align_corners,
            )

        # Channel-wise softmax -> geometry-class probability distributions.
        q_p2 = F.softmax(polarity_p2 / self.temperature, dim=1)  # [B, 4, H, W]
        q_p4 = F.softmax(p4_up / self.temperature, dim=1)        # [B, 4, H, W]

        # KL(q_p2 || q_p4) = sum_i q_p2 * (log q_p2 - log q_p4); eps avoids log(0).
        log_q_p2 = torch.log(q_p2 + self.eps)
        log_q_p4 = torch.log(q_p4 + self.eps)
        kl = (q_p2 * (log_q_p2 - log_q_p4)).sum(dim=1, keepdim=True)
        # kl: [B, 1, H, W]

        if mask is not None:
            if mask.shape[-2:] != target_size:
                raise ValueError(
                    "mask spatial resolution must match polarity_p2"
                )
            kl = kl * mask
            n_valid = mask.sum().clamp(min=1.0)
        else:
            n_valid = torch.tensor(
                float(kl.numel()), device=kl.device, dtype=kl.dtype
            )

        return kl.sum() / n_valid

    def extra_repr(self) -> str:
        return (
            f"temperature={self.temperature}, "
            f"upsample_mode='{self.upsample_mode}'"
        )
