# -*- coding: utf-8 -*-
"""PGI: Polarity-Guided Injection (paper Section 3.4).

PGI takes the polarity map P from PSPK (P2 stage, 4 channels, P2 resolution),
downsamples it to the P3 / P4 / P5 scales, and additively injects it into the
matching neck PAN-FPN features via a 4x4 geometry mixing matrix W_geo, a 1x1
channel-expansion projection W_proj, and a scalar gate eta. The
(W_geo, W_proj, eta) parameters are independent per scale.

Formulation
-----------
Eq. (3.11), channel-wise max-pool multi-scale polarity downsampling:

    P^(s)(x) = max_{(u,v) in R_s(x)} P_c(u, v)   (spatial max, per channel)

Eq. (3.13), additive injection:

    F_out^(s)(x) = F_in^(s)(x)
                   + eta^(s) * W_proj^(s) * W_geo^(s) * P^(s)(x)

where W_geo^(s) is a 4x4 geometry-class mixing matrix, W_proj^(s) a 1x1 conv
(C_s x 4), and eta^(s) a scalar gate.

Eq. (3.14), Frobenius distance metric:

    d_F^(s) = ||W_geo^(s) - I_4||_F

used to study the projection-matrix training dynamics in Section 4.5.

Notes
-----
W_geo is initialised to the identity I_4, W_proj to a small zero-mean Gaussian,
and eta to 0.5 for a mild initial injection (eta is wrapped by a sigmoid to stay
in (0, 1)). The two-layer projection W_proj * W_geo is mathematically equivalent
to a single (C_s, 4) matrix, but keeping the factorisation lets W_geo be read as
a geometry-class mixing matrix (Section 3.4.3). Channel-wise max-pool is used
(not average-pool) to avoid polarity cancellation.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SinglePGI(nn.Module):
    """Single-scale PGI injection branch.

    Parameters
    ----------
    target_channels : int
        Neck feature channel count C_s at the target scale.
    eta_init : float, default 0.5
        Initial gate value (pre-sigmoid logit equivalent).

    Notes
    -----
    Learnable parameters: W_geo (4x4 = 16), W_proj (C_s x 4 + C_s with bias,
    ~5*C_s), eta (1), i.e. 17 + 5*C_s per branch. Across three scales, about
    51 + 5*(C_P3 + C_P4 + C_P5); Section 3.4.3 reports ~0.43 GMac added.
    """

    def __init__(self, target_channels: int, eta_init: float = 0.5) -> None:
        super().__init__()
        self.target_channels = int(target_channels)

        # W_geo: 4x4 geometry-class mixing matrix, initialised to the identity.
        self.W_geo = nn.Parameter(torch.eye(4))

        # W_proj: 1x1 conv 4 -> target_channels, projecting the 4-D geometric
        # prior into the semantic channel space of Eq. (3.13).
        self.W_proj = nn.Conv2d(
            in_channels=4,
            out_channels=target_channels,
            kernel_size=1,
            bias=True,
        )
        with torch.no_grad():
            # Small zero-mean Gaussian for a mild initial injection.
            self.W_proj.weight.normal_(mean=0.0, std=0.01)
            if self.W_proj.bias is not None:
                self.W_proj.bias.zero_()

        # eta: scalar gate constrained to (0, 1) via sigmoid. A logit of 0 gives
        # sigmoid(0) = 0.5; recover the logit from eta_init via log(p/(1-p)).
        if not (0.0 < eta_init < 1.0):
            raise ValueError(f"eta_init must be in (0, 1), got {eta_init}")
        eta_logit = float(
            torch.log(torch.tensor(eta_init / (1.0 - eta_init))).item()
        )
        self.eta_logit = nn.Parameter(torch.tensor(eta_logit))

    @property
    def eta(self) -> torch.Tensor:
        """Current gate value, sigmoid(eta_logit) in (0, 1)."""
        return torch.sigmoid(self.eta_logit)

    def frobenius_distance_to_identity(self) -> torch.Tensor:
        """Frobenius distance of W_geo to the identity, d_F (Eq. 3.14)."""
        return torch.norm(self.W_geo - torch.eye(4, device=self.W_geo.device), p="fro")

    def forward(
        self, feat_in: torch.Tensor, polarity_at_scale: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        feat_in : torch.Tensor
            Neck feature at the target scale, [B, C_s, H_s, W_s].
        polarity_at_scale : torch.Tensor
            Polarity map already downsampled to this scale, [B, 4, H_s, W_s].

        Returns
        -------
        feat_out : torch.Tensor
            Injected feature, [B, C_s, H_s, W_s].
        """
        # 1) W_geo mixes along the channel (geometry-class) dimension; spatial
        #    dims unchanged. Treat the channel dim as last for the matmul.
        b, c4, h, w = polarity_at_scale.shape
        if c4 != 4:
            raise ValueError(f"PGI expects 4-channel polarity, got {c4}")
        # [B, H, W, 4] @ W_geo^T, then back to [B, 4, H, W].
        p = polarity_at_scale.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 4]
        p = p @ self.W_geo.t()  # equivalent to W_geo @ p_vec, class-wise mixing
        p = p.permute(0, 3, 1, 2).contiguous()  # [B, 4, H, W]

        # 2) W_proj: 1x1 conv expanding channels 4 -> C_s.
        injection = self.W_proj(p)  # [B, C_s, H, W]

        # 3) Gated additive injection.
        feat_out = feat_in + self.eta * injection
        return feat_out

    def extra_repr(self) -> str:
        return (
            f"target_channels={self.target_channels}, "
            f"eta(current)={float(self.eta):.4f}"
        )


class PGI(nn.Module):
    """Multi-scale Polarity-Guided Injection (paper Section 3.4).

    Parameters
    ----------
    target_channels : tuple of int
        Neck feature channels for the three scales (P3, P4, P5), e.g.
        (64, 128, 256) or (128, 256, 512) depending on the configuration.
    eta_init : tuple of float, default (0.5, 0.5, 0.5)
        Per-scale initial gate values, allowing a "shallow-strong, deep-weak"
        initial bias.

    Notes
    -----
    Multi-scale downsampling uses channel-wise max-pool (Section 3.4.2). Max-pool
    backpropagates gradients only to the argmax locations, so non-maximal
    positions rely on the detection path; this avoids the polarity cancellation
    of average-pool.
    """

    SCALES: Tuple[str, str, str] = ("P3", "P4", "P5")

    def __init__(
        self,
        target_channels: Tuple[int, int, int],
        eta_init: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__()
        if len(target_channels) != 3 or len(eta_init) != 3:
            raise ValueError(
                "PGI requires length-3 tuples of channels and eta_init for the "
                "three scales (P3, P4, P5)"
            )
        self.target_channels = tuple(int(c) for c in target_channels)

        # Independent per-scale branches, keyed by scale name for readability.
        self.branches = nn.ModuleDict(
            {
                self.SCALES[i]: _SinglePGI(
                    target_channels=self.target_channels[i],
                    eta_init=float(eta_init[i]),
                )
                for i in range(3)
            }
        )

    @staticmethod
    def downsample_polarity(
        polarity: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Channel-wise max-pool multi-scale downsampling (Eq. 3.11).

        Parameters
        ----------
        polarity : torch.Tensor
            Source-scale polarity map, [B, 4, H_src, W_src].
        target_size : (int, int)
            Target spatial resolution (H_tgt, W_tgt).

        Returns
        -------
        downsampled : torch.Tensor
            [B, 4, H_tgt, W_tgt].

        Notes
        -----
        ``F.adaptive_max_pool2d`` pools each channel independently, matching the
        per-channel max of Eq. (3.11).
        """
        return F.adaptive_max_pool2d(polarity, output_size=target_size)

    def forward(
        self,
        feats: List[torch.Tensor],
        polarity: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        feats : list of torch.Tensor
            Length-3 list [P3, P4, P5] of neck features, each [B, C_s, H_s, W_s].
        polarity : torch.Tensor
            Polarity map from the PSPK P2 stage, [B, 4, H_P2, W_P2].

        Returns
        -------
        outputs : list of torch.Tensor
            Injected features, same shapes as ``feats``.
        polarities_per_scale : list of torch.Tensor
            Per-scale downsampled polarity maps for the cross-scale consistency
            loss L_polar, each [B, 4, H_s, W_s].
        """
        if len(feats) != 3:
            raise ValueError(f"PGI expects 3 neck scales, got {len(feats)}")

        outputs: List[torch.Tensor] = []
        polarities_per_scale: List[torch.Tensor] = []

        for i, scale_name in enumerate(self.SCALES):
            feat_in = feats[i]
            target_h, target_w = feat_in.shape[-2], feat_in.shape[-1]
            # Channel-wise max-pool downsampling.
            p_scale = self.downsample_polarity(polarity, (target_h, target_w))
            # Injection.
            feat_out = self.branches[scale_name](feat_in, p_scale)
            outputs.append(feat_out)
            polarities_per_scale.append(p_scale)

        return outputs, polarities_per_scale

    def frobenius_distances(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-scale Frobenius distances of W_geo (Eq. 3.14).

        Returns
        -------
        (d_P3, d_P4, d_P5) : tuple of torch.Tensor (0-dim)
        """
        return tuple(
            self.branches[s].frobenius_distance_to_identity()
            for s in self.SCALES
        )

    def etas(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Current per-scale gate values."""
        return tuple(self.branches[s].eta for s in self.SCALES)
