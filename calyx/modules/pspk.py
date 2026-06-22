# -*- coding: utf-8 -*-
"""PSPK block: Photometric Shape-Prior Kernels (paper Section 3.3).

The PSPK block is embedded at the backbone P2 (stride 4) and P4 (stride 16)
stages. On the SADR-recovered features it extracts four local geometry-
grayscale signatures (concave / convex / saddle / flat) with four physically
initialised 4x4 kernels, producing a four-channel polarity map P that is
concatenated with the original channels and passed to the next backbone block.

Physical basis
--------------
Section 3.1.3 partitions local geometry by the signs of the principal
curvatures:

    S_1 concave: kappa_1 < 0, kappa_2 < 0
    S_2 convex:  kappa_1 > 0, kappa_2 > 0
    S_3 saddle:  kappa_1 * kappa_2 < 0
    S_4 flat:    |kappa_1|, |kappa_2| below a threshold

Section 3.3.2 gives the four 4x4 physically initialised kernels
(Eqs. 3.8/3.9/3.10):

    K_1 (concave): centre 2x2 = -1, outer 12 pixels = +1/3 (zero mean),
                   the "dark centre, bright ring" interreflection signature
    K_2 (convex):  K_2 = -K_1 (first-order symmetric approximation)
    K_3 (saddle):  alternating +/- diagonal blocks, encoding an axis-aligned
                   saddle
    K_4 (flat):    1/16 * J_4 + epsilon * E, with J_4 the all-ones matrix and
                   E a small asymmetric perturbation breaking gradient symmetry

Notes
-----
Input feature F is [B, C, H, W]. A learnable 1x1 conv W_g: C -> 1 projects F
to a single-channel geometry-sensitive map g(F); four 4x4 kernels then produce
the four-channel polarity map P. The 4x4 even kernels have no exact centre, so
a (1, 2, 1, 2) reflect padding keeps the output resolution equal to the input.
``forward`` returns (output, polarity): output is [B, C+4, H, W] for the
backbone; polarity is [B, 4, H, W] for PGI and the DAR head. Kernel values are
not frozen by default, so backprop may refine the K_2 = -K_1 and axis-aligned
K_3 simplifications to the data distribution.

References
----------
Cole F, Sanik K, DeCarlo D, et al. How well do line drawings depict shape?
ACM SIGGRAPH 2009.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Physically initialised kernel constructors
# ----------------------------------------------------------------------

def _build_K1() -> torch.Tensor:
    """Concave kernel K_1 (Eq. 3.8).

    Centre 2x2 block = -1.0, outer 12 pixels = +1/3. The kernel is zero mean:
    4*(-1) + 12*(1/3) = 0.

    Returns
    -------
    k : torch.Tensor, shape [4, 4]
    """
    k = torch.full((4, 4), 1.0 / 3.0)
    k[1:3, 1:3] = -1.0
    return k


def _build_K2() -> torch.Tensor:
    """Convex kernel K_2 = -K_1 (Eq. 3.9, first-order symmetric approx)."""
    return -_build_K1()


def _build_K3() -> torch.Tensor:
    """Saddle kernel K_3 (Eq. 3.10).

    Diagonal-block design: top-left and bottom-right 2x2 blocks take +1/sqrt(8),
    bottom-left and top-right take -1/sqrt(8). This encodes an axis-aligned
    saddle; saddles whose principal directions deviate from the axes give an
    attenuated response (Section 3.3.2). The 1/sqrt(8) magnitude matches the
    Frobenius norm of K_1/K_2 so that responses stay comparable during training.

    Returns
    -------
    k : torch.Tensor, shape [4, 4]
    """
    val = 1.0 / (8.0 ** 0.5)
    k = torch.zeros(4, 4)
    k[0:2, 0:2] = +val   # top-left
    k[2:4, 2:4] = +val   # bottom-right
    k[0:2, 2:4] = -val   # top-right
    k[2:4, 0:2] = -val   # bottom-left
    return k


def _build_K4() -> torch.Tensor:
    """Flat kernel K_4 (Eq. 3.10).

    1/16 * J_4 plus a tiny asymmetric perturbation E (J_4 the all-ones matrix).
    The 1e-3 perturbation breaks symmetry early in training to avoid gradient
    degeneracy and is far smaller than the weights, preserving the uniform-
    response meaning.

    Returns
    -------
    k : torch.Tensor, shape [4, 4]
    """
    # Uniform body.
    k = torch.full((4, 4), 1.0 / 16.0)
    # Deterministic (non-random) asymmetric perturbation for reproducibility.
    perturb = torch.tensor(
        [
            [+1, -1, +1, -1],
            [-1, +1, -1, +1],
            [+1, -1, +1, -1],
            [-1, +1, -1, +1],
        ],
        dtype=torch.float32,
    )
    k = k + 1e-3 * perturb
    return k


def build_pspk_kernel_bank() -> torch.Tensor:
    """Stack the four physically initialised kernels into [4, 1, 4, 4].

    The shape matches the ``F.conv2d`` weight layout [out_channels=4,
    in_channels=1, kH=4, kW=4].

    Returns
    -------
    bank : torch.Tensor, shape [4, 1, 4, 4]
    """
    k1 = _build_K1()
    k2 = _build_K2()
    k3 = _build_K3()
    k4 = _build_K4()
    bank = torch.stack([k1, k2, k3, k4], dim=0).unsqueeze(1)  # [4, 1, 4, 4]
    return bank


# ----------------------------------------------------------------------
# PSPK block
# ----------------------------------------------------------------------

class PSPKBlock(nn.Module):
    """Photometric Shape-Prior Kernels block (paper Section 3.3).

    Parameters
    ----------
    in_channels : int
        Number of input feature channels C.
    freeze_kernels : bool, default False
        If True, the four physically initialised kernels are excluded from
        gradient updates and only the 1x1 projection W_g and downstream modules
        adapt. Default False allows the kernels to refine the K_2 = -K_1 and
        axis-aligned K_3 simplifications.

    Notes
    -----
    Added parameters: a 1x1 conv W_g (in_channels -> 1, ~in_channels params
    with bias) and four 4x4 single-input kernels (4*16 = 64), about
    ``in_channels + 65`` in total. PSPK is embedded once at P2 and once at P4
    (Section 3.3), adding roughly 0.32 GMac.
    """

    def __init__(self, in_channels: int, freeze_kernels: bool = False) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        self.in_channels = int(in_channels)

        # ---- 1x1 projection: multi-channel feature -> single-channel g(F) ----
        # Section 3.3.2 reads this as a learnable grayscale operator whose output
        # approximates the reflectance field I_tilde in feature space.
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )
        # Channel-mean initialisation as a training start point.
        with torch.no_grad():
            self.proj.weight.fill_(1.0 / float(in_channels))
            if self.proj.bias is not None:
                self.proj.bias.zero_()

        # ---- four 4x4 physically initialised kernels ----
        # Registered as a Parameter so they can train; freeze via requires_grad.
        bank = build_pspk_kernel_bank()  # [4, 1, 4, 4]
        self.kernels = nn.Parameter(bank, requires_grad=not freeze_kernels)

        # Even 4x4 kernels have no exact centre; a left-top aligned (1, 2, 1, 2)
        # padding keeps the output resolution equal to the input.
        self._pad = (1, 2, 1, 2)  # (left, right, top, bottom) for F.pad

    def polarity_map(self, feat: torch.Tensor) -> torch.Tensor:
        """Polarity map P(x) = (P_1, P_2, P_3, P_4).

        Parameters
        ----------
        feat : torch.Tensor
            Input feature, [B, C, H, W].

        Returns
        -------
        polarity : torch.Tensor
            [B, 4, H, W].
        """
        # 1) 1x1 projection to the single-channel geometry-sensitive map g(F).
        g = self.proj(feat)  # [B, 1, H, W]

        # 2) Explicit left-top aligned padding to preserve spatial resolution.
        g_padded = F.pad(g, self._pad, mode="reflect")

        # 3) Apply the four 4x4 kernels jointly -> four-channel output.
        polarity = F.conv2d(
            g_padded,
            self.kernels,
            bias=None,
            stride=1,
            padding=0,
        )
        # polarity: [B, 4, H, W]
        return polarity

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        feat : torch.Tensor
            Input feature, [B, C, H, W].

        Returns
        -------
        output : torch.Tensor
            Concatenated feature [B, C+4, H, W] for the next backbone block.
        polarity : torch.Tensor
            Polarity map [B, 4, H, W] for PGI and the DAR head.
        """
        if feat.dim() != 4:
            raise ValueError(
                f"PSPK expects input [B, C, H, W], got {tuple(feat.shape)}"
            )
        if feat.size(1) != self.in_channels:
            raise ValueError(
                f"PSPK in_channels={self.in_channels} does not match input "
                f"channels {feat.size(1)}"
            )
        polarity = self.polarity_map(feat)
        output = torch.cat([feat, polarity], dim=1)  # [B, C+4, H, W]
        return output, polarity

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"trainable_kernels={self.kernels.requires_grad}"
        )
