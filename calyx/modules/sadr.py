# -*- coding: utf-8 -*-
"""SADR: Specular-Aware Diffuse Recovery (paper Section 3.2).

SADR sits at the backbone input. It takes a single RGB frame and returns a
diffuse-recovered RGB image, suppressing the specular highlights produced by
conveyor top-lighting so that the downstream PSPK block receives an input on
which the geometry-dominance assumption approximately holds.

Formulation
-----------
Following the dichromatic reflection model and the maximum-chromaticity
analysis of Tan & Ikeuchi (2005), the achromatic specular strength is
estimated per pixel. Under near-white illumination the specular term adds
nearly equally across the RGB channels, so it is modelled as an achromatic
offset s(x) with diffuse part D(x) = I(x) - s(x)*1. Requiring the maximum
chromaticity to return to a diffuse reference Lambda_ref after removing s:

    (I_max - s) / (sum_I - 3 s) = Lambda_ref

and writing the maximum chromaticity c_max(x) = I_max(x) / sum_I(x), solving
for s gives Eq. (3.5):

    s(x) = clamp( sum_I(x) * (Lambda_ref - c_max(x)) / (3 Lambda_ref - 1),
                  0, I_min(x) )

- Lambda_ref (> 1/3) is the diffuse reference maximum chromaticity. Specular
  pixels have c_max biased toward the achromatic point 1/3 (below Lambda_ref),
  so s > 0; purely diffuse, highly saturated pixels have c_max >= Lambda_ref
  and are clamped to s = 0, matching the "no correction for diffuse pixels"
  semantics of max(0, .).
- The upper bound I_min(x) keeps the recovered image non-negative.
- By default Lambda_ref is estimated per image as a high quantile (default
  0.95) of c_max over sufficiently bright pixels, i.e. the cleanest surface
  chromaticity in that image; under stable line lighting it may instead be
  fixed via ``ref_chroma`` for determinism and lower latency.

Soft-gated substitution, Eqs. (3.6)-(3.7):

    alpha(x) = sigmoid(gamma * (s(x) / s_0 - tau))
    I_tilde(x) = I(x) - alpha(x) * s(x) * 1,   1 = [1, 1, 1]^T

Default hyper-parameters follow Section 3.2.3: s_0 = 0.20, gamma = 12.0,
tau = 0.85.

Notes
-----
Input tensors are [B, 3, H, W] with RGB values in [0, 1]. Since s and s_0
share the same scale, the [0, 1] convention is assumed; rescale s_0 and
bright_eps if inputs are in [0, 255]. SADR has no learnable parameters.

References
----------
Tan R T, Ikeuchi K. Separating reflection components of textured surfaces
using a single image. IEEE TPAMI, 27(2):178-193, 2005.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SADR(nn.Module):
    """Specular-Aware Diffuse Recovery (paper Section 3.2).

    Parameters
    ----------
    s0 : float, default 0.20
        Specular-strength normalisation scale in Eq. (3.6); same scale as the
        input ([0, 1]).
    gamma : float, default 12.0
        Steepness of the sigmoid soft gate.
    tau : float, default 0.85
        Normalised gating threshold.
    eps : float, default 1e-6
        Numerical-stability term in denominators.
    ref_chroma : float or None, default None
        Diffuse reference maximum chromaticity Lambda_ref. None -> per-image
        estimate (a high quantile of c_max over bright pixels); float -> fixed
        calibrated value. Values are clamped to (1/3 + ref_margin, 1].
    ref_quantile : float, default 0.95
        Quantile of c_max used for the per-image Lambda_ref estimate.
    bright_eps : float, default 0.05
        Lower brightness (sum_I) bound for pixels entering the Lambda_ref
        estimate, excluding near-black pixels with unreliable chromaticity.
    ref_margin : float, default 0.02
        Minimum gap of Lambda_ref above the achromatic point 1/3, avoiding a
        vanishing denominator (3*Lambda_ref - 1).

    Notes
    -----
    No learnable parameters. ``forward`` takes [B, 3, H, W] RGB in [0, 1] and
    returns (I_tilde, s, alpha), exposing intermediates for visualisation and
    the hyper-parameter sensitivity study of Appendix A.2.
    """

    def __init__(
        self,
        s0: float = 0.20,
        gamma: float = 12.0,
        tau: float = 0.85,
        eps: float = 1e-6,
        ref_chroma: Optional[float] = None,
        ref_quantile: float = 0.95,
        bright_eps: float = 0.05,
        ref_margin: float = 0.02,
    ) -> None:
        super().__init__()
        # Gating constants are non-learnable; registered as buffers so they
        # serialise with the state_dict.
        self.register_buffer("s0", torch.tensor(float(s0)))
        self.register_buffer("gamma", torch.tensor(float(gamma)))
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.eps = float(eps)
        if not (0.0 < ref_quantile < 1.0):
            raise ValueError(f"ref_quantile must be in (0, 1), got {ref_quantile}")
        if ref_chroma is not None and not (1.0 / 3.0 < float(ref_chroma) <= 1.0):
            raise ValueError(f"ref_chroma must be in (1/3, 1], got {ref_chroma}")
        self.ref_chroma = None if ref_chroma is None else float(ref_chroma)
        self.ref_quantile = float(ref_quantile)
        self.bright_eps = float(bright_eps)
        self.ref_margin = float(ref_margin)

    # ---- sub-steps kept separate for unit testing and visualisation ----

    def reference_chroma(
        self, c_max: torch.Tensor, i_sum: torch.Tensor
    ) -> torch.Tensor:
        """Diffuse reference maximum chromaticity Lambda_ref.

        Parameters
        ----------
        c_max : torch.Tensor
            Per-pixel maximum chromaticity, [B, 1, H, W].
        i_sum : torch.Tensor
            Per-pixel channel sum sum_I, [B, 1, H, W], used for the brightness
            mask.

        Returns
        -------
        lam : torch.Tensor
            [B, 1, 1, 1] (per image) or [1, 1, 1, 1] (fixed), clamped to
            (1/3 + ref_margin, 1].
        """
        lo = 1.0 / 3.0
        if self.ref_chroma is not None:
            lam = torch.as_tensor(
                self.ref_chroma, device=c_max.device, dtype=c_max.dtype
            )
            return lam.clamp(min=lo, max=1.0).view(1, 1, 1, 1)

        b = c_max.shape[0]
        flat_c = c_max.reshape(b, -1)
        flat_b = i_sum.reshape(b, -1)
        lams = []
        for i in range(b):
            cm = flat_c[i]
            m = flat_b[i] > self.bright_eps
            vals = cm[m] if bool(m.any()) else cm
            lams.append(torch.quantile(vals, self.ref_quantile))
        lam = torch.stack(lams).reshape(b, 1, 1, 1)
        return lam.clamp(min=lo, max=1.0)

    def specular_strength(self, image: torch.Tensor) -> torch.Tensor:
        """Per-pixel achromatic specular strength s(x), Eq. (3.5).

        Parameters
        ----------
        image : torch.Tensor
            [B, 3, H, W], values in [0, 1].

        Returns
        -------
        s : torch.Tensor
            [B, 1, H, W], non-negative and bounded by the minimum channel
            I_min.
        """
        eps = self.eps
        i_max = image.max(dim=1, keepdim=True).values    # [B, 1, H, W]
        i_min = image.min(dim=1, keepdim=True).values    # [B, 1, H, W]
        i_sum = image.sum(dim=1, keepdim=True)            # [B, 1, H, W]
        c_max = i_max / (i_sum + eps)                     # max chromaticity

        lam = self.reference_chroma(c_max, i_sum)         # Lambda_ref [B,1,1,1]
        denom = 3.0 * lam - 1.0                            # > 0 iff Lambda_ref > 1/3

        # s = sum_I * (Lambda_ref - c_max) / (3*Lambda_ref - 1)
        s = i_sum * (lam - c_max) / denom.clamp(min=self.eps)
        # Lower bound 0 (no correction for diffuse / saturated pixels);
        # upper bound I_min (keeps recovery non-negative).
        s = torch.clamp(s, min=0.0)
        s = torch.minimum(s, i_min)
        # Degenerate guard: if the diffuse chromaticity reference is too weak
        # (Lambda_ref ~ 1/3, e.g. a near-achromatic image), chromaticity alone
        # cannot identify specularity, so output s = 0 for that image.
        s = torch.where(denom > self.ref_margin, s, torch.zeros_like(s))
        return s

    def soft_gate(self, s: torch.Tensor) -> torch.Tensor:
        """Sigmoid soft gate alpha(x), Eq. (3.6).

        Parameters
        ----------
        s : torch.Tensor
            [B, 1, H, W] from ``specular_strength``.

        Returns
        -------
        alpha : torch.Tensor
            [B, 1, H, W] in (0, 1).
        """
        # alpha = sigmoid(gamma * (s / s0 - tau)); alpha -> 1 when s >> s0*tau
        # (strong correction) and -> 0 when s << s0*tau (almost none).
        return torch.sigmoid(self.gamma * (s / self.s0 - self.tau))

    def diffuse_recover(
        self, image: torch.Tensor, s: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Diffuse recovery I_tilde(x), Eq. (3.7).

        Parameters
        ----------
        image : torch.Tensor
            Original RGB image, [B, 3, H, W].
        s : torch.Tensor
            Specular-strength map, [B, 1, H, W].
        alpha : torch.Tensor
            Soft gate, [B, 1, H, W].

        Returns
        -------
        i_tilde : torch.Tensor
            [B, 3, H, W].

        Notes
        -----
        Eq. (3.7) is I_tilde = I - alpha*s*1 with 1 the all-ones vector: the
        same scalar alpha*s is subtracted from all three channels, so s
        broadcasts over the channel dimension. Since s <= I_min and
        alpha in (0, 1), the result stays non-negative.
        """
        return image - alpha * s

    # ---- main entry ----

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        image : torch.Tensor
            [B, 3, H, W] RGB input in [0, 1].

        Returns
        -------
        i_tilde : torch.Tensor
            Diffuse-recovered image, [B, 3, H, W].
        s : torch.Tensor
            Specular-strength map, [B, 1, H, W] (for visualisation/ablation).
        alpha : torch.Tensor
            Soft-gate values, [B, 1, H, W].
        """
        if image.dim() != 4 or image.size(1) != 3:
            raise ValueError(
                f"SADR expects input [B, 3, H, W], got {tuple(image.shape)}"
            )
        s = self.specular_strength(image)
        alpha = self.soft_gate(s)
        i_tilde = self.diffuse_recover(image, s, alpha)
        return i_tilde, s, alpha

    def extra_repr(self) -> str:
        ref = "per-image" if self.ref_chroma is None else f"{self.ref_chroma:.3f}"
        return (
            f"s0={float(self.s0):.3f}, gamma={float(self.gamma):.2f}, "
            f"tau={float(self.tau):.3f}, eps={self.eps:.0e}, "
            f"ref_chroma={ref}, ref_quantile={self.ref_quantile:.2f}"
        )
