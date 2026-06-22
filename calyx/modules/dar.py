# -*- coding: utf-8 -*-
"""DAR head: Depth-Alignment Regularization (paper Section 3.5).

(The loss itself is in ``calyx/losses/silog.py``.)

The DAR head predicts a relative depth map D_hat from the PSPK P3-stage
features, using the zero-shot Depth-Anything-V2 output D_star as a soft target.
Depth accuracy is not the objective; the objective is that the relative depth
ordering of concave vs convex regions agrees with geometric intuition, a
falsifiable check on the physical meaning of the PSPK polarity maps
(Section 3.5.1).

DAR is active only during training and is explicitly removed at inference via
ONNX subgraph pruning, leaving no DAR operators in the forward graph
(Section 3.5.4).

Notes
-----
The head consumes the injected P3 features (PSPK + PGI fused, so semantic
channels and the geometric prior are coupled). Structure: two 1x1 convs with
GELU and a single-channel output; ``softplus`` keeps depth non-negative. The
simplicity of the head means pruning needs no special operators and leaves no
residue in ONNX/TensorRT. ``forward`` is permitted in ``eval`` mode to support
the training-dynamics study of Section 4.5.

References
----------
Yang L, et al. Depth Anything V2. NeurIPS 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DARHead(nn.Module):
    """Depth-Alignment Regularization head (paper Section 3.5.1).

    Parameters
    ----------
    in_channels : int
        Input feature channels, usually the injected P3 channels.
    hidden_channels : int, default 64
        Hidden channel count.
    output_size : tuple of int or None, default None
        If given, ``F.interpolate`` upsamples the depth map to this resolution
        to align with the pseudo-depth soft target. None keeps the input
        resolution and lets the loss handle alignment.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        output_size: tuple = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.output_size = output_size

        # Lightweight two-layer 1x1 conv head with single-channel output. Its
        # simplicity lets the whole subgraph be pruned at inference without any
        # special operator, leaving no residue in ONNX/TensorRT.
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        feat : torch.Tensor
            [B, C, H, W], the injected P3 features.

        Returns
        -------
        depth_pred : torch.Tensor
            [B, 1, H_out, W_out], non-negative relative depth.

        Notes
        -----
        In ``eval`` mode the inference graph is usually being exported and the
        DAR head should not run; remove it via ``del model.dar_head`` or ONNX
        subgraph pruning before export. The call is not blocked here so that the
        training-dynamics evaluation of Section 4.5 can use it.
        """
        if feat.dim() != 4 or feat.size(1) != self.in_channels:
            raise ValueError(
                f"DARHead expects input [B, {self.in_channels}, H, W], "
                f"got {tuple(feat.shape)}"
            )

        x = F.gelu(self.conv1(feat))
        x = F.gelu(self.conv2(x))
        # softplus keeps depth non-negative (for the subsequent log); the small
        # threshold is handled inside the SiLog loss.
        depth = F.softplus(self.conv_out(x))

        if self.output_size is not None:
            depth = F.interpolate(
                depth,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )
        return depth

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"output_size={self.output_size}"
        )
