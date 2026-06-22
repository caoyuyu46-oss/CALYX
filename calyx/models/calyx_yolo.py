# -*- coding: utf-8 -*-
"""CalyxYOLO: end-to-end integration.

A reference integration of SADR / PSPK / PGI / DAR with a YOLO11n
backbone + neck + head. This module gives a minimal working assembly of the
CALYX framework, showing where each Section-3 module attaches and the data flow.

Data flow (Section 3.1.1):
  RGB I -> SADR -> I_tilde -> backbone (PSPK at P2 and P4)
       -> neck PAN-FPN
       -> PGI three-scale injection (P3/P4/P5)
       -> detection head (4-class output)
       -> grading head (aggregated from detections)
  The P2 polarity map also feeds the DAR head and the cross-scale consistency
  loss during training.

Notes
-----
Two integration modes are provided:
1. ``CalyxYOLOSkeleton``: a pure-PyTorch skeleton with no ultralytics
   dependency, defining the attachment points and data flow of all CALYX
   modules. The backbone and neck are placeholder implementations (a simplified
   ResNet-style backbone + PAN-FPN) for unit tests and exposition.
2. With ``ultralytics`` installed, see ``register.py`` to register PSPK / PGI as
   YOLO11 custom layers and load a real YOLO11n backbone via
   ``configs/calyx_yolo11n.yaml``.

References
----------
Paper Sections 3.1.1, 3.3, 3.4, 3.5.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from calyx.modules.sadr import SADR
from calyx.modules.pspk import PSPKBlock
from calyx.modules.pgi import PGI
from calyx.modules.dar import DARHead


# ----------------------------------------------------------------------
# Simplified skeleton for exposition and unit testing
# ----------------------------------------------------------------------

class _ConvBNAct(nn.Module):
    """Standard Conv + BN + SiLU block, a backbone placeholder."""

    def __init__(
        self, in_c: int, out_c: int, k: int = 3, s: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


@dataclass
class CalyxOutputs:
    """CalyxYOLO forward outputs."""
    detections: torch.Tensor
    """Detection-head output; shape depends on the head. The skeleton returns a
       placeholder tensor concatenating the multi-scale features."""

    polarity_p2: torch.Tensor
    """PSPK P2-stage polarity map, [B, 4, H/4, W/4], for ``L_polar`` and the
       DAR head."""

    polarity_p4: torch.Tensor
    """PSPK P4-stage polarity map, [B, 4, H/16, W/16]."""

    polarity_per_scale: List[torch.Tensor]
    """PGI downsampled polarity maps, length 3, for P3 / P4 / P5."""

    depth_pred: Optional[torch.Tensor]
    """DAR-head depth prediction, [B, 1, H/8, W/8] during training, None at
       inference."""

    sadr_intermediate: Tuple[torch.Tensor, torch.Tensor]
    """(s, alpha) tuple: specular-strength map and soft gate, for ablation."""


class CalyxYOLOSkeleton(nn.Module):
    """Minimal CALYX integration skeleton (exposition/test version).

    Parameters
    ----------
    nc : int, default 4
        Number of classes, fixed to 4 by the paper (pear / peduncle / calyx /
        defect). The concave/convex/saddle/flat local-geometry classes are only
        intermediate PSPK channels, not detection classes (Section 3.1.1).
    base_channels : int, default 32
        Backbone starting channels; the simplified version doubles per stage.
    p3_channels : int, default 128
    p4_channels : int, default 256
    p5_channels : int, default 512
        Neck channels for the three scales, near the YOLO11n scale.
    enable_dar : bool, default True
        Whether to enable the DAR head. Set False at inference export to strip
        the auxiliary branch.
    """

    def __init__(
        self,
        nc: int = 4,
        base_channels: int = 32,
        p3_channels: int = 128,
        p4_channels: int = 256,
        p5_channels: int = 512,
        enable_dar: bool = True,
    ) -> None:
        super().__init__()
        if nc != 4:
            # Section 3.1.1 fixes 4 classes. Other values are allowed but keep
            # nc=4 in practice for physical alignment with PSPK.
            pass

        self.nc = int(nc)
        self.enable_dar = bool(enable_dar)

        # ---- SADR (Section 3.2) ----
        self.sadr = SADR()

        # ---- simplified backbone: five stride-2 stages (P1..P5) ----
        c0 = base_channels
        c1 = base_channels  # P1, stride 2
        c2 = base_channels * 2  # P2, stride 4
        c3 = p3_channels  # P3, stride 8
        c4 = p4_channels  # P4, stride 16
        c5 = p5_channels  # P5, stride 32

        self.stem = _ConvBNAct(3, c0, k=3, s=2)         # P1: stride 2
        self.b1 = _ConvBNAct(c0, c1, k=3, s=1)
        self.down_p2 = _ConvBNAct(c1, c2, k=3, s=2)     # P2: stride 4
        self.b2 = _ConvBNAct(c2, c2, k=3, s=1)
        self.down_p3 = _ConvBNAct(c2 + 4, c3, k=3, s=2)  # +4 from PSPK
        self.b3 = _ConvBNAct(c3, c3, k=3, s=1)
        self.down_p4 = _ConvBNAct(c3, c4, k=3, s=2)
        self.b4 = _ConvBNAct(c4, c4, k=3, s=1)
        self.down_p5 = _ConvBNAct(c4 + 4, c5, k=3, s=2)  # +4 from PSPK
        self.b5 = _ConvBNAct(c5, c5, k=3, s=1)

        # ---- PSPK at P2 and P4 (Section 3.3) ----
        self.pspk_p2 = PSPKBlock(in_channels=c2)
        self.pspk_p4 = PSPKBlock(in_channels=c4)

        # ---- simplified PAN-FPN neck ----
        # top-down
        self.lateral_p4 = nn.Conv2d(c4 + 4, c4, kernel_size=1)
        self.lateral_p3 = nn.Conv2d(c3, c3, kernel_size=1)
        self.merge_p4 = _ConvBNAct(c4 + c5, c4, k=3, s=1)
        self.merge_p3 = _ConvBNAct(c3 + c4, c3, k=3, s=1)
        # bottom-up
        self.bu_p4 = _ConvBNAct(c3, c4, k=3, s=2)
        self.bu_p5 = _ConvBNAct(c4, c5, k=3, s=2)

        # ---- PGI three-scale injection (Section 3.4) ----
        self.pgi = PGI(target_channels=(c3, c4, c5))

        # ---- detection head: simplified multi-scale conv outputs ----
        # Replace with the YOLO11 decoupled head in real deployment.
        self.head_p3 = nn.Conv2d(c3, (5 + nc), kernel_size=1)
        self.head_p4 = nn.Conv2d(c4, (5 + nc), kernel_size=1)
        self.head_p5 = nn.Conv2d(c5, (5 + nc), kernel_size=1)

        # ---- DAR head (Section 3.5.1) ----
        if self.enable_dar:
            self.dar_head = DARHead(in_channels=c3)
        else:
            self.dar_head = None

    def forward(self, image: torch.Tensor) -> CalyxOutputs:
        """Forward pass.

        Parameters
        ----------
        image : torch.Tensor
            [B, 3, H, W].

        Returns
        -------
        CalyxOutputs
        """
        # ---- SADR ----
        i_tilde, s_map, alpha = self.sadr(image)

        # ---- backbone ----
        x = self.stem(i_tilde)        # P1
        x = self.b1(x)
        x = self.down_p2(x)            # P2
        x = self.b2(x)

        # PSPK at P2
        x_p2_cat, polarity_p2 = self.pspk_p2(x)  # [B, c2+4, H/4, W/4]

        x = self.down_p3(x_p2_cat)     # P3
        x = self.b3(x)
        feat_p3 = x

        x = self.down_p4(x)            # P4
        x = self.b4(x)

        # PSPK at P4
        x_p4_cat, polarity_p4 = self.pspk_p4(x)  # [B, c4+4, H/16, W/16]

        x = self.down_p5(x_p4_cat)     # P5
        x = self.b5(x)
        feat_p5 = x

        # ---- PAN-FPN neck (simplified) ----
        # top-down: P5 -> P4
        p4_lateral = self.lateral_p4(x_p4_cat)
        p5_up = F.interpolate(feat_p5, scale_factor=2, mode="nearest")
        p4_td = self.merge_p4(torch.cat([p4_lateral, p5_up], dim=1))

        # top-down: P4 -> P3
        p3_lateral = self.lateral_p3(feat_p3)
        p4_up = F.interpolate(p4_td, scale_factor=2, mode="nearest")
        p3_td = self.merge_p3(torch.cat([p3_lateral, p4_up], dim=1))

        # bottom-up: P3 -> P4 -> P5
        p4_bu = p4_td + self.bu_p4(p3_td)
        p5_bu = feat_p5 + self.bu_p5(p4_bu)

        # ---- PGI three-scale injection ----
        pgi_inputs = [p3_td, p4_bu, p5_bu]
        pgi_outputs, polarity_per_scale = self.pgi(
            pgi_inputs, polarity_p2
        )
        feat_p3_inj, feat_p4_inj, feat_p5_inj = pgi_outputs

        # ---- detection head ----
        det_p3 = self.head_p3(feat_p3_inj)
        det_p4 = self.head_p4(feat_p4_inj)
        det_p5 = self.head_p5(feat_p5_inj)
        # Simplified: stack the three scales; a real head would integrate these.
        detections = torch.cat(
            [
                det_p3.flatten(start_dim=2),
                det_p4.flatten(start_dim=2),
                det_p5.flatten(start_dim=2),
            ],
            dim=2,
        )  # [B, 5+nc, N_anchors_total]

        # ---- DAR head (active during training) ----
        if self.training and self.dar_head is not None:
            depth_pred = self.dar_head(feat_p3_inj)
        else:
            depth_pred = None

        return CalyxOutputs(
            detections=detections,
            polarity_p2=polarity_p2,
            polarity_p4=polarity_p4,
            polarity_per_scale=polarity_per_scale,
            depth_pred=depth_pred,
            sadr_intermediate=(s_map, alpha),
        )

    def export_inference_only(self) -> "CalyxYOLOSkeleton":
        """Strip the DAR head and training-only branches for inference export.

        Section 3.5.4 removes DAR and the calyx-end probe at inference via ONNX
        subgraph pruning. This returns the model with the DAR head removed,
        ready for ``torch.onnx.export``. The calyx-end probe is usually managed
        by the training script (outside the nn.Module), so this mainly strips
        the DAR head.

        Returns
        -------
        self : CalyxYOLOSkeleton
            DAR head removed in place; returns self for chaining.
        """
        self.enable_dar = False
        if self.dar_head is not None:
            del self.dar_head
            self.dar_head = None
        return self
