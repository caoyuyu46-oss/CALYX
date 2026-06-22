# -*- coding: utf-8 -*-
"""CALYX: Calyx-Aware Lighting-Yielded eXplainable framework.

CALYX is a physics-guided single-stage detection framework for production-line
surface-defect grading of Korla pears, integrating physical optical priors,
interpretable representation learning, and the detection task in one framework.

Module layout
-------------
- ``calyx.modules.sadr.SADR``: specular-aware diffuse recovery (Section 3.2)
- ``calyx.modules.pspk.PSPKBlock``: photometric shape-prior kernel block
  (Section 3.3)
- ``calyx.modules.pgi.PGI``: polarity-guided class-prior injection (Section 3.4)
- ``calyx.modules.dar.DARHead``: depth-alignment auxiliary head (Section 3.5.1)
- ``calyx.losses``: the four end-to-end losses (SiLog / Polar / Probe / Total)
- ``calyx.models.calyx_yolo.CalyxYOLOSkeleton``: end-to-end reference integration
- ``calyx.models.register.register_calyx_modules``: ultralytics registrar

References
----------
This repository accompanies the paper; the dataset and training checkpoints will
be released after acceptance.
"""

__version__ = "1.0.0"
__all__ = [
    "SADR",
    "PSPKBlock",
    "PGI",
    "DARHead",
    "SiLogLoss",
    "PolarConsistencyLoss",
    "CalyxLinearProbe",
    "CalyxTotalLoss",
    "CalyxLossWeights",
    "CalyxLossOutput",
    "CalyxYOLOSkeleton",
    "CalyxOutputs",
    "register_calyx_modules",
]

from calyx.modules.sadr import SADR
from calyx.modules.pspk import PSPKBlock
from calyx.modules.pgi import PGI
from calyx.modules.dar import DARHead

from calyx.losses.silog import SiLogLoss
from calyx.losses.polar_consistency import PolarConsistencyLoss
from calyx.losses.calyx_probe import CalyxLinearProbe
from calyx.losses.total import (
    CalyxTotalLoss,
    CalyxLossWeights,
    CalyxLossOutput,
)

from calyx.models.calyx_yolo import CalyxYOLOSkeleton, CalyxOutputs
from calyx.models.register import register_calyx_modules
