# -*- coding: utf-8 -*-
"""CALYX loss functions (paper Section 3.5)."""
from calyx.losses.silog import SiLogLoss
from calyx.losses.polar_consistency import PolarConsistencyLoss
from calyx.losses.calyx_probe import CalyxLinearProbe
from calyx.losses.total import CalyxTotalLoss, CalyxLossWeights, CalyxLossOutput

__all__ = [
    "SiLogLoss",
    "PolarConsistencyLoss",
    "CalyxLinearProbe",
    "CalyxTotalLoss",
    "CalyxLossWeights",
    "CalyxLossOutput",
]
