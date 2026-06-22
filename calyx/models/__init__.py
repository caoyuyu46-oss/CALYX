# -*- coding: utf-8 -*-
"""CALYX end-to-end model integration."""
from calyx.models.calyx_yolo import CalyxYOLOSkeleton, CalyxOutputs
from calyx.models.register import register_calyx_modules, is_registered

__all__ = [
    "CalyxYOLOSkeleton",
    "CalyxOutputs",
    "register_calyx_modules",
    "is_registered",
]
