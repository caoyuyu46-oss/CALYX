# -*- coding: utf-8 -*-
"""Ultralytics custom-module registration.

For the ``PSPKBlock`` and ``PGI`` names in ``configs/calyx_yolo11n.yaml`` to be
recognised by the ultralytics YOLO11 parser, call ``register_calyx_modules``
once before building the model.

ultralytics parses the YAML via ``parse_model`` in ``ultralytics.nn.tasks``,
which raises ``KeyError`` on unknown layer names. ``register_calyx_modules``
injects the CALYX custom layers into the parser's global symbol table for
transparent loading.

Example
-------
.. code-block:: python

    from calyx.models.register import register_calyx_modules
    register_calyx_modules()

    from ultralytics import YOLO
    model = YOLO("configs/calyx_yolo11n.yaml")

References
----------
Section 4.2 uses YOLO11n as the baseline backbone.
"""

from typing import Optional

from calyx.modules.pspk import PSPKBlock
from calyx.modules.pgi import PGI
from calyx.modules.dar import DARHead


def register_calyx_modules(verbose: bool = False) -> None:
    """Register the CALYX custom modules with the ultralytics parser.

    Parameters
    ----------
    verbose : bool, default False
        Print registration details.

    Notes
    -----
    Returns silently when ultralytics is unavailable, so this module can be
    imported in a pure-PyTorch environment for exposition and testing.
    """
    try:
        # Access the ultralytics parser's global symbol injection point. The
        # path may vary across versions; this is the most compatible form.
        from ultralytics.nn import tasks as _tasks
    except ImportError:
        if verbose:
            print(
                "[CALYX] ultralytics not installed; skipping custom-module "
                "registration. Use CalyxYOLOSkeleton for pure-PyTorch "
                "integration."
            )
        return

    # ultralytics parse_model looks up layer classes via globals(); injecting
    # them into the ``_tasks`` module symbol table makes the YAML names resolve.
    setattr(_tasks, "PSPKBlock", PSPKBlock)
    setattr(_tasks, "PGI", PGI)
    setattr(_tasks, "DARHead", DARHead)

    if verbose:
        print(
            "[CALYX] Registered custom modules PSPKBlock, PGI, DARHead "
            "with ultralytics.nn.tasks"
        )


def is_registered() -> bool:
    """Check whether the CALYX custom modules are registered."""
    try:
        from ultralytics.nn import tasks as _tasks
    except ImportError:
        return False
    return all(hasattr(_tasks, name) for name in ("PSPKBlock", "PGI", "DARHead"))
