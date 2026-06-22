# -*- coding: utf-8 -*-
"""Visualization utilities.

Tools for visualising the PSPK physically initialised kernels and polarity
maps, reproducing paper Figs. 3-4.

Dependencies
------------
``matplotlib`` is required; others are optional.
"""

from typing import Optional, Tuple

import numpy as np


def visualize_pspk_kernels(save_path: Optional[str] = None):
    """Visualise the four PSPK physically initialised kernels.

    Produces a 1x4 grid showing, left to right, K_1 (concave), K_2 (convex),
    K_3 (saddle), K_4 (flat), with a red-blue diverging colour scale.

    Parameters
    ----------
    save_path : str or None, default None
        If given, save the figure (png / pdf / svg); otherwise leave the plt
        state to the caller.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from calyx.modules.pspk import build_pspk_kernel_bank

    bank = build_pspk_kernel_bank().squeeze(1).numpy()  # [4, 4, 4]
    titles = [
        "$K_1$ Concave",
        "$K_2$ Convex",
        "$K_3$ Saddle",
        "$K_4$ Planar",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    vmax = float(np.abs(bank).max())
    for i, ax in enumerate(axes):
        im = ax.imshow(bank[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        for u in range(4):
            for v in range(4):
                ax.text(
                    v,
                    u,
                    f"{bank[i, u, v]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(bank[i, u, v]) < vmax / 2 else "white",
                )
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    fig.suptitle("PSPK Physical Initialization Kernels", fontsize=12)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def visualize_polarity_map(
    polarity: "torch.Tensor",  # noqa: F821
    save_path: Optional[str] = None,
    sample_idx: int = 0,
):
    """Visualise the four polarity-map channels of one sample.

    Parameters
    ----------
    polarity : torch.Tensor
        [B, 4, H, W], returned by PSPK forward.
    save_path : str or None
    sample_idx : int, default 0
        Which batch sample to visualise.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import torch

    if polarity.dim() != 4 or polarity.size(1) != 4:
        raise ValueError(
            f"polarity expects shape [B, 4, H, W], got {tuple(polarity.shape)}"
        )

    p = polarity[sample_idx].detach().cpu().numpy()  # [4, H, W]
    titles = [
        "$P_1$ Concave Response",
        "$P_2$ Convex Response",
        "$P_3$ Saddle Response",
        "$P_4$ Planar Response",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for i, ax in enumerate(axes):
        im = ax.imshow(p[i], cmap="viridis")
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def visualize_sadr_pipeline(
    image: "torch.Tensor",  # noqa: F821
    save_path: Optional[str] = None,
    sample_idx: int = 0,
):
    """Visualise the SADR pipeline (reproducing Fig. 3).

    Produces four panels: input / specular strength / soft gate / diffuse
    recovery.

    Parameters
    ----------
    image : torch.Tensor
        [B, 3, H, W], values in [0, 1].
    save_path : str or None
    sample_idx : int, default 0

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from calyx.modules.sadr import SADR

    sadr = SADR()
    sadr.eval()

    import torch
    with torch.no_grad():
        i_tilde, s, alpha = sadr(image)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    img_show = image[sample_idx].detach().cpu().numpy().transpose(1, 2, 0)
    img_show = np.clip(img_show, 0, 1)
    axes[0].imshow(img_show)
    axes[0].set_title("(a) Input $I(x)$")

    axes[1].imshow(s[sample_idx, 0].detach().cpu().numpy(), cmap="hot")
    axes[1].set_title("(b) Specular Strength $s(x)$")

    axes[2].imshow(
        alpha[sample_idx, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1
    )
    axes[2].set_title("(c) Soft Gate $\\alpha(x)$")

    out_show = i_tilde[sample_idx].detach().cpu().numpy().transpose(1, 2, 0)
    out_show = np.clip(out_show, 0, 1)
    axes[3].imshow(out_show)
    axes[3].set_title("(d) Diffuse $\\tilde{I}(x)$")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
