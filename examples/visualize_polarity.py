#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_polarity.py
=====================

Visualize either the PSPK kernel bank (no checkpoint required) or the
polarity map output of a trained CALYX model on a given input image.

Two modes::

    --mode kernels                 (default; no checkpoint required)
        Plots K_1, K_2, K_3, K_4 in a 1×4 grid using the diverging RdBu_r
        colormap, following paper Eqs. (3.8)-(3.10).

    --mode polarity                (requires checkpoint and image)
        Loads a trained CALYX model, runs the input image through SADR +
        backbone + PSPK at P2, and plots the 4-channel polarity map next
        to the input (paper Section 3.3).

Examples::

    # Plot the PSPK kernels (no training required)
    python examples/visualize_polarity.py --mode kernels --output kernels.png

    # Visualize polarity on a test image (requires checkpoint)
    python examples/visualize_polarity.py --mode polarity \\
        --checkpoint checkpoints/calyx.pt \\
        --image data/test_pear.jpg \\
        --output polarity.png
"""

import argparse
import sys
from typing import Optional

import numpy as np
import torch


def plot_kernels(save_path: Optional[str]) -> None:
    """Mode 1: plot the four physically-initialized PSPK kernels."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is required. Install via `pip install matplotlib`.")
        sys.exit(1)

    from calyx.modules.pspk import build_pspk_kernel_bank

    bank = build_pspk_kernel_bank().squeeze(1).numpy()  # [4, 4, 4]
    titles = [
        "$K_1$ Concave",
        "$K_2$ Convex",
        "$K_3$ Saddle",
        "$K_4$ Planar",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    vmax = float(np.abs(bank).max())
    for i, ax in enumerate(axes):
        im = ax.imshow(bank[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(titles[i], fontsize=12)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.tick_params(labelsize=8)
        # annotate each cell
        for r in range(4):
            for c in range(4):
                v = bank[i, r, c]
                ax.text(
                    c, r, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(v) > vmax * 0.45 else "black",
                )

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle(
        "PSPK Physically-Initialized Kernels (Paper Eq. 3.8–3.10)",
        fontsize=13,
    )

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Kernel figure saved to: {save_path}")
    else:
        plt.show()


def plot_polarity(checkpoint_path: str, image_path: str, save_path: Optional[str]) -> None:
    """Mode 2: plot polarity map output for a given image."""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("ERROR: matplotlib and Pillow are required.")
        sys.exit(1)

    from calyx import CalyxYOLOSkeleton

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0

    # Resize to 640x640 (or any multiple of 32) for the skeleton model
    target_size = 640
    pil = Image.fromarray((img_np * 255.0).astype(np.uint8)).resize(
        (target_size, target_size), Image.BILINEAR
    )
    img_resized = np.asarray(pil).astype(np.float32) / 255.0
    image_tensor = (
        torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0)
    )  # [1, 3, H, W]

    # Load model
    model = CalyxYOLOSkeleton(nc=4)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        out = model(image_tensor)
    polarity_p2 = out.polarity_p2.squeeze(0).numpy()  # [4, H/4, W/4]

    # Plot
    titles = ["$P_1$ Concave", "$P_2$ Convex", "$P_3$ Saddle", "$P_4$ Planar"]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.5))

    axes[0].imshow(img_resized)
    axes[0].set_title("Input image", fontsize=12)
    axes[0].axis("off")

    vmax = float(np.abs(polarity_p2).max())
    for i in range(4):
        im = axes[i + 1].imshow(polarity_p2[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i + 1].set_title(titles[i], fontsize=12)
        axes[i + 1].axis("off")
    fig.colorbar(im, ax=axes[1:], fraction=0.025, pad=0.02)
    fig.suptitle("Polarity map at P2 stage", fontsize=13)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Polarity figure saved to: {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CALYX kernel and polarity visualization."
    )
    parser.add_argument(
        "--mode",
        choices=["kernels", "polarity"],
        default="kernels",
        help="`kernels` plots K_1..K_4; `polarity` plots polarity map for an image.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to trained CALYX checkpoint (.pt). Required when mode=polarity.",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image. Required when mode=polarity.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path (.png/.pdf/.svg). If omitted, opens an interactive window.",
    )
    args = parser.parse_args()

    if args.mode == "kernels":
        plot_kernels(args.output)
    elif args.mode == "polarity":
        if args.checkpoint is None or args.image is None:
            parser.error("--checkpoint and --image are required when mode=polarity")
        plot_polarity(args.checkpoint, args.image, args.output)


if __name__ == "__main__":
    main()
