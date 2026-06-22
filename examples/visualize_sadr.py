#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_sadr.py
=================

Visualize the SADR pipeline as a 4-panel figure (paper Eqs. 3.5-3.7) showing
input RGB → specular strength s(x) → soft gate α(x) → recovered Ĩ(x).

SADR has no learnable parameters, so this script does not require a
trained checkpoint. It works on any RGB input image.

Usage::

    python examples/visualize_sadr.py --image path/to/pear.jpg --output sadr.png

If ``--output`` is omitted, an interactive matplotlib window opens.

Hyperparameters (paper Section 3.2.3 defaults)::

    s0    = 0.20
    gamma = 12.0
    tau   = 0.85

Override via ``--s0``, ``--gamma``, ``--tau`` for sensitivity demonstrations.
"""

import argparse
import sys

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SADR pipeline visualization (paper Eqs. 3.5-3.7)."
    )
    parser.add_argument("--image", type=str, required=True, help="Input RGB image.")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output figure path. If omitted, opens an interactive window.",
    )
    parser.add_argument("--s0", type=float, default=0.20, help="SADR s0.")
    parser.add_argument("--gamma", type=float, default=12.0, help="SADR gamma.")
    parser.add_argument("--tau", type=float, default=0.85, help="SADR tau.")
    parser.add_argument(
        "--resize", type=int, default=None,
        help="Optional square resize size before running SADR. "
             "Useful for very large images.",
    )
    args = parser.parse_args()

    # ----- imports (deferred to allow `--help` without optional deps) -----
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("ERROR: matplotlib and Pillow are required.")
        print("Install via: pip install matplotlib Pillow")
        sys.exit(1)

    from calyx.modules.sadr import SADR

    # ----- load image -----
    img = Image.open(args.image).convert("RGB")
    if args.resize is not None:
        img = img.resize((args.resize, args.resize), Image.BILINEAR)
    img_np = np.asarray(img).astype(np.float32) / 255.0
    image = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # ----- run SADR -----
    sadr = SADR(s0=args.s0, gamma=args.gamma, tau=args.tau)
    with torch.no_grad():
        i_tilde, s, alpha = sadr(image)

    # Convert back to numpy for plotting
    i_tilde_np = i_tilde.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    s_np = s.squeeze().numpy()
    alpha_np = alpha.squeeze().numpy()

    # ----- plot 4-panel figure -----
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    axes[0].imshow(img_np)
    axes[0].set_title("(a) Input RGB $I(x)$", fontsize=12)
    axes[0].axis("off")

    im_s = axes[1].imshow(s_np, cmap="viridis")
    axes[1].set_title(
        rf"(b) Specular strength $s(x)$" + "\n"
        rf"(Eq. 3.5)",
        fontsize=11,
    )
    axes[1].axis("off")
    plt.colorbar(im_s, ax=axes[1], fraction=0.046, pad=0.04)

    im_a = axes[2].imshow(alpha_np, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title(
        rf"(c) Soft gate $\alpha(x)$" + "\n"
        rf"(Eq. 3.6, $s_0={args.s0}$, $\gamma={args.gamma}$, $\tau={args.tau}$)",
        fontsize=11,
    )
    axes[2].axis("off")
    plt.colorbar(im_a, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(i_tilde_np)
    axes[3].set_title(r"(d) Recovered $\tilde{I}(x)$" + "\n(Eq. 3.7)", fontsize=11)
    axes[3].axis("off")

    fig.suptitle(
        "SADR pipeline (Eqs. 3.5-3.7)",
        fontsize=14, y=1.03,
    )

    fig.tight_layout()

    # ----- print summary stats -----
    sp = (s_np > 1e-3).mean()
    print("Summary:")
    print(f"  Image shape (after resize)  : {img_np.shape}")
    print(f"  Specular strength range     : [{s_np.min():.4f}, {s_np.max():.4f}]")
    print(f"  Specular pixel ratio (s>1e-3): {sp * 100:.2f}%")
    print(f"  Soft gate range             : [{alpha_np.min():.4f}, {alpha_np.max():.4f}]")
    print(f"  Mean recovery delta (|Ĩ−I|) : {np.mean(np.abs(i_tilde_np - img_np)):.4f}")

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Figure saved to: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
