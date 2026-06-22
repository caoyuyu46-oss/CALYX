#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""quick_start.py.

End-to-end minimal demonstration of the CALYX framework using the standalone
skeleton model (no Ultralytics dependency).

This script:
  1. builds ``CalyxYOLOSkeleton`` with default parameters;
  2. runs a forward pass on synthetic input;
  3. prints the shape of every output tensor;
  4. computes the four loss components with the real loss modules and
     aggregates them with ``CalyxTotalLoss``;
  5. performs a single optimizer step to verify the backward graph.

Run::

    python examples/quick_start.py

A few seconds on CPU for a single batch. Useful for checking installation and
API integrity. For real training, see ``docs/usage.md``.
"""

import torch

from calyx import (
    CalyxLinearProbe,
    CalyxLossWeights,
    CalyxTotalLoss,
    CalyxYOLOSkeleton,
    PolarConsistencyLoss,
    SiLogLoss,
)


def main() -> None:
    # ----------------------------------------------------------------
    # 1. Build the model
    # ----------------------------------------------------------------
    print("=" * 60)
    print("CALYX quick-start demo")
    print("=" * 60)

    torch.manual_seed(42)

    model = CalyxYOLOSkeleton(nc=4)
    # Train mode so the DAR head produces a depth prediction.
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[1/5] Skeleton model built. Trainable parameters: {n_params:,}")

    # ----------------------------------------------------------------
    # 2. Forward pass on synthetic input
    # ----------------------------------------------------------------
    image = torch.rand(2, 3, 640, 640)  # batch=2, RGB, 640x640
    out = model(image)

    print("\n[2/5] Forward pass complete. Output shapes:")
    print(f"        detections      : {tuple(out.detections.shape)}")
    print(f"        polarity (P2)   : {tuple(out.polarity_p2.shape)}")
    print(f"        polarity (P4)   : {tuple(out.polarity_p4.shape)}")
    print(f"        polarity scales : {[tuple(p.shape) for p in out.polarity_per_scale]}")
    if out.depth_pred is not None:
        print(f"        depth_pred      : {tuple(out.depth_pred.shape)}")
    s, alpha = out.sadr_intermediate
    print(f"        SADR s          : {tuple(s.shape)}, range "
          f"[{s.min().item():.4f}, {s.max().item():.4f}]")
    print(f"        SADR alpha      : {tuple(alpha.shape)}, range "
          f"[{alpha.min().item():.4f}, {alpha.max().item():.4f}]")

    # ----------------------------------------------------------------
    # 3. Construct dummy supervisory targets
    # ----------------------------------------------------------------
    # In production, replace these with real targets parsed from your dataset
    # and a YOLO-style detection loss for L_det. Here synthetic targets exercise
    # the full backward graph.
    silog = SiLogLoss()
    polar = PolarConsistencyLoss()
    probe = CalyxLinearProbe()

    # Placeholder detection loss (replace with the real YOLO loss).
    l_det = out.detections.abs().mean()

    # Synthetic positive depth target (e.g. from Depth-Anything-V2).
    depth_target = torch.rand_like(out.depth_pred) + 0.5
    l_dar = silog(out.depth_pred, depth_target)

    # Cross-scale polarity consistency between P2 and P4.
    l_polar = polar(out.polarity_p2, out.polarity_p4)

    # Calyx-end probe: pool the P3 polarity into per-sample 4-D ROI vectors and
    # pair them with synthetic binary labels.
    roi_vecs = out.polarity_per_scale[0].mean(dim=(2, 3))  # [B, 4]
    roi_labels = torch.randint(0, 2, (roi_vecs.size(0),))
    l_probe = probe(roi_vecs, roi_labels)

    print("\n[3/5] Dummy targets prepared and component losses computed.")

    # ----------------------------------------------------------------
    # 4. Aggregate the total loss
    # ----------------------------------------------------------------
    loss_fn = CalyxTotalLoss(weights=CalyxLossWeights())
    print(f"\n[4/5] Loss weights: {loss_fn.weights}")

    loss_out = loss_fn(
        loss_det=l_det,
        loss_dar=l_dar,
        loss_polar=l_polar,
        loss_probe=l_probe,
    )

    print(f"        L_det      = {loss_out.det.item():.4f}")
    print(f"        L_DAR      = {loss_out.dar.item():.4f}")
    print(f"        L_polar    = {loss_out.polar.item():.4f}")
    print(f"        L_probe    = {loss_out.probe.item():.4f}")
    print(f"        L_total    = {loss_out.total.item():.4f}")

    # ----------------------------------------------------------------
    # 5. Backward + single optimizer step
    # ----------------------------------------------------------------
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    optim.zero_grad()
    loss_out.total.backward()
    optim.step()

    print("\n[5/5] Backward pass + 1 optimizer step done.")
    print("\nQuick-start demo completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
