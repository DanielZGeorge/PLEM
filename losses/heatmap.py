"""
Point-Heatmap Regression Loss — the training-side counterpart to
metrics/point_f1.py, NOT a differentiable relaxation of it.

point_f1.py matches predicted vs. GT point-class blobs via connected-
component centroid extraction (`scipy.ndimage.label` + `center_of_mass`) then
tolerance-restricted Hungarian one-to-one assignment
(`scipy.optimize.linear_sum_assignment`) — a combinatorial procedure with no
useful gradient path back to per-pixel logits. Sinkhorn-style differentiable
relaxations of Hungarian exist in the literature but are typically still used
for matching-then-detach rather than true end-to-end gradients, so this
module does not attempt one.

Instead, point supervision is reformulated entirely as dense Gaussian-heatmap
regression, CenterNet-style (Zhou et al. 2019) / CornerNet-style (Law & Deng
2018): burn each GT point instance's centroid into a Gaussian blob on a
target heatmap, and train the point class's own softmax probability channel
to match it with a pixelwise focal loss. `point_f1.py`'s Hungarian matching
is untouched and remains a pure eval-time metric — this is a genuine
train/eval formulation mismatch, documented as such rather than papered over.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label as cc_label, center_of_mass


_MAX_INSTANCES_WARNED = False


def gt_centroids_to_heatmap(
    gt_point_mask: torch.Tensor, sigma: float = 2.0, max_instances: int = 100
) -> torch.Tensor:
    """
    (B, H, W) binary GT point-class mask -> (B, H, W) float Gaussian heatmap
    in [0, 1]. Computed under `no_grad` via `scipy.ndimage.label` +
    `center_of_mass` on the CPU — exactly the same connected-component +
    centroid extraction `metrics/point_f1.py::_instance_centroids` already
    does at eval time, but here purely for TARGET construction, not inside
    the loss's gradient path. Overlapping Gaussians from nearby GT points are
    combined via elementwise max (standard CenterNet convention — avoids
    peak inflation where points cluster). Each instance additionally stamps its
    nearest integer pixel to exactly 1.0 so `heatmap_focal_loss`'s positive
    term (gated on `gt_heatmap >= 1.0`) always fires — see the loop below.

    `max_instances` bounds worst-case cost: the per-centroid loop below does
    one full (H, W) Gaussian burn per instance, so a mask with a pathological
    number of tiny/scattered components (e.g. dense per-pixel noise, never
    seen on real point-feature data — real Potsdam crops top out around a
    few dozen instances per 256x256 tile) could otherwise silently stall
    training for minutes per batch with no error. Instances beyond this cap
    are dropped (with a one-time warning) rather than processed, trading
    target completeness for a bounded worst case on malformed input.
    """
    device = gt_point_mask.device
    mask_np = gt_point_mask.detach().to("cpu").numpy()
    B, H, W = mask_np.shape

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    heatmaps = np.zeros((B, H, W), dtype=np.float32)

    for b in range(B):
        labeled, n = cc_label(mask_np[b] > 0)
        if n == 0:
            continue
        if n > max_instances:
            global _MAX_INSTANCES_WARNED
            if not _MAX_INSTANCES_WARNED:
                print(
                    f"gt_centroids_to_heatmap: {n} connected components in one sample exceeds "
                    f"max_instances={max_instances} -- dropping the excess (this should not "
                    f"happen on real point-feature data; check upstream labels if it does)."
                )
                _MAX_INSTANCES_WARNED = True
            n = max_instances
        centroids = np.asarray(
            center_of_mass(mask_np[b] > 0, labeled, np.arange(1, n + 1))
        ).reshape(-1, 2)
        for cy, cx in centroids:
            g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
            heatmaps[b] = np.maximum(heatmaps[b], g)
            # Stamp the nearest integer pixel to EXACTLY 1.0. `center_of_mass`
            # returns sub-pixel float coords for any blob wider than one pixel
            # (i.e. essentially every real Potsdam tree/car blob), so the
            # Gaussian above peaks at < 1.0 on the integer grid (e.g. ~0.977 for
            # a 0.3 px centroid offset) and no pixel ever reaches 1.0.
            # `heatmap_focal_loss` gates its ENTIRE positive term on
            # `gt_heatmap >= 1.0`; without this stamp a fractional centroid gets
            # zero positive supervision, the point channel receives only
            # suppressive (push-to-zero) gradient, and the model learns to
            # predict no point pixels at all — exactly what was observed on the
            # first real train_unet_joint_scaled.ipynb run (0 predicted point
            # px on every source; heatmap sub-term 1.13 -> 0.0004 after epoch 1).
            ry = int(min(H - 1, max(0, round(float(cy)))))
            rx = int(min(W - 1, max(0, round(float(cx)))))
            heatmaps[b, ry, rx] = 1.0

    return torch.from_numpy(heatmaps).to(device=device, dtype=torch.float32)


def heatmap_focal_loss(
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    (B, H, W), (B, H, W) -> scalar. Penalty-reduced pixelwise focal loss
    (CornerNet/CenterNet formulation): full penalty at exact GT peaks
    (`gt_heatmap == 1`), reduced penalty near peaks (weighted by
    `(1 - gt_heatmap)^beta`), so near-misses are punished less than spurious
    detections far from any GT point. Sidesteps instance matching entirely.

    Normalized by the TOTAL positive-peak count across the whole batch (the
    standard CenterNet reduction), not by a per-sample mean. This matters for
    PLEM's mixed-source batches: only Potsdam tiles annotate the point class,
    and of those only a minority contain any point instances, so a per-sample
    `.mean()` divides the point signal by the full batch size (~20-40x here)
    and the term is swamped by the CE/Dice base. Global normalization keeps the
    gradient magnitude tied to how many points are actually in the batch.
    Falls back to the raw loss sum when the batch has zero GT points at all.
    """
    pred = pred_heatmap.clamp(eps, 1 - eps)
    pos_mask = (gt_heatmap >= 1.0).float()
    neg_mask = 1.0 - pos_mask

    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
    neg_loss = -((1 - gt_heatmap) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg_mask

    num_pos = pos_mask.sum()
    total = pos_loss.sum() + neg_loss.sum()
    return total / num_pos.clamp_min(1.0)


class PointHeatmapLoss(nn.Module):
    """
    Point-F1 analog — a genuinely different formulation (dense heatmap
    regression), not a relaxation of Hungarian matching (see module
    docstring). Uses the point class's own softmax probability channel
    directly as the predicted heatmap (no extra head).
    """

    def __init__(self, point_classes: list, sigma: float = 2.0, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.point_classes = point_classes
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.point_classes:
            return torch.zeros((), device=logits.device)
        probs = torch.softmax(logits, dim=1)

        losses = []
        for c in self.point_classes:
            gt_mask_c = (target == c).float()
            with torch.no_grad():
                gt_heatmap = gt_centroids_to_heatmap(gt_mask_c, self.sigma)
            losses.append(heatmap_focal_loss(probs[:, c], gt_heatmap, self.alpha, self.beta))
        return torch.stack(losses).mean()
