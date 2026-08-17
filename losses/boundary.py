"""
Soft Boundary Loss — differentiable analog of metrics/boundary_f1.py.

boundary_f1.py extracts boundary pixels via `skimage.morphology.disk`-based
binary dilation (`_extract_boundary`), then applies the same tolerance-radius
EDT test as DTAF1, restricted to those boundary pixels. Both the boundary
extraction and the tolerance test are hard/discrete.

This module reuses `losses.tolerance.soft_tolerance_pair_loss` (the same
precision/recall-against-a-dilated-band primitive DTAF1's loss uses) applied
to a *soft boundary map* instead of the full class mask — mostly wiring on
top of that primitive, not new math. The soft boundary map is the standard
differentiable stand-in for morphological boundary extraction: the
morphological gradient `dilate(x) - erode(x)` (reusing the same soft
erosion/dilation primitives `losses.cldice` uses for soft-skeletonization),
which is 0 in flat interior/exterior regions and saturates to exactly 1.0 at
a hard, confident class edge (dilated=1, eroded=0 there) — this is actually a
close differentiable match for `boundary_f1.py`'s `_extract_boundary`, which
is itself effectively a discrete dilate-minus-erode formula (`dilated & ~m |
(m & ~dilated(~m))`). An earlier `|x - avgpool3x3(x)|` formulation was tried
and rejected: its peak response for a hard step edge is well below 1.0 (~0.56
for a straight edge under a 3x3 average), which silently caps the recall
term's achievable value even for a perfectly-aligned prediction — caught by
`tests/test_losses_sanity.py::TestSoftBoundaryLoss::test_perfect_prediction_near_zero`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.cldice import soft_erode, soft_dilate
from losses.tolerance import soft_tolerance_pair_loss


def soft_boundary_map(mask: torch.Tensor) -> torch.Tensor:
    """
    (B, H, W) -> (B, H, W). Morphological gradient `dilate(x) - erode(x)`:
    high (saturating to 1.0 for a hard/confident edge) where `mask` (a soft
    probability channel or a binary GT channel) has a sharp 3x3-neighborhood
    transition, 0 in flat regions. Applying this to a soft probability map
    keeps the whole path differentiable; applying it to a binary GT mask
    (under `no_grad`, since GT is fixed) gives a boundary ring in the same
    spirit as boundary_f1.py's dilation-based extraction.
    """
    x = mask.unsqueeze(1)
    boundary = soft_dilate(x) - soft_erode(x)
    return boundary[:, 0]


class SoftBoundaryLoss(nn.Module):
    """
    Differentiable analog of metrics/boundary_f1.py. `class_config` uses the
    same `{class_id: {"name": ..., "tolerance": ...}}` shape as
    `losses.tolerance.ToleranceBandLoss` — typically the polygon classes,
    with a boundary-positional tolerance (e.g. matching `building_tolerance`
    in metrics/unified.py::evaluate_all).
    """

    def __init__(self, class_config: dict, eps: float = 1e-6):
        super().__init__()
        self.class_config = class_config
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        gt_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        losses = []
        for c, cfg in self.class_config.items():
            pred_boundary = soft_boundary_map(probs[:, c])
            with torch.no_grad():
                gt_boundary = (soft_boundary_map(gt_onehot[:, c]) > 1e-6).float()
            losses.append(
                soft_tolerance_pair_loss(pred_boundary, gt_boundary, cfg["tolerance"], self.eps)
            )
        return torch.stack(losses).mean()
