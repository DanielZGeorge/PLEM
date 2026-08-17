"""
Soft clDice — differentiable analog of metrics/cldice.py.

Reference: Shit et al. "clDice — a Novel Topology-Preserving Loss Function
for Tubular Structure Segmentation", CVPR 2021.

metrics/cldice.py computes tprec/tsens/clDice on HARD skeletons produced by
`skimage.morphology.skeletonize` (discrete Lee thinning — not differentiable,
and iterated to full convergence regardless of input width, which is exactly
what makes eval-time clDice width-invariant). This module reimplements the
paper's own "soft-skeletonization" trick instead: iterated morphological
erosion (`soft_erode`) alternated with dilation-of-the-eroded-result
(`soft_open`) applied directly to the model's continuous probability map via
`F.max_pool2d`, fully differentiable, no discrete `skeletonize` call.

Known limitation (unlike the hard eval-time version): soft-skeletonization
runs for a FIXED number of iterations (`iters`), not to convergence, so width
invariance only holds up to whatever width that fixed iteration count can
fully erode away — a road several times wider than `iters` px will not be
skeletonized down to a true 1px centerline. `iters` should be chosen relative
to the widest class-width perturbation the loss is expected to stay robust
to; see `tests/test_losses_sanity.py::TestSoftClDiceLoss`'s width-
insensitivity regression test for the range this is validated over.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) -> (B, 1, H, W). Erosion via -maxpool(-x): shrinks foreground."""
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) -> (B, 1, H, W). Dilation via maxpool: grows foreground."""
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    """Morphological opening = dilate(erode(x)) — removes structure thinner
    than one erosion step, keeping only what survives it."""
    return soft_dilate(soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """
    (B, 1, H, W) -> (B, 1, H, W) soft skeleton in [0, 1]. Shit et al.'s
    iterative algorithm: at each scale, the part of `x` NOT recovered by
    opening it (`relu(x - open(x))`) is skeleton mass at that scale; erode
    `x` and repeat, fuzzy-OR-accumulating the skeleton response seen at each
    pixel across scales.
    """
    x1 = soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iters):
        x = soft_erode(x)
        x1 = soft_open(x)
        delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(
    pred_prob_c: torch.Tensor, gt_hard_c: torch.Tensor, iters: int = 10, eps: float = 1e-6
) -> torch.Tensor:
    """
    pred_prob_c, gt_hard_c: (B, H, W). Literal soft analog of
    metrics/cldice.py's `cldice()`: tprec = |skel_pred . gt| / |skel_pred|,
    tsens = |skel_gt . pred| / |skel_gt|, cl = 2*tprec*tsens/(tprec+tsens) —
    computed on SOFT skeletons of soft probability masks instead of
    `skimage.skeletonize()`'d binary masks. Returns a scalar (batch-mean
    clDice score, in [0, 1] — NOT yet a loss; callers take `1 - soft_cldice`).
    """
    pred = pred_prob_c.unsqueeze(1)
    gt = gt_hard_c.unsqueeze(1)

    skel_pred = soft_skeletonize(pred, iters)
    # gt has requires_grad=False already (it's a fixed label-derived tensor),
    # so no_grad here is purely documentation of intent, not load-bearing.
    with torch.no_grad():
        skel_gt = soft_skeletonize(gt, iters)

    tprec = ((skel_pred * gt).sum(dim=(1, 2, 3)) + eps) / (skel_pred.sum(dim=(1, 2, 3)) + eps)
    tsens = ((skel_gt * pred).sum(dim=(1, 2, 3)) + eps) / (skel_gt.sum(dim=(1, 2, 3)) + eps)

    cl = 2 * tprec * tsens / (tprec + tsens + eps)
    return cl.mean()


class SoftClDiceLoss(nn.Module):
    """
    Differentiable analog of metrics/cldice.py, applied per linear class id,
    macro-averaged (mirrors `cldice_multiclass` + `mean_cldice(reduction=
    "macro")`).
    """

    def __init__(self, linear_classes: list, iters: int = 10, eps: float = 1e-6):
        super().__init__()
        self.linear_classes = linear_classes
        self.iters = iters
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.linear_classes:
            return torch.zeros((), device=logits.device)
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        gt_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        losses = [
            1.0 - soft_cldice(probs[:, c], gt_onehot[:, c], self.iters, self.eps)
            for c in self.linear_classes
        ]
        return torch.stack(losses).mean()
