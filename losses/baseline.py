"""
Conventional-training baseline for the scaled joint model: the loss a
typical segmentation pipeline would use, with none of PLEM's geometric
terms. It is the comparison point for losses/multitask.py's
PLEMMultiTaskLoss(point_head="sigmoid") and deliberately differs from it in
the loss only:

    segmentation channels : CE + soft Dice (accuracy + an IoU surrogate),
                            the same `ce_dice_loss` the joint loss uses,
                            under the same per-source `_mask_logits` masking.
    point channel         : per-pixel BCE + soft Dice of sigmoid(logits[:, point])
                            against the stamped intersection mask, i.e. points
                            treated as an ordinary rare binary class -- no
                            Gaussian heatmap, no focal term.

Per-source masking is kept on purpose: ignoring unlabelled classes is
standard partial-label practice, and without it SpaceNet6 tiles would teach
"no road" everywhere, confounding the comparison with a labelling effect.

forward(logits, target, class_mask, point_target) has PLEMMultiTaskLoss's
signature and returns the same kind of dict ("loss" plus detached terms), so
a training loop can swap one for the other.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.multitask import _mask_logits, ce_dice_loss


class ConventionalSegLoss(nn.Module):
    """
    Parameters
    ----------
    point_class : channel index of the point logit; must be the last channel.
                  Channels before it are the segmentation softmax.
    pos_weight  : BCE weight on positive point pixels (the usual remedy for a
                  rare foreground, typically the background-to-point pixel
                  ratio of the training set). 1.0 = plain BCE.
    weights     : optional {"ce_dice", "point_bce_dice": float}, default 1.0 each.
    dice_eps    : smoothing for both Dice terms.
    """

    def __init__(self, point_class: int = 3, pos_weight: float = 1.0,
                 weights: dict = None, dice_eps: float = 1e-6):
        super().__init__()
        self.point_class = point_class
        self.pos_weight = float(pos_weight)
        self.dice_eps = dice_eps
        self.weights = {"ce_dice": 1.0, "point_bce_dice": 1.0}
        if weights:
            self.weights.update(weights)

    def _point_bce_dice(self, point_logits, point_target, sample_mask):
        """BCE + soft Dice over the samples whose source annotates points."""
        keep = sample_mask > 0
        if not keep.any():
            # Zero, but still attached to the graph so backward() works.
            return point_logits.sum() * 0.0
        logit = point_logits[keep].float()
        gt = (point_target[keep] > 0).float()
        pos_weight = torch.tensor(self.pos_weight, device=logit.device)
        bce = F.binary_cross_entropy_with_logits(logit, gt, pos_weight=pos_weight)
        prob = torch.sigmoid(logit)
        intersection = (prob * gt).sum()
        dice = 1.0 - (2 * intersection + self.dice_eps) / (prob.sum() + gt.sum() + self.dice_eps)
        return bce + dice

    def forward(self, logits, target, class_mask, point_target=None) -> dict:
        c = self.point_class
        if c != logits.shape[1] - 1:
            raise ValueError("ConventionalSegLoss needs the point class to be the last channel")
        if point_target is None:
            raise ValueError("ConventionalSegLoss needs a point_target mask")
        seg_logits = _mask_logits(logits[:, :c], class_mask[:, :c])
        terms = {
            "ce_dice": ce_dice_loss(seg_logits, target, self.dice_eps),
            "point_bce_dice": self._point_bce_dice(logits[:, c], point_target, class_mask[:, c]),
        }
        total = sum(self.weights.get(name, 1.0) * value for name, value in terms.items())
        out = {"loss": total}
        out.update({name: float(value.detach().item()) for name, value in terms.items()})
        return out
