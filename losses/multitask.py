"""
Combined masked multi-task training loss — the trainable counterpart to
metrics/unified.py::evaluate_all(). Sums a base CE+Dice term (adapted from
notebooks/train_unet.ipynb's `combined_loss`) with three geometry-aware
terms, each a differentiable analog of one PLEM eval metric:

    tolerance.ToleranceBandLoss  <-> metrics/dtaf1.py    (linear + polygon classes)
    cldice.SoftClDiceLoss        <-> metrics/cldice.py   (linear classes)
    heatmap.PointHeatmapLoss     <-> metrics/point_f1.py (point classes, reformulated)

APLS / DTAF1-Topo (metrics/apls.py, metrics/dtaf1_topo.py) have no
differentiable analog here by design — graph shortest-path recomputation has
no tractable lightweight relaxation — and stay eval-only permanently.

Per-source class masking (SpaceNet tiles annotate road+building only;
Potsdam tiles annotate building+point only — see datasets/joint.py's
SOURCE_CLASSES) is implemented by additively biasing a sample's masked-out
class channels to a large negative logit BEFORE any softmax is taken
(`_mask_logits`), rather than post-hoc zeroing four separately-computed
per-class losses. Every term here is a function of softmax probabilities, so
driving a masked channel's pre-softmax logit to -inf makes its post-softmax
probability — and therefore its gradient contribution to every term at once
— vanish identically: one mechanism shared by all four loss terms instead of
four bespoke masking implementations. This relies on target label maps never
containing a class id a sample's source doesn't annotate, which holds by
construction for both datasets/spacenet.py and datasets/potsdam.py's
rasterizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.tolerance import ToleranceBandLoss
from losses.cldice import SoftClDiceLoss
from losses.heatmap import PointHeatmapLoss


def _mask_logits(logits: torch.Tensor, class_mask: torch.Tensor, neg_value: float = -1e4) -> torch.Tensor:
    """
    (B, C, H, W), (B, C) 0/1 -> (B, C, H, W): adds `(1 - class_mask) *
    neg_value` to each channel, broadcast over H, W. See module docstring.
    """
    bias = (1.0 - class_mask.float()).unsqueeze(-1).unsqueeze(-1) * neg_value
    return logits + bias


class PLEMMultiTaskLoss(nn.Module):
    """
    Parameters
    ----------
    class_config    : {class_id: {"name": ..., "tolerance": ...}} — same shape
                       as metrics/dtaf1.py's class_config; entries for
                       linear_classes + polygon_classes are used by the
                       tolerance-band term.
    linear_classes  : e.g. [1] (road) — supervised by ToleranceBandLoss + SoftClDiceLoss.
    polygon_classes : e.g. [2] (building) — supervised by ToleranceBandLoss only.
    point_classes   : e.g. [3] (point) — supervised by PointHeatmapLoss only.
    weights         : optional {"ce_dice", "tolerance", "cldice", "heatmap": float}
                       overriding the default 1.0 weight on each term.

    forward(logits, target, class_mask) returns a dict with "loss" (the
    total, for `.backward()`) plus each active sub-term as a detached float
    for logging — mirrors metrics/'s "always return a dict of named scalars"
    convention on the training side too.
    """

    def __init__(
        self,
        class_config: dict,
        linear_classes: list,
        polygon_classes: list,
        point_classes: list,
        weights: dict = None,
        dice_eps: float = 1e-6,
    ):
        super().__init__()
        self.linear_classes = list(linear_classes)
        self.polygon_classes = list(polygon_classes)
        self.point_classes = list(point_classes)
        self.dice_eps = dice_eps

        tolerance_config = {
            c: class_config[c]
            for c in self.linear_classes + self.polygon_classes
            if c in class_config
        }
        self.tolerance_loss = ToleranceBandLoss(tolerance_config) if tolerance_config else None
        self.cldice_loss = SoftClDiceLoss(self.linear_classes) if self.linear_classes else None
        self.heatmap_loss = PointHeatmapLoss(self.point_classes) if self.point_classes else None

        self.weights = {"ce_dice": 1.0, "tolerance": 1.0, "cldice": 1.0, "heatmap": 1.0}
        if weights:
            self.weights.update(weights)

    def _ce_dice(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Base term, adapted from train_unet.ipynb's `combined_loss`
        (CrossEntropyLoss + soft multiclass Dice, unweighted sum). Operates
        on already class-masked logits (see `forward`), so no separate
        masking logic is needed here.
        """
        num_classes = logits.shape[1]
        ce = F.cross_entropy(logits, target)

        probs = torch.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        union = probs.sum(dims) + target_onehot.sum(dims)
        dice_per_class = (2 * intersection + self.dice_eps) / (union + self.dice_eps)
        dice = 1.0 - dice_per_class.mean()

        return ce + dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor, class_mask: torch.Tensor) -> dict:
        masked_logits = _mask_logits(logits, class_mask)

        terms = {"ce_dice": self._ce_dice(masked_logits, target)}
        if self.tolerance_loss is not None:
            terms["tolerance"] = self.tolerance_loss(masked_logits, target)
        if self.cldice_loss is not None:
            terms["cldice"] = self.cldice_loss(masked_logits, target)
        if self.heatmap_loss is not None:
            terms["heatmap"] = self.heatmap_loss(masked_logits, target)

        total = sum(self.weights.get(name, 1.0) * value for name, value in terms.items())

        out = {"loss": total}
        out.update({name: float(value.detach().item()) for name, value in terms.items()})
        return out
