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
    point_head      : "softmax" (default) — the point class is one more
                       softmax channel, competing with the segmentation classes
                       (the original formulation; `target` contains the point
                       class id). "sigmoid" — the point channel is an
                       INDEPENDENT head: the segmentation terms see only the
                       non-point channels, and the heatmap term sees only
                       `sigmoid(logits[:, point_class])` against a separate
                       `point_target` mask. Neither side's logits enter the
                       other's loss, so points cannot take pixels from
                       road/building (the run-3 flooding failure) and a point
                       may coincide with a segmentation class — required for
                       road intersections, which lie on road pixels. Needs a
                       single point class that is the LAST channel.
    point_sigma     : Gaussian sigma (px) of the heatmap target.
    point_neg_norm  : negative-term normalizer for the sigmoid head, "positives"
                       (CenterNet's, default) or "pixels" -- see
                       losses/heatmap.py::heatmap_focal_loss.
    weights         : optional {"ce_dice", "tolerance", "cldice", "heatmap",
                       "point_suppress": float} overriding each term's default
                       weight (1.0, except "point_suppress" which defaults to
                       0.0 = off; see `_point_suppress`).

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
        point_head: str = "softmax",
        point_sigma: float = 2.0,
        point_neg_norm: str = "positives",
    ):
        super().__init__()
        self.linear_classes = list(linear_classes)
        self.polygon_classes = list(polygon_classes)
        self.point_classes = list(point_classes)
        self.dice_eps = dice_eps
        if point_head not in ("softmax", "sigmoid"):
            raise ValueError(f"point_head must be 'softmax' or 'sigmoid', got {point_head!r}")
        if point_head == "sigmoid" and len(self.point_classes) != 1:
            raise ValueError("point_head='sigmoid' needs exactly one point class")
        self.point_head = point_head
        self.point_neg_norm = point_neg_norm

        tolerance_config = {
            c: class_config[c]
            for c in self.linear_classes + self.polygon_classes
            if c in class_config
        }
        self.tolerance_loss = ToleranceBandLoss(tolerance_config) if tolerance_config else None
        self.cldice_loss = SoftClDiceLoss(self.linear_classes) if self.linear_classes else None
        self.heatmap_loss = (
            PointHeatmapLoss(self.point_classes, sigma=point_sigma) if self.point_classes else None
        )

        self.weights = {"ce_dice": 1.0, "tolerance": 1.0, "cldice": 1.0, "heatmap": 1.0,
                        "point_suppress": 0.0}
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

    def _point_suppress(self, logits: torch.Tensor, class_mask: torch.Tensor) -> torch.Tensor:
        """
        Weak "no points here" prior on samples whose source does NOT annotate a
        point class: mean BCE-toward-0, `-log(1 - p_point)`, of the point
        class's softmax probability, computed on the RAW (unmasked) logits --
        i.e. the same 4-way softmax prediction sees at inference.

        Why it exists: `_mask_logits` gives a masked channel no gradient at
        all, which is right for road on Potsdam (roads there are merely
        unlabelled) but left the point channel completely unconstrained on
        SpaceNet/SpaceNet6 imagery (~90% of training patches). On the third
        real train_unet_joint_scaled.ipynb run it became confident "point"
        over large parts of those sources and flooded predictions. This trades
        a known label-noise cost (SpaceNet does contain unlabelled trees/cars)
        for suppressing that; point_f1 is never scored on those sources, and
        Potsdam's positive supervision still defines what a point looks like.

        The gradient on the non-point logits is `p_k - p_k / (1 - p_point)`,
        proportional to each one's own share, so it lowers the point
        probability without reordering road/building/background. Off by
        default (weight 0.0) so existing callers and masking guarantees are
        unchanged.
        """
        log_probs = torch.log_softmax(logits.float(), dim=1)
        losses = []
        for c in self.point_classes:
            unannotated = (class_mask[:, c] == 0)
            if not unannotated.any():
                continue
            # log(1 - p_c) = logsumexp over the other channels' log-probs, stable.
            others = torch.cat([log_probs[:, :c], log_probs[:, c + 1:]], dim=1)
            log_not_c = torch.logsumexp(others, dim=1)  # (B, H, W)
            losses.append(-log_not_c[unannotated].mean())
        if not losses:
            return torch.zeros((), device=logits.device)
        return torch.stack(losses).mean()

    def _forward_sigmoid_head(self, logits, target, class_mask, point_target) -> dict:
        """`point_head="sigmoid"` path — see the class docstring."""
        c = self.point_classes[0]
        if c != logits.shape[1] - 1:
            raise ValueError("point_head='sigmoid' needs the point class to be the last channel")
        if point_target is None:
            raise ValueError("point_head='sigmoid' needs a point_target mask")
        seg_logits = _mask_logits(logits[:, :c], class_mask[:, :c])

        terms = {"ce_dice": self._ce_dice(seg_logits, target)}
        if self.tolerance_loss is not None:
            terms["tolerance"] = self.tolerance_loss(seg_logits, target)
        if self.cldice_loss is not None:
            terms["cldice"] = self.cldice_loss(seg_logits, target)
        # No logit masking here: non-annotating samples are excluded outright
        # by sample_mask, so their point logit gets exactly zero gradient.
        terms["heatmap"] = self.heatmap_loss.forward_prob(
            torch.sigmoid(logits[:, c].float()), point_target, sample_mask=class_mask[:, c],
            neg_norm=self.point_neg_norm,
        )
        return terms

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, class_mask: torch.Tensor,
        point_target: torch.Tensor = None,
    ) -> dict:
        if self.point_head == "sigmoid":
            terms = self._forward_sigmoid_head(logits, target, class_mask, point_target)
            total = sum(self.weights.get(name, 1.0) * value for name, value in terms.items())
            out = {"loss": total}
            out.update({name: float(value.detach().item()) for name, value in terms.items()})
            return out

        masked_logits = _mask_logits(logits, class_mask)

        terms = {"ce_dice": self._ce_dice(masked_logits, target)}
        if self.tolerance_loss is not None:
            terms["tolerance"] = self.tolerance_loss(masked_logits, target)
        if self.cldice_loss is not None:
            terms["cldice"] = self.cldice_loss(masked_logits, target)
        if self.heatmap_loss is not None:
            # class_mask also restricts the heatmap's normalizers to point-annotating
            # samples; logit masking alone zeroes masked samples' contribution but
            # would still count their pixels in the negative term's denominator.
            terms["heatmap"] = self.heatmap_loss(masked_logits, target, class_mask)
            if self.weights.get("point_suppress", 0.0) > 0:
                # Deliberately on the UNMASKED logits -- see _point_suppress.
                terms["point_suppress"] = self._point_suppress(logits, class_mask)

        total = sum(self.weights.get(name, 1.0) * value for name, value in terms.items())

        out = {"loss": total}
        out.update({name: float(value.detach().item()) for name, value in terms.items()})
        return out
