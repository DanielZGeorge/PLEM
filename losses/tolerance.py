"""
Soft Tolerance-Band Loss — differentiable analog of metrics/dtaf1.py.

DTAF1 (metrics/dtaf1.py::_class_dtf1) marks a predicted pixel as a true
positive if a ground-truth pixel of the same class exists within a
class-specific Euclidean tolerance d_c, and vice versa for recall, via
scipy.ndimage.distance_transform_edt. Both `dist_from_gt <= d` and
`dist_from_pred <= d` are hard, discrete tests — not differentiable.

The key observation that makes this tractable: `dist_from_gt <= d` is exactly
the definition of a binary dilation of the GT mask by radius d, which can be
precomputed once per batch (GT is a fixed target, not a function of the
model's output) and used as a static weight field against the model's own
SOFT probability map:

    precision_c = (pred_c · dilate(gt_c, d) + eps) / (sum(pred_c) + eps)

mirrors DTAF1's precision = tp_pred / n_pred (tp_pred = pred pixels within d
of GT) — this direction needs only a GT-side dilation, exactly as
`dist_from_gt` in dtaf1.py does, so it is fully differentiable with no
approximation.

The recall direction is the harder one: dtaf1.py's recall uses
`dist_from_pred`, an EDT of the *prediction* itself. Rather than falling back
to a detached/stop-gradient hard threshold, this module dilates the model's
own continuous probability map directly via iterated `F.max_pool2d` — max-pool
is differentiable, so this "soft dilation" keeps the recall term fully
differentiable too, with gradients flowing into the model on both terms:

    recall_c = (gt_c · soft_dilate(pred_c, d) + eps) / (sum(gt_c) + eps)

Both denominators use epsilon smoothing (matching notebooks/train_unet.ipynb's
existing dice_loss convention) rather than explicit both-empty branching: when
a class is entirely absent from both pred and GT, both terms reduce cleanly to
~1.0 (~0 loss), matching dtaf1's "both empty -> perfect" rule without a
separate code path.

Operates on standard PyTorch segmentation tensors: (B, C, H, W) logits,
(B, H, W) int64 label maps — NOT metrics/'s H×W numpy label-map convention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dilate_binary(x: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Binary dilation by `radius` px via iterated 3x3 max-pool. Each max-pool
    step grows the foreground region by 1px in every direction, so `radius`
    iterations approximate a disk-ish dilation of that radius — cheaper than
    a true Euclidean distance transform, and there is no differentiable EDT
    primitive in stock torch anyway.
    """
    radius = int(round(radius))
    for _ in range(max(radius, 0)):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x


def soft_dilate_prob(x: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Same iterated-max-pool dilation as `_dilate_binary`, applied to a
    continuous [0, 1] probability map instead of a binary mask. Deliberately
    NOT wrapped in `no_grad`: `F.max_pool2d` is differentiable, so applying it
    directly to the model's soft output lets gradients flow back through the
    dilation — this is what makes the recall direction below fully
    differentiable without a stop-gradient/detached hard-threshold
    approximation.
    """
    return _dilate_binary(x, radius)


def gt_tolerance_band(gt_onehot: torch.Tensor, class_ids, radii: dict) -> torch.Tensor:
    """
    Precompute a per-class binary dilation-by-radius mask of the GT (the
    analog of dtaf1.py's `dist_from_gt <= d`), under `no_grad` since GT is
    fixed target construction, not a function of the model's output.

    Parameters
    ----------
    gt_onehot : (B, C, H, W) float 0/1 one-hot GT
    class_ids : iterable of channel indices (== class ids) to compute a band for
    radii     : {class_id: tolerance_px}

    Returns
    -------
    (B, C, H, W) float 0/1 tensor — channels not in class_ids are left as-is.
    """
    with torch.no_grad():
        band = gt_onehot.clone()
        for c in class_ids:
            band[:, c:c + 1] = _dilate_binary(gt_onehot[:, c:c + 1], radii[c])
    return band


def soft_tolerance_pair_loss(
    pred_soft: torch.Tensor,
    gt_hard: torch.Tensor,
    radius: float,
    eps: float = 1e-6,
    precision_weight: float = 0.5,
) -> torch.Tensor:
    """
    Differentiable tolerance-band precision/recall loss between one predicted
    SOFT channel (B, H, W) and one GT HARD binary channel (B, H, W) — the
    shared primitive behind both `ToleranceBandLoss` (whole-class masks,
    below) and `SoftBoundaryLoss` (boundary-ring masks, losses/boundary.py).

    See module docstring for the precision/recall formulas. Returns a scalar
    (batch-mean loss).
    """
    gt_band = _dilate_binary(gt_hard.unsqueeze(1), radius)[:, 0]
    dilated_pred = soft_dilate_prob(pred_soft.unsqueeze(1), radius)[:, 0]

    p_sum = pred_soft.sum(dim=(1, 2))
    g_sum = gt_hard.sum(dim=(1, 2))
    prec_num = (pred_soft * gt_band).sum(dim=(1, 2))
    rec_num = (gt_hard * dilated_pred).sum(dim=(1, 2))

    precision = (prec_num + eps) / (p_sum + eps)
    recall = (rec_num + eps) / (g_sum + eps)

    loss = precision_weight * (1.0 - precision) + (1.0 - precision_weight) * (1.0 - recall)
    return loss.mean()


class ToleranceBandLoss(nn.Module):
    """
    Differentiable analog of metrics/dtaf1.py, applied per-class over
    `class_config` — the SAME `{class_id: {"name": ..., "tolerance": ...}}`
    shape as dtaf1.py's `ROAD_BUILDING_CONFIG`, so a caller can literally
    share one `class_config` object between eval (`dtaf1()`) and training
    (this class).
    """

    def __init__(self, class_config: dict, precision_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.class_config = class_config
        self.precision_weight = precision_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        gt_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        losses = [
            soft_tolerance_pair_loss(
                probs[:, c], gt_onehot[:, c], cfg["tolerance"], self.eps, self.precision_weight
            )
            for c, cfg in self.class_config.items()
        ]
        return torch.stack(losses).mean()
