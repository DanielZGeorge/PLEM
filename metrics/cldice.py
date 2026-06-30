"""
Centerline Dice (clDice) for linear / tubular feature evaluation.

Reference: Shit et al. "clDice — a Novel Topology-Preserving Loss Function
for Tubular Structure Segmentation", CVPR 2021.

clDice evaluates how well the predicted centerline (skeleton) is covered by
the ground-truth region, and vice versa. This makes it:
  - Tolerant to road width errors (a too-thick prediction does not help)
  - Sensitive to broken centerlines (a gap in the predicted road is penalised)
  - Far more appropriate for roads than pixel-wise IoU

Skeletonization via skimage.morphology.skeletonize (Lee's thinning algorithm).
"""

import numpy as np
from skimage.morphology import skeletonize


def _soft_skeleton(mask: np.ndarray) -> np.ndarray:
    """Return binary skeleton of a binary mask."""
    return skeletonize(mask > 0)


def cldice(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute clDice between a predicted binary mask and a ground-truth mask.

    Parameters
    ----------
    pred : H×W binary or integer array (>0 treated as foreground)
    gt   : H×W binary or integer array (>0 treated as foreground)

    Returns
    -------
    dict with keys:
        "cldice"    : scalar score in [0, 1]
        "tprec"     : topology precision  (skeleton of pred covered by GT region)
        "tsens"     : topology sensitivity (skeleton of GT covered by pred region)
    """
    p = (pred > 0)
    g = (gt > 0)

    skel_p = _soft_skeleton(p)
    skel_g = _soft_skeleton(g)

    n_skel_p = skel_p.sum()
    n_skel_g = skel_g.sum()

    # Both empty → perfect score
    if n_skel_p == 0 and n_skel_g == 0:
        return {"cldice": 1.0, "tprec": 1.0, "tsens": 1.0}

    # One empty, one not → zero score
    if n_skel_p == 0 or n_skel_g == 0:
        return {"cldice": 0.0, "tprec": 0.0, "tsens": 0.0}

    # Topology precision: what fraction of the predicted skeleton lies inside GT?
    tprec = float((skel_p & g).sum() / n_skel_p)
    # Topology sensitivity: what fraction of the GT skeleton lies inside prediction?
    tsens = float((skel_g & p).sum() / n_skel_g)

    denom = tprec + tsens
    cl = float(2 * tprec * tsens / denom) if denom > 0 else 0.0

    return {"cldice": cl, "tprec": tprec, "tsens": tsens}


def cldice_multiclass(pred: np.ndarray, gt: np.ndarray, linear_classes: list) -> dict:
    """
    Apply clDice to selected classes in a multiclass label map.

    Parameters
    ----------
    pred           : H×W integer label map
    gt             : H×W integer label map
    linear_classes : list of integer class IDs to evaluate with clDice

    Returns
    -------
    dict  {class_id: cldice_result_dict}
    """
    results = {}
    for cls in linear_classes:
        results[cls] = cldice(pred == cls, gt == cls)
    return results


def mean_cldice(pred: np.ndarray, gt: np.ndarray, linear_classes: list) -> float:
    """Macro-average clDice over the specified linear classes."""
    per_class = cldice_multiclass(pred, gt, linear_classes)
    scores = [v["cldice"] for v in per_class.values()]
    return float(np.mean(scores)) if scores else 0.0
