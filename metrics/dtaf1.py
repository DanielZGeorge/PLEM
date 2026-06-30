"""
Distance-Tolerance Adaptive F1 (DTAF1)

Unified metric for multiclass extraction containing both linear features
(e.g. roads) and polygonal features (e.g. buildings).

Key insight: a predicted pixel is a true positive if a ground-truth pixel
of the same class exists within a class-specific Euclidean tolerance d_c.
Linear classes get a larger tolerance (positional wiggle is acceptable);
polygonal classes get a tighter tolerance (shape accuracy matters).

All tolerance values should be expressed in pixels and ideally tied to the
ground sample distance (GSD) of the imagery.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


# Default tolerances (pixels). Override via the `tolerances` argument.
# Tie these to GSD in practice: d = physical_tolerance_metres / GSD_metres
DEFAULT_TOLERANCES = {
    "road":     10,   # roads: ~10 m at 1 m/px GSD
    "building":  2,   # buildings: ~2 m at 1 m/px GSD
}


def _class_dtf1(pred_mask: np.ndarray, gt_mask: np.ndarray, d: float) -> dict:
    """
    Compute tolerance-based precision, recall, and F1 for one class.

    A predicted positive pixel is TP if a GT positive pixel exists within d px.
    A GT positive pixel is covered if a predicted positive pixel exists within d px.

    Returns dict with keys: precision, recall, f1, tp_pred, tp_gt, n_pred, n_gt.
    """
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)

    n_pred = int(p.sum())
    n_gt = int(g.sum())

    if n_pred == 0 and n_gt == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "tp_pred": 0, "tp_gt": 0, "n_pred": 0, "n_gt": 0}
    if n_pred == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp_pred": 0, "tp_gt": 0, "n_pred": 0, "n_gt": n_gt}
    if n_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp_pred": 0, "tp_gt": 0, "n_pred": n_pred, "n_gt": 0}

    # Distance from every pixel to the nearest GT positive pixel
    dist_from_gt = distance_transform_edt(1 - g)
    # Distance from every pixel to the nearest predicted positive pixel
    dist_from_pred = distance_transform_edt(1 - p)

    tp_pred = int(((p == 1) & (dist_from_gt <= d)).sum())
    tp_gt = int(((g == 1) & (dist_from_pred <= d)).sum())

    precision = tp_pred / n_pred
    recall = tp_gt / n_gt
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_pred": tp_pred,
        "tp_gt": tp_gt,
        "n_pred": n_pred,
        "n_gt": n_gt,
    }


def dtaf1(
    pred: np.ndarray,
    gt: np.ndarray,
    class_config: dict,
    reduction: str = "macro",
) -> dict:
    """
    Compute DTAF1 over all classes.

    Parameters
    ----------
    pred : H×W integer array  (0 = background)
    gt   : H×W integer array  (0 = background)
    class_config : mapping of {class_id (int): {"name": str, "tolerance": float}}
        Example:
            {1: {"name": "road",     "tolerance": 10},
             2: {"name": "building", "tolerance":  2}}
    reduction : "macro"  — unweighted mean of per-class F1 scores
                "weighted" — weight by number of GT pixels per class

    Returns
    -------
    dict with keys:
        "dtaf1"        : scalar unified score
        "per_class"    : {class_id: per-class result dict}
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shapes differ: {pred.shape} vs {gt.shape}")
    if pred.ndim != 2:
        raise ValueError("pred and gt must be 2-D (H×W) label maps")

    per_class = {}
    for cls_id, cfg in class_config.items():
        d = cfg.get("tolerance", DEFAULT_TOLERANCES.get(cfg.get("name", ""), 5))
        pred_mask = (pred == cls_id)
        gt_mask = (gt == cls_id)
        result = _class_dtf1(pred_mask, gt_mask, d)
        result["name"] = cfg.get("name", str(cls_id))
        result["tolerance"] = d
        per_class[cls_id] = result

    f1_scores = [r["f1"] for r in per_class.values()]
    if reduction == "macro":
        score = float(np.mean(f1_scores))
    elif reduction == "weighted":
        weights = np.array([r["n_gt"] for r in per_class.values()], dtype=float)
        total = weights.sum()
        score = float(np.dot(weights, f1_scores) / total) if total > 0 else 0.0
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}. Use 'macro' or 'weighted'.")

    return {"dtaf1": score, "per_class": per_class}


# ---------------------------------------------------------------------------
# Convenience wrapper for the canonical road + building two-class case
# ---------------------------------------------------------------------------

ROAD_BUILDING_CONFIG = {
    1: {"name": "road",     "tolerance": 10},
    2: {"name": "building", "tolerance":  2},
}


def dtaf1_road_building(
    pred: np.ndarray,
    gt: np.ndarray,
    road_tolerance: float = 10,
    building_tolerance: float = 2,
    reduction: str = "macro",
) -> dict:
    """Convenience wrapper: class 1 = road, class 2 = building."""
    cfg = {
        1: {"name": "road",     "tolerance": road_tolerance},
        2: {"name": "building", "tolerance": building_tolerance},
    }
    return dtaf1(pred, gt, cfg, reduction=reduction)
