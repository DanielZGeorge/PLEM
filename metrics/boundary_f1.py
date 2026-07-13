"""
Boundary F1 Score (BF score) for polygonal feature evaluation.

Evaluates shape accuracy by comparing the boundary pixels of the prediction
against the boundary pixels of the ground truth, within a positional tolerance τ.

This is more sensitive to shape distortion than area-based IoU:
  - A building predicted as a perfect square when the GT is a rectangle scores
    lower under BF than a prediction that captures the actual boundary well.
  - A slight area under/over-estimation is only penalised proportionally.

Reference: Csurka et al. "What is a good evaluation measure for semantic
segmentation?", BMVC 2013.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.morphology import disk


def _extract_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """
    Extract the boundary pixels of a binary mask.

    A pixel is on the boundary if it is foreground and any pixel in its
    `thickness`-pixel neighbourhood is background.
    """
    m = mask > 0
    if not m.any():
        return np.zeros_like(m)
    dilated = binary_dilation(m, structure=disk(thickness))
    return dilated & ~m | (m & ~binary_dilation(~m, structure=disk(thickness)))


def boundary_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance: float = 2.0,
    boundary_thickness: int = 1,
) -> dict:
    """
    Compute Boundary F1 between a predicted binary mask and a GT mask.

    Parameters
    ----------
    pred               : H×W binary or integer array (>0 = foreground)
    gt                 : H×W binary or integer array (>0 = foreground)
    tolerance          : max distance (px) for a boundary pixel to count as matched
    boundary_thickness : morphological thickness used to extract boundaries

    Returns
    -------
    dict with keys:
        "bf"        : Boundary F1 score in [0, 1]
        "precision" : fraction of pred boundary pixels within tolerance of GT boundary
        "recall"    : fraction of GT boundary pixels within tolerance of pred boundary
    """
    p = (pred > 0)
    g = (gt > 0)

    bp = _extract_boundary(p, boundary_thickness)
    bg = _extract_boundary(g, boundary_thickness)

    n_bp = bp.sum()
    n_bg = bg.sum()

    # Both empty → perfect
    if n_bp == 0 and n_bg == 0:
        return {"bf": 1.0, "precision": 1.0, "recall": 1.0}
    # One side has no boundary → zero
    if n_bp == 0 or n_bg == 0:
        return {"bf": 0.0, "precision": 0.0, "recall": 0.0}

    # Distance from every pixel to the nearest GT boundary pixel
    dist_from_gt_boundary = distance_transform_edt(~bg)
    # Distance from every pixel to the nearest pred boundary pixel
    dist_from_pred_boundary = distance_transform_edt(~bp)

    precision = float((bp & (dist_from_gt_boundary <= tolerance)).sum() / n_bp)
    recall = float((bg & (dist_from_pred_boundary <= tolerance)).sum() / n_bg)

    denom = precision + recall
    bf = float(2 * precision * recall / denom) if denom > 0 else 0.0

    return {"bf": bf, "precision": precision, "recall": recall}


def boundary_f1_multiclass(
    pred: np.ndarray,
    gt: np.ndarray,
    polygon_classes: list,
    tolerance: float = 2.0,
    boundary_thickness: int = 1,
) -> dict:
    """
    Apply BF score to selected classes in a multiclass label map.

    Returns dict {class_id: boundary_f1_result_dict}, each result dict
    additionally carries "n_gt" (GT pixel count for that class) for
    pixel-area-weighted aggregation.
    """
    results = {}
    for cls in polygon_classes:
        gt_mask = (gt == cls)
        result = boundary_f1(pred == cls, gt_mask, tolerance, boundary_thickness)
        result["n_gt"] = int(gt_mask.sum())
        results[cls] = result
    return results


def mean_boundary_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    polygon_classes: list,
    tolerance: float = 2.0,
    reduction: str = "macro",
) -> float:
    """
    Average BF score over the specified polygonal classes.

    reduction : "macro"    — unweighted mean of per-class BF scores
                "weighted" — weight by number of GT pixels per class
    """
    per_class = boundary_f1_multiclass(pred, gt, polygon_classes, tolerance)
    scores = [v["bf"] for v in per_class.values()]
    if not scores:
        return 0.0
    if reduction == "macro":
        return float(np.mean(scores))
    elif reduction == "weighted":
        weights = np.array([v["n_gt"] for v in per_class.values()], dtype=float)
        total = weights.sum()
        return float(np.dot(weights, scores) / total) if total > 0 else 0.0
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}. Use 'macro' or 'weighted'.")


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Standard pixel-wise IoU for a single binary class (reference baseline)."""
    p = pred > 0
    g = gt > 0
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter / union) if union > 0 else 1.0
