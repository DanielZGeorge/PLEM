"""
Point F1 — tolerance-radius instance matching for point-like features.

For small, discrete objects (trees, lamp posts, manhole covers, etc.) that
only span a handful of pixels and are best evaluated by "is there an object
here", not by shape (BF score) or centerline (clDice). Ground truth and
predictions are still H×W integer label maps — a class id marks the pixels
of each blob — so a connected-component blob is first reduced to a
centroid, then centroids are matched between pred and GT via a tolerance-
restricted optimal (Hungarian) one-to-one assignment.

Hungarian assignment (scipy.optimize.linear_sum_assignment), not greedy
nearest-neighbor, is used because point objects legitimately cluster (e.g.
several manholes a few pixels apart): greedy matching can lock a predicted
blob onto the wrong nearby GT point and silently undercount true positives.
Hungarian gives a deterministic, globally optimal one-to-one assignment,
and is computationally trivial at the scale relevant here (tens to low
hundreds of instances per scene).

Tolerance should be expressed in pixels and tied to GSD as with the other
PLEM metrics: d = physical_metres / GSD_metres_per_pixel.
"""

import numpy as np
from scipy.ndimage import label, center_of_mass, sum as ndi_sum
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


DEFAULT_TOLERANCE = 5  # pixels
DEFAULT_MAX_INSTANCES = 500  # guardrail against speckle blow-up, not a real-data limit

_MAX_INSTANCES_WARNED = False


def _instance_centroids(mask: np.ndarray, min_area: int = 1,
                        max_instances: int = DEFAULT_MAX_INSTANCES) -> np.ndarray:
    """
    Connected-component blobs in a binary mask -> (k, 2) array of (row, col)
    centroids.

    `max_instances` bounds the cost of the downstream `cdist` +
    `linear_sum_assignment`: a noisy/dark model prediction can scatter 10^4+
    single-pixel components in the point channel, which would otherwise make a
    dark-test sweep (severities x tiles) crawl. Real point-feature tiles carry
    at most a few dozen instances, so hitting this cap means the input is
    speckle -- the largest-area blobs are kept and a one-time warning printed.
    """
    labeled, n = label(mask > 0)
    if n == 0:
        return np.zeros((0, 2))
    ids = np.arange(1, n + 1)
    if min_area > 1 or n > max_instances:
        sizes = ndi_sum(np.ones_like(labeled), labeled, index=ids)
        if min_area > 1:
            keep = sizes >= min_area
            ids, sizes = ids[keep], sizes[keep]
        if len(ids) > max_instances:
            global _MAX_INSTANCES_WARNED
            if not _MAX_INSTANCES_WARNED:
                print(
                    f"point_f1._instance_centroids: {len(ids)} components exceeds "
                    f"max_instances={max_instances} -- keeping the {max_instances} "
                    f"largest (input looks like speckle, not real point features)."
                )
                _MAX_INSTANCES_WARNED = True
            ids = ids[np.argsort(sizes)[::-1][:max_instances]]
        if len(ids) == 0:
            return np.zeros((0, 2))
    return np.asarray(center_of_mass(mask > 0, labeled, ids)).reshape(-1, 2)


def _match_centroids(gt_c: np.ndarray, pred_c: np.ndarray, tolerance: float) -> int:
    """
    Tolerance-restricted Hungarian assignment between GT and predicted centroids.

    Returns the number of matched pairs (true positives): pairs that are
    both within `tolerance` of each other and part of the globally optimal
    (minimum total distance) one-to-one assignment.
    """
    if len(gt_c) == 0 or len(pred_c) == 0:
        return 0

    dist = cdist(gt_c, pred_c)
    # Discourage (but don't forbid) out-of-tolerance pairs so linear_sum_assignment
    # still returns a full assignment; filter by tolerance afterwards.
    big = tolerance * 1e6 + 1.0
    cost = np.where(dist <= tolerance, dist, big)
    row_ind, col_ind = linear_sum_assignment(cost)
    valid = dist[row_ind, col_ind] <= tolerance
    return int(valid.sum())


def point_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
    min_area: int = 1,
    max_instances: int = DEFAULT_MAX_INSTANCES,
) -> dict:
    """
    Compute tolerance-radius instance F1 between a predicted and GT point mask.

    A predicted blob is matched to a GT blob if their centroids are within
    `tolerance` px of each other and the pairing is part of the globally
    optimal one-to-one assignment (Hungarian). Unmatched predicted blobs are
    false positives; unmatched GT blobs are false negatives.

    Parameters
    ----------
    pred      : H×W binary or integer array (>0 = foreground blob pixels)
    gt        : H×W binary or integer array (>0 = foreground blob pixels)
    tolerance : max centroid distance (px) for a predicted blob to count as matched
    min_area  : minimum connected-component pixel area to count as an instance
                (drops speckle noise below this size)

    Returns
    -------
    dict with keys:
        "point_f1"  : scalar F1 score in [0, 1]
        "precision" : fraction of predicted instances matched to a GT instance
        "recall"    : fraction of GT instances matched to a predicted instance
        "tp", "fp", "fn" : matched / unmatched predicted / unmatched GT instance counts
        "n_pred", "n_gt" : total instance counts
        "tolerance" : tolerance used (px)
    """
    p = pred > 0
    g = gt > 0

    gt_c = _instance_centroids(g, min_area, max_instances)
    pred_c = _instance_centroids(p, min_area, max_instances)

    n_gt = len(gt_c)
    n_pred = len(pred_c)

    if n_gt == 0 and n_pred == 0:
        return {
            "point_f1": 1.0, "precision": 1.0, "recall": 1.0,
            "tp": 0, "fp": 0, "fn": 0,
            "n_pred": 0, "n_gt": 0, "tolerance": tolerance,
        }
    if n_gt == 0 or n_pred == 0:
        return {
            "point_f1": 0.0, "precision": 0.0, "recall": 0.0,
            "tp": 0, "fp": n_pred, "fn": n_gt,
            "n_pred": n_pred, "n_gt": n_gt, "tolerance": tolerance,
        }

    tp = _match_centroids(gt_c, pred_c, tolerance)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / n_pred
    recall = tp / n_gt
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "point_f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "tolerance": tolerance,
    }


def point_f1_multiclass(
    pred: np.ndarray,
    gt: np.ndarray,
    point_classes: list,
    tolerance: float = DEFAULT_TOLERANCE,
    min_area: int = 1,
    max_instances: int = DEFAULT_MAX_INSTANCES,
) -> dict:
    """
    Apply point_f1 to selected classes in a multiclass label map.

    Returns dict {class_id: point_f1_result_dict}.
    """
    results = {}
    for cls in point_classes:
        results[cls] = point_f1(pred == cls, gt == cls, tolerance, min_area, max_instances)
    return results


def mean_point_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    point_classes: list,
    tolerance: float = DEFAULT_TOLERANCE,
    min_area: int = 1,
    max_instances: int = DEFAULT_MAX_INSTANCES,
) -> float:
    """Macro-average point_f1 over the specified point classes."""
    per_class = point_f1_multiclass(pred, gt, point_classes, tolerance, min_area, max_instances)
    scores = [v["point_f1"] for v in per_class.values()]
    return float(np.mean(scores)) if scores else 0.0
