"""
DTAF1-Topo: connectivity-aware companion to DTAF1.

Blends DTAF1's per-class tolerance-radius F1 with the raster-skeleton APLS
approximation (metrics/apls.py) for linear classes: harmonic_mean(f1, apls)
replaces plain F1 for classes in `linear_classes`, so a class only scores
well if it is both positionally accurate AND topologically connected.
Non-linear classes (building, point) keep their plain DTAF1 F1 unchanged --
APLS/connectivity has no defined meaning for polygon/point classes.

Addresses DTAF1's documented under-penalisation of road breakage (see
CLAUDE.md's "Known limitation of DTAF1"): scattered surviving pixels after
heavy random dropout still fall within DTAF1's tolerance radius of some GT
pixel, so plain per-pixel F1 stays high even when the road network is
disconnected; APLS collapses because shortest-path length between sampled
control points blows up or the path disappears once the skeleton fragments.

Non-breaking / additive by construction: this module only calls dtaf1() and
apls_multiclass() -- it does not modify metrics/dtaf1.py, metrics/apls.py,
metrics/unified.py, or metrics/__init__.py.
"""

import numpy as np

from metrics.dtaf1 import dtaf1
from metrics.apls import apls_multiclass, DEFAULT_N_PAIRS, DEFAULT_SEED


def _harmonic_mean(a: float, b: float) -> float:
    denom = a + b
    return float(2 * a * b / denom) if denom > 0 else 0.0


def dtaf1_topo(
    pred: np.ndarray,
    gt: np.ndarray,
    class_config: dict,
    linear_classes: list,
    reduction: str = "macro",
    apls_n_pairs: int = DEFAULT_N_PAIRS,
    apls_seed: int = DEFAULT_SEED,
) -> dict:
    """
    DTAF1 with a connectivity-aware blend for linear classes.

    Parameters
    ----------
    pred, gt        : H×W integer label maps (0 = background)
    class_config    : same {class_id: {"name", "tolerance"}} dtaf1() takes
    linear_classes  : subset of class_config's keys blended with APLS via
                       harmonic_mean(f1, apls); all other classes keep plain F1.
    reduction       : "macro" | "weighted" -- same semantics as dtaf1()
    apls_n_pairs, apls_seed : forwarded to apls_multiclass()

    Returns
    -------
    dict with keys:
        "dtaf1_topo"          : scalar blended score, per `reduction`
        "dtaf1_topo_macro"    : unweighted mean of per-class blended scores
        "dtaf1_topo_weighted" : GT-pixel-count-weighted mean of blended scores
        "per_class"           : {class_id: {...dtaf1 per-class fields...,
                                              "apls": float|None, "blended": float}}
        "dtaf1_result"        : the complete, unmodified dtaf1() return value
                                 (for direct side-by-side comparison)
        "apls_detail"         : {class_id: apls() result dict}, linear classes only
    """
    dtaf1_result = dtaf1(pred, gt, class_config, reduction=reduction)

    tolerances = {cls: class_config[cls]["tolerance"] for cls in linear_classes}
    apls_detail = apls_multiclass(
        pred, gt, linear_classes, tolerances, n_pairs=apls_n_pairs, seed=apls_seed
    )

    per_class, blended_scores, weights = {}, [], []
    for cls_id in class_config:
        base = dtaf1_result["per_class"][cls_id]
        entry = dict(base)
        if cls_id in linear_classes:
            a = apls_detail[cls_id]["apls"]
            entry["apls"] = a
            entry["blended"] = _harmonic_mean(base["f1"], a)
        else:
            entry["apls"] = None
            entry["blended"] = base["f1"]
        per_class[cls_id] = entry
        blended_scores.append(entry["blended"])
        weights.append(base["n_gt"])

    macro_score = float(np.mean(blended_scores))
    weights_arr = np.array(weights, dtype=float)
    total = weights_arr.sum()
    weighted_score = float(np.dot(weights_arr, blended_scores) / total) if total > 0 else 0.0

    if reduction == "macro":
        score = macro_score
    elif reduction == "weighted":
        score = weighted_score
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}. Use 'macro' or 'weighted'.")

    return {
        "dtaf1_topo": score,
        "dtaf1_topo_macro": macro_score,
        "dtaf1_topo_weighted": weighted_score,
        "per_class": per_class,
        "dtaf1_result": dtaf1_result,
        "apls_detail": apls_detail,
    }
