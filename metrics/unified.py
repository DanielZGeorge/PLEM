"""
Unified composite metrics for joint linear + polygonal extraction evaluation.

Two composite metrics are implemented:

  CBHM  — clDice + Boundary F1 Harmonic Mean
           Best-fit metric per feature type; harmonic mean prevents either
           class from masking poor performance in the other.

  DTAF1 — Distance-Tolerance Adaptive F1  (re-exported for convenience)
           Single formula across all classes; per-class tolerance radius
           encodes the geometric precision expected for each feature type.

Both return a scalar in [0, 1], where 1 is perfect extraction and 0 is
complete failure.
"""

import numpy as np

from metrics.cldice import mean_cldice, cldice_multiclass
from metrics.boundary_f1 import mean_boundary_f1, boundary_f1_multiclass
from metrics.point_f1 import point_f1_multiclass
from metrics.dtaf1 import dtaf1, dtaf1_road_building  # re-export


def cbhm(
    pred: np.ndarray,
    gt: np.ndarray,
    linear_classes: list,
    polygon_classes: list,
    road_tolerance: float = 10.0,
    building_tolerance: float = 2.0,
) -> dict:
    """
    Compute CBHM: harmonic mean of mean clDice (linear) and mean BF (polygonal).

    Parameters
    ----------
    pred             : H×W integer label map (0 = background)
    gt               : H×W integer label map (0 = background)
    linear_classes   : list of class IDs to evaluate with clDice  (e.g. [1])
    polygon_classes  : list of class IDs to evaluate with BF score (e.g. [2])
    road_tolerance   : BF tolerance applied to linear classes (px) — kept for
                       potential future extension; clDice itself is tolerance-free
    building_tolerance : BF positional tolerance (px)

    Returns
    -------
    dict with keys:
        "cbhm"           : scalar unified score
        "cldice_mean"    : mean clDice across linear classes
        "bf_mean"        : mean Boundary F1 across polygon classes
        "cldice_detail"  : per-class clDice dicts
        "bf_detail"      : per-class BF dicts
    """
    cldice_detail = cldice_multiclass(pred, gt, linear_classes)
    bf_detail = boundary_f1_multiclass(
        pred, gt, polygon_classes, tolerance=building_tolerance
    )

    cldice_scores = [v["cldice"] for v in cldice_detail.values()]
    bf_scores = [v["bf"] for v in bf_detail.values()]

    cl_mean = float(np.mean(cldice_scores)) if cldice_scores else 0.0
    bf_mean = float(np.mean(bf_scores)) if bf_scores else 0.0

    denom = cl_mean + bf_mean
    score = float(2 * cl_mean * bf_mean / denom) if denom > 0 else 0.0

    return {
        "cbhm": score,
        "cldice_mean": cl_mean,
        "bf_mean": bf_mean,
        "cldice_detail": cldice_detail,
        "bf_detail": bf_detail,
    }


def evaluate_all(
    pred: np.ndarray,
    gt: np.ndarray,
    linear_classes: list = None,
    polygon_classes: list = None,
    point_classes: list = None,
    dtaf1_config: dict = None,
    building_tolerance: float = 2.0,
    point_tolerance: float = 5.0,
) -> dict:
    """
    Run all metrics and return a consolidated report.

    Defaults assume: class 1 = road (linear), class 2 = building (polygonal).

    Parameters
    ----------
    pred              : H×W integer label map
    gt                : H×W integer label map
    linear_classes    : class IDs for linear features  (default: [1])
    polygon_classes   : class IDs for polygonal features (default: [2])
    point_classes     : class IDs for point features (e.g. trees, manholes).
                        Optional — omitted entirely from the report unless given.
    dtaf1_config      : class_config dict for dtaf1(); uses road/building defaults.
                        Note: dtaf1() is class-agnostic and already supports a
                        point class today with zero code changes — just add e.g.
                        {3: {"name": "point", "tolerance": d}} to this config
                        yourself if you want DTAF1's macro average to include it.
                        This is not done automatically, so existing 2-class DTAF1
                        results stay comparable when point_classes is passed.
    building_tolerance: positional tolerance (px) for BF score
    point_tolerance   : centroid-matching tolerance (px) for point_f1

    Returns
    -------
    dict with keys: "cbhm", "dtaf1", "cldice_mean", "bf_mean", "point_f1_mean",
    "per_class_detail". "point_f1_mean" is reported as an independent figure
    alongside "cbhm" — CBHM's harmonic mean stays 2-way (clDice, BF) by design,
    since it's an intentional foil ("harmonic mean prevents either class from
    masking poor performance in the other"); folding in an unbalanced 3rd class
    would blur that comparison. "point_f1_mean" is None when point_classes is
    not given.
    """
    if linear_classes is None:
        linear_classes = [1]
    if polygon_classes is None:
        polygon_classes = [2]
    if dtaf1_config is None:
        dtaf1_config = {
            cls: {"name": "road",     "tolerance": 10} for cls in linear_classes
        }
        dtaf1_config.update({
            cls: {"name": "building", "tolerance": building_tolerance}
            for cls in polygon_classes
        })

    cbhm_result = cbhm(
        pred, gt, linear_classes, polygon_classes,
        building_tolerance=building_tolerance,
    )
    dtaf1_result = dtaf1(pred, gt, dtaf1_config)

    point_detail = {}
    point_f1_mean = None
    if point_classes:
        point_detail = point_f1_multiclass(
            pred, gt, point_classes, tolerance=point_tolerance
        )
        point_f1_mean = float(
            np.mean([v["point_f1"] for v in point_detail.values()])
        )

    return {
        "cbhm":          cbhm_result["cbhm"],
        "dtaf1":         dtaf1_result["dtaf1"],
        "cldice_mean":   cbhm_result["cldice_mean"],
        "bf_mean":       cbhm_result["bf_mean"],
        "point_f1_mean": point_f1_mean,
        "per_class_detail": {
            "cldice":  cbhm_result["cldice_detail"],
            "bf":      cbhm_result["bf_detail"],
            "dtaf1":   dtaf1_result["per_class"],
            "point":   point_detail,
        },
    }


__all__ = [
    "cbhm",
    "evaluate_all",
    "dtaf1",
    "dtaf1_road_building",
]
