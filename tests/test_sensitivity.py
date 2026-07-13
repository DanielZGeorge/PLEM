"""
Sensitivity sweep experiments for PLEM metrics.

Each experiment systematically degrades a synthetic ground-truth mask,
records metric values at each degradation level, and optionally plots the
results. This script can be run standalone or imported as a module.

Run:  python tests/test_sensitivity.py
      python tests/test_sensitivity.py --plot      (saves PNG figures)

These experiments support the paper's claim that DTAF1 and clDice respond
more appropriately to geometric degradations than pixel-wise IoU.
"""

import argparse
import sys
import os
import numpy as np
from scipy.ndimage import binary_erosion, shift as nd_shift

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics.dtaf1 import dtaf1
from metrics.cldice import cldice
from metrics.boundary_f1 import boundary_f1, iou
from metrics.point_f1 import point_f1
from metrics.unified import cbhm


# ---------------------------------------------------------------------------
# Synthetic scene builders
# ---------------------------------------------------------------------------

SHAPE = (128, 128)


def make_road(col=64, thickness=3, shape=SHAPE):
    m = np.zeros(shape, dtype=np.uint8)
    half = thickness // 2
    m[:, col - half: col + half + 1] = 1
    return m


def make_building(top=20, left=20, height=40, width=40, shape=SHAPE):
    m = np.zeros(shape, dtype=np.uint8)
    m[top: top + height, left: left + width] = 2
    return m


def make_gt(shape=SHAPE):
    r = make_road(shape=shape)
    b = make_building(shape=shape)
    gt = np.zeros(shape, dtype=np.uint8)
    gt[r > 0] = 1
    gt[b > 0] = 2
    return gt


def make_points(coords, shape=SHAPE, radius=1):
    """Small square blobs at the given (row, col) coords -- a scattered point scene."""
    m = np.zeros(shape, dtype=np.uint8)
    for r, c in coords:
        m[max(0, r - radius): r + radius + 1, max(0, c - radius): c + radius + 1] = 1
    return m


ROAD_CONFIG = {
    1: {"name": "road",     "tolerance": 10},
    2: {"name": "building", "tolerance":  2},
}

GT = make_gt()

# Independent scattered-points scene (not tied to the road/building GT above),
# with margin kept from the canvas edge so jitter/clutter sweeps don't clip.
POINT_COORDS = [
    (20, 20), (20, 60), (60, 20), (60, 60), (40, 40),
    (10, 100), (100, 10), (100, 100), (110, 20), (30, 110),
]
GT_POINTS = make_points(POINT_COORDS)


# ---------------------------------------------------------------------------
# Helper: compute all metrics for one (pred, gt) pair, return dict of scalars
# ---------------------------------------------------------------------------

def _score(pred: np.ndarray, gt: np.ndarray) -> dict:
    dtaf1_r = dtaf1(pred, gt, ROAD_CONFIG)
    road_iou = iou((pred == 1), (gt == 1))
    bld_iou = iou((pred == 2), (gt == 2))
    cl_r = cldice((pred == 1), (gt == 1))
    bf_r = boundary_f1((pred == 2), (gt == 2), tolerance=2)
    cbhm_r = cbhm(pred, gt, linear_classes=[1], polygon_classes=[2])

    return {
        "dtaf1":          dtaf1_r["dtaf1"],
        "dtaf1_weighted": dtaf1_r["dtaf1_weighted"],
        "cbhm":           cbhm_r["cbhm"],
        "cbhm_soft":      cbhm_r["cbhm_soft"],
        "road_iou":       road_iou,
        "bld_iou":        bld_iou,
        "cldice":         cl_r["cldice"],
        "bf":             bf_r["bf"],
    }


def _point_score(pred_points: np.ndarray, gt_points: np.ndarray, tolerance: float = 5.0) -> dict:
    """Separate from _score() -- point scenes are independent of the road/building GT
    the other sweeps depend on, so this doesn't touch their existing numeric behavior."""
    r = point_f1(pred_points, gt_points, tolerance=tolerance)
    return {"point_f1": r["point_f1"], "precision": r["precision"], "recall": r["recall"]}


# ---------------------------------------------------------------------------
# Experiment 1: Road centerline offset sweep
# ---------------------------------------------------------------------------

def sweep_road_offset(offsets=None) -> dict:
    """
    Shift the predicted road horizontally by 0…N pixels.
    Demonstrates that DTAF1 / clDice tolerate small offsets while IoU collapses.
    """
    if offsets is None:
        offsets = list(range(0, 25, 2))

    gt_road = make_road()
    results = {k: [] for k in ["offset", "dtaf1", "cbhm", "road_iou", "cldice"]}

    for offset in offsets:
        pred_road = np.roll(gt_road, offset, axis=1)  # horizontal shift
        pred = np.zeros(SHAPE, dtype=np.uint8)
        pred[pred_road > 0] = 1
        pred[GT == 2] = 2  # keep buildings perfect

        s = _score(pred, GT)
        results["offset"].append(offset)
        results["dtaf1"].append(s["dtaf1"])
        results["cbhm"].append(s["cbhm"])
        results["road_iou"].append(s["road_iou"])
        results["cldice"].append(s["cldice"])

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Road breakage sweep (random pixel deletion)
# ---------------------------------------------------------------------------

def sweep_road_breakage(fractions=None, seed=42) -> dict:
    """
    Randomly delete 0…100% of road pixels.
    A topologically meaningful error: broken road network.
    """
    if fractions is None:
        fractions = [i / 10 for i in range(11)]

    rng = np.random.default_rng(seed)
    road_pixels = np.argwhere(GT == 1)
    results = {k: [] for k in ["fraction", "dtaf1", "cbhm", "road_iou", "cldice"]}

    for frac in fractions:
        pred = GT.copy()
        if frac > 0:
            n_remove = int(frac * len(road_pixels))
            idx = rng.choice(len(road_pixels), n_remove, replace=False)
            for r, c in road_pixels[idx]:
                pred[r, c] = 0

        s = _score(pred, GT)
        results["fraction"].append(frac)
        results["dtaf1"].append(s["dtaf1"])
        results["cbhm"].append(s["cbhm"])
        results["road_iou"].append(s["road_iou"])
        results["cldice"].append(s["cldice"])

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Building erosion sweep
# ---------------------------------------------------------------------------

def sweep_building_erosion(radii=None) -> dict:
    """
    Progressively erode the predicted building mask.
    Measures sensitivity to shape/area degradation.
    """
    if radii is None:
        radii = list(range(0, 12))

    gt_b = (GT == 2).astype(np.uint8)
    results = {k: [] for k in ["erosion_px", "dtaf1", "cbhm", "bld_iou", "bf"]}

    for r in radii:
        pred_b = binary_erosion(gt_b, iterations=r).astype(np.uint8) if r > 0 else gt_b
        pred = np.zeros(SHAPE, dtype=np.uint8)
        pred[GT == 1] = 1      # road stays perfect
        pred[pred_b > 0] = 2

        s = _score(pred, GT)
        results["erosion_px"].append(r)
        results["dtaf1"].append(s["dtaf1"])
        results["cbhm"].append(s["cbhm"])
        results["bld_iou"].append(s["bld_iou"])
        results["bf"].append(s["bf"])

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Road width sweep (over-thick prediction)
# ---------------------------------------------------------------------------

def sweep_road_thickness(thicknesses=None) -> dict:
    """
    Predict road with increasing thickness while GT has fixed width.
    IoU collapses; clDice / DTAF1 should stay high.
    """
    if thicknesses is None:
        thicknesses = list(range(1, 22, 2))

    gt_road = make_road(thickness=3)
    gt_map = np.zeros(SHAPE, dtype=np.uint8)
    gt_map[gt_road > 0] = 1
    gt_map[GT == 2] = 2

    results = {k: [] for k in ["thickness", "dtaf1", "cbhm", "road_iou", "cldice"]}

    for t in thicknesses:
        pred_road = make_road(thickness=t)
        pred = np.zeros(SHAPE, dtype=np.uint8)
        pred[pred_road > 0] = 1
        pred[GT == 2] = 2

        s = _score(pred, gt_map)
        results["thickness"].append(t)
        results["dtaf1"].append(s["dtaf1"])
        results["cbhm"].append(s["cbhm"])
        results["road_iou"].append(s["road_iou"])
        results["cldice"].append(s["cldice"])

    return results


# ---------------------------------------------------------------------------
# Experiment 5: Class imbalance — vary road-to-building area ratio
# ---------------------------------------------------------------------------

def sweep_class_imbalance(road_lengths=None) -> dict:
    """
    Vary how much of the image is road vs building, with an *imperfect* road
    prediction (road entirely missing) at every ratio. With a perfect
    prediction the averaging logic is never exercised (every score is 1.0
    regardless of imbalance) -- this sweep instead shows how the harsh,
    unweighted "cbhm"/"dtaf1" (macro) and the lenient, area-weighted
    "cbhm_soft"/"dtaf1_weighted" diverge as the road's pixel share shrinks:
    the weighted variants should recover toward 1.0 as road_frac -> 0 (since
    the correct, dominant building class should count for more), while the
    unweighted variants stay flat regardless of road_frac.
    """
    if road_lengths is None:
        road_lengths = [16, 32, 64, 96, 128]

    results = {k: [] for k in ["road_frac", "dtaf1", "dtaf1_weighted", "cbhm", "cbhm_soft"]}

    for length in road_lengths:
        gt = np.zeros(SHAPE, dtype=np.uint8)
        gt[:length, 62:66] = 1   # partial road
        gt[20:60, 20:60] = 2     # constant building

        pred = gt.copy()
        pred[pred == 1] = 0      # road entirely missing; building stays perfect

        s = _score(pred, gt)
        road_frac = (gt == 1).mean()
        results["road_frac"].append(float(road_frac))
        results["dtaf1"].append(s["dtaf1"])
        results["dtaf1_weighted"].append(s["dtaf1_weighted"])
        results["cbhm"].append(s["cbhm"])
        results["cbhm_soft"].append(s["cbhm_soft"])

    return results


# ---------------------------------------------------------------------------
# Experiment 5b: Sparse-class offset sweep
# ---------------------------------------------------------------------------

def sweep_sparse_class_offset(offsets=None) -> dict:
    """
    Mirrors the real-data failure found in
    notebooks/composite_vs_submetric_report.ipynb (Khartoum_img371): a very
    sparse road (~0.02% of the image) alongside a large, dominant building,
    with the road progressively offset. Once the offset exceeds clDice's
    effective tolerance the skeleton comparison collapses to exactly 0, which
    collapses the harsh "cbhm" harmonic mean to 0 even though the dominant
    building class is still perfect. "cbhm_soft"/"dtaf1_weighted" should
    degrade far more gracefully, staying high throughout.
    """
    if offsets is None:
        offsets = list(range(0, 21, 2))

    gt = np.zeros(SHAPE, dtype=np.uint8)
    gt[3:7, 3:4] = 1            # sparse road stub, outside the building's rows
    gt[10:110, 10:110] = 2      # large, dominant building

    results = {k: [] for k in ["offset", "dtaf1", "dtaf1_weighted", "cbhm", "cbhm_soft"]}

    for offset in offsets:
        pred = np.zeros(SHAPE, dtype=np.uint8)
        pred[3:7, 3 + offset:4 + offset] = 1
        pred[10:110, 10:110] = 2

        s = _score(pred, gt)
        results["offset"].append(offset)
        results["dtaf1"].append(s["dtaf1"])
        results["dtaf1_weighted"].append(s["dtaf1_weighted"])
        results["cbhm"].append(s["cbhm"])
        results["cbhm_soft"].append(s["cbhm_soft"])

    return results


# ---------------------------------------------------------------------------
# Experiment 6: Point positional jitter sweep
# ---------------------------------------------------------------------------

def sweep_point_jitter(offsets=None) -> dict:
    """
    Shift every predicted point horizontally by 0…N pixels.
    Demonstrates that point_f1 tolerates small jitter and collapses once the
    shift exceeds the tolerance radius.
    """
    if offsets is None:
        offsets = list(range(0, 21, 2))

    results = {k: [] for k in ["offset", "point_f1", "precision", "recall"]}
    for offset in offsets:
        pred_coords = [(r, c + offset) for r, c in POINT_COORDS]
        pred_points = make_points(pred_coords)

        s = _point_score(pred_points, GT_POINTS, tolerance=5.0)
        results["offset"].append(offset)
        results["point_f1"].append(s["point_f1"])
        results["precision"].append(s["precision"])
        results["recall"].append(s["recall"])

    return results


# ---------------------------------------------------------------------------
# Experiment 7: Point dropout sweep (missed detections)
# ---------------------------------------------------------------------------

def sweep_point_dropout(fractions=None, seed=42) -> dict:
    """
    Randomly drop 0…100% of predicted points (simulate missed detections).
    Recall should fall roughly linearly; precision should stay high since the
    remaining predictions are still correct.
    """
    if fractions is None:
        fractions = [i / 10 for i in range(11)]

    rng = np.random.default_rng(seed)
    n_points = len(POINT_COORDS)
    results = {k: [] for k in ["fraction", "point_f1", "precision", "recall"]}

    for frac in fractions:
        n_keep = max(0, int(round(n_points * (1 - frac))))
        keep_idx = rng.choice(n_points, n_keep, replace=False) if n_keep > 0 else []
        pred_coords = [POINT_COORDS[i] for i in keep_idx]
        pred_points = make_points(pred_coords)

        s = _point_score(pred_points, GT_POINTS, tolerance=5.0)
        results["fraction"].append(frac)
        results["point_f1"].append(s["point_f1"])
        results["precision"].append(s["precision"])
        results["recall"].append(s["recall"])

    return results


# ---------------------------------------------------------------------------
# Experiment 8: Point clutter sweep (spurious false positives)
# ---------------------------------------------------------------------------

def sweep_point_clutter(n_extra=None, seed=7) -> dict:
    """
    Add an increasing number of spurious predicted points scattered randomly
    across the canvas, on top of the correct predictions. Precision should
    fall roughly linearly; recall should stay at 1.0 since every GT point is
    still correctly predicted.
    """
    if n_extra is None:
        n_extra = list(range(0, 21, 2))

    rng = np.random.default_rng(seed)
    results = {k: [] for k in ["n_extra", "point_f1", "precision", "recall"]}

    for n in n_extra:
        pred_coords = list(POINT_COORDS)
        extra_rc = rng.integers(0, SHAPE[0], size=(n, 2))
        pred_coords += [tuple(rc) for rc in extra_rc]
        pred_points = make_points(pred_coords)

        s = _point_score(pred_points, GT_POINTS, tolerance=5.0)
        results["n_extra"].append(n)
        results["point_f1"].append(s["point_f1"])
        results["precision"].append(s["precision"])
        results["recall"].append(s["recall"])

    return results


# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------

def plot_all(save_dir: str = "."):
    """Generate and save sensitivity curve figures."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    os.makedirs(save_dir, exist_ok=True)

    # --- Experiment 1: offset ---
    r1 = sweep_road_offset()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r1["offset"], r1["road_iou"], "r--",  label="Road IoU")
    ax.plot(r1["offset"], r1["cldice"],   "b-",   label="clDice")
    ax.plot(r1["offset"], r1["dtaf1"],    "g-",   label="DTAF1 (d=10px)")
    ax.set_xlabel("Road offset (pixels)")
    ax.set_ylabel("Score")
    ax.set_title("Road offset sweep\n(IoU collapses; clDice/DTAF1 tolerant within d)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "sweep_road_offset.png"), dpi=150)
    plt.close(fig)

    # --- Experiment 2: breakage ---
    r2 = sweep_road_breakage()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r2["fraction"], r2["road_iou"], "r--",  label="Road IoU")
    ax.plot(r2["fraction"], r2["cldice"],   "b-",   label="clDice")
    ax.plot(r2["fraction"], r2["dtaf1"],    "g-",   label="DTAF1")
    ax.set_xlabel("Fraction of road pixels removed")
    ax.set_ylabel("Score")
    ax.set_title("Road breakage sweep\n(all metrics penalise pixel deletion)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "sweep_road_breakage.png"), dpi=150)
    plt.close(fig)

    # --- Experiment 3: erosion ---
    r3 = sweep_building_erosion()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r3["erosion_px"], r3["bld_iou"], "r--", label="Building IoU")
    ax.plot(r3["erosion_px"], r3["bf"],       "b-",  label="Boundary F1")
    ax.plot(r3["erosion_px"], r3["dtaf1"],    "g-",  label="DTAF1")
    ax.set_xlabel("Erosion radius (pixels)")
    ax.set_ylabel("Score")
    ax.set_title("Building erosion sweep")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "sweep_building_erosion.png"), dpi=150)
    plt.close(fig)

    # --- Experiment 4: thickness ---
    r4 = sweep_road_thickness()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r4["thickness"], r4["road_iou"], "r--", label="Road IoU")
    ax.plot(r4["thickness"], r4["cldice"],   "b-",  label="clDice")
    ax.plot(r4["thickness"], r4["dtaf1"],    "g-",  label="DTAF1 (d=10px)")
    ax.axvline(x=3, color="k", linestyle=":", label="GT thickness")
    ax.set_xlabel("Predicted road thickness (pixels)")
    ax.set_ylabel("Score")
    ax.set_title("Road width sweep\n(IoU penalises over-thick; clDice/DTAF1 do not)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "sweep_road_thickness.png"), dpi=150)
    plt.close(fig)

    # --- Experiment 5b: sparse-class offset ---
    r5b = sweep_sparse_class_offset()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r5b["offset"], r5b["cbhm"],           "r--", label="CBHM (harsh)")
    ax.plot(r5b["offset"], r5b["cbhm_soft"],      "r-",  label="CBHM-soft (area-weighted)")
    ax.plot(r5b["offset"], r5b["dtaf1"],          "b--", label="DTAF1 (macro)")
    ax.plot(r5b["offset"], r5b["dtaf1_weighted"], "b-",  label="DTAF1 (weighted)")
    ax.set_xlabel("Sparse road offset (pixels)")
    ax.set_ylabel("Score")
    ax.set_title("Sparse-class offset sweep\n(reproduces the Khartoum_img371 real-data CBHM collapse)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "sweep_sparse_class_offset.png"), dpi=150)
    plt.close(fig)

    # --- Experiment 6: point jitter ---
    r6 = sweep_point_jitter()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r6["offset"], r6["precision"], "r--", label="Precision")
    ax.plot(r6["offset"], r6["recall"],    "b--", label="Recall")
    ax.plot(r6["offset"], r6["point_f1"],  "g-",  label="point_f1 (tol=5px)")
    ax.set_xlabel("Point jitter (pixels)")
    ax.set_ylabel("Score")
    ax.set_title("Point jitter sweep\n(tolerant within tolerance radius, collapses beyond it)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "sweep_point_jitter.png"), dpi=150)
    plt.close(fig)

    # --- Experiment 7: point dropout ---
    r7 = sweep_point_dropout()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r7["fraction"], r7["precision"], "r--", label="Precision")
    ax.plot(r7["fraction"], r7["recall"],    "b--", label="Recall")
    ax.plot(r7["fraction"], r7["point_f1"],  "g-",  label="point_f1")
    ax.set_xlabel("Fraction of predicted points dropped")
    ax.set_ylabel("Score")
    ax.set_title("Point dropout sweep (missed detections)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "sweep_point_dropout.png"), dpi=150)
    plt.close(fig)

    # --- Experiment 8: point clutter ---
    r8 = sweep_point_clutter()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r8["n_extra"], r8["precision"], "r--", label="Precision")
    ax.plot(r8["n_extra"], r8["recall"],    "b--", label="Recall")
    ax.plot(r8["n_extra"], r8["point_f1"],  "g-",  label="point_f1")
    ax.set_xlabel("Number of spurious predicted points")
    ax.set_ylabel("Score")
    ax.set_title("Point clutter sweep (false positives)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "sweep_point_clutter.png"), dpi=150)
    plt.close(fig)

    print(f"Figures saved to {save_dir}/")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def print_table(name: str, results: dict):
    keys = list(results.keys())
    x_key = keys[0]
    metric_keys = keys[1:]
    header = f"{'':>12}" + "".join(f"{k:>12}" for k in metric_keys)
    print(f"\n=== {name} ===")
    print(header)
    for i, x in enumerate(results[x_key]):
        row = f"{x:>12.3f}" + "".join(f"{results[k][i]:>12.4f}" for k in metric_keys)
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLEM metric sensitivity sweeps")
    parser.add_argument("--plot", action="store_true", help="Save PNG figures")
    parser.add_argument("--plot-dir", default="notebooks/figures",
                        help="Directory for figure output")
    args = parser.parse_args()

    print_table("Road offset sweep",     sweep_road_offset())
    print_table("Road breakage sweep",   sweep_road_breakage())
    print_table("Building erosion sweep", sweep_building_erosion())
    print_table("Road thickness sweep",  sweep_road_thickness())
    print_table("Class imbalance sweep", sweep_class_imbalance())
    print_table("Sparse-class offset sweep", sweep_sparse_class_offset())
    print_table("Point jitter sweep",    sweep_point_jitter())
    print_table("Point dropout sweep",   sweep_point_dropout())
    print_table("Point clutter sweep",   sweep_point_clutter())

    if args.plot:
        plot_all(save_dir=args.plot_dir)
