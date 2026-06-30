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


ROAD_CONFIG = {
    1: {"name": "road",     "tolerance": 10},
    2: {"name": "building", "tolerance":  2},
}

GT = make_gt()


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
        "dtaf1":      dtaf1_r["dtaf1"],
        "cbhm":       cbhm_r["cbhm"],
        "road_iou":   road_iou,
        "bld_iou":    bld_iou,
        "cldice":     cl_r["cldice"],
        "bf":         bf_r["bf"],
    }


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
    Vary how much of the image is road vs building.
    Checks that neither class dominates the unified score.
    """
    if road_lengths is None:
        road_lengths = [16, 32, 64, 96, 128]

    results = {k: [] for k in ["road_frac", "dtaf1", "cbhm"]}

    for length in road_lengths:
        gt = np.zeros(SHAPE, dtype=np.uint8)
        gt[:length, 62:66] = 1   # partial road
        gt[20:60, 20:60] = 2     # constant building

        pred = gt.copy()  # perfect pred; just checking score isn't class-dominated
        s = _score(pred, gt)
        road_frac = (gt == 1).mean()
        results["road_frac"].append(float(road_frac))
        results["dtaf1"].append(s["dtaf1"])
        results["cbhm"].append(s["cbhm"])

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

    if args.plot:
        plot_all(save_dir=args.plot_dir)
