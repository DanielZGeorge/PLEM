"""
Sensitivity sweep experiments for the losses/ package (differentiable
training losses) -- the training-signal counterpart to test_sensitivity.py's
eval-metric sweeps.

Each experiment systematically degrades a synthetic prediction (mirroring the
exact same synthetic scenes/perturbation families as test_sensitivity.py),
and records LOSS VALUE (not accuracy/score) at each degradation level for
five configurations built from PLEMMultiTaskLoss's own sub-term breakdown:

    ce_dice_only    -- the base term alone (no geometric awareness)
    plus_tolerance  -- + ToleranceBandLoss (DTAF1 analog)
    plus_cldice     -- + SoftClDiceLoss (clDice analog)
    plus_heatmap    -- + PointHeatmapLoss (point-F1 analog)
    full            -- all four terms combined

This is a model-free, controlled way to validate loss *design*: it operates
directly on GT-vs-perturbed-GT tensor pairs (via a "confident" logit encoding
of the perturbed label map, same trick as
tests/test_losses_sanity.py::perfect_logits), no trained model in the loop.

Expected qualitative story (see CLAUDE.md): plus_cldice's loss should rise
sharply under road width/offset perturbation where ce_dice_only stays
comparatively flat (mirrors the eval-side clDice-vs-IoU story from
notebooks/metric_comparison.ipynb, now on the training signal), and
plus_heatmap should be the term most distinctly shape-sensitive to point
jitter/dropout/clutter. A FLAT ablation curve where one is expected is a
signal something in the loss math is wrong -- chase it as a bug, don't report
it as a finding.

Run:  python tests/test_loss_sensitivity.py
      python tests/test_loss_sensitivity.py --plot   (saves PNG figures)
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from losses.multitask import PLEMMultiTaskLoss
from test_sensitivity import SHAPE, make_road, make_building, make_gt, make_points, GT, POINT_COORDS


# ---------------------------------------------------------------------------
# Shared loss config + label -> logits encoding
# ---------------------------------------------------------------------------

NUM_CLASSES = 4  # background, road, building, point

CLASS_CONFIG = {
    1: {"name": "road", "tolerance": 10},
    2: {"name": "building", "tolerance": 2},
}

MULTITASK_LOSS = PLEMMultiTaskLoss(
    CLASS_CONFIG, linear_classes=[1], polygon_classes=[2], point_classes=[3]
)

_FULL_MASK = torch.ones((1, NUM_CLASSES))


def _logits_from_label(label: np.ndarray, scale: float = 4.0) -> torch.Tensor:
    """
    (H, W) uint8 label -> (1, C, H, W) logits whose softmax is confidently
    peaked at one-hot(label) (~99.9% at scale=4), but NOT saturated all the
    way to the heatmap_focal_loss's 1e-6 clamp boundary the way
    tests/test_losses_sanity.py's scale=12 `perfect_logits` is. A confidently
    WRONG pixel under PointHeatmapLoss's focal formulation is penalized by
    `-log(1 - pred)`, which is unbounded as pred -> 1 and only capped by the
    clamp epsilon -- at scale=12 essentially every "wrong" pixel hits that
    clamp ceiling exactly, so a handful of false-positive point blobs (9px
    each, from make_points_class3's radius=1) can dominate a sweep's total by
    two orders of magnitude versus every other term. Real model logits are
    never this saturated, so scale=4 is a more representative "confident but
    not maximally certain" prediction for a controlled sensitivity sweep.
    """
    label_t = torch.from_numpy(label.astype(np.int64))
    onehot = torch.nn.functional.one_hot(label_t, NUM_CLASSES).permute(2, 0, 1).float()
    logits = onehot * scale - (1.0 - onehot) * scale
    return logits.unsqueeze(0)


def make_points_class3(coords, shape=SHAPE, radius=1) -> np.ndarray:
    """Same as test_sensitivity.py's make_points, but burns class id 3 (point)
    instead of 1 -- matches PLEMMultiTaskLoss's point_classes=[3]."""
    m = np.zeros(shape, dtype=np.uint8)
    for r, c in coords:
        m[max(0, r - radius): r + radius + 1, max(0, c - radius): c + radius + 1] = 3
    return m


def _score_loss(pred_label: np.ndarray, gt_label: np.ndarray) -> dict:
    """
    Score one (pred, gt) label-map pair, returning both the five cumulative
    ablation configurations (for the printed table, matching the
    ce_dice_only / +tolerance / +cldice / +heatmap / full framing) AND each
    term's own RAW, unsummed value (for plotting -- see RAW_KEYS below).
    Cumulative sums are dominated visually by a growing ce_dice baseline once
    charted (a term's own marginal contribution can be small relative to the
    baseline's absolute scale even when it's the dominant qualitative
    driver), so the two views serve different purposes: the table shows
    "how much does adding this term change the total loss," the plot shows
    "how does this term's own signal move under this perturbation."
    """
    target = torch.from_numpy(gt_label.astype(np.int64)).unsqueeze(0)
    logits = _logits_from_label(pred_label)
    with torch.no_grad():
        out = MULTITASK_LOSS(logits, target, _FULL_MASK)

    ce = out["ce_dice"]
    tol = out["tolerance"]
    cl = out["cldice"]
    hm = out["heatmap"]
    return {
        "ce_dice_only": ce,
        "plus_tolerance": ce + tol,
        "plus_cldice": ce + cl,
        "plus_heatmap": ce + hm,
        "full": ce + tol + cl + hm,
        "ce_dice": ce,
        "tolerance": tol,
        "cldice": cl,
        "heatmap": hm,
    }


CONFIG_KEYS = ["ce_dice_only", "plus_tolerance", "plus_cldice", "plus_heatmap", "full"]
RAW_KEYS = ["ce_dice", "tolerance", "cldice", "heatmap"]


# ---------------------------------------------------------------------------
# Experiment 1: Road centerline offset sweep
# ---------------------------------------------------------------------------

def sweep_road_offset_loss(offsets=None) -> dict:
    """Shift the predicted road horizontally by 0...N pixels (mirrors
    test_sensitivity.py::sweep_road_offset). plus_tolerance's loss should
    stay comparatively low within DTAF1's 10px tolerance and rise beyond it;
    plus_cldice should stay low far longer (clDice is zero-tolerant to
    positional shift only in the sense that it still requires SOME overlap of
    skeleton/mask -- it degrades once the shift exceeds the road's own width)."""
    if offsets is None:
        offsets = list(range(0, 25, 2))

    gt_road = make_road()
    results = {k: [] for k in ["offset"] + CONFIG_KEYS + RAW_KEYS}

    for offset in offsets:
        pred_road = np.roll(gt_road, offset, axis=1)
        pred = np.zeros(SHAPE, dtype=np.uint8)
        pred[pred_road > 0] = 1
        pred[GT == 2] = 2

        s = _score_loss(pred, GT)
        results["offset"].append(offset)
        for k in CONFIG_KEYS + RAW_KEYS:
            results[k].append(s[k])

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Road breakage sweep (random pixel deletion)
# ---------------------------------------------------------------------------

def sweep_road_breakage_loss(fractions=None, seed=42) -> dict:
    """Randomly delete 0...100% of road pixels (mirrors
    test_sensitivity.py::sweep_road_breakage). Neither plus_tolerance nor
    plus_cldice have a connectivity/topology term (that's APLS/dtaf1_topo,
    deliberately out of scope for the differentiable loss -- see
    CLAUDE.md/losses/multitask.py), so both should degrade roughly with pixel
    count lost, not collapse sharply the way APLS does on the eval side."""
    if fractions is None:
        fractions = [i / 10 for i in range(11)]

    rng = np.random.default_rng(seed)
    road_pixels = np.argwhere(GT == 1)
    results = {k: [] for k in ["fraction"] + CONFIG_KEYS + RAW_KEYS}

    for frac in fractions:
        pred = GT.copy()
        if frac > 0:
            n_remove = int(frac * len(road_pixels))
            idx = rng.choice(len(road_pixels), n_remove, replace=False)
            for r, c in road_pixels[idx]:
                pred[r, c] = 0

        s = _score_loss(pred, GT)
        results["fraction"].append(frac)
        for k in CONFIG_KEYS + RAW_KEYS:
            results[k].append(s[k])

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Road width sweep (over-thick prediction)
# ---------------------------------------------------------------------------

def sweep_road_thickness_loss(thicknesses=None) -> dict:
    """Predict road with increasing thickness while GT has fixed width
    (mirrors test_sensitivity.py::sweep_road_thickness). This is the primary
    plus_cldice regression sweep: its loss should stay low well past the
    point where ce_dice_only's plain-Dice term starts climbing, up to
    losses/cldice.py's documented iters-bounded width-invariance limit."""
    if thicknesses is None:
        thicknesses = list(range(1, 22, 2))

    gt_road = make_road(thickness=3)
    gt_map = np.zeros(SHAPE, dtype=np.uint8)
    gt_map[gt_road > 0] = 1
    gt_map[GT == 2] = 2

    results = {k: [] for k in ["thickness"] + CONFIG_KEYS + RAW_KEYS}

    for t in thicknesses:
        pred_road = make_road(thickness=t)
        pred = np.zeros(SHAPE, dtype=np.uint8)
        pred[pred_road > 0] = 1
        pred[GT == 2] = 2

        s = _score_loss(pred, gt_map)
        results["thickness"].append(t)
        for k in CONFIG_KEYS + RAW_KEYS:
            results[k].append(s[k])

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Building erosion sweep
# ---------------------------------------------------------------------------

def sweep_building_erosion_loss(radii=None) -> dict:
    """Progressively erode the predicted building mask (mirrors
    test_sensitivity.py::sweep_building_erosion). plus_tolerance should track
    this closely (DTAF1's building tolerance is tight, 2px, so this behaves
    similarly to ce_dice_only's Dice term); included mainly as a negative
    control showing plus_cldice (a linear-class-only term) stays flat here."""
    if radii is None:
        radii = list(range(0, 12))

    gt_b = (GT == 2).astype(np.uint8)
    results = {k: [] for k in ["erosion_px"] + CONFIG_KEYS + RAW_KEYS}

    for r in radii:
        pred_b = binary_erosion(gt_b, iterations=r).astype(np.uint8) if r > 0 else gt_b
        pred = np.zeros(SHAPE, dtype=np.uint8)
        pred[GT == 1] = 1
        pred[pred_b > 0] = 2

        s = _score_loss(pred, GT)
        results["erosion_px"].append(r)
        for k in CONFIG_KEYS + RAW_KEYS:
            results[k].append(s[k])

    return results


# ---------------------------------------------------------------------------
# Experiment 5: Point positional jitter sweep
# ---------------------------------------------------------------------------

def sweep_point_jitter_loss(offsets=None) -> dict:
    """Shift every predicted point horizontally by 0...N pixels (mirrors
    test_sensitivity.py::sweep_point_jitter, class id 3 instead of 1). This
    is the primary plus_heatmap regression sweep: it should be the config
    most distinctly shape-sensitive to jitter, since ce_dice_only's Dice term
    also responds (point pixels are a tiny fraction of the image, so plain
    Dice is somewhat sensitive too) but plus_heatmap's Gaussian-peak-based
    penalty should show a more pronounced, more clearly saturating response."""
    if offsets is None:
        offsets = list(range(0, 21, 2))

    gt_points = make_points_class3(POINT_COORDS)
    results = {k: [] for k in ["offset"] + CONFIG_KEYS + RAW_KEYS}

    for offset in offsets:
        pred_coords = [(r, c + offset) for r, c in POINT_COORDS]
        pred_points = make_points_class3(pred_coords)

        s = _score_loss(pred_points, gt_points)
        results["offset"].append(offset)
        for k in CONFIG_KEYS + RAW_KEYS:
            results[k].append(s[k])

    return results


# ---------------------------------------------------------------------------
# Experiment 6: Point dropout sweep (missed detections)
# ---------------------------------------------------------------------------

def sweep_point_dropout_loss(fractions=None, seed=42) -> dict:
    """Randomly drop 0...100% of predicted points (mirrors
    test_sensitivity.py::sweep_point_dropout, class id 3)."""
    if fractions is None:
        fractions = [i / 10 for i in range(11)]

    rng = np.random.default_rng(seed)
    n_points = len(POINT_COORDS)
    gt_points = make_points_class3(POINT_COORDS)
    results = {k: [] for k in ["fraction"] + CONFIG_KEYS + RAW_KEYS}

    for frac in fractions:
        n_keep = max(0, int(round(n_points * (1 - frac))))
        keep_idx = rng.choice(n_points, n_keep, replace=False) if n_keep > 0 else []
        pred_coords = [POINT_COORDS[i] for i in keep_idx]
        pred_points = make_points_class3(pred_coords)

        s = _score_loss(pred_points, gt_points)
        results["fraction"].append(frac)
        for k in CONFIG_KEYS + RAW_KEYS:
            results[k].append(s[k])

    return results


# ---------------------------------------------------------------------------
# Experiment 7: Point clutter sweep (spurious false positives)
# ---------------------------------------------------------------------------

def sweep_point_clutter_loss(n_extra=None, seed=7) -> dict:
    """Add an increasing number of spurious predicted points (mirrors
    test_sensitivity.py::sweep_point_clutter, class id 3)."""
    if n_extra is None:
        n_extra = list(range(0, 21, 2))

    rng = np.random.default_rng(seed)
    gt_points = make_points_class3(POINT_COORDS)
    results = {k: [] for k in ["n_extra"] + CONFIG_KEYS + RAW_KEYS}

    for n in n_extra:
        pred_coords = list(POINT_COORDS)
        extra_rc = rng.integers(0, SHAPE[0], size=(n, 2))
        pred_coords += [tuple(rc) for rc in extra_rc]
        pred_points = make_points_class3(pred_coords)

        s = _score_loss(pred_points, gt_points)
        results["n_extra"].append(n)
        for k in CONFIG_KEYS + RAW_KEYS:
            results[k].append(s[k])

    return results


# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------

_RAW_COLORS = {
    "ce_dice": "0.5",
    "tolerance": "tab:blue",
    "cldice": "tab:green",
    "heatmap": "tab:orange",
}
_RAW_TITLES = {
    "ce_dice": "ce_dice (base term)",
    "tolerance": "tolerance (DTAF1 analog)",
    "cldice": "cldice (clDice analog)",
    "heatmap": "heatmap (point-F1 analog)",
}


def _plot_sweep(results: dict, x_key: str, xlabel: str, title: str, path: str):
    """
    Small-multiples layout, one subplot per RAW term with its OWN y-scale
    (mirrors the small-multiples convention notebooks/real_data_evaluation.ipynb
    already uses for eval-metric sweeps) -- deliberately not a single shared
    axis: the four terms' magnitudes differ by orders of magnitude (e.g.
    heatmap's focal-loss formulation vs. the bounded-[0,1]-ratio terms), so a
    shared axis would visually flatten every non-dominant term to ~0. Each
    subplot's own scale makes that term's shape legible regardless.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, k in zip(axes, RAW_KEYS):
        ax.plot(results[x_key], results[k], "-", color=_RAW_COLORS[k])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loss value")
        ax.set_title(_RAW_TITLES[k])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all(save_dir: str = "notebooks/figures/loss_sensitivity"):
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        print("matplotlib not installed -- skipping plots")
        return

    os.makedirs(save_dir, exist_ok=True)

    _plot_sweep(
        sweep_road_offset_loss(), "offset", "Road offset (pixels)",
        "Road offset sweep (loss)", os.path.join(save_dir, "sweep_road_offset_loss.png"),
    )
    _plot_sweep(
        sweep_road_breakage_loss(), "fraction", "Fraction of road pixels removed",
        "Road breakage sweep (loss)", os.path.join(save_dir, "sweep_road_breakage_loss.png"),
    )
    _plot_sweep(
        sweep_road_thickness_loss(), "thickness", "Predicted road thickness (pixels)",
        "Road thickness sweep (loss)\n(plus_cldice should stay low longer than ce_dice_only)",
        os.path.join(save_dir, "sweep_road_thickness_loss.png"),
    )
    _plot_sweep(
        sweep_building_erosion_loss(), "erosion_px", "Erosion radius (pixels)",
        "Building erosion sweep (loss)", os.path.join(save_dir, "sweep_building_erosion_loss.png"),
    )
    _plot_sweep(
        sweep_point_jitter_loss(), "offset", "Point jitter (pixels)",
        "Point jitter sweep (loss)\n(plus_heatmap should be most shape-sensitive)",
        os.path.join(save_dir, "sweep_point_jitter_loss.png"),
    )
    _plot_sweep(
        sweep_point_dropout_loss(), "fraction", "Fraction of predicted points dropped",
        "Point dropout sweep (loss)", os.path.join(save_dir, "sweep_point_dropout_loss.png"),
    )
    _plot_sweep(
        sweep_point_clutter_loss(), "n_extra", "Number of spurious predicted points",
        "Point clutter sweep (loss)", os.path.join(save_dir, "sweep_point_clutter_loss.png"),
    )

    print(f"Figures saved to {save_dir}/")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def print_table(name: str, results: dict):
    """Prints only the 5 cumulative CONFIG_KEYS columns (results also carries
    RAW_KEYS, used by _plot_sweep, but showing all 9 in one table is noisy)."""
    x_key = list(results.keys())[0]
    header = f"{'':>12}" + "".join(f"{k:>16}" for k in CONFIG_KEYS)
    print(f"\n=== {name} ===")
    print(header)
    for i, x in enumerate(results[x_key]):
        row = f"{x:>12.3f}" + "".join(f"{results[k][i]:>16.4f}" for k in CONFIG_KEYS)
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLEM losses/ sensitivity sweeps")
    parser.add_argument("--plot", action="store_true", help="Save PNG figures")
    parser.add_argument("--plot-dir", default="notebooks/figures/loss_sensitivity",
                        help="Directory for figure output")
    args = parser.parse_args()

    print_table("Road offset sweep (loss)", sweep_road_offset_loss())
    print_table("Road breakage sweep (loss)", sweep_road_breakage_loss())
    print_table("Road thickness sweep (loss)", sweep_road_thickness_loss())
    print_table("Building erosion sweep (loss)", sweep_building_erosion_loss())
    print_table("Point jitter sweep (loss)", sweep_point_jitter_loss())
    print_table("Point dropout sweep (loss)", sweep_point_dropout_loss())
    print_table("Point clutter sweep (loss)", sweep_point_clutter_loss())

    if args.plot:
        plot_all(save_dir=args.plot_dir)
