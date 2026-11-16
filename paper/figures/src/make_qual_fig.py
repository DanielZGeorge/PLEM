"""Figure: GT vs conventional vs joint predictions on SpaceNet test tiles where
mIoU and PLEM's metrics disagree. Reads data/spacenet_cases/ (GT, imagery),
results/scaled/preds/{baseline,plem}/ (predictions) and results/scaled/*_test.csv."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, label as cc_label

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
TILES = ["Vegas_img1407", "Vegas_img1505", "Shanghai_img1166"]

ROAD = np.array([0xeb, 0x68, 0x34]) / 255      # orange
BLDG = np.array([0x2a, 0x78, 0xd6]) / 255      # blue
PT_FILL, PT_GT = "#ffd84d", "white"
INK = "#1f1f1e"

scores = {v: pd.read_csv(root / f"results/scaled/{v}_test.csv").set_index("tile")
          for v in ("baseline", "plem")}


def overlay(img, seg, valid=None):
    base = img.astype(float) / 255 * 0.55            # darken imagery so labels read
    o = base.copy()
    for cls, col in ((1, ROAD), (2, BLDG)):
        m = seg == cls
        o[m] = 0.25 * base[m] + 0.75 * col
    if valid is not None:                              # building labels unknown here
        u = valid == 0
        o[u] = 0.6 * o[u] + 0.4 * np.array([0.75, 0.75, 0.75])
    return np.clip(o, 0, 1)


def centroids(mask):
    lab, n = cc_label(mask > 0)
    return np.array(center_of_mass(mask > 0, lab, range(1, n + 1))) if n else np.zeros((0, 2))


fig, axes = plt.subplots(len(TILES), 3, figsize=(7.0, 7.3), constrained_layout=True)
for r, tile in enumerate(TILES):
    city = tile.split("_")[0]
    gt = np.load(root / "data/spacenet_cases" / city / f"{tile}.npz")
    img, lab, valid, gpts = gt["image"], gt["label"], gt["building_valid"], gt["points"]
    panels = [("Ground truth", lab, None)]
    for v, name in (("baseline", "Conventional"), ("plem", "Joint loss")):
        p = np.load(root / "results/scaled/preds" / v / f"{tile}.npz")
        seg = p["seg"].copy()
        seg[(seg == 2) & (valid == 0)] = 0             # as scored: buildings only where labelled
        panels.append((name, seg, centroids(p["points"]), v))
    for c, panel in enumerate(panels):
        ax = axes[r, c]
        title, seg = panel[0], panel[1]
        ax.imshow(overlay(img, seg, valid))
        ax.scatter(gpts[:, 1], gpts[:, 0], s=9, facecolors="none", edgecolors=PT_GT, linewidths=0.7)
        if c == 0:
            ax.set_ylabel(tile.replace("_", " "), fontsize=7, color=INK)
            sub = "GT intersections: white rings"
        else:
            pp, v = panel[2], panel[3]
            if len(pp):
                ax.scatter(pp[:, 1], pp[:, 0], s=10, marker="s", c=PT_FILL, edgecolors=INK,
                           linewidths=0.4)
            s = scores[v].loc[tile]
            sub = (f"mIoU {s.miou:.3f} · DTAF1-Topo {s.dtaf1_topo:.3f}\n"
                   f"APLS {s.road_apls:.3f} · int. F1 {s.point_f1:.2f} ({int(s.n_tp_points)}/{int(s.n_gt_points)})")
        ax.set_title(f"{title}\n{sub}" if r == 0 or c else sub, fontsize=6.3, color=INK, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)

fig.savefig(out, dpi=250, facecolor="white")
for tile in TILES:
    b, j = scores["baseline"].loc[tile], scores["plem"].loc[tile]
    print(tile, {k: (round(b[k], 3), round(j[k], 3)) for k in
                 ["miou", "iou_road", "iou_building", "dtaf1_topo", "road_apls", "bf_mean", "point_f1",
                  "n_pred_points", "n_tp_points", "n_gt_points"]})
