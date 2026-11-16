"""Figure 1: per-term training curves of the production joint model and the
conventional baseline, from results/scaled/*_history.csv."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(sys.argv[1])
out = Path(sys.argv[2])
plem = pd.read_csv(root / "results/scaled/plem_history.csv")
base = pd.read_csv(root / "results/scaled/baseline_history.csv")

JOINT, CONV = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

best_joint = int(plem.loc[plem.val_select.idxmin(), "epoch"])
best_conv = int(base.loc[base.val_miou.idxmax(), "epoch"])

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "font.family": "DejaVu Sans",
})

panels = [
    ("ce_dice", "CE+Dice (train)", True),
    ("tolerance", "Tolerance (train)", False),
    ("cldice", "Soft-clDice (train)", False),
    ("heatmap", "Heatmap (train)", False),
    ("val_miou", "Validation mIoU", True),
]
fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.55), constrained_layout=True)
for ax, (col, title, both) in zip(axes, panels):
    ax.plot(plem.epoch, plem[col], color=JOINT, lw=1.4, label="joint loss")
    if both:
        ax.plot(base.epoch, base[col], color=CONV, lw=1.4, ls=(0, (4, 2)),
                label="conventional")
    # best checkpoint of each model
    ax.plot(best_joint, plem.loc[plem.epoch == best_joint, col].item(), "o",
            ms=3.5, color=JOINT, mec="white", mew=0.8, zorder=3)
    if both:
        ax.plot(best_conv, base.loc[base.epoch == best_conv, col].item(), "o",
                ms=3.5, color=CONV, mec="white", mew=0.8, zorder=3)
    ax.set_title(title, loc="left", pad=3)
    ax.set_xlabel("epoch")
    ax.set_xlim(0, 41)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper right", ncol=2, frameon=False,
           fontsize=7)
fig.savefig(out, dpi=300, facecolor="white")
print("best joint epoch", best_joint, "best conv epoch", best_conv)
for col, *_ in panels:
    print(col, "joint", round(plem[col].iloc[0], 3), "->", round(plem[col].iloc[-1], 3),
          "| conv", round(base[col].iloc[0], 3) if col in base else "-", "->",
          round(base[col].iloc[-1], 3) if col in base else "-")
