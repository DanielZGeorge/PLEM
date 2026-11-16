# Figure sources

Run from the repo root with the project's `.venv`.

| Figure | Command | Inputs |
|---|---|---|
| `training_curves.png` | `python paper/figures/src/make_training_curves.py . paper/figures/training_curves.png` | `results/scaled/{plem,baseline}_history.csv` |
| `qualitative_baseline_vs_joint.png` | `python paper/figures/src/build_fig_tiles.py .` then `python paper/figures/src/make_qual_fig.py . paper/figures/qualitative_baseline_vs_joint.png` | `results/scaled/preds/` (gitignored, from CARC), `results/scaled/*_test.csv`, and the three test tiles that `build_fig_tiles.py` re-derives into `data/spacenet_cases/` with the same `datasets/spacenet.py` functions as the training cache |

`build_cases.py` caches `Khartoum_img371`/`img333`, the sparse-class case-study tiles that
`notebooks/real_data_dtaf1_topo.ipynb` scores separately.
