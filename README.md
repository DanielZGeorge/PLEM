# PLEM — Polygonal-Linear Extraction Metrics

A research library implementing evaluation metrics for **multiclass geospatial segmentation maps** that contain both linear features (roads) and polygonal features (buildings).

## The Core Problem

Standard segmentation evaluation leans on IoU (Intersection over Union). IoU works reasonably well for compact, area-like objects such as buildings, but it is fundamentally the wrong tool for **linear** features like roads:

- A road predicted a few pixels off-center from the true centerline is geometrically almost perfect — the network topology and connectivity are correct — but a thin linear shape shifted sideways has very little pixel overlap with itself, so IoU collapses.
- A road predicted with a slightly different **width** than ground truth (e.g. 2px wider) is often still a completely usable extraction, but again the extra/missing width area tanks IoU because a road's area is small relative to its length.
- Neither of these failure modes says anything about whether the road network is actually *broken* (disconnected) — which is a real, important failure mode that pixel-wise IoU also doesn't directly capture.

PLEM implements a family of **tolerance-based, topology-aware, and class-appropriate** metrics so that linear and polygonal features are each scored by criteria that match how they actually fail, and combines them into unified composite scores for consolidated reporting.

### Conventions

- All inputs are plain 2-D NumPy arrays: `H×W`, dtype `uint8`.
- Label convention: `0` = background, `1` = road (linear), `2` = building (polygonal), `3` = point feature (small discrete objects — trees, lamp posts, manhole covers; optional, only used when the data actually has a point class).
- Tolerances are expressed in pixels, tied to ground sample distance (GSD): `d = physical_metres / GSD_metres_per_pixel`.
- Every public metric function returns a **dict** of named scalar fields (never a bare scalar), so results are easy to log, inspect, and compare side by side.

---

## Project Timeline

Dated by actual commit history (`git log`), oldest first.

| Date | Commit | What changed |
|---|---|---|
| **2026-06-30** | `d91f5f5` | **Initial implementation of PLEM metric library.** First versions of DTAF1, clDice, boundary F1/IoU, and the unified CBHM composite; synthetic-scene sanity tests and sensitivity sweeps; `metric_comparison.ipynb` demonstration notebook. |
| 2026-06-30 | `950c199` | Removed committed `__pycache__` directories (housekeeping). |
| 2026-06-30 | `ec2a32a` | Fixed `AttributeError` from a deprecated Matplotlib API (`plt.cm.get_cmap()` → `plt.colormaps[]`), keeping the comparison notebook runnable on current library versions. |
| 2026-06-30 | `6c80d7f` | Executed `metric_comparison.ipynb` end-to-end and committed the run outputs. |
| **2026-07-05** | `c97f863` | **Added the point-feature metric and the real-data evaluation pipeline.** Introduced `point_f1.py` (Hungarian-matching instance-level F1 for point objects), the `datasets/` package for pulling real SpaceNet/Potsdam imagery, and the first real-data evaluation notebooks. |
| **2026-07-07** | `a645f68` | Added human-readable perturbation descriptions and colorized small-multiples charts to `real_data_evaluation.ipynb`, making the real-data sweep results directly comparable to the synthetic sweeps. |
| **2026-07-12** | `94ef38e` | **Added area-weighted, collapse-resistant companion metrics** — `dtaf1_weighted` and `cbhm_soft` — after real-data testing (`Khartoum_img371`) showed the harsh, unweighted `dtaf1`/`cbhm` composites could be misled by a single sparse class. See "Area-weighting and zero-collapse" below. |
| **2026-07-13** | `848b215`, `5ce8968` | Made `train_unet.ipynb` runnable standalone on Google Colab CPU — PLEM's first trained-model pipeline (a small multiclass U-Net for road/building segmentation), so metrics can be evaluated against a real extractor's output instead of only perturbed ground truth. |
| **2026-08-01** | `890b791` | **Added the raster-skeleton APLS connectivity term**, closing DTAF1's known road-breakage blind spot (see below). New `metrics/apls.py` and `metrics/dtaf1_topo.py`, plus the `TestRoadBreakageRegression` regression tests and an updated `sweep_road_breakage()` sensitivity sweep. |
| **2026-08-02** | `e65001e` | Added a visual-confirmation section (3f) to `metric_comparison.ipynb` charting `dtaf1_topo`/`road_apls` collapsing sharply under road breakage where plain `dtaf1` stays pinned near 1.0 — a direct, plotted cross-check of the `TestRoadBreakageRegression` assertions. |

---

## Repository Layout

```
PLEM/
├── metrics/          # The metric library (see below)
├── datasets/         # Real-data acquisition & rasterization (SpaceNet, Potsdam)
├── tests/            # Unit/sanity tests + sensitivity sweep experiments
├── notebooks/        # Synthetic demos, real-data prep/eval, model training
├── data/             # Gitignored — cached downloaded/rasterized real-data samples
├── models/           # Gitignored — trained U-Net checkpoints
├── requirements.txt
└── CLAUDE.md         # Detailed engineering/architecture reference for this repo
```

---

## The Metrics (`metrics/`)

### `dtaf1.py` — DTAF1 (primary unified metric)

Distance-Tolerant, Area/class-Fused F1. For each class, computes a tolerance-radius F1 score: a predicted pixel counts as a true positive if it falls within `tolerance` pixels of *any* GT pixel of that class (and symmetrically for recall), using `scipy.ndimage.distance_transform_edt` for the distance computation. The `class_config` dict maps `class_id → {name, tolerance}`, so scoring an additional class (e.g. a point class) costs zero code changes — just add its entry.

`dtaf1()` always computes **both**:
- `dtaf1_macro` — unweighted mean of per-class F1 (every class counts equally regardless of size)
- `dtaf1_weighted` — GT-pixel-count-weighted mean (large classes dominate the average)

The top-level `dtaf1` key is driven by the `reduction` parameter (default `"macro"`, preserving backward compatibility).

**Known limitation:** random pixel deletion (road breakage) is under-penalized, because scattered surviving pixels still fall within the tolerance radius of some GT pixel even once the road network is topologically disconnected. Confirmed on real SpaceNet data: DTAF1 stayed at 1.0 even with 75% of real road pixels deleted, while CBHM correctly collapsed to ~0.57. This is what `apls.py`/`dtaf1_topo.py` (below) were built to fix.

### `cldice.py` — clDice (centerline Dice, linear features)

Skeletonizes both prediction and ground truth via `skimage.morphology.skeletonize`, then measures mutual coverage between each skeleton and the other mask. Zero-tolerant to positional shift in the sense that it directly measures the extracted centerline against the true one; insensitive to road *width* (a 1px-wide and 20px-wide correct road score identically). `cldice_multiclass()` results carry an `"n_gt"` pixel count per class; `mean_cldice(..., reduction="macro"|"weighted")` mirrors DTAF1's reduction API.

**Known brittleness:** on real, very sparse (~0.1% road-pixel) tiles, a moderate positional offset can collapse clDice to exactly 0, because skeleton-based comparison is brittle on short/sparse road segments — DTAF1 degrades much more gracefully in that same scenario. Neither metric is uniformly more trustworthy.

### `boundary_f1.py` — BF score (boundary F1, polygonal features) + `iou()` baseline

Extracts boundary pixels of a shape morphologically, then checks within-tolerance matches using distance transforms — the polygon analogue of clDice's tolerance-based matching. Also exports plain `iou()`, used throughout the library and tests as the "naive baseline" comparator that the tolerance-based metrics are shown to outperform for roads. `boundary_f1_multiclass()`/`mean_boundary_f1()` carry the same `"n_gt"`/`reduction` support as `cldice.py`.

### `point_f1.py` — Point F1 (instance-level matching, point features)

For small discrete objects (trees, lamp posts, manhole covers), pixel-level tolerance matching would over- or under-count when points cluster closely. Instead, `point_f1.py` reduces each connected-component blob to a centroid (`scipy.ndimage.label` + `center_of_mass`), then performs **tolerance-restricted Hungarian assignment** (`scipy.optimize.linear_sum_assignment`) between predicted and GT centroids — deliberately not greedy nearest-neighbor, which can lock a predicted blob onto the wrong nearby GT point when points cluster together (see `TestPointInstanceMatching` for the adversarial case this avoids).

### `unified.py` — CBHM and `evaluate_all()`

**CBHM** (Composite Boundary-Harmonic Metric) is the harmonic mean of mean clDice (across linear classes) and mean BF (across polygon classes). It deliberately stays a strict 2-way harmonic mean even when point classes are present — folding in an unbalanced 3rd class would blur the linear-vs-polygon comparison it's designed to make. Because it's a harmonic mean, **CBHM collapses toward zero if either feature type scores near-zero** — a much harsher composite than DTAF1's macro average. The two are deliberately kept as foils: DTAF1 lenient, CBHM harsh.

`cbhm()` also always computes:
- `cldice_mean_weighted` / `bf_mean_weighted` — within-type, GT-pixel-count-weighted means
- **`cbhm_soft`** — a weighted *arithmetic* mean across linear vs. polygon types (weighted by each type's share of total GT pixels)

**`evaluate_all()`** runs DTAF1 + CBHM together, optionally runs `point_f1_multiclass` if `point_classes` is passed (reported as an independent `point_f1_mean` figure, never folded into CBHM's harmonic mean), and surfaces `cbhm_soft`/`dtaf1_weighted` alongside `cbhm`/`dtaf1` for a single consolidated report.

### `apls.py` — APLS (Average Path Length Similarity)

Added 2026-08-01 to close DTAF1's road-breakage blind spot. No vector road graph exists anywhere in this pipeline (`datasets/common.py`'s `rasterize_lines` collapses roads to a flat pixel mask), so this is a **raster-skeleton approximation** of real APLS:

1. Skeletonize pred/GT masks (`skimage.morphology.skeletonize`, same as `cldice.py`).
2. Build an 8-connected pixel-adjacency graph via `networkx`.
3. Sample control-point pairs within the same GT connected component, grouped by source node so a single `nx.single_source_dijkstra_path_length` call amortizes across many pairs instead of one Dijkstra call per pair.
4. Snap-match each GT control point to its nearest pred-graph node via `scipy.spatial.cKDTree`, within a tolerance radius.
5. Score each pair by the GT-vs-pred shortest-path-length ratio (scored 0 if either endpoint is unmatched, or no pred path exists between the matched nodes).

`apls_multiclass()`/`mean_apls()` carry the same `"n_gt"`/`reduction` support as the other multiclass metric wrappers.

### `dtaf1_topo.py` — DTAF1-Topo (connectivity-aware companion to DTAF1)

Additive and non-breaking, like `dtaf1_weighted`/`cbhm_soft` — does **not** modify `dtaf1.py`. `dtaf1_topo()` calls `dtaf1()` unchanged, then for classes listed in `linear_classes` replaces the plain per-class F1 with `harmonic_mean(f1, apls)` before macro/weighted-averaging. Non-linear classes keep their plain F1 (APLS has no defined meaning for polygon/point classes). Returns `dtaf1_topo`/`dtaf1_topo_macro`/`dtaf1_topo_weighted`, plus the complete unmodified `dtaf1()` result under `"dtaf1_result"` for direct side-by-side comparison.

**Validated on synthetic data:** on the synthetic `sweep_road_breakage()` sweep (128×128, 4px road stripe, random pixel dropout), `dtaf1` stays pinned at 1.000 across nearly the entire fraction range, while `dtaf1_topo`/`road_apls` collapse sharply starting around 30% dropout (0.938 → 0.397 → 0.057 → 0.010 at fractions 0.3 / 0.5 / 0.7) — even more aggressively than CBHM, since shortest paths across the fragmenting skeleton either lengthen sharply or disappear entirely once the road disconnects. Confirmed both by `TestRoadBreakageRegression` (`tests/test_metrics_sanity.py`) and visually in section 3f of `metric_comparison.ipynb` (added 2026-08-02).

**Not yet validated on real data** and **not yet wired into `evaluate_all()`/`metrics/__init__.py`** — treat it as a validated-on-synthetic-data experimental addition, not yet a drop-in replacement for `dtaf1` in the consolidated report.

### `__init__.py`

Re-exports the main public API. `apls`/`dtaf1_topo` are intentionally not yet included, pending real-data validation (see above).

---

## Area-Weighting and Zero-Collapse (added 2026-07-12)

`dtaf1`'s default macro average and `cbhm`'s `cldice_mean`/`bf_mean` originally treated every class equally regardless of its GT pixel-area share, and `cbhm`'s harmonic mean collapses entirely to 0 if a single class scores 0 — even a sparse one next to an otherwise-perfect dominant class. `dtaf1_weighted` and `cbhm_soft` are additive companion figures that address this (`dtaf1`/`cbhm` themselves are left unchanged, preserving the harsh-vs-lenient foil pairing and all pre-existing tests).

`cbhm_soft` deliberately uses a weighted **arithmetic** mean, not geometric — a geometric mean still collapses to exactly 0 whenever one input is 0, same as a harmonic mean.

**Caveat found on real data** (`notebooks/composite_vs_submetric_report.ipynb`): area-weighting only helps when the *failing* class is also the pixel-count *minority* within that tile. On `Khartoum_img371` (the tile that motivated this fix), the road actually has more GT pixels than the building despite covering a tiny share of the image's total area, so `cbhm_soft` only partially recovers there (0.000 → 0.285), while it recovers much further (e.g. 0.48 → 0.85) on tiles where the building genuinely dominates by pixel count.

See `TestAreaWeighting`/`TestSparseClassCollapse` in `tests/test_metrics_sanity.py` and the `sweep_class_imbalance`/`sweep_sparse_class_offset` sensitivity sweeps.

---

## Tests (`tests/`)

### `test_metrics_sanity.py` — 50 pytest tests

Covers: perfect/null predictions, class isolation (missing road/building/point), road width insensitivity (IoU vs clDice/DTAF1), road offset tolerance, building erosion, monotonicity, point positional jitter tolerance, point instance matching (including the Hungarian-vs-greedy adversarial case), area-weighted reduction (`TestAreaWeighting`), sparse-class harmonic-mean collapse vs. `cbhm_soft`/`dtaf1_weighted` (`TestSparseClassCollapse`), APLS primitive edge cases (`TestAPLS`), `dtaf1_topo` composite structure and non-mutation of `dtaf1()` (`TestDtaf1Topo`), and the road-breakage regression proving `dtaf1_topo`/`apls` collapse where plain `dtaf1` stays high (`TestRoadBreakageRegression`).

```bash
.venv/Scripts/python.exe -m pytest tests/test_metrics_sanity.py -v
```

### `test_sensitivity.py` — standalone sweep script, also importable as a module

Systematically degrades a synthetic ground-truth mask and records metric values at each severity level. Exports:

| Sweep | What it perturbs |
|---|---|
| `sweep_road_offset()` | Horizontal pixel shift of the road centerline |
| `sweep_road_breakage()` | Random road-pixel deletion; also reports `dtaf1_topo`/`road_apls` alongside `dtaf1`/`cbhm` |
| `sweep_building_erosion()` | Progressive binary erosion of the building mask |
| `sweep_road_thickness()` | Over/under-thick road prediction vs. fixed-width GT |
| `sweep_class_imbalance()` | Road entirely missing, road's pixel-fraction of the image varied |
| `sweep_sparse_class_offset()` | Reproduces the real `Khartoum_img371` CBHM collapse at small scale |
| `sweep_point_jitter()` | Positional jitter of point-feature centroids |
| `sweep_point_dropout()` | Missed point detections |
| `sweep_point_clutter()` | Spurious extra point detections |

```bash
.venv/Scripts/python.exe tests/test_sensitivity.py            # print tables to stdout
.venv/Scripts/python.exe tests/test_sensitivity.py --plot      # also save PNGs to notebooks/figures/
```

---

## Real-Data Acquisition (`datasets/`)

Decoupled from `metrics/` — produces plain `H×W` integer label maps from real public datasets. Instance/centroid extraction stays in `metrics/point_f1.py`, not here.

| File | What it does |
|---|---|
| `spacenet.py` | Pulls SpaceNet (SN2 buildings + SN3 roads) from the public, unauthenticated `spacenet-dataset` S3 bucket. SN2 building tiles (650×650px) and SN3 road tiles (1300×1300px) use different tiling grids even for the same city, so `img{N}` is not the same tile between them — `build_spacenet_sample()` resolves this with a coarse-then-dense nearest-neighbor spatial search over tile centroids, verified against the rasterized pixel count (not just a lon/lat bbox overlap, which can false-positive on merely-touching tiles). |
| `potsdam.py` | Pulls ISPRS Potsdam (6cm/px, has real `Tree`/`Car` classes) via a Kaggle mirror, since SpaceNet has no point-object class. Needs a one-time user-side Kaggle API token (`~/.kaggle/kaggle.json`). Thresholds Potsdam's flat colored label rasters to a target class's palette color, connected-component labels blobs, filters by `min_area`/`max_area`. **Known limitation:** touching instances of the same class merge into one connected component (no watershed splitting). |
| `common.py` | Shared helpers: anonymous HTTPS download, geojson loading, lon/lat → pixel-space reprojection, `rasterize_polygons`/`rasterize_lines` (roads buffered by a physically-motivated width from each segment's `lane_number` property). |

Downloaded/cached data lands under a gitignored `data/` directory (curated sample: up to ~100 SpaceNet tiles across Vegas, Khartoum, Paris, and Shanghai, plus 15 Potsdam crops). No cloud compute needed — everything is a local, curated sample fetched via plain HTTPS. Trained model checkpoints land under a separate gitignored `models/` directory.

---

## Notebooks (`notebooks/`)

| Notebook | Purpose |
|---|---|
| **`metric_comparison.ipynb`** | Synthetic-only demonstration. Builds a 128×128 synthetic scene, runs all sanity checks, runs all sweeps — including the class-imbalance/sparse-class-offset section demonstrating `cbhm_soft`/`dtaf1_weighted`, and (added 2026-08-02) a section 3f rerunning `sweep_road_breakage()` to chart `dtaf1_topo`/`road_apls` collapsing sharply where plain `dtaf1` stays pinned near 1.0 — and renders a colour-coded summary table with pandas `Styler`. Kept synthetic-only by design. |
| **`spacenet_data_prep.ipynb`** / **`potsdam_data_prep.ipynb`** | Build and visually sanity-check the curated real-data samples described in `datasets/` above. Run these first, in this order, before any real-data evaluation notebook — outputs are cached under `data/`, so re-runs are fast/free. |
| **`real_data_evaluation.ipynb`** | Applies the same perturbation families as `test_sensitivity.py`'s synthetic sweeps to real GT (there's no trained extractor used here — "pred" is real GT run through offset/erosion/dropout/thickening/jitter/clutter at fixed severities) across the full curated sample, to check whether real, irregular geometry changes metric behavior compared to idealized synthetic shapes. Each sweep section documents what its perturbations physically simulate (registration drift, network fragmentation, footprint erosion, missed/spurious point detections, etc.) and is followed by a colorized small-multiples chart (metric value vs. severity, mean ± std band across the sample), with a fixed metric→color mapping shared across sections — solid lines are harsh/unweighted `cbhm`/`dtaf1`, dashed lines are the area-weighted `cbhm_soft`/`dtaf1_weighted` companions. |
| **`composite_vs_submetric_report.ipynb`** | Compares each composite metric (CBHM, DTAF1) against its own sub-metrics on the real-data sweep results, flags the largest-divergence cases, and records findings — including the DTAF1/CBHM failure-mode notes above and the `Khartoum_img371` caveat. |
| **`train_unet.ipynb`** | PLEM's first real trained-model pipeline (everything above scores ground truth run through synthetic perturbations, not an actual extractor). Trains a small, CPU-only, single multiclass U-Net (3-level encoder/decoder, base=16 channels, one softmax head over bg/road/building) on 256×256 patches cut from the curated SpaceNet sample, split 70/15/15 train/val/test **at the tile level** (patch-level splitting would leak adjacent patches across the split and inflate held-out scores). Loss is class-weighted cross-entropy + soft Dice. Test-tile predictions are stitched back into full-size label maps and scored with `evaluate_all()`, then a handful of test tiles spanning the score range get 3-panel image/GT-overlay/prediction-overlay qualitative figures. Checkpoints save to gitignored `models/unet_road_building.pt`. Runnable standalone on Colab CPU (added 2026-07-13). **Known limitation:** ~100 curated real tiles with no pretraining is a small dataset for segmentation — treat test-set scores as a pipeline sanity check, not a benchmark result. |

All five real-data notebooks have been run to completion at least once against the actual curated sample (not just written) — see their output cells for the executed results.

---

## Setup & Commands

All commands run from the project root using the `.venv` interpreter.

```bash
# Install dependencies
.venv/Scripts/python.exe -m pip install -r requirements.txt

# torch is CPU-only in this environment (no GPU) — install from the CPU wheel
# index, or a plain `pip install -r requirements.txt` may pull a much larger CUDA build
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Run all sanity/unit tests
.venv/Scripts/python.exe -m pytest tests/test_metrics_sanity.py -v

# Run a single test class or test
.venv/Scripts/python.exe -m pytest tests/test_metrics_sanity.py::TestRoadWidthInsensitivity -v
.venv/Scripts/python.exe -m pytest tests/test_metrics_sanity.py::TestRoadWidthInsensitivity::test_thick_road_iou_low -v

# Run sensitivity sweeps (prints tables to stdout)
.venv/Scripts/python.exe tests/test_sensitivity.py

# Run sweeps and save PNG figures to notebooks/figures/
.venv/Scripts/python.exe tests/test_sensitivity.py --plot

# Launch the synthetic-data comparison notebook
.venv/Scripts/jupyter.exe lab notebooks/metric_comparison.ipynb

# Build the real-data samples (run in this order the first time; cached
# under data/, which is gitignored, so re-runs are fast/free)
.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace notebooks/spacenet_data_prep.ipynb
.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace notebooks/potsdam_data_prep.ipynb   # needs a Kaggle API token, see the notebook
.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace notebooks/real_data_evaluation.ipynb
.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace notebooks/composite_vs_submetric_report.ipynb

# Train and evaluate the small road/building U-Net (CPU-only; run
# spacenet_data_prep.ipynb first). Checkpoints land under gitignored models/.
.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace notebooks/train_unet.ipynb
```

---

## Adding a New Metric

1. Implement in `metrics/<name>.py`; return a dict with at minimum a scalar key matching the metric name.
2. Import and expose in `metrics/__init__.py`.
3. Wire into `evaluate_all()` in `metrics/unified.py` if it should be part of the consolidated report — decide deliberately whether it belongs inside an existing composite's formula (e.g. CBHM's harmonic mean) or should be surfaced as an independent figure alongside it (see how `point_f1_mean` was added as precedent).
4. Add sanity tests in `tests/test_metrics_sanity.py` (perfect → 1.0, null → 0.0, class isolation).
5. Add a sweep function to `tests/test_sensitivity.py` and a notebook cell to `notebooks/metric_comparison.ipynb`.
6. If new dependencies are needed, add them to `requirements.txt`.
7. Commit and push the change.

For the fuller engineering/architecture reference (including exact file-by-file design rationale), see [`CLAUDE.md`](CLAUDE.md).
