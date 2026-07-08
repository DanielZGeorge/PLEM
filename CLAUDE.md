# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**PLEM** (Polygonal-Linear Extraction Metrics) — a research library implementing evaluation metrics for multiclass geospatial segmentation maps that contain both linear features (roads) and polygonal features (buildings). The core problem: IoU is inappropriate for linear features because small positional offsets and width errors collapse the score even when the road centerline is correctly extracted.

All inputs are 2-D NumPy arrays (`H×W`, dtype `uint8`). Label convention: `0` = background, `1` = road (linear), `2` = building (polygonal), `3` = point feature (small discrete objects — trees, lamp posts, manhole covers — spanning only a few pixels; optional, only used where the data actually has a point class). Tolerances are expressed in pixels; tie to ground sample distance (GSD) via `d = physical_metres / GSD_metres_per_pixel`.

## Commands

All commands are run from the project root using the `.venv` interpreter. Dependencies are listed in `requirements.txt` (`.venv/Scripts/python.exe -m pip install -r requirements.txt`).

```bash
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
```

## Architecture

### `metrics/` — the metric library

All public functions take `H×W` integer label maps and return dicts with named scalar fields (never bare scalars), making results easy to log and inspect.

| File | What it implements |
|---|---|
| `dtaf1.py` | **DTAF1** — primary unified metric. Per-class tolerance-radius F1 via `scipy.ndimage.distance_transform_edt`. The `class_config` dict maps `class_id → {name, tolerance}`. Class-agnostic: scoring a point class costs zero code changes, just add its entry to `class_config`. |
| `cldice.py` | **clDice** — centerline Dice for linear features. Skeletonizes both pred and GT via `skimage.morphology.skeletonize`, then measures mutual coverage. Zero-tolerant to positional shift; insensitive to road width. |
| `boundary_f1.py` | **BF score** — boundary F1 for polygonal features. Extracts boundary pixels morphologically, then checks within-tolerance matches using distance transforms. Also exports `iou()` as the baseline comparator. |
| `point_f1.py` | **Point F1** — tolerance-radius *instance* matching for point features (trees, lamp posts, manhole covers). Reduces each connected-component blob to a centroid (`scipy.ndimage.label` + `center_of_mass`), then does tolerance-restricted Hungarian assignment (`scipy.optimize.linear_sum_assignment`) between predicted and GT centroids — not greedy nearest-neighbor, which can lock a predicted blob onto the wrong nearby GT point when points cluster (see `TestPointInstanceMatching` for the adversarial case this avoids). |
| `unified.py` | **CBHM** — harmonic mean of mean clDice (linear classes) and mean BF (polygon classes); deliberately stays 2-way even when point classes are present (see below). **`evaluate_all()`** — runs DTAF1 + CBHM, and optionally `point_f1_multiclass` if `point_classes` is passed, reported as an independent `point_f1_mean` figure. |
| `__init__.py` | Re-exports the main public API. |

**Key design decision:** CBHM uses a harmonic mean so that failure on either feature type collapses the score to zero. DTAF1 uses a macro average, which is more lenient. Both are needed as foils. Point features are deliberately kept out of CBHM's harmonic mean (folding in an unbalanced 3rd class would blur that comparison) — `evaluate_all()` surfaces `point_f1_mean` alongside `cbhm`/`dtaf1`, never inside `cbhm`. Point matching also intentionally differs from DTAF1/clDice/BF: it works on discrete *instances* (via Hungarian assignment), not per-pixel tolerance, because point objects are numerous and pixel-based nearest-neighbor could double-count multiple GT points near one predicted blob.

**Known limitation of DTAF1:** random pixel deletion (road breakage) is under-penalised because scattered remaining pixels still fall within the tolerance radius of GT pixels. Confirmed on real SpaceNet data in `notebooks/composite_vs_submetric_report.ipynb`: DTAF1 stays at 1.0 even with 75% of real road pixels deleted, while CBHM correctly collapses to ~0.57. A connectivity/topology component (APLS or connected-component ratio) would be needed to capture fragmentation. The same notebook also found the opposite failure mode for CBHM: on a real, very sparse (~0.1% road-pixel) tile, a moderate road offset collapsed clDice (and therefore CBHM) to exactly 0 while DTAF1 degraded gracefully — clDice's skeleton-based comparison is brittle on short/sparse road segments. Neither composite is uniformly more trustworthy; check both.

### `tests/`

- `test_metrics_sanity.py` — 33 pytest tests covering: perfect/null predictions, class isolation (missing road/building/point), road width insensitivity (IoU vs clDice/DTAF1), road offset tolerance, building erosion, monotonicity, point positional jitter tolerance, and point instance matching (including the Hungarian-vs-greedy adversarial case).
- `test_sensitivity.py` — standalone sweep script; also importable as a module. Exports `sweep_road_offset()`, `sweep_road_breakage()`, `sweep_building_erosion()`, `sweep_road_thickness()`, `sweep_class_imbalance()`, `sweep_point_jitter()`, `sweep_point_dropout()`, `sweep_point_clutter()` — each returns a dict of lists suitable for plotting.

### `datasets/` — real-data acquisition and rasterization

Parallel to `metrics/`; produces plain `H×W` integer label maps from real public datasets, decoupled from `metrics/` (instance/centroid extraction stays in `point_f1.py`, not here).

| File | What it does |
|---|---|
| `spacenet.py` | Pulls SpaceNet (SN2 buildings + SN3 roads) from the public, unauthenticated `spacenet-dataset` S3 bucket. **Caveat:** SN2 building tiles (650×650px) and SN3 road tiles (1300×1300px) use different tiling grids even for the same city — `img{N}` is not the same tile between them. `build_spacenet_sample()` resolves this with a coarse-then-dense nearest-neighbor spatial search over tile centroids (tile ids are assigned in roughly raster-scan order, verified empirically) rather than downloading every building tile, and verifies the *rasterized pixel count* (not just a lon/lat bbox overlap, which can false-positive on merely-touching tiles) before accepting a match. |
| `potsdam.py` | Pulls ISPRS Potsdam (6cm/px, has real `Tree`/`Car` classes) via a Kaggle mirror, since SpaceNet has no point-object class and at its ~0.3-0.5m/px GSD objects like manholes/lamp posts would be sub-pixel anyway. Needs a one-time user-side Kaggle API token (`~/.kaggle/kaggle.json`) — not scriptable, see `load_kaggle_dataset()`'s docstring. Thresholds Potsdam's flat colored label rasters to a target class's palette color, connected-component labels blobs, filters by `min_area`/`max_area`. **Known limitation:** touching instances of the same class merge into one connected component (no watershed splitting) — real Potsdam "tree" blobs are sometimes contiguous canopy, not isolated point trees. |
| `common.py` | Shared helpers: anonymous HTTPS download, geojson loading, lon/lat → pixel-space reprojection (`rasterio.warp.transform_geom` + inverse affine), `rasterize_polygons`/`rasterize_lines` (roads buffered by a physically-motivated width from each segment's `lane_number` property). |

Downloaded/cached data lands under a gitignored top-level `data/` directory (~1GB for the curated sample used so far: 24 SpaceNet tiles across Vegas + Khartoum, 15 Potsdam crops). No cloud compute needed — everything here is a local, curated sample fetched via plain HTTPS.

### `notebooks/`

- `metric_comparison.ipynb` — synthetic-only demonstration: builds a 128×128 synthetic scene, runs all sanity checks, runs all sweeps, and renders a colour-coded summary table with pandas `Styler`. Kept synthetic-only by design; real-data work lives in the notebooks below.
- `spacenet_data_prep.ipynb` / `potsdam_data_prep.ipynb` — build and visually sanity-check the curated real-data samples (see `datasets/` above).
- `real_data_evaluation.ipynb` — applies the same perturbation families as `test_sensitivity.py`'s synthetic sweeps to real GT (there's no trained extractor in this repo, so "pred" is real GT run through offset/erosion/dropout/thickening/jitter/clutter at fixed severities) across the full curated sample, to see whether real, irregular geometry changes how a metric behaves compared to idealized synthetic shapes. Each sweep section has a markdown cell describing what its perturbations physically simulate (registration drift, network fragmentation, footprint erosion, missed/spurious point detections, etc.) and the severity units used, and is followed by a colorized small-multiples chart (metric value vs. severity, one subplot per perturbation, mean ± std band across the sample) using a fixed metric→color mapping shared across both the road/building and point-feature sections.
- `composite_vs_submetric_report.ipynb` — compares each composite metric (CBHM, DTAF1) against its own sub-metrics on the real-data sweep results, flags the largest-divergence cases, and records findings (see the DTAF1/CBHM failure-mode note above).

All four real-data notebooks were run to completion at least once against the actual curated sample (not just written) — see their output cells for the executed results.

## Adding a new metric

1. Implement in `metrics/<name>.py`; return a dict with at minimum a scalar key matching the metric name.
2. Import and expose in `metrics/__init__.py`.
3. Wire into `evaluate_all()` in `metrics/unified.py` if it should be part of the consolidated report — decide deliberately whether it belongs inside an existing composite's formula (e.g. CBHM's harmonic mean) or should be surfaced as an independent figure alongside it (see how `point_f1_mean` was added as a precedent).
4. Add sanity tests in `tests/test_metrics_sanity.py` (perfect → 1.0, null → 0.0, class isolation).
5. Add a sweep function to `tests/test_sensitivity.py` and a notebook cell to `notebooks/metric_comparison.ipynb`.
6. If new dependencies are needed, add them to `requirements.txt`.
7. Commit and push the change (see "Workflow" below).

## Workflow

After making a change to this repo, update this CLAUDE.md file to reflect it (new modules, changed design decisions, new known limitations discovered), then commit and push to `origin/master`.
