# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**PLEM** (Polygonal-Linear Extraction Metrics) — a research library implementing evaluation metrics for multiclass geospatial segmentation maps that contain both linear features (roads) and polygonal features (buildings). The core problem: IoU is inappropriate for linear features because small positional offsets and width errors collapse the score even when the road centerline is correctly extracted.

All inputs are 2-D NumPy arrays (`H×W`, dtype `uint8`). Label convention: `0` = background, `1` = road (linear), `2` = building (polygonal). Tolerances are expressed in pixels; tie to ground sample distance (GSD) via `d = physical_metres / GSD_metres_per_pixel`.

## Commands

All commands are run from the project root using the `.venv` interpreter.

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

# Launch the comparison notebook
.venv/Scripts/jupyter.exe lab notebooks/metric_comparison.ipynb
```

## Architecture

### `metrics/` — the metric library

All public functions take `H×W` integer label maps and return dicts with named scalar fields (never bare scalars), making results easy to log and inspect.

| File | What it implements |
|---|---|
| `dtaf1.py` | **DTAF1** — primary unified metric. Per-class tolerance-radius F1 via `scipy.ndimage.distance_transform_edt`. The `class_config` dict maps `class_id → {name, tolerance}`. |
| `cldice.py` | **clDice** — centerline Dice for linear features. Skeletonizes both pred and GT via `skimage.morphology.skeletonize`, then measures mutual coverage. Zero-tolerant to positional shift; insensitive to road width. |
| `boundary_f1.py` | **BF score** — boundary F1 for polygonal features. Extracts boundary pixels morphologically, then checks within-tolerance matches using distance transforms. Also exports `iou()` as the baseline comparator. |
| `unified.py` | **CBHM** — harmonic mean of mean clDice (linear classes) and mean BF (polygon classes). **`evaluate_all()`** — convenience wrapper that runs both DTAF1 and CBHM in one call. |
| `__init__.py` | Re-exports the main public API. |

**Key design decision:** CBHM uses a harmonic mean so that failure on either feature type collapses the score to zero. DTAF1 uses a macro average, which is more lenient. Both are needed as foils.

**Known limitation of DTAF1:** random pixel deletion (road breakage) is under-penalised because scattered remaining pixels still fall within the tolerance radius of GT pixels. A connectivity/topology component (APLS or connected-component ratio) would be needed to capture fragmentation.

### `tests/`

- `test_metrics_sanity.py` — 24 pytest tests covering: perfect/null predictions, class isolation (missing road vs missing building), road width insensitivity (IoU vs clDice/DTAF1), road offset tolerance, building erosion, and monotonicity.
- `test_sensitivity.py` — standalone sweep script; also importable as a module. Exports `sweep_road_offset()`, `sweep_road_breakage()`, `sweep_building_erosion()`, `sweep_road_thickness()`, `sweep_class_imbalance()` — each returns a dict of lists suitable for plotting.

### `notebooks/`

`metric_comparison.ipynb` — end-to-end demonstration: builds a 128×128 synthetic scene, runs all sanity checks, runs all four sweeps, and renders a colour-coded summary table with pandas `Styler`.

## Adding a new metric

1. Implement in `metrics/<name>.py`; return a dict with at minimum a scalar key matching the metric name.
2. Import and expose in `metrics/__init__.py`.
3. Wire into `evaluate_all()` in `metrics/unified.py` if it should be part of the consolidated report.
4. Add sanity tests in `tests/test_metrics_sanity.py` (perfect → 1.0, null → 0.0, class isolation).
5. Add a sweep function to `tests/test_sensitivity.py` and a notebook cell to `notebooks/metric_comparison.ipynb`.
