# PLEM: Polygonal Linear Evaluation Metric

*(working subtitle: Tolerance- and Topology-Aware Metrics for Joint Linear/Polygonal Feature*
*Extraction in Geospatial Segmentation)*

> **Status:** Outline only, formatted to mirror the section structure and rigor of the project's
> reference ACM SIGSPATIAL template (`Unified_Learning_for_0D_1D_2D_Map_Features.pdf` — its title
> page, "PLEM: Polygonal Linear Evaluation Metric," already belongs to this paper; its body is
> boilerplate from an unrelated point-encoder draft, reused here only for structure). Sections
> marked *(draft later)* have no prose yet. Reported "final" metrics throughout: **DTAF1-Topo**
> (latest DTAF1 iteration) and **`cbhm_soft`** (latest CBHM iteration). Base `dtaf1` and `cbhm`
> are retained as motivating baselines that expose the two failure modes these iterations fix.
> All numeric values below were computed directly against the live code in this session (see
> the extraction script referenced per-section) rather than carried over from memory or notes.

---

## Front Matter

**Title:** PLEM: Polygonal Linear Evaluation Metric

**Authors:** Daniel George (dzgeorge@usc.edu), John Krumm (jkrumm@usc.edu) — Department of
Computer Science, Viterbi School of Engineering, University of Southern California, Los Angeles,
CA, USA.

**Venue:** Proceedings of the 34th ACM SIGSPATIAL International Conference on Advances in
Geographic Information Systems (SIGSPATIAL 2026), November 3–6, 2026, Riverside, CA.

### Abstract *(~200 words, replaces the reference template's bracketed placeholder)*

Pixel intersection-over-union (IoU) is the wrong evaluation tool for linear features such as
roads: a small centerline offset or a width error collapses the score even when the extracted
road network is geometrically and topologically correct, while the same offset is a reasonable
penalty for a compact polygonal feature such as a building footprint. Evaluating multiclass
geospatial segmentation maps that mix linear and polygonal feature types therefore needs metrics
tolerant of exactly the errors each feature type can absorb, and no more. We introduce **DTAF1**,
a distance-tolerant, per-class F1 score built on tolerance-radius matching via Euclidean distance
transforms that is class-agnostic across road, building, and point classes, and **CBHM**, a
harmonic-mean composite of centerline Dice (clDice) and boundary F1 (BF) deliberately designed as
DTAF1's harsh foil: CBHM collapses to zero if either feature type fails outright, where DTAF1's
macro average is more forgiving. Applying both metrics to real SpaceNet and ISPRS Potsdam imagery
surfaced two failure modes: CBHM's harmonic mean can collapse to exactly zero from a single
sparse, offset class even when a dominant class is scored perfectly, and DTAF1's per-pixel
tolerance matching under-penalizes road breakage, since scattered surviving pixels after heavy
dropout still fall within tolerance of some ground-truth pixel — confirmed directly in this paper
(§4.2): at 33% random road-pixel deletion in a controlled synthetic scene, DTAF1 still reports
**1.000**. We ship an additive, non-breaking fix for each: `cbhm_soft`, a weighted arithmetic
mean across feature types, and **DTAF1-Topo**, which blends in a raster-skeleton approximation of
APLS as a connectivity term for linear classes. We frame this loop — find a real failure, ship a
targeted fix, regression-test it — as a contribution in its own right. The library is validated
with 50 unit tests, 9 synthetic sensitivity sweeps, a curated real-data sample, and a first
end-to-end trained U-Net pipeline scored by the library.

### CCS Concepts

- **Information systems** → Geographic information systems; Evaluation of retrieval results.
- **Theory of computation** → Computational geometry.
- **Computing methodologies** → Neural networks; Image segmentation.

### Keywords

geospatial segmentation evaluation, road extraction, building extraction, tolerance-radius
metrics, topology-aware metrics, centerline Dice, boundary F1, Average Path Length Similarity,
multiclass segmentation

### ACM Reference Format

> Daniel George and John Krumm. 2026. PLEM: Polygonal Linear Evaluation Metric. In *Proceedings of
> the 34th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems
> (SIGSPATIAL 2026)*. ACM, New York, NY, USA, [N] pages. https://doi.org/XXXXXXX.XXXXXXX

---

## 1. Introduction

Segmentation maps of overhead imagery routinely mix feature types with fundamentally different
geometry: **linear** features such as roads, one-dimensional structures extruded to a small
width; **polygonal** features such as building footprints, genuinely two-dimensional regions; and
optionally **point** features — trees, lamp posts, manhole covers — small discrete objects
spanning only a handful of pixels. A single evaluation tile covering a city block can legitimately
contain all three. Yet the standard tool for scoring segmentation quality, pixel IoU, treats every
class identically as a region-overlap problem, and that assumption breaks down specifically for
the linear case: because a road is only a few pixels wide, IoU denominators are dominated by
exactly the pixels most sensitive to sub-pixel-scale registration noise, centerline jitter, and
predicted-width error, none of which reflect whether the road network was actually extracted
correctly.

Table 1 previews five concrete scenarios on a controlled 128×128 synthetic scene (a 3px-wide
vertical road stripe plus a 40×40 building rectangle; full setup in §4.1, `notebooks/
metric_comparison.ipynb`), with values computed directly against the live metrics library for
this paper (script: see §4.2 methodology note). It shows where IoU's verdict diverges from actual
prediction quality, and — importantly — a case where **DTAF1 itself is fooled** even though it
was designed to fix IoU's problem: at 33% random road-pixel deletion, DTAF1 still reports a
perfect **1.000**, motivating DTAF1-Topo in §6.

**Table 1: IoU and DTAF1 failure modes for linear and polygonal features.**

| Scenario | Road IoU | clDice | DTAF1 | CBHM | CBHM-soft | Correct read |
|---|---|---|---|---|---|---|
| Perfect prediction | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | baseline |
| Road shifted 5px | 0.000 | 0.000 | **1.000** | 0.000 | 0.806 | centerline correct; IoU wrongly fails it, CBHM's harmonic mean wrongly fails it too |
| Road 3.3× too thick (10px vs. 3px GT) | 0.281 | 1.000 | 1.000 | 1.000 | 1.000 | centerline correct, only width wrong — IoU alone misleads |
| Road 33% pixels deleted | 0.667 | 0.759 | **1.000** | 0.863 | 0.953 | network is measurably degraded; **DTAF1 misses this entirely** — motivates §6 |
| Building eroded 5px | 0.562† | — | 0.917 | 0.939 | 0.907 | shape genuinely worse; DTAF1/CBHM both degrade proportionally (correct) |
| Sparse road (15px offset, ~0.4% of image) | 0.000 | — | 0.500 | **0.000** | 1.000 | building perfect, road wrong; CBHM's own collapse failure mode (§5) |
| Null prediction | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | baseline |

*†Building IoU, not road IoU, for this row.*

We adopt a simple, uniform data representation throughout: every input is an `H×W` `uint8` NumPy
array of integer class labels, with `0` = background, `1` = road, `2` = building, and an optional
`3` = point feature, used only where the underlying dataset actually has a point class. Every
distance-based tolerance in the library is expressed in pixels but is meant to be tied to a
tile's ground sample distance (GSD): `d = physical_metres / GSD_metres_per_pixel`, so the same
physical tolerance (e.g., "3 metres of road centerline slop") applies consistently across imagery
captured at different resolutions.

This paper makes six contributions:

1. **DTAF1** (§3.1) — a single, class-agnostic tolerance-radius F1 formula spanning linear and
   polygonal classes with one code path; scoring a newly added feature type is a configuration
   change, not a code change.
2. **CBHM** (§3.6) — a harmonic-mean composite of clDice and boundary F1, deliberately designed
   as DTAF1's "harsh" foil, so a single feature type failing outright is not masked by the other
   type succeeding.
3. **`dtaf1_weighted` / `cbhm_soft`** (§5.2) — GT-pixel-count-weighted companions to DTAF1 and
   CBHM that soften single-class collapse, motivated by a real failure case found on SpaceNet
   imagery (`Khartoum_img371`).
4. A raster-only approximation of **APLS** (§3.5) — no vector road graph survives anywhere in
   this pipeline, see §7 — and **DTAF1-Topo** (§6.3), which blends this connectivity term into
   DTAF1 and closes the road-breakage blind spot demonstrated in Table 1.
5. **`point_f1`** (§3.4), a point-feature metric using Hungarian instance matching rather than
   greedy nearest-neighbor assignment, avoiding a specific adversarial failure mode.
6. Empirical validation (§8–§9) spanning nine synthetic sensitivity sweeps, a curated real-data
   sample from SpaceNet (four cities) and ISPRS Potsdam, and a first trained-model pipeline scored
   end-to-end by the library.

The rest of the paper is organized as follows. §2 places this work relative to prior evaluation
metrics *(not yet drafted)*. §3 defines each metric primitive, following the same
"one-subsection-per-primitive" structure as the reference template's encoder definitions. §4
empirically motivates why these primitives are necessary, mirroring the reference's own
motivating experiment. §5 and §6 present two analysis-and-design studies — composite robustness
and topological robustness — mirroring the reference's two "Analysis and Design" sections. §7
describes the implementation. §8–§9 give the full experimental setup and results across synthetic,
real-data, and trained-model settings. §10 discusses limitations honestly. §11 concludes.

---

## 2. Related Work *(draft later — scaffold only, per prior instruction)*

Each subsection below is left as a scaffold with a named citation target, so prose can be dropped
in without restructuring.

- **2.1 IoU-family segmentation metrics** — the region-overlap baseline this paper argues against
  for linear features; standard semantic-segmentation benchmarks (Cityscapes, PASCAL VOC) as the
  incumbent evaluation convention.
- **2.2 Topology-/skeleton-aware metrics for linear structures** — clDice and related centerline
  methods. *Citation target: Shit et al., CVPR 2021.*
- **2.3 Boundary-based metrics for polygonal features** — Boundary F1 / BF score.
  *Citation target: Csurka et al., BMVC 2013.*
- **2.4 Road-network extraction and connectivity metrics** — APLS, TOPO, and the SpaceNet
  road-extraction challenge line of work that motivates §3.5/§6.
  *Citation target: the SpaceNet APLS metric definition.*
- **2.5 Point/instance detection metrics** — COCO-style average precision, and the optimal
  bipartite-matching literature underlying `point_f1`'s Hungarian assignment.
  *Citation target: Kuhn, 1955 (the Hungarian algorithm).*
- **2.6 Composite / multi-task segmentation evaluation** — positions the gap PLEM fills: existing
  work evaluates linear and polygonal feature types with separate, non-comparable metrics rather
  than a single class-agnostic formula (DTAF1) plus a deliberately harsh cross-type composite
  (CBHM).

---

## 3. Metric Definitions

*(Mirrors the reference template's §3 "Encoder Definitions": one subsection per primitive, each
giving the general formula followed by the exact parameters/code used in this paper.)*

**Shared notation.** Let `P` and `G` be `H×W` integer label maps (prediction and ground truth)
over the class set `C = {0, 1, 2, [3]}`. For class `c`, define binary masks `P_c = (P == c)` and
`G_c = (G == c)`. Every primitive in this section is applied per-class and then reduced (macro or
GT-pixel-count-weighted mean) across the classes relevant to it.

**Shared edge-case convention.** Every primitive below implements the same rule, verified
identical across `dtaf1.py:43-51`, `cldice.py:51-56`, `boundary_f1.py:68-72`,
`point_f1.py:109-120`, and `apls.py:141-144`:

```
both P_c and G_c empty        → score = 1.0   (vacuously correct — nothing to find, nothing predicted)
exactly one of P_c, G_c empty → score = 0.0   (total failure — false positives or a total miss)
```

Stated once here as a cross-cutting design decision rather than repeated per metric.

### 3.1 DTAF1 — Distance-Tolerant, per-class F1

Tolerance-radius matching via Euclidean distance transforms (`scipy.ndimage.distance_transform_edt`):

```
dist_from_gt(x)   = EDT(1 - G_c)(x)      # distance from x to nearest GT-positive pixel
dist_from_pred(x) = EDT(1 - P_c)(x)      # distance from x to nearest predicted-positive pixel

TP_pred = |{ x in P_c : dist_from_gt(x)   <= d_c }|
TP_gt   = |{ x in G_c : dist_from_pred(x) <= d_c }|

Precision_c = TP_pred / |P_c|
Recall_c    = TP_gt   / |G_c|
F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
```

Reduction across classes (both are always computed; `reduction` just selects which is surfaced as
the top-level `dtaf1` key):

```
DTAF1_macro    = (1/|C|) * sum_c F1_c
DTAF1_weighted = ( sum_c n_c * F1_c ) / ( sum_c n_c ),   n_c = |G_c|
```

**Code** (`metrics/dtaf1.py:53-64`, the tolerance-matching core):

```python
dist_from_gt = distance_transform_edt(1 - g)
dist_from_pred = distance_transform_edt(1 - p)

tp_pred = int(((p == 1) & (dist_from_gt <= d)).sum())
tp_gt = int(((g == 1) & (dist_from_pred <= d)).sum())

precision = tp_pred / n_pred
recall = tp_gt / n_gt
f1 = (2 * precision * recall / (precision + recall)
      if (precision + recall) > 0 else 0.0)
```

And the macro/weighted reduction (`metrics/dtaf1.py:123-127`):

```python
f1_scores = [r["f1"] for r in per_class.values()]
macro_score = float(np.mean(f1_scores))
weights = np.array([r["n_gt"] for r in per_class.values()], dtype=float)
total = weights.sum()
weighted_score = float(np.dot(weights, f1_scores) / total) if total > 0 else 0.0
```

`DEFAULT_TOLERANCES = {"road": 10, "building": 2}` (pixels; see §5.3 for how these were chosen).
Function signature: `dtaf1(pred, gt, class_config, reduction="macro") -> dict`, returning
`dtaf1`, `dtaf1_macro`, `dtaf1_weighted`, `per_class` (per-class `precision/recall/f1/tp_pred/
tp_gt/n_pred/n_gt/name/tolerance`). **Class-agnostic by construction** — scoring a new class costs
zero code changes, just an added `class_config` entry.

### 3.2 clDice — Centerline Dice

*(cite Shit et al., CVPR 2021, in §2 — not here)*

```
S_p = skeletonize(P_c),  S_g = skeletonize(G_c)      # skimage.morphology.skeletonize
T_prec = |S_p ∩ G_c| / |S_p|      # topology precision
T_sens = |S_g ∩ P_c| / |S_g|      # topology sensitivity
clDice = 2 * T_prec * T_sens / (T_prec + T_sens)
```

**Code** (`metrics/cldice.py:58-64`):

```python
tprec = float((skel_p & g).sum() / n_skel_p)
tsens = float((skel_g & p).sum() / n_skel_g)

denom = tprec + tsens
cl = float(2 * tprec * tsens / denom) if denom > 0 else 0.0
```

Width-insensitive (a predicted road 3.3× too thick still scores `clDice = 1.000`, Table 1 row 2)
but offset-brittle: skeletons of the exact same road shifted 5px share **zero** overlap
(`clDice = 0.000`, Table 1 row 1) — this offset-sensitivity is exactly what motivates DTAF1's
distance-tolerant matching instead of a raw skeleton intersection.

### 3.3 Boundary F1 (BF) and the IoU Baseline

*(cite Csurka et al., BMVC 2013, in §2 — not here)*

```
B_p = boundary(P_c),  B_g = boundary(G_c)     # morphological boundary extraction
Precision = |{x in B_p : dist(x, B_g) <= tau}| / |B_p|
Recall    = |{x in B_g : dist(x, B_p) <= tau}| / |B_g|
BF = 2 * Precision * Recall / (Precision + Recall)
```

**Code** (`metrics/boundary_f1.py:79-85`):

```python
precision = float((bp & (dist_from_gt_boundary <= tolerance)).sum() / n_bp)
recall = float((bg & (dist_from_pred_boundary <= tolerance)).sum() / n_bg)
denom = precision + recall
bf = float(2 * precision * recall / denom) if denom > 0 else 0.0
```

`iou()` is introduced here explicitly as the pixel-overlap baseline comparator used throughout
§4 and §8 (`metrics/boundary_f1.py:138-144`):

```python
def iou(pred, gt):
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter / union) if union > 0 else 1.0
```

### 3.4 Point F1 — Hungarian Instance Matching

For the optional point class (trees, lamp posts, manhole covers): connected-component blobs are
reduced to centroids (`scipy.ndimage.label` + `center_of_mass`), then matched via
tolerance-restricted Hungarian assignment (`scipy.optimize.linear_sum_assignment`), not greedy
nearest-neighbor:

```
cost(i,j) = dist(gt_i, pred_j)              if dist <= tolerance
cost(i,j) = tolerance * 1e6 + 1              otherwise (discouraged, not forbidden — keeps a full assignment)
assignment = linear_sum_assignment(cost)     # Hungarian algorithm
TP = |{ assigned pairs (i,j) : dist(gt_i, pred_j) <= tolerance }|
Precision = TP / n_pred,  Recall = TP / n_gt,  F1 = harmonic mean of Precision, Recall
```

**Code** (`metrics/point_f1.py:58-65`):

```python
dist = cdist(gt_c, pred_c)
big = tolerance * 1e6 + 1.0
cost = np.where(dist <= tolerance, dist, big)
row_ind, col_ind = linear_sum_assignment(cost)
valid = dist[row_ind, col_ind] <= tolerance
return int(valid.sum())
```

Justified by the clustered-points adversarial case in `TestPointInstanceMatching`: greedy
nearest-neighbor gets `TP=1` where Hungarian finds the globally optimal `TP=2` (formalized in
Appendix A.6).

### 3.5 APLS — Raster-Skeleton Average Path Length Similarity

No vector road graph survives anywhere in this pipeline — `datasets/common.py::rasterize_lines`
collapses roads straight to a flat pixel mask, discarding segment/graph structure — so APLS is
approximated entirely from raster skeletons (`metrics/apls.py`):

```
skeletonize(P_c), skeletonize(G_c)        # skimage.morphology.skeletonize
build an 8-connected pixel-adjacency graph (networkx); edge weight = Euclidean pixel distance

for sampled GT control-point pairs (u, v) within one GT connected component:
    L_gt(u,v)   = shortest_path_length_G(u, v)
    u', v'      = nearest pred-graph nodes to u, v within tolerance (scipy.spatial.cKDTree)
    L_pred(u,v) = shortest_path_length_P(u', v')     if both snap-matched and a path exists
    score(u,v)  = max(0, 1 - |L_pred(u,v) - L_gt(u,v)| / L_gt(u,v))   if matched, else 0

APLS_c = mean over sampled pairs of score(u,v)
```

Control-point pairs are sampled **grouped by source** (`_sample_control_pairs`), so one
`nx.single_source_dijkstra_path_length` call amortizes across many targets instead of one
independent Dijkstra call per pair — see complexity analysis in Appendix B.6.

**Code** (`metrics/apls.py:170-185`, the grouped-Dijkstra path-ratio scoring loop):

```python
scores, n_matched, n_sampled = [], 0, 0
for source, targets in grouped_pairs:
    gt_dist = nx.single_source_dijkstra_path_length(gt_graph, source, weight="weight")
    pu = _nearest_within_tolerance(pred_tree, pred_nodes, source, tolerance)
    pred_dist = (nx.single_source_dijkstra_path_length(pred_graph, pu, weight="weight")
                 if pu is not None else None)
    for target in targets:
        n_sampled += 1
        gt_len = gt_dist[target]
        pv = _nearest_within_tolerance(pred_tree, pred_nodes, target, tolerance)
        if pred_dist is None or pv is None or pv not in pred_dist:
            scores.append(0.0)
            continue
        pred_len = pred_dist[pv]
        n_matched += 1
        scores.append(max(0.0, 1.0 - abs(pred_len - gt_len) / gt_len) if gt_len > 0 else 0.0)
```

This is deliberately sensitive to a failure DTAF1's per-pixel tolerance matching misses entirely
(quantified in §4.2/Table 1): once the skeleton fragments, shortest paths between control points
either lengthen sharply or vanish, so APLS collapses where DTAF1 does not.

### 3.6 CBHM — Composite clDice/Boundary-F1 Harmonic Mean

```
clDice_mean = mean( clDice_c for c in linear_classes )
BF_mean     = mean( BF_c     for c in polygon_classes )
CBHM = 2 * clDice_mean * BF_mean / (clDice_mean + BF_mean)     if denom > 0, else 0
```

**Code** (`metrics/unified.py:77-81`):

```python
cl_mean = float(np.mean(cldice_scores)) if cldice_scores else 0.0
bf_mean = float(np.mean(bf_scores)) if bf_scores else 0.0
denom = cl_mean + bf_mean
score = float(2 * cl_mean * bf_mean / denom) if denom > 0 else 0.0
```

The harmonic mean is a **deliberate design choice**: a single class scoring 0 collapses the whole
composite to 0. Positioned explicitly as the "harsh" foil to DTAF1's more lenient macro average
(Table 1's "Road shifted 5px" row: DTAF1 = 1.000 but CBHM = 0.000, on the *same* input).

**Composition note.** These six primitives compose into the two "final iteration" composites
covered later rather than here — `cbhm_soft` in §5.2 and DTAF1-Topo in §6.3 — exactly as the
reference template introduces its two "simplified encoders" in its own later analysis sections
rather than alongside the base encoder definitions.

---

## 4. Why These Metrics? Empirical Motivation

*(Mirrors the reference template's §4, "Why Encode Points?": a controlled experiment showing the
baseline tool fails, followed by an interpretation of why.)*

### 4.1 Experimental Setup

All numbers in this section come from the 128×128 synthetic scene defined in
`notebooks/metric_comparison.ipynb`: a 3px-wide vertical road stripe (class 1) through the tile
center, plus a 40×40 building rectangle (class 2) in the upper-left quadrant. Nine sweep functions
in `tests/test_sensitivity.py` perturb this scene along one axis at a time:

| Sweep | Perturbs | Range used |
|---|---|---|
| `sweep_road_offset` | horizontal road shift | 0–24px |
| `sweep_road_breakage` | random road pixel dropout (seed 42) | 0–100% |
| `sweep_building_erosion` | building erosion radius | 0–11px |
| `sweep_road_thickness` | predicted road thickness (GT fixed 3px) | 1–21px |
| `sweep_class_imbalance` | road:building GT pixel-area ratio, road entirely missing | 0.4%–3.1% road-pixel share |
| `sweep_sparse_class_offset` | sparse road (~0.4% of image) offset | 0–20px |
| `sweep_point_jitter` | horizontal point shift | 0–20px |
| `sweep_point_dropout` | fraction of predicted points removed | 0–100% |
| `sweep_point_clutter` | spurious predicted points added | 0–20 |

### 4.2 Results

**Table 1 (§1) is reproduced from a direct run of the library** against this scene (not
hand-computed or estimated) — see the summary table cell in `metric_comparison.ipynb` §4 for the
executed reference and the companion script used to re-verify these exact numbers for this paper.
Three additional single-axis sweeps sharpen the picture beyond Table 1's discrete scenarios:

**Road offset sweep** (`sweep_road_offset`, tolerance = 10px): Road IoU and clDice both collapse
to `0.000` immediately at offset = 4px (any offset larger than the road's own 3px width already
destroys pixel/skeleton overlap), while DTAF1 stays at `1.000` through offset = 10px exactly at
its tolerance boundary, then degrades — `0.667` at 12px, `0.500` from 16px onward (reflecting the
building's still-perfect class dragging the macro average, since `1.0 + 0.0` over 2 classes
averages to `0.5` once road F1 hits exactly 0).

**Road breakage sweep** (`sweep_road_breakage`, seed 42): Road IoU degrades roughly linearly with
dropout fraction (`1.000 → 0.500` at 50% → `0.201` at 80%), clDice degrades faster and less evenly
(skeleton connectivity breaks unevenly as pixels vanish), but **DTAF1 remains pinned at exactly
`1.000` for every dropout fraction from 0% through 90%**, only dropping (to `0.500`, again the
building-only floor) once 100% of road pixels are gone. This is the blind spot in its starkest
form: DTAF1 cannot distinguish a perfectly intact road from one with 90% of its pixels randomly
deleted, because the ~10% of pixels that remain are scattered widely enough that *some* surviving
pixel still falls within the 10px tolerance radius of nearly every GT pixel.

**Road thickness sweep** (`sweep_road_thickness`, GT fixed at 3px): Road IoU falls from `1.000`
at the correct thickness to `0.157` at 21px predicted thickness, while both clDice and DTAF1 stay
at `1.000` through 13px and only start softening at very extreme over-thickness (clDice `0.694` at
17px) — confirming width-insensitivity holds over a wide practical range.

**Building erosion sweep** (`sweep_building_erosion`): all three metrics degrade together and
roughly proportionally with erosion radius (Building IoU `1.000→0.203`, BF `1.000→0.512`, DTAF1
`1.000→0.728` at radius 11) — the one scenario in this section where IoU's verdict was never wrong
to begin with, included as a control.

### 4.3 Interpretation via Distance-Transform Geometry

*(Mirrors the reference template's NTK-based interpretation subsection: explains* why *the
observed behavior occurs, not just that it occurs.)*

DTAF1's tolerance radius `d_c` defines, for every pixel, a disc of admissible positional error:
a predicted pixel counts as correct if *any* GT pixel of the same class lies within `d_c`, and
vice versa for recall. This is precisely why width and small-offset errors are absorbed (§4.2):
every pixel of an over-thick or slightly-shifted road still has some true road pixel within
`d_c`, so precision and recall both stay near 1. The same geometry is also precisely why DTAF1 is
blind to breakage: the tolerance radius has no notion of *which* surviving pixels are connected to
each other — it only asks "is there a same-class pixel nearby," a purely local, per-pixel
question. Random dropout removes pixels roughly uniformly, so even at high dropout fractions the
*remaining* pixels tend to still lie within `d_c` of most GT locations, exactly as a spatial
Poisson-thinning argument would predict: for a road of width `w` and tolerance `d_c ≫ w`, the
probability that some surviving pixel lies within `d_c` of a given GT pixel stays high until the
surviving fraction becomes small relative to `w / d_c`. clDice does encode a *global* connectivity
signal (via `skeletonize`), which is why it degrades faster and less evenly than DTAF1 under the
same dropout sweep — but clDice's zero-tolerance to any positional offset (§3.2) makes it unusable
as DTAF1's replacement outright. This tension — a tolerance radius that must be wide enough to
absorb positional/width noise but is then necessarily blind to the connectivity question at that
same radius — is the structural reason a *separate* connectivity term (APLS, §3.5) is needed
rather than simply shrinking `d_c`; §6 develops this formally.

---

## 5. Analysis and Design for Composite Robustness

*(Mirrors the reference template's §5, "Analysis and Design for Inference": a controlled-
complexity task family, an efficient corrective proxy, then a parameter-design application.)*

### 5.1 Class-Imbalance–Controlled Tasks

`sweep_class_imbalance` and `sweep_sparse_class_offset` parameterize a controlled-complexity axis
analogous to the reference template's frequency-parameterized task family: instead of varying a
task's spatial frequency `β`, we vary the road's *share of GT pixels* (0.4%–3.1% in
`sweep_class_imbalance`, with the road entirely missing from the prediction) or its *offset*
(`sweep_sparse_class_offset`, road share fixed at ~0.4%). In both sweeps, `cbhm` (harsh, harmonic)
sits at exactly `0.000` for every non-trivial perturbation level — confirmed directly:
`road_frac ∈ {0.0039, ..., 0.0312}` all give `cbhm = 0.000`, and `offset ∈ {2, 4, ..., 20}` all
give `cbhm = 0.000` — while `cbhm_soft` tracks the dominant (building) class instead, staying
between `0.758` and `1.000` across the same ranges. As a scalar summary analogous to the
reference's frequency capacity `β_max(τ)`, define the **harsh-composite collapse threshold**: the
smallest perturbation magnitude at which `cbhm` first reaches exactly 0. Empirically this
threshold is effectively immediate (any nonzero road offset, or any road-area share tested down to
0.4%) — `cbhm`'s harmonic mean has essentially zero tolerance once one class is fully wrong,
regardless of how small that class's share of the scene is.

### 5.2 Weighted Reduction as an Efficient Corrective Proxy

```
clDice_mean_weighted = ( sum_i n_i * clDice_i ) / ( sum_i n_i )    over linear classes i
BF_mean_weighted     = ( sum_j n_j * BF_j )     / ( sum_j n_j )    over polygon classes j

w_linear  = N_linear  / (N_linear + N_polygon)     # share of total GT pixels that are linear
w_polygon = N_polygon / (N_linear + N_polygon)     # (0.5 / 0.5 if both totals are 0)

cbhm_soft = w_linear * clDice_mean_weighted + w_polygon * BF_mean_weighted
```

**Code** (`metrics/unified.py:88-102`):

```python
cl_mean_weighted = (
    float(np.dot(cldice_weights, cldice_scores) / cl_total) if cl_total > 0 else 0.0
)
bf_mean_weighted = (
    float(np.dot(bf_weights, bf_scores) / bf_total) if bf_total > 0 else 0.0
)
type_total = cl_total + bf_total
if type_total > 0:
    w_cl = float(cl_total / type_total)
    w_bf = float(bf_total / type_total)
else:
    w_cl = w_bf = 0.5
cbhm_soft = w_cl * cl_mean_weighted + w_bf * bf_mean_weighted
```

The key structural move, and the analogue of the reference template's KRR-as-cheap-proxy
argument: `cbhm_soft` is a **reweighting of already-computed per-class scores**, not a
redesign of the harmonic-mean composite itself. It costs nothing beyond the per-class scores
DTAF1/CBHM already compute, exactly as the reference used a closed-form regression surrogate to
avoid retraining a neural network for every candidate encoder parameter. **Stated carefully, not
oversold**: `cbhm_soft` is a weighted *arithmetic* mean, not geometric — a geometric mean still
collapses to exactly 0 whenever one input is 0, same as harmonic (formalized in Appendix A.3–A.4).

**Worked example — the real failure that motivated this fix.** On real SpaceNet imagery
(`composite_vs_submetric_report.ipynb`, tile `Khartoum_img371`), a 12px road offset gave
`cbhm = 0.000` despite `dtaf1 = 0.934` (per-class road F1 = 0.867, precision = 0.872,
recall = 0.863) — the road is only ~0.1% of image pixels, and clDice collapsed entirely on this
sparse, offset road, dragging the harmonic mean to exactly zero even though the dominant building
class was essentially perfect. `cbhm_soft` recovers only partially here (**0.000 → 0.285**),
because the road actually has *more* GT pixels than the building despite its tiny share of image
*area* — area-weighting only helps when the failing class is also the pixel-count minority.
Contrast `Khartoum_img333`, where the building genuinely dominates by pixel count:
`cbhm` 0.48–0.52 → `cbhm_soft` **0.85**, a clean recovery. This nuance is treated as a real
limitation, not a clean win — expanded in §10.

### 5.3 Choosing Per-Class Tolerance Radii

`DEFAULT_TOLERANCES = {"road": 10, "building": 2}` (pixels) were validated, not guessed, against
the offset and erosion sweeps in §4.2: the road offset sweep shows DTAF1 holding at `1.000`
through exactly the 10px boundary and softening immediately past it (§4.2), confirming the
tolerance behaves as intended rather than being either so tight it rejects correct predictions or
so loose it never penalizes anything. This mirrors the reference template's "efficient design of
simplified encoders" in spirit — using a cheap, already-available sweep as the design proxy
instead of an expensive full retrain (or, here, a full real-data re-annotation) for every
candidate tolerance value — though PLEM's tolerances are fixed physically-motivated constants
(§1's `d = physical_metres / GSD_metres_per_pixel`) rather than a learned/optimized parameter;
§10 lists learned/adaptive tolerance radii as future work.

---

## 6. Analysis and Design for Topological Robustness

*(Mirrors the reference template's §6, "Analysis and Design for Decoding": a structural property
the primitive lacks, a sensitivity analysis, then a closed-form-plus-numerical design section.)*

### 6.1 The Connectivity Blind Spot

Where the reference template's §6.1 *proves* a property its encoders *have* (injectivity), this
subsection formally states a property DTAF1 **lacks**: pointwise EDT tolerance-matching is not
"connectivity-injective." Concretely (proof sketch; full statement in Appendix A):

> For a road mask `G_c` of width `w` and tolerance `d_c`, there exist predicted masks `P_c`
> obtained from `G_c` by deleting an arbitrarily large fraction of pixels — up to and including
> masks in which `G_c`'s connected skeleton is fully disconnected into many isolated fragments —
> such that `DTAF1_c(P_c, G_c) = 1.0`, provided the surviving pixels remain within `d_c` of every
> GT pixel.

§4.2's road-breakage sweep is the empirical witness: DTAF1 stays at exactly `1.000` through 90%
random dropout on the synthetic scene. This is symmetric to, not merely inspired by, the reference
template's treatment of injectivity: there, the question was "can two distinct inputs share an
encoding" (answered no, by construction, for every encoder studied); here, the question is "can
two topologically distinct predictions (connected vs. fully fragmented) share a score" (answered
yes, by construction, for DTAF1 alone — motivating §6.2's fix).

### 6.2 APLS as Connectivity Recovery, and Its Sensitivity to Fragmentation

APLS (§3.5) recovers exactly the signal DTAF1 lacks: its score depends on shortest-path length
*between* control points, not merely on point-to-point proximity, so it is sensitive to whether a
path exists at all. Framing `sweep_road_breakage`'s dropout fraction as the "noise" axis (the
structural analogue of the reference template's additive-encoding-noise experiment, though APLS
has no Jacobian to derive a first-order prediction from — the analogy is the *shape* of the
sensitivity curve, not the math), the empirical curve has the same two-regime character the
reference found for its encoders under noise: a near-flat, low-sensitivity regime while the
skeleton is still mostly connected, followed by a sharp departure once fragmentation actually
occurs. Measured directly against the live code:

| Dropout fraction | `dtaf1` | `cbhm` | `road_apls` | `dtaf1_topo` |
|---|---|---|---|---|
| 0.0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.1 | 1.000 | 0.965 | 0.898 | 0.946 |
| 0.2 | 1.000 | 0.919 | 0.883 | 0.938 |
| 0.3 | 1.000 | 0.909 | **0.247** | **0.397** |
| 0.5 | 1.000 | 0.805 | 0.029 | 0.057 |
| 0.7 | 1.000 | 0.606 | 0.005 | 0.010 |
| 1.0 | 0.500 | 0.000 | 0.000 | 0.000 |

The transition is sharp between 0.2 and 0.3: `road_apls` falls from `0.883` to `0.247` in that one
step — the point at which the road skeleton's largest connected fragment actually detaches from
enough control-point pairs that shortest paths start disappearing rather than merely lengthening.
`dtaf1` shows no analogous transition anywhere in the swept range.

### 6.3 Harmonic-Blending Design for DTAF1-Topo

*(Mirrors the reference template's §6.3, "Sensitivity-Based Encoder Design," which had one
closed-form parameter-optimization case and one numerical-sweep case.)*

**Code** (`metrics/dtaf1_topo.py:29-31`, the blending primitive, and `:74-87`, the per-class loop):

```python
def _harmonic_mean(a: float, b: float) -> float:
    denom = a + b
    return float(2 * a * b / denom) if denom > 0 else 0.0

# per class in linear_classes:
entry["blended"] = _harmonic_mean(base["f1"], a)   # a = apls_detail[cls_id]["apls"]
# non-linear classes keep entry["blended"] = base["f1"] unchanged (APLS undefined there)
```

```
blended_c = HarmonicMean(F1_c, APLS_c)     if c is a linear class
blended_c = F1_c                            otherwise
DTAF1-Topo_macro / DTAF1-Topo_weighted = same reduction as DTAF1, applied to blended_c
```

#### 6.3.1 Closed-Form Boundary Behavior

The harmonic mean of `F1` and `APLS` has a simple, exactly-analyzable shape at its boundaries —
the closed-form analogue of the reference template's constant-Jacobian trigonometric-encoder case
(§6.3.1 there):

- `APLS = 0, F1 = 1` (the DTAF1 blind-spot regime from §6.1, worst case): `blended = 0`. A
  perfectly tolerance-matched but fully disconnected road scores exactly zero once blended,
  regardless of how high `F1` is.
- `APLS = 1, F1 = 1`: `blended = 1`, recovering DTAF1's own perfect score unchanged.
- For fixed `F1 = 1`, `blended(APLS) = 2·APLS / (1 + APLS)` — concave and always `≤ APLS`, meaning
  the harmonic blend never *rewards* connectivity beyond what APLS itself reports; it can only
  ever penalize a topologically-correct-looking `F1` down toward the (generally lower) APLS score,
  never the reverse. This asymmetry is intentional: it is the mechanism by which DTAF1-Topo closes
  the blind spot rather than merely averaging two independent opinions.

#### 6.3.2 Numerical Validation via the Road-Breakage Sweep

Unlike the closed-form boundary analysis in §6.3.1, the *practical* question — where along the
dropout axis does the blend actually diverge from plain DTAF1? — has no closed form and is
answered numerically, exactly as the reference template's DoG-encoder width parameter `σ` was
optimized by sweeping rather than solving in closed form (its §6.3.2). Table in §6.2 already gives
the answer directly: `dtaf1_topo` tracks `road_apls` closely (both driven to near-zero by fraction
0.5–0.7) while `dtaf1` stays at `1.000` throughout — this *is* the fix, demonstrated on the same
sweep that exposed the blind spot in §4.2/§6.1, and cross-checked against the assertions in
`tests/test_metrics_sanity.py::TestRoadBreakageRegression`. **Validated on synthetic data only** —
see §10 for the real-data caveat (`dtaf1_topo`/`apls` have not yet been re-run on the 24-tile real
SpaceNet sample used elsewhere in §9).

---

## 7. Implementation

- All inputs/outputs are plain NumPy `H×W` `uint8` arrays — no framework lock-in.
- Every public function returns a **dict of named scalars**, never a bare float, for
  logging/inspection.
- Tech stack, grouped by role:

  | Role | Libraries |
  |---|---|
  | Core numerics | `numpy`, `scipy` (`scipy.ndimage` for distance transforms/erosion/labeling) |
  | Skeletonization/morphology | `scikit-image` |
  | Graph algorithms | `networkx` (APLS shortest paths), `scipy.spatial.cKDTree` (snap-matching) |
  | Optimal assignment | `scipy.optimize.linear_sum_assignment` (Point F1) |
  | Geospatial I/O | `rasterio`, `shapely` |
  | Data acquisition | `requests` (anonymous S3), `kaggle` (Potsdam mirror) |
  | Trained-model pipeline | `torch` (CPU-only) |
  | Testing/analysis | `pytest`, `pandas`, `matplotlib`, `jupyterlab` |

- Package structure/exports note: `dtaf1_topo`, `apls`/`apls_multiclass`/`mean_apls` are **not**
  re-exported at the `metrics/__init__.py` top level; `cbhm_soft` exists only as a dict key inside
  `cbhm()`'s return, not a separate function — deliberate, framed as "validated on synthetic data
  only, not yet a drop-in replacement in the consolidated report" (`evaluate_all()`, expanded in
  §10).

---

## 8. Experimental Setup

### 8.1 Synthetic Benchmark

128×128 scene: vertical road stripe (class 1) + 40×40 building rectangle (class 2), from
`notebooks/metric_comparison.ipynb`. The nine sweep functions listed in §4.1
(`tests/test_sensitivity.py`).

### 8.2 Real-Data Sample

SpaceNet SN2 (buildings, 650×650px) + SN3 (roads, 1300×1300px) from the public unauthenticated S3
bucket, cities Vegas/Khartoum/Paris/Shanghai, up to ~100 tiles curated, the two tiling grids
reconciled via coarse-then-dense nearest-neighbor spatial matching, verified by rasterized pixel
overlap (not just bbox overlap). Plus ISPRS Potsdam (6cm/px) for the point class (trees), 15 crops
via a Kaggle mirror, thresholded by palette color + connected-component labeled.

No trained extractor was available for most of this evaluation, so "pred" = real GT run through
the same perturbation families as the synthetic sweeps (`notebooks/real_data_evaluation.ipynb`),
at fixed severities:

- `road_offset` ∈ {0, 4, 8, 12, 16} px
- `road_dropout` ∈ {0, 25, 50, 75}%
- `road_thicken` ∈ {0, 2, 4, 6} px
- `building_erode` ∈ {0, 1, 2, 3} px
- `point_jitter` ∈ {0, 3, 6, 10, 15} px
- `point_dropout` ∈ {0, 25, 50, 75, 100}%
- `point_clutter` ∈ {0, 5, 10, 15, 20} extra blobs

### 8.3 Trained Model (First Real Prediction Source)

Small 3-level encoder/decoder U-Net, base=16 channels doubling per level, single softmax head over
`{bg, road, building}`, **483,475 parameters**. 100 SpaceNet tiles split **70/15/15 at the tile
level** (avoids patch-leakage inflating held-out scores), patchified to 256px
(2520 train / 540 val patches). Loss = class-weighted cross-entropy + soft Dice; class weights
`[0.1, 1.313, 1.602]` for (bg, road, building), from pixel counts
`[147742496, 9565097, 7843127]`. CPU-only, 15 epochs. *(Citation target for the base architecture:
Ronneberger et al., 2015.)*

**Limitation stated explicitly here**: "~100 curated real tiles with no pretraining is a small
dataset for segmentation; treat as a pipeline sanity check, not a benchmark result."

---

## 9. Results

### 9.1 Synthetic Sensitivity Results

Table 1 (§1) and the extended sweep results in §4.2/§6.2 are this section's synthetic results,
computed directly against the live library for this paper rather than reused from prior notes.

### 9.2 Real-Data Case Studies

- **Finding #1 — DTAF1 blind spot, confirmed on real data.** Averaged across 24 real SpaceNet
  tiles (Vegas + Khartoum), 75% road-pixel dropout leaves plain DTAF1 at **1.000** while CBHM
  collapses to **~0.57** (driven by `cldice_mean` falling to ~0.40). `dtaf1_weighted` does **not**
  fix this — also stays at 1.000, since area-weighting only changes how per-class scores combine,
  not the per-pixel tolerance-matching logic within a class. **Open item**: `dtaf1_topo` has not
  yet been re-validated on this real 24-tile sample (synthetic-only so far, §6.3.2) — flagged as
  future work (§10/§11) rather than reporting a real-data number that doesn't exist yet.
- **Finding #2 — CBHM sparse-class collapse, `Khartoum_img371` case study.** Detailed in §5.2:
  12px road offset → `cbhm = 0.000` vs. `dtaf1 = 0.934`; `cbhm_soft` recovers only to **0.285**
  because the road has more GT pixels than the building despite its tiny area share.
- **Contrast case (`Khartoum_img333`)**: `cbhm` 0.48–0.52 → `cbhm_soft` **0.85** — a clean win
  where the building genuinely dominates by pixel count.
- **Aggregate table** across all 24 tiles: `road_offset`@16px (`cbhm` 0.669 → `cbhm_soft` 0.712),
  `road_dropout`@0.75 (`cbhm` 0.568 → `cbhm_soft` 0.646).
- **Point-feature results** on 15 Potsdam crops: `point_dropout` (precision flat at 1.000, recall
  falls linearly to 0), `point_clutter` (recall flat at 1.000, precision 1.0 → 0.36 → 0.13 at
  5/20 spurious blobs), `point_jitter` (drops to ~0.38 by 6px on real data — faster than the
  synthetic sweep, attributed to real objects being packed closer together).

### 9.3 Applied Pipeline: Trained U-Net Results

Full test-set table (15 held-out tiles, mean ± std):

| Metric | mean | std |
|---|---|---|
| `cbhm` | 0.316 | 0.126 |
| `cbhm_soft` | 0.378 | 0.130 |
| `dtaf1` | 0.440 | 0.134 |
| `dtaf1_weighted` | 0.476 | 0.141 |
| `cldice_mean` | 0.409 | 0.193 |
| `bf_mean` | 0.308 | 0.156 |

Best tile `Shanghai_img1375` (`cbhm` = 0.617), worst `Shanghai_img1711` (`cbhm` = 0.163). Framed
explicitly as "first end-to-end validation that the metric library scores an actual trained
model's output sensibly," not as a SOTA benchmark claim.

---

## 10. Discussion & Limitations

- DTAF1's road-breakage blind spot (§4.2, §6.1) is fixed by DTAF1-Topo (§6.3), but DTAF1-Topo is
  validated on **synthetic data only** (§6.3.2) — not yet re-run on the real 24-tile sample
  (§9.2's Finding #1), and not yet wired into `evaluate_all()`.
- `cbhm_soft` (§5.2) only helps when the failing class is also the pixel-count minority; when a
  class sparse by *area* is not sparse by *pixel count* (`Khartoum_img371`), the fix is partial.
- clDice is brittle on short/sparse real road segments (§3.2) — moderate offsets can collapse it
  to exactly 0 even when DTAF1 degrades gracefully. Neither composite is uniformly more
  trustworthy — recommend reporting `cbhm`, `cbhm_soft`, `dtaf1`, and `dtaf1_weighted` together,
  not picking one.
- U-Net results (§9.3) are a pipeline sanity check (small dataset, no pretraining, CPU-only), not
  a benchmark result — do not oversell.
- APLS (§3.5) here is a raster-skeleton approximation, not true vector-graph APLS, because no
  vector road graph survives the `rasterize_lines` step anywhere in the pipeline (§7).
- The harmonic-blend asymmetry proved in §6.3.1 (DTAF1-Topo can only ever pull a score *down*
  toward APLS, never up) is intentional, but means DTAF1-Topo inherits every one of APLS's own
  failure modes (e.g., a genuinely correct but very short/sparse road segment with too few
  sampled control-point pairs to estimate APLS reliably) — not yet characterized empirically.

---

## 11. Conclusion & Future Work

Restate the core contribution: a class-agnostic, tolerance/topology-aware metric library that
found and fixed two real failure modes through an explicit synthetic-sweep → real-data validation
→ targeted additive fix → regression-test loop (§4→§9 of this paper, end to end).

Future work:

- Wire `dtaf1_topo` into `evaluate_all()` and `metrics/__init__.py` after real-data validation
  (§9.2's Finding #1, §10).
- Re-run `dtaf1_topo`/`apls` on the real 24-tile SpaceNet sample already used for every other
  real-data finding in §9.2, closing the "synthetic-only" caveat in §6.3.2/§10.
- Train on the full SpaceNet corpus with GPU/pretraining (§8.3/§9.3).
- Extend the point-feature pipeline to more Potsdam classes (cars).
- Explore learned/adaptive tolerance radii instead of fixed per-class constants (§5.3).
- Characterize DTAF1-Topo's inherited APLS failure modes on short/sparse real road segments
  (§10's last point) with a dedicated sensitivity sweep, mirroring `sweep_sparse_class_offset`'s
  treatment of CBHM's analogous weakness (§5.1).

---

## Acknowledgments

*(Mirrors the reference template's own disclosure norm — it explicitly states "parts of the text
and parts of the code for experiments & plotting in this paper were generated with ChatGPT and
carefully checked by the author," with the author accepting responsibility for correctness.)*
Parts of the implementation, notebooks, and this outline were developed with the assistance of
Claude Code, and were checked and are the responsibility of the author(s).

---

## References *(empty — matches §2 being left blank; six anticipated citations)*

- Shit et al., "clDice — a Novel Topology-Preserving Loss Function for Tubular Structure
  Segmentation," CVPR 2021. *(§2.2, §3.2)*
- Csurka et al., "What is a good evaluation measure for semantic segmentation?," BMVC 2013
  (Boundary F1 / BF score). *(§2.3, §3.3)*
- SpaceNet challenge / APLS metric paper. *(§2.4, §3.5, §8.2)*
- ISPRS Potsdam benchmark paper. *(§8.2)*
- Kuhn, H. W., "The Hungarian method for the assignment problem," 1955. *(§2.5, §3.4)*
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," 2015.
  *(§8.3, §9.3)*

---

## Appendix A: Formal Properties and Proof Sketches

*(Mirrors the reference template's Appendix A, "Injectivity of Encoders": one short proof per
primitive, matching the exact parameterizations used in §3.)*

### A.1 Shared Edge-Case Correctness

**Claim:** every primitive in §3 returns `1.0` when both `P_c` and `G_c` are empty, and `0.0` when
exactly one is empty.

**Proof sketch:** verified by direct inspection of the identical guard pattern in each
implementation — `dtaf1.py:43-51` (`n_pred == 0 and n_gt == 0` → all of precision/recall/f1 = 1.0;
either alone zero → all 0.0), `cldice.py:51-56`, `boundary_f1.py:68-72` (same shape, on skeleton
and boundary pixel counts respectively), `point_f1.py:109-120` (on instance counts), and
`apls.py:141-144` (on foreground pixel counts, prior to any graph construction). Since every
downstream composite (CBHM, DTAF1-Topo, `cbhm_soft`) is built from weighted means and harmonic
means of these primitives, this shared base case is what makes "both classes vacuously correct" a
well-defined `1.0` at every level of the library, not just at the leaves.

### A.2 DTAF1 Boundedness and Reduction Equivalence

**Claim:** `dtaf1_macro`, `dtaf1_weighted` ∈ `[0, 1]` always; and `dtaf1_weighted = dtaf1_macro`
when GT pixel counts are equal across all classes in `class_config`.

**Proof sketch:** each per-class `F1_c` is itself a harmonic mean of two quantities in `[0,1]`
(§3.1), hence `F1_c ∈ [0,1]`; both the unweighted mean (`dtaf1.py:124`) and the pixel-count-
weighted mean (`dtaf1.py:125-127`) are convex combinations of the `F1_c` values, hence also in
`[0,1]`. When every class's `n_c` (GT pixel count) is equal, the weights `n_c / sum(n_c)` reduce
to `1/|C|` for every class, making the weighted sum in `dtaf1.py:127` numerically identical to the
unweighted mean in `dtaf1.py:124`.

### A.3 CBHM Zero-Collapse

**Claim:** `cbhm = 0` if and only if `cldice_mean = 0` or `bf_mean = 0` (or both).

**Proof sketch:** directly from `unified.py:80-81` — `score = 2*cl_mean*bf_mean/denom if denom > 0
else 0.0`. If either `cl_mean` or `bf_mean` is exactly 0, the numerator `2*cl_mean*bf_mean` is 0
while `denom = cl_mean + bf_mean` is the other (generally nonzero) term, so `score = 0/denom = 0`
exactly — regardless of how close to 1 the other mean is. Conversely, if both means are strictly
positive, both numerator and denominator are strictly positive, so `score > 0`. This is the formal
statement behind Table 1's "Road shifted 5px" row (`clDice = 0` forces `CBHM = 0` even though
`DTAF1 = 1.000` on the identical input).

### A.4 `cbhm_soft` Non-Collapse

**Claim:** `cbhm_soft = 0` only if *both* `cl_mean_weighted = 0` and `bf_mean_weighted = 0`
(unlike CBHM's single-sided collapse in A.3).

**Proof sketch:** from `unified.py:102`, `cbhm_soft = w_cl * cl_mean_weighted + w_bf *
bf_mean_weighted` is a weighted arithmetic mean with non-negative weights summing to 1 (or 0.5/0.5
when both totals are 0, `unified.py:96-100`). A weighted arithmetic mean of two non-negative terms
is 0 only if every term with positive weight is itself 0. Since `w_cl, w_bf ≥ 0` and at least one
is strictly positive whenever any GT pixels exist, `cbhm_soft = 0` requires whichever of
`cl_mean_weighted`/`bf_mean_weighted` has positive weight to be 0 — and if only one type is
present (the other has zero GT pixels and thus zero weight), `cbhm_soft` simply equals that one
type's weighted mean, never forced toward 0 by the *other*, absent type. This is the mechanism
behind the `Khartoum_img371`/`Khartoum_img333` contrast in §5.2: `cbhm_soft` can only be dragged
down by a failing class in proportion to that class's own pixel-count weight, not collapsed
outright by it.

### A.5 APLS Score Boundedness

**Claim:** every per-pair APLS score, and hence the aggregate `apls` score, lies in `[0, 1]`.

**Proof sketch:** each per-pair score is either `0.0` (unmatched, `apls.py:181`) or
`max(0.0, 1.0 - abs(pred_len - gt_len) / gt_len)` (`apls.py:185`) — the outer `max(0.0, ...)`
clamps the lower bound, and the term inside is `≤ 1.0` whenever `pred_len ≥ 0` (shortest-path
lengths are non-negative by construction), so every individual score lies in `[0,1]`. The
aggregate `apls` score is `np.mean(scores)` (`apls.py:188`), a convex combination of values in
`[0,1]`, hence itself in `[0,1]`.

### A.6 Point F1: Hungarian Optimality vs. Greedy Suboptimality

**Claim:** there exist configurations of GT/predicted point centroids where greedy nearest-
neighbor matching strictly under-counts true positives relative to the Hungarian assignment used
in `point_f1` (`point_f1.py:58-65`).

**Proof sketch (formalizing `TestPointInstanceMatching`'s adversarial case):** consider two GT
points `g1, g2` and one predicted point `p` positioned such that `dist(g1, p) < dist(g2, p)` but a
*second* predicted point `p'` exists with `dist(g2, p') < tolerance` while `dist(g1, p') >
tolerance`. A greedy matcher processing predictions in an order that assigns `p → g1` first (the
globally nearer pair) can leave `g2` unmatched to `p'` if it was already "claimed" — or, in the
canonical clustered-points case, greedy assigns the single nearest GT to *both* predictions'
shared nearest neighbor, yielding `TP=1`. The Hungarian assignment instead minimizes total
assignment cost over the *complete* bipartite graph simultaneously (`linear_sum_assignment` on the
full cost matrix, `point_f1.py:63`), correctly recovering the one-to-one pairing `p↔g1, p'↔g2`
that gives `TP=2` — strictly more true positives from the same input, because it considers all
pairings jointly rather than greedily committing to the locally-best match first.

---

## Appendix B: Algorithms and Computational Complexity

*(Mirrors the reference template's Appendix B, "Decoding and Jacobians": one subsection per
primitive. PLEM's primitives have no decoding-Jacobian analogue, since they are not invertible
encodings — this appendix instead gives the per-primitive computational complexity, which plays
the same "intrinsic property independent of any particular task" role the reference's Jacobians
played for its encoders.)*

### B.1 General Framework and Notation

Let `N = H×W` be the pixel count of one tile, and (where relevant) `k` the number of point-class
instances in a scene (typically tens to low hundreds, `k ≪ N`).

### B.2 DTAF1

Two calls to `scipy.ndimage.distance_transform_edt`, each `O(N)` (the exact Euclidean distance
transform algorithm used by SciPy runs in linear time in the number of pixels via a two-pass
separable algorithm); TP/precision/recall computation is a constant number of `O(N)` boolean-mask
reductions. Total per class: `O(N)`.

### B.3 clDice

`skimage.morphology.skeletonize` (Lee's thinning algorithm) is near-linear in `N`; the subsequent
set intersections (`skel_p & g`, `skel_g & p`) are `O(N)`. Total per class: `O(N)`.

### B.4 Boundary F1

Morphological boundary extraction (`binary_dilation` with a small structuring element) is `O(N)`;
two `distance_transform_edt` calls are `O(N)` each as in B.2. Total per class: `O(N)`.

### B.5 Point F1

Connected-component labeling (`scipy.ndimage.label`) is `O(N)`. The subsequent Hungarian
assignment (`linear_sum_assignment`) operates on a `k × k` cost matrix and runs in `O(k³)` — cheap
in practice since `k` is an *instance* count, not a pixel count (§B.1), making this the one
primitive in the library whose dominant cost term is independent of `N` for realistic point-class
densities.

### B.6 APLS

Skeletonization is `O(N)` as in B.3; building the 8-connected adjacency graph is a single pass
over skeleton pixels, `O(N)`. Control-point pairs are sampled grouped by source
(`_sample_control_pairs`, §3.5) specifically so that `nx.single_source_dijkstra_path_length` — run
once per unique sampled source at `O((V+E) log V)` where `V, E` are the skeleton graph's node/edge
counts — is amortized across many targets per source rather than paying one independent Dijkstra
call (`O((V+E) log V)` each) per sampled *pair*. With `n_pairs` total pairs grouped into
`O(√n_pairs)` sources (the default sampling policy, `apls.py:73-74`), this reduces total Dijkstra
work from `O(n_pairs · (V+E) log V)` (naive, one call per pair) to `O(√n_pairs · (V+E) log V)`.
cKDTree snap-matching (`_nearest_within_tolerance`) is `O(log V)` per query after an `O(V log V)`
tree build.

### B.7 DTAF1-Topo / `cbhm_soft`

Both composites add no new asymptotic cost beyond summing their already-analyzed constituent
primitives' costs: DTAF1-Topo is `dtaf1()` (§B.2, `O(N)` per class) plus `apls_multiclass()`
(§B.6) plus an `O(|linear_classes|)` harmonic-mean blending pass — no additional raster-level
work. `cbhm_soft` is a handful of extra `O(1)`-per-class weighted-sum arithmetic operations on
scores CBHM already computed (§3.6) — strictly cheaper than either primitive it reweights.
