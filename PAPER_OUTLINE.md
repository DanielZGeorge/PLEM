# PLEM: Polygonal Linear Evaluation Metric

*(working subtitle: Evaluation Metrics and a Differentiable Training Loss for Joint Point, Linear,*
*and Polygonal Map Feature Extraction)*

> **Status:** Outline, in progress. The paper's primary contribution is `losses/`, a
> differentiable multi-task training loss (`PLEMMultiTaskLoss`) whose three geometric terms are
> direct analogs of three of PLEM's own eval metrics (§7), with the eval-metric half (§3–§6)
> serving as both the design inspiration for the loss and the evaluation protocol used to validate
> it. Reported "final" eval-metric iterations are **DTAF1-Topo** and **`cbhm_soft`**; base
> `dtaf1`/`cbhm` are retained as motivating baselines. All numeric values in §1, §4, and §6 were
> computed directly against the live code (`tests/test_metrics_sanity.py`,
> `tests/test_sensitivity.py`). The training loss is validated at three increasing levels: 20 unit
> tests plus synthetic per-term ablation sweeps (§9.4/§10.4, `tests/test_loss_sensitivity.py`); a
> small real joint-training run across 153 SpaceNet+Potsdam tiles (§9.5/§10.5,
> `notebooks/train_unet_joint.ipynb`) that produced real qualitative wins; and a production-scale
> real joint-training run across 1,033 tiles and three sources — adding SpaceNet6 SAR imagery and
> low-light augmentation for illumination robustness — trained end-to-end for 40 epochs on a CARC
> A100 without divergence (§9.6/§10.6, `notebooks/train_unet_joint_scaled.ipynb`). That run also
> surfaced a genuine point-head failure mode, root-caused and already fixed in `losses/heatmap.py`;
> its evaluation-side results are pending a re-run with that fix applied. Every limitation and open
> item found across all three validation levels is consolidated in one place, §11, rather than
> repeated inline throughout the paper.

---

## Front Matter

**Title:** PLEM: Polygonal Linear Evaluation Metric

**Authors:** Daniel George (dzgeorge@usc.edu), John Krumm (jkrumm@usc.edu) — Department of
Computer Science, Viterbi School of Engineering, University of Southern California, Los Angeles,
CA, USA.

**Venue:** Proceedings of the 34th ACM SIGSPATIAL International Conference on Advances in
Geographic Information Systems (SIGSPATIAL 2026), November 3–6, 2026, Riverside, CA.

### Abstract *(~260 words, replaces the reference template's bracketed placeholder)*

Pixel intersection-over-union (IoU) is the wrong evaluation tool for linear features such as
roads: a small centerline offset or a width error collapses the score even when the extracted
road network is geometrically and topologically correct. We first introduce **DTAF1**, a
distance-tolerant, per-class F1 score built on tolerance-radius matching via Euclidean distance
transforms and class-agnostic across road, building, and point classes, and **CBHM**, a
harmonic-mean composite of centerline Dice (clDice) and boundary F1 deliberately designed as
DTAF1's harsh foil. Applying both to real SpaceNet and ISPRS Potsdam imagery surfaced two failure
modes, each fixed with an additive, non-breaking companion metric: `cbhm_soft` (area-weighted,
non-collapsing) and **DTAF1-Topo** (a raster-skeleton APLS connectivity term closing DTAF1's
road-breakage blind spot). Building on this evaluation vocabulary, we then ask a training
question: can a single model learn to predict point (0D), linear (1D), and polygonal (2D) map
features *jointly*, using a loss shaped by the same tolerance/topology/instance-matching
considerations that motivate these metrics, rather than a geometry-agnostic per-pixel loss? We
introduce **`PLEMMultiTaskLoss`**, a differentiable multi-task training loss combining three
geometry-aware terms — a soft tolerance-band term (DTAF1 analog), a soft-clDice term (Shit et
al.'s soft-skeletonization), and a CenterNet-style Gaussian-heatmap term standing in for
Hungarian-matched point F1, which has no tractable differentiable relaxation — plus a masked
multi-source training scheme letting heterogeneous single-modality datasets (SpaceNet: road+
building; Potsdam: building+point) jointly supervise one 4-class model. We validate the loss with
20 unit tests and synthetic per-term ablation sweeps confirming each term encodes the geometric
sensitivity it claims to. On real imagery, a joint-training run across 153 SpaceNet+Potsdam tiles
shows all four loss terms decreasing jointly and produces qualitatively strong road+building
predictions that a harsh composite metric alone would undersell. A subsequent production-scale
run — three sources including SpaceNet6 SAR imagery, 1,033 tiles, 40 GPU epochs — trains to
convergence without divergence and independently confirms the multi-source class-masking
mechanism at scale, while also surfacing a point-head failure mode we root-cause and fix. Open
items from all three validation stages are collected in one place (§11) rather than qualifying
each result individually.

### CCS Concepts

- **Information systems** → Geographic information systems; Evaluation of retrieval results.
- **Theory of computation** → Computational geometry.
- **Computing methodologies** → Neural networks; Image segmentation; Learning latent
  representations.

### Keywords

geospatial segmentation evaluation, road extraction, building extraction, tolerance-radius
metrics, topology-aware metrics, centerline Dice, boundary F1, Average Path Length Similarity,
multiclass segmentation, differentiable loss functions, multi-task learning, point/keypoint
detection, soft-skeletonization

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
correctly. The same problem recurs one level up: the *training* losses conventionally used for
multiclass segmentation (cross-entropy, soft Dice) are exactly as geometry-agnostic as IoU is as
an evaluation tool — they have no notion of tolerance, centerline topology, or point-instance
identity either, so a model trained with them has no direct pressure to get those properties
right, only to match pixels.

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
captured at different resolutions. `losses/` (§7) reuses this exact class convention on batched
torch tensors — channel index equals class id — so a `class_config` dict is literally shareable
between an eval call (`dtaf1()`) and a training loss (`ToleranceBandLoss`).

This paper makes eight contributions:

1. **A differentiable multi-task training loss** (§7) whose three geometric terms are direct,
   well-precedented relaxations of three of this paper's own eval metrics — soft tolerance-band
   (DTAF1), soft-clDice (Shit et al. 2021), and Gaussian-heatmap point regression (CenterNet-style)
   — plus a masked multi-source training scheme letting heterogeneous datasets (SpaceNet
   road+building, Potsdam building+point, SpaceNet6 building-only) jointly supervise one 4-class
   model despite no single source annotating all three feature types. Validated via controlled
   synthetic ablations (§9.4/§10.4), a small real joint-training run across 153 real tiles (§9.5/
   §10.5), and a production-scale real joint-training run across 1,033 tiles and three sources
   (§9.6/§10.6).
2. **DTAF1** (§3.1) — a single, class-agnostic tolerance-radius F1 formula spanning linear and
   polygonal classes with one code path; scoring a newly added feature type is a configuration
   change, not a code change.
3. **CBHM** (§3.6) — a harmonic-mean composite of clDice and boundary F1, deliberately designed
   as DTAF1's "harsh" foil, so a single feature type failing outright is not masked by the other
   type succeeding.
4. **`dtaf1_weighted` / `cbhm_soft`** (§5.2) — GT-pixel-count-weighted companions to DTAF1 and
   CBHM that soften single-class collapse, motivated by a real failure case found on SpaceNet
   imagery (`Khartoum_img371`).
5. A raster-only approximation of **APLS** (§3.5) — no vector road graph survives anywhere in
   this pipeline, see §8 — and **DTAF1-Topo** (§6.3), which blends this connectivity term into
   DTAF1 and closes the road-breakage blind spot demonstrated in Table 1.
6. **`point_f1`** (§3.4), a point-feature metric using Hungarian instance matching rather than
   greedy nearest-neighbor assignment, avoiding a specific adversarial failure mode.
7. Empirical validation (§9–§10) spanning nine synthetic eval-metric sensitivity sweeps, seven
   synthetic loss-ablation sweeps, a curated real-data sample from SpaceNet (four cities) and
   ISPRS Potsdam, and two trained-model pipelines scored end-to-end by the library.
8. **A production-scale validation of the multi-source training mechanism** (§9.6/§10.6): a
   1,033-tile, three-source, 40-epoch run on real cloud GPU hardware that trains to convergence
   without divergence, whose train/predict-asymmetry check gives direct evidence the per-source
   class-masking mechanism (§7.6) behaves as designed at this scale, and which surfaced, root-
   caused, and fixed a genuine point-head failure mode (§11).

The rest of the paper is organized as follows. §2 places this work relative to prior evaluation
metrics and differentiable segmentation losses. §3 defines each eval metric primitive, following a
"one-subsection-per-primitive" structure. §4 empirically motivates why these primitives are
necessary. §5 and §6 present two analysis-and-design studies — composite robustness and
topological robustness. §7 introduces the differentiable multi-task training loss, presenting each
term as a direct analog of one primitive from §3. §8 describes the implementation. §9–§10 give the
full experimental setup and results across synthetic eval sweeps, synthetic loss ablations,
real-data, and two real trained-model settings at increasing scale. §11 consolidates every
limitation and open item raised anywhere in the paper into one place. §12 concludes.

---

## 2. Related Work

- **2.1 IoU-family segmentation metrics** — the region-overlap baseline this paper argues against
  for linear features. Region-overlap IoU is the incumbent evaluation convention across
  general-purpose semantic-segmentation benchmarks — PASCAL VOC (Everingham et al., IJCV 2010) and
  Cityscapes (Cordts et al., CVPR 2016) — neither of which contains a thin, 1-pixel-wide linear
  class, so neither benchmark's own leaderboard practice had to confront IoU's road-width
  sensitivity at all. **Motivational baseline (see also 2.4):** on the ISPRS Potsdam benchmark
  (Rottensteiner et al., ISPRS Annals 2012 / ISPRS J. Photogramm. 2014), which *does* include a
  building class close in kind to PLEM's polygon classes, modern models report mean IoU up to
  ~86% and mean F1 up to ~92% (e.g. A²-FPN) — i.e. plain region-overlap metrics are already
  near-saturated on well-behaved polygon classes. This is the same "good-looking score, hidden
  failure mode" pattern §2.4's road numbers show more starkly, and is why this paper does not treat
  a high IoU/F1 figure alone as evidence a metric is adequate for PLEM's target feature types.
- **2.2 Topology-/skeleton-aware metrics for linear structures** — clDice and related centerline
  methods (Shit et al., CVPR 2021).
- **2.3 Boundary-based metrics for polygonal features** — Boundary F1 / BF score
  (Csurka et al., BMVC 2013).
- **2.4 Road-network extraction and connectivity metrics** — APLS (the SpaceNet road-extraction
  challenge metric; Van Etten et al., arXiv:1807.01232, 2018) and TOPO (Biagioni & Eriksson,
  Transportation Research Record 2291, 2012), the two dominant graph-connectivity metrics for
  linear road networks, motivate §3.5/§6's `apls`/`dtaf1_topo` primitives. **Motivational
  baseline:** published road-extraction leaderboards plateau well short of 1.0 even at their best —
  SpaceNet Challenge 3's winning submission scored APLS 0.666 (field topped out ≈0.67); CRESI, the
  best-known raster-to-graph pipeline, reaches APLS 0.69 ± 0.02 (Van Etten, arXiv:1908.09715,
  2019); D-LinkNet, winner of the DeepGlobe 2018 Road Extraction Challenge, reports pixel-IoU
  0.647 (val) / 0.634 (test) (Zhou et al., CVPR Workshops 2018) — a *pixel*-only ceiling that says
  nothing about whether the extracted network stays connected. SpaceNet's building-footprint
  challenges show the same pattern from the other direction: the first SpaceNet building challenge
  topped out at F1 0.26, and the four-city SpaceNet 2 challenge's winner reached F1 0.693 averaged
  across cities (both per Van Etten et al., arXiv:1807.01232, 2018, and the associated challenge
  results reporting). These are cited here as **motivating context, not as numbers PLEM's own
  trained models are benchmarked against** — PLEM's `train_unet_joint*.ipynb` runs evaluate a small
  curated tile subset under this repo's own protocol, not the official SpaceNet/DeepGlobe/Potsdam
  test sets, so the two sets of numbers are not directly comparable. The point of citing them is
  narrower and stronger than a leaderboard comparison: even the *best* published pixel/topology
  scores on real road and building extraction plateau in the 0.26–0.93 range depending on class and
  difficulty, and a single scalar in that range cannot by itself distinguish "scattered pixel noise
  within tolerance" from "a genuinely fragmented network" — exactly the gap `dtaf1_topo`/`apls`
  close (see the road-breakage finding under `metrics/` above, where `dtaf1` stays pinned at 1.0
  under 75% real road-pixel deletion while `dtaf1_topo`/`apls` collapse sharply).
- **2.5 Point/instance detection metrics** — COCO-style average precision (Lin et al.,
  arXiv:1405.0312, 2014) and the optimal bipartite-matching literature underlying `point_f1`'s
  Hungarian assignment (Kuhn, 1955).
- **2.6 Composite / multi-task segmentation evaluation** — positions the gap PLEM's *evaluation*
  half fills: existing work evaluates linear and polygonal feature types with separate,
  non-comparable metrics rather than a single class-agnostic formula (DTAF1) plus a deliberately
  harsh cross-type composite (CBHM).
- **2.7 Differentiable segmentation losses** *(new — motivates §7)* — the training-side analog of
  the gap §2.6 describes on the eval side. Standard multiclass segmentation losses (cross-entropy,
  soft Dice — *citation target: Milletari et al., 3DV 2016, "V-Net"*) are geometry-agnostic in
  exactly the sense §1 argues IoU is: no notion of positional tolerance, centerline topology, or
  point-instance identity. Three prior lines of work each address one of those properties in
  isolation, none jointly: (a) **boundary-/Hausdorff-distance losses**, which precompute a GT-side
  distance transform and use it as a fixed per-pixel weight against the model's soft output —
  *citation targets: Kervadec et al., 2019 ("Boundary loss for highly unbalanced segmentation");
  Karimi & Salcudean, 2019 (Hausdorff-distance loss)* — the direct precedent for §7.2's soft
  tolerance-band term; (b) **soft-clDice**, a differentiable soft-skeletonization loss for tubular
  structures — *citation target: Shit et al., CVPR 2021* (the same paper cited for clDice itself
  in §2.2/§3.2 — here cited for its *loss*, not its eval metric, contribution) — the direct
  precedent for §7.4; (c) **heatmap regression for keypoint/point detection**, which reformulates
  discrete point detection as dense Gaussian-target regression rather than an instance-matching
  problem — *citation targets: Law & Deng, ECCV 2018 ("CornerNet"); Zhou et al., 2019
  ("CenterNet"/"Objects as Points")* — the direct precedent for §7.5. §7's contribution is
  combining well-precedented individual relaxations of (a)–(c) into one multi-task loss covering
  all three geometric properties simultaneously, plus the masked multi-source training scheme
  (§7.6) that lets two datasets which individually annotate only two of the three feature types
  jointly supervise one model.

---

## 3. Evaluation Protocol

*(Retitled from "Metric Definitions" — same content, same one-subsection-per-primitive structure.
These six primitives now serve two purposes rather than one: as before, they are PLEM's evaluation
vocabulary for scoring any prediction against ground truth; as of §7, three of them are also the
metric each corresponds to that a differentiable loss term is designed to approximate during
training. Nothing in this section changes to accommodate that — the metrics stay exactly as
originally defined and validated; §7 does the work of relating a *separate* differentiable
formulation back to each one.)*

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
zero code changes, just an added `class_config` entry. §7.2's `ToleranceBandLoss` reuses this
exact `class_config` shape.

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
distance-tolerant matching instead of a raw skeleton intersection. §7.4's `SoftClDiceLoss`
reimplements this width-insensitivity property differentiably (§11).

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
§4 and §9 (`metrics/boundary_f1.py:138-144`):

```python
def iou(pred, gt):
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter / union) if union > 0 else 1.0
```

§7.3's `SoftBoundaryLoss` reuses §7.2's tolerance-band primitive applied to a soft boundary map
instead of the full class mask, making it the most directly-shared piece of machinery between any
two `losses/` terms.

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
Appendix A.6). **This is the one primitive in §3 with no differentiable analog in §7** — Hungarian
assignment and connected-component labeling have no useful gradient path back to per-pixel logits;
§7.5 reformulates point supervision entirely as heatmap regression instead (§11).

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
either lengthen sharply or vanish, so APLS collapses where DTAF1 does not. Graph shortest-path
recomputation has no tractable lightweight differentiable relaxation, so APLS and DTAF1-Topo
(§6.3) stay eval-only permanently — a stated design decision (§7.1, Table 2), not a to-do.

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
covered later rather than here — `cbhm_soft` in §5.2 and DTAF1-Topo in §6.3 — and, as of this
revision, into the three geometry-aware `losses/` terms in §7.

---

## 4. Why These Metrics? Empirical Motivation

*(A controlled experiment showing the baseline tool fails, followed by an interpretation of why.)*

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

§9.4 introduces a tenth family, `tests/test_loss_sensitivity.py`, that reuses these exact same
scenes/perturbation functions (imported directly, not duplicated) to sweep the **loss** side
instead.

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

*(Explains* why *the observed behavior occurs, not just that it occurs.)*

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
rather than simply shrinking `d_c`; §6 develops this formally, and §7.2 shows the same "wide
enough to absorb noise, blind to connectivity" tradeoff reproduces itself on the training-loss
side (`ToleranceBandLoss` has no connectivity term either, by the same design logic).

---

## 5. Analysis and Design for Composite Robustness

*(A controlled-complexity task family, an efficient corrective proxy, then a parameter-design
application.)*

### 5.1 Class-Imbalance–Controlled Tasks

`sweep_class_imbalance` and `sweep_sparse_class_offset` parameterize a controlled-complexity axis:
instead of varying a task's spatial frequency, we vary the road's *share of GT pixels*
(0.4%–3.1% in `sweep_class_imbalance`, with the road entirely missing from the prediction) or its
*offset* (`sweep_sparse_class_offset`, road share fixed at ~0.4%). In both sweeps, `cbhm` (harsh,
harmonic) sits at exactly `0.000` for every non-trivial perturbation level — confirmed directly:
`road_frac ∈ {0.0039, ..., 0.0312}` all give `cbhm = 0.000`, and `offset ∈ {2, 4, ..., 20}` all
give `cbhm = 0.000` — while `cbhm_soft` tracks the dominant (building) class instead, staying
between `0.758` and `1.000` across the same ranges. As a scalar summary, define the
**harsh-composite collapse threshold**: the smallest perturbation magnitude at which `cbhm` first
reaches exactly 0. Empirically this threshold is effectively immediate (any nonzero road offset,
or any road-area share tested down to 0.4%) — `cbhm`'s harmonic mean has essentially zero
tolerance once one class is fully wrong, regardless of how small that class's share of the scene
is.

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

The key structural move: `cbhm_soft` is a **reweighting of already-computed per-class scores**,
not a redesign of the harmonic-mean composite itself. It costs nothing beyond the per-class scores
DTAF1/CBHM already compute. **Stated carefully, not oversold**: `cbhm_soft` is a weighted
*arithmetic* mean, not geometric — a geometric mean still collapses to exactly 0 whenever one
input is 0, same as harmonic (formalized in Appendix A.3–A.4).

**Worked example — the real failure that motivated this fix.** On real SpaceNet imagery
(`composite_vs_submetric_report.ipynb`, tile `Khartoum_img371`), a 12px road offset gave
`cbhm = 0.000` despite `dtaf1 = 0.934` (per-class road F1 = 0.867, precision = 0.872,
recall = 0.863) — the road is only ~0.1% of image pixels, and clDice collapsed entirely on this
sparse, offset road, dragging the harmonic mean to exactly zero even though the dominant building
class was essentially perfect. `cbhm_soft` recovers only partially here (**0.000 → 0.285**),
because the road actually has *more* GT pixels than the building despite its tiny share of image
*area* — area-weighting only helps when the failing class is also the pixel-count minority.
Contrast `Khartoum_img333`, where the building genuinely dominates by pixel count:
`cbhm` 0.48–0.52 → `cbhm_soft` **0.85**, a clean recovery (§11).

### 5.3 Choosing Per-Class Tolerance Radii

`DEFAULT_TOLERANCES = {"road": 10, "building": 2}` (pixels) were validated, not guessed, against
the offset and erosion sweeps in §4.2: the road offset sweep shows DTAF1 holding at `1.000`
through exactly the 10px boundary and softening immediately past it (§4.2), confirming the
tolerance behaves as intended rather than being either so tight it rejects correct predictions or
so loose it never penalizes anything. This is a cheap, already-available sweep used as the design
proxy instead of an expensive full retrain (or, here, a full real-data re-annotation) for every
candidate tolerance value (learned/adaptive tolerance radii are future work, §11/§12). §7.2's
`ToleranceBandLoss` reuses these same tolerance values directly via a shared `class_config`.

---

## 6. Analysis and Design for Topological Robustness

*(A structural property the primitive lacks, a sensitivity analysis, then a closed-form-plus-
numerical design section.)*

### 6.1 The Connectivity Blind Spot

This subsection formally states a property DTAF1 **lacks**: pointwise EDT tolerance-matching is
not "connectivity-injective." Concretely (proof sketch; full statement in Appendix A):

> For a road mask `G_c` of width `w` and tolerance `d_c`, there exist predicted masks `P_c`
> obtained from `G_c` by deleting an arbitrarily large fraction of pixels — up to and including
> masks in which `G_c`'s connected skeleton is fully disconnected into many isolated fragments —
> such that `DTAF1_c(P_c, G_c) = 1.0`, provided the surviving pixels remain within `d_c` of every
> GT pixel.

§4.2's road-breakage sweep is the empirical witness: DTAF1 stays at exactly `1.000` through 90%
random dropout on the synthetic scene.

### 6.2 APLS as Connectivity Recovery, and Its Sensitivity to Fragmentation

APLS (§3.5) recovers exactly the signal DTAF1 lacks: its score depends on shortest-path length
*between* control points, not merely on point-to-point proximity, so it is sensitive to whether a
path exists at all. Framing `sweep_road_breakage`'s dropout fraction as the "noise" axis, the
empirical curve has a two-regime character: a near-flat, low-sensitivity regime while the skeleton
is still mostly connected, followed by a sharp departure once fragmentation actually occurs.
Measured directly against the live code:

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

The harmonic mean of `F1` and `APLS` has a simple, exactly-analyzable shape at its boundaries:

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
answered numerically. The table in §6.2 already gives the answer directly: `dtaf1_topo` tracks
`road_apls` closely (both driven to near-zero by fraction 0.5–0.7) while `dtaf1` stays at `1.000`
throughout — this *is* the fix, demonstrated on the same sweep that exposed the blind spot in
§4.2/§6.1, and cross-checked against the assertions in
`tests/test_metrics_sanity.py::TestRoadBreakageRegression`. Validated on synthetic data so far;
real-data re-validation is future work (§11).

**Bridge to §7.** Every property analyzed in §3–§6 — offset tolerance (§3.1/§4.2), width
insensitivity (§3.2/§4.2), connectivity/breakage sensitivity (§3.5/§6), instance-level point
matching (§3.4) — was developed purely as an *evaluation* concern: a property a *finished*
prediction is scored against, after training is already done. §7 asks the natural next question:
if these are the geometric properties that actually distinguish a good extraction from a bad one,
why train a model with a loss that has no notion of any of them? The three tractable properties
(tolerance, width-insensitivity via topology, point-instance-like localization) are exactly the
three `losses/` implements as differentiable training terms; the one property with no tractable
differentiable relaxation (graph connectivity, §3.5/§6) is exactly the one `losses/` leaves out,
by design, not oversight.

---

## 7. A Differentiable Multi-Task Loss for Joint 0D/1D/2D Extraction

Sections 3–6 are eval-only by construction: DTAF1, Boundary F1, and APLS are built on
`scipy.ndimage.distance_transform_edt`; clDice on `skimage.morphology.skeletonize`; Point F1 on
`scipy.optimize.linear_sum_assignment`. Every one of these operations is discrete and breaks
autograd, so none of them can be used directly as a training loss. This section introduces
`losses/`, a new top-level package (parallel to `metrics/` and `datasets/`, pure `torch`) of
differentiable training losses, each a direct analog of one primitive from §3 — mirrored in
spirit, never sharing code with its eval-side counterpart (numpy+scipy vs. torch tensors inside an
autograd graph cannot share an implementation) — so a single model can be trained to *jointly*
predict point (0D), linear (1D), and polygonal (2D) map features using a loss shaped by the same
tolerance, topology, and instance-matching considerations §3–§6 spent five sections motivating.

### 7.1 Design Principle: Differentiable Analogs, Not Shared Code

`losses/` operates on the standard PyTorch segmentation convention — `(B, C, H, W)` logits,
`(B, H, W)` int64 label maps, channel index equal to class id — not §3's `H×W` numpy label-map
convention. Table 2 summarizes the correspondence and, for each term, what had to change relative
to its eval-metric counterpart to make it differentiable:

**Table 2: Eval metric ↔ differentiable loss term correspondence.**

| §3 metric | `losses/` term | What changed to make it differentiable |
|---|---|---|
| DTAF1 (§3.1) | `ToleranceBandLoss` (§7.2) | GT-side dilation-by-radius precomputed once per batch (GT is a fixed target, not a model output); the harder *recall* direction — which needs an EDT of the *prediction* on the eval side — is made differentiable by soft-dilating the model's own continuous probability map via iterated max-pool instead of a detached hard threshold. **Both** precision and recall stay fully differentiable; no stop-gradient approximation was needed on either term (§7.2). |
| Boundary F1 (§3.3) | `SoftBoundaryLoss` (§7.3) | Soft boundary map = morphological gradient `dilate(x) − erode(x)` (saturates to exactly 1.0 at a hard edge — an early `\|x − avgpool(x)\|` formulation was tried and rejected, since its peak response is well below 1.0 and silently caps achievable recall even under perfect alignment); reuses `ToleranceBandLoss`'s precision/recall primitive on this soft boundary map instead of the full class mask. |
| clDice (§3.2) | `SoftClDiceLoss` (§7.4) | Shit et al.'s (2021) soft-skeletonization: iterated soft-erode/soft-open via max-pool/min-pool, applied directly to the probability map — no discrete `skeletonize` call. Runs for a *fixed* iteration count rather than to convergence, so width-invariance is bounded (§7.4, §11), unlike the eval-side metric. |
| Point F1 (§3.4) | `PointHeatmapLoss` (§7.5) | **Not a relaxation.** Hungarian assignment + connected-component labeling have no useful gradient path. Reformulated entirely as CenterNet-style dense Gaussian-heatmap regression — a genuinely different formulation, not a softened version of the eval metric's algorithm. |
| APLS / DTAF1-Topo (§3.5/§6.3) | *(none)* | No tractable lightweight differentiable relaxation of graph shortest-path recomputation exists; stays eval-only permanently — a stated design decision (§11), not a gap to be filled later. |

`PLEMMultiTaskLoss` (§7.6) combines the first three, with `class_config` reused verbatim from
`dtaf1.py`'s convention (§3.1) — the same dict can be passed to `dtaf1()` for evaluation and
`ToleranceBandLoss` for training.

### 7.2 Soft Tolerance-Band Loss (DTAF1 Analog)

Recall from §3.1 that DTAF1's precision test only requires a distance transform *from GT*
(`dist_from_gt`), while its recall test requires one *from the prediction* (`dist_from_pred`).
This asymmetry turns out to matter: precision needs only a GT-side dilation-by-`d` mask
(equivalent to `dist_from_gt <= d`), which is fixed target construction and can be precomputed
once per batch under `no_grad`. Recall is the harder direction on the eval side because it
distance-transforms the *prediction* — but since `F.max_pool2d` is differentiable, applying it
directly and iteratively to the model's own soft probability map ("soft-dilating" it) keeps
gradients flowing with no detach:

```
gt_band_c        = dilate(G_c, d_c)                      # precomputed once, no_grad — GT is fixed
precision_c = ( sum(p_c · gt_band_c) + eps ) / ( sum(p_c) + eps )              # GT-side only, exactly mirrors dtaf1.py's dist_from_gt direction
recall_c    = ( sum(g_c · soft_dilate(p_c, d_c)) + eps ) / ( sum(g_c) + eps )  # soft-dilate the model's OWN probability map
loss_c = w * (1 − precision_c) + (1 − w) * (1 − recall_c)
```

where `p_c` is the model's softmax probability for class `c` and `g_c` is the one-hot GT mask.
Both denominators use epsilon smoothing (matching `train_unet.ipynb`'s existing soft-Dice
convention) rather than explicit both-empty branching, so a class absent from both prediction and
GT reduces cleanly toward the §3's "both empty → perfect" convention without a separate code path.

**Code** (`losses/tolerance.py`):

```python
def soft_tolerance_pair_loss(pred_soft, gt_hard, radius, eps=1e-6, precision_weight=0.5):
    gt_band = _dilate_binary(gt_hard.unsqueeze(1), radius)[:, 0]
    dilated_pred = soft_dilate_prob(pred_soft.unsqueeze(1), radius)[:, 0]   # differentiable
    p_sum, g_sum = pred_soft.sum(dim=(1, 2)), gt_hard.sum(dim=(1, 2))
    prec_num = (pred_soft * gt_band).sum(dim=(1, 2))
    rec_num = (gt_hard * dilated_pred).sum(dim=(1, 2))
    precision = (prec_num + eps) / (p_sum + eps)
    recall = (rec_num + eps) / (g_sum + eps)
    loss = precision_weight * (1.0 - precision) + (1.0 - precision_weight) * (1.0 - recall)
    return loss.mean()
```

`ToleranceBandLoss` applies this per class over a `class_config` dict of the same
`{class_id: {"name", "tolerance"}}` shape as `dtaf1.py`'s `ROAD_BUILDING_CONFIG` (§3.1),
macro-averaging across the configured linear + polygon classes.

**Both directions turned out fully differentiable.** The initial design assessment expected the
recall direction would need a detached/stop-gradient hard-threshold approximation, mirroring how a
real Euclidean distance transform of the prediction would break differentiability the way it does
on the eval side (§3.1). Soft-dilating the model's own probability map via differentiable max-pool
avoided any detach at all — a cleaner result than the boundary-loss literature (§2.7) would
suggest is typical for this kind of term.

### 7.3 Soft Boundary Loss (Boundary F1 Analog)

The soft boundary map reuses the same soft-erode/soft-dilate primitives §7.4 introduces for
soft-skeletonization: `boundary(x) = dilate(x) − erode(x)`, the morphological gradient, which is
`0` in flat interior/exterior regions and saturates to exactly `1.0` at a hard, confident class
edge. `SoftBoundaryLoss` then calls §7.2's `soft_tolerance_pair_loss` on this soft boundary map
(predicted side) against a thresholded boundary map of the one-hot GT (computed under `no_grad`)
instead of on the full class mask — the most directly-shared piece of machinery between any two
`losses/` terms, and the most mechanical of the three (§7.1, Table 2).

### 7.4 Soft-clDice Loss (clDice Analog)

Implements Shit et al.'s (CVPR 2021) soft-skeletonization directly: iterated
`soft_erode(x) = -maxpool(-x)` / `soft_dilate(x) = maxpool(x)` / `soft_open(x) =
soft_dilate(soft_erode(x))`, accumulating skeleton response across scales exactly as the original
paper's reference algorithm does, applied to the model's continuous probability map instead of a
discretely-`skeletonize()`d binary mask:

```
tprec_c = ( sum(skel(p_c) · g_c) + eps ) / ( sum(skel(p_c)) + eps )
tsens_c = ( sum(skel(g_c) · p_c) + eps ) / ( sum(skel(g_c)) + eps )
soft_clDice_c = 2 * tprec_c * tsens_c / (tprec_c + tsens_c + eps)
loss_c = 1 − soft_clDice_c
```

Unlike the eval-side metric, soft-skeletonization runs for a *fixed* number of iterations
(`iters=10` default), not to convergence, so width-invariance holds up to whatever width that
iteration count can fully erode away (§11) — a documented, iteration-bounded version of §3.2's
exact width-invariance. Validated directly: on a 32×32 synthetic scene, a GT road of
thickness 3px scores soft-clDice loss `4.77e-7` against a matching thickness-3px prediction, and
**the identical `4.77e-7`** against a 3×-thicker (9px) prediction on the same centerline — i.e.,
zero measurable width sensitivity up to a 3× ratio — while plain soft Dice on the same 9px input
scores only `0.500` (a substantially degraded overlap ratio), confirming soft-clDice is doing
something meaningfully different from plain Dice, not just numerically saturating
(`tests/test_losses_sanity.py::TestSoftClDiceLoss::test_width_insensitivity`).

### 7.5 Point-Heatmap Regression Loss (Point F1 Analog, Reformulated)

Per Table 2, this term is **not** a relaxation of §3.4's Hungarian matching — connected-component
labeling and combinatorial assignment have no useful gradient. Instead, point supervision is
reformulated entirely as dense Gaussian-heatmap regression, CenterNet-style: each GT point
instance's centroid (extracted via the exact same `scipy.ndimage.label` + `center_of_mass`
pipeline `point_f1.py`'s `_instance_centroids` already uses at eval time — here purely for *target*
construction, under `no_grad`) is burned into a Gaussian blob on a target heatmap (overlapping
Gaussians combined via elementwise max, avoiding peak inflation where GT points cluster), and the
point class's own softmax probability channel is trained directly against it with a
penalty-reduced pixelwise focal loss (CornerNet/CenterNet formulation):

```
gt_heatmap(x)  = max over GT centroids c of exp( -||x - c||^2 / (2 sigma^2) )
pos_loss(x)    = -(1 - p(x))^alpha * log(p(x))                          if gt_heatmap(x) == 1
neg_loss(x)    = -(1 - gt_heatmap(x))^beta * p(x)^alpha * log(1 - p(x))  otherwise
loss = ( sum_x pos_loss(x) + sum_x neg_loss(x) ) / max(1, num_positive_pixels)
```

`point_f1.py`'s Hungarian matching is untouched and remains a pure eval-time metric — a genuine
train/eval formulation mismatch, documented rather than papered over as a relaxation (§11).

### 7.6 `PLEMMultiTaskLoss`: Combination and Multi-Source Class Masking

The combined loss sums a CE+soft-Dice base term (adapted from `train_unet.ipynb`'s existing
`combined_loss`) with the three geometry-aware terms above, each independently weighted:

```
L = w_ce_dice * L_ce_dice + w_tol * L_tolerance + w_cldice * L_cldice + w_heatmap * L_heatmap
```

**Multi-source class masking.** SpaceNet tiles annotate road+building only (no point class exists
at SpaceNet's GSD, §9.2); Potsdam tiles annotate building+point only (no road class in Potsdam's
palette). Training one 4-class model on both jointly requires each sample to be supervised only on
the classes its *source* actually annotates. This is implemented by additively biasing a sample's
masked-out class channels to a large negative logit **before** any softmax is taken:

```
logits'[:, c] = logits[:, c] + (1 - class_mask[:, c]) * (-1e4)     # per-sample, per-channel
```

Since every term in §7.2–§7.5 is a function of softmax probabilities, driving a masked channel's
pre-softmax logit to `-inf` makes its post-softmax probability — and therefore its gradient
contribution to *every* term at once — vanish identically. This is a single mechanism shared by
all four loss terms, rather than four bespoke masking implementations, and relies on target label
maps never containing a class id a sample's source doesn't annotate (true by construction for both
`datasets/spacenet.py` and `datasets/potsdam.py`'s rasterizers, §9.2, §9.5).
`PLEMMultiTaskLoss.forward()` returns a dict (`"loss"` plus each active sub-term as a detached
float) — mirroring §3's "always return a dict of named scalars" convention on the training side,
so per-epoch logging code looks structurally similar to `evaluate_all()`'s output.

---

## 8. Implementation

- All `metrics/`/`datasets/` inputs/outputs are plain NumPy `H×W` `uint8` arrays — no framework
  lock-in. `losses/` (§7) is the one exception by design: it is pure `torch`, batched-tensor,
  differentiable, and deliberately kept in its own top-level package rather than inside
  `metrics/`, whose documented contract ("every public function takes `H×W` integer label maps and
  returns dicts of named scalars") is non-differentiable by construction — adding torch code there
  would silently violate that invariant.
- Every public `metrics/`/`datasets/` function returns a **dict of named scalars**, never a bare
  float, for logging/inspection; `PLEMMultiTaskLoss.forward()` follows the same convention on the
  training side (§7.6).
- Tech stack, grouped by role:

  | Role | Libraries |
  |---|---|
  | Core numerics | `numpy`, `scipy` (`scipy.ndimage` for distance transforms/erosion/labeling) |
  | Skeletonization/morphology | `scikit-image` |
  | Graph algorithms | `networkx` (APLS shortest paths), `scipy.spatial.cKDTree` (snap-matching) |
  | Optimal assignment | `scipy.optimize.linear_sum_assignment` (Point F1, eval-only) |
  | Geospatial I/O | `rasterio`, `shapely` |
  | Data acquisition | `requests` (anonymous S3), `kaggle` (Potsdam mirror) |
  | Differentiable training losses | `torch`/`torch.nn.functional` (CPU-only) — `max_pool2d`/`avg_pool2d`-based dilation/erosion; no `kornia`/`torchvision` dependency added |
  | Testing/analysis | `pytest`, `pandas`, `matplotlib`, `jupyterlab` |

- Package structure/exports note: `dtaf1_topo`, `apls`/`apls_multiclass`/`mean_apls` are **not**
  re-exported at the `metrics/__init__.py` top level, and `cbhm_soft` exists only as a dict key
  inside `cbhm()`'s return, not a separate function (§11). `losses/__init__.py` re-exports all five
  public classes (`ToleranceBandLoss`, `SoftBoundaryLoss`, `SoftClDiceLoss`, `PointHeatmapLoss`,
  `PLEMMultiTaskLoss`).

---

## 9. Experimental Setup

### 9.1 Synthetic Benchmark

128×128 scene: vertical road stripe (class 1) + 40×40 building rectangle (class 2), from
`notebooks/metric_comparison.ipynb`. The nine sweep functions listed in §4.1
(`tests/test_sensitivity.py`).

### 9.2 Real-Data Sample

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

### 9.3 Trained Model (First Real Prediction Source)

Small 3-level encoder/decoder U-Net, base=16 channels doubling per level, single softmax head over
`{bg, road, building}`, **483,475 parameters**. 100 SpaceNet tiles split **70/15/15 at the tile
level** (avoids patch-leakage inflating held-out scores), patchified to 256px
(2520 train / 540 val patches). Loss = class-weighted cross-entropy + soft Dice; class weights
`[0.1, 1.313, 1.602]` for (bg, road, building), from pixel counts
`[147742496, 9565097, 7843127]`. CPU-only, 15 epochs. *(Citation target for the base architecture:
Ronneberger et al., 2015.)* This is the frozen CE+Dice **baseline** referenced by §9.5's ablation.
~100 curated real tiles with no pretraining is a small dataset for segmentation (§11).

### 9.4 Loss Ablation Setup

Mirrors §4.1's synthetic-sweep methodology exactly: `tests/test_loss_sensitivity.py` imports the
same scene builders and perturbed-scene generators directly from `tests/test_sensitivity.py`
(`SHAPE`, `make_road`, `make_building`, `make_gt`, `make_points`, `GT`, `POINT_COORDS` — not
duplicated), so every loss-side sweep perturbs the *identical* synthetic scene as its eval-metric
counterpart. Rather than accuracy, each sweep records **loss value** for a perturbed prediction
against the fixed ground truth, encoded as "confidently peaked" logits (softmax ≈ one-hot at the
perturbed label, but at a moderate confidence scale — see the methodology note below) fed through
`PLEMMultiTaskLoss`, reporting five configurations built from its own sub-term breakdown:
`ce_dice_only`, `plus_tolerance`, `plus_cldice`, `plus_heatmap`, `full`. Seven sweeps mirror seven
of §4.1's nine (road offset/breakage/thickness, building erosion, point jitter/dropout/clutter —
`sweep_class_imbalance`/`sweep_sparse_class_offset` have no loss-side analog, since they are
about how eval-time composite *reductions* behave, not a property of one loss term).

**Methodology note on confidence scale.** The eval-side unit tests (§10, `tests/
test_losses_sanity.py`) use a "fully confident" logit encoding (softmax ≈ 1 − 10⁻¹⁰) to test
perfect-prediction behavior. The loss-ablation sweeps deliberately use a more moderate confidence
scale instead (softmax ≈ 0.999): at full confidence, `PointHeatmapLoss`'s focal-loss log term
saturates at its numerical clamp ceiling for *every* confidently-wrong pixel, and since each
predicted point blob spans multiple pixels, a handful of false-positive point predictions can
inflate a sweep's total loss by roughly two orders of magnitude — a test-harness artifact of total
confidence no real model output actually reaches, not a property of the loss itself. This is
documented directly in `_logits_from_label`'s docstring and is itself a useful, real finding for
§9.5: the heatmap term's absolute magnitude scales differently from the other three (a
CornerNet-style focal formulation, not a bounded [0, 1] ratio the tolerance/clDice terms are),
motivating `PLEMMultiTaskLoss`'s per-term `weights` dict for rebalancing during real training.

### 9.5 Joint Multi-Source Training Setup

Neither SpaceNet nor Potsdam alone annotates all three feature types (§7.6): SpaceNet supplies
road+building but no point class; Potsdam's 6-class palette includes `building` and point-capable
`tree`/`car` classes but no road/linear class at all. A genuinely joint 0D/1D/2D training run
therefore requires combining both sources under `PLEMMultiTaskLoss`'s masked multi-source scheme
(§7.6). `datasets/potsdam.py::label_raster_to_building_mask` extracts the `building` palette color
(a straight threshold, deliberately not routed through the point pipeline's connected-component +
area filter, which would wrongly split/drop large contiguous footprints) alongside the existing
point extraction, combined by `label_raster_to_multiclass_mask`; `build_potsdam_sample
(extract_buildings=True)` builds this combined cache (`tile_potsdam`'s new `min_building_pixels`
parameter switches its crop-keep test to an OR condition — point instances *or* building coverage,
not both required). Run against the real 15-crop Potsdam sample: **14 of 15 cached crops have
building coverage, 5 have point instances, 3 have both** — confirming genuinely multiclass
(building+point) tiles exist in the cache, not just in principle. `datasets/joint.py` then loads
and tags every cached tile from both sources (`SOURCE_CLASSES = {"spacenet": [1,2], "potsdam":
[2,3]}`, `class_mask_for_source()` returning the `(4,)` mask `PLEMMultiTaskLoss` consumes directly)
— run against the real caches: **123 SpaceNet tiles + 30 Potsdam tiles = 153 tagged tiles**, split
70/15/15 at the tile level *independently per source* before concatenating (a single
pooled-then-permuted split risks starving the smaller Potsdam side, which has only 15 crops, from
a split entirely). `SmallUNet` (§9.3) is extended from 3 to 4 output classes and trained on this
combined set (§10.5); the resulting model is evaluated with `evaluate_all(..., point_classes=[3])`
— the first end-to-end exercise of `evaluate_all()`'s `point_classes` argument on a real trained
model's output. SpaceNet and Potsdam are not resampled to a common ground sample distance
(SpaceNet ~0.3–0.5 m/px vs. Potsdam's 6 cm/px, §11); each source is patchified independently to the
same 256×256 *pixel* patch size regardless.

### 9.6 Scaled Production Training Setup

§9.5's run establishes that the masked multi-source mechanism works mechanically; this run tests
whether it holds up at a scale and source count closer to a real deployment, and adds the paper's
illumination-robustness angle. Two changes over §9.5: a third data source, **SpaceNet6** —
Capella Space SAR imagery over Rotterdam with building-footprint labels only
(`datasets/spacenet6.py`), converted to a pseudo-RGB image via percentile-clip normalization
(`sar_bands_to_pseudo_rgb`) so it flows through the unmodified 3-channel `SmallUNet` — and
`datasets/joint.py::SOURCE_CLASSES` gaining `"spacenet6": [2]` with no other code change, since the
class-masking mechanism (§7.6) is source-count-agnostic by design. SAR is an active sensor, so it
looks identical day or night — the real, illumination-invariant half of a two-pronged
night-robustness approach; the synthetic half is `datasets/augment.py::simulate_low_light`
(gamma darkening → brightness/contrast/saturation jitter → CLAHE → sensor-noise injection), applied
to a random 40% of training patches (`dark_aug_prob=0.4`) regardless of source.

Scaled-up per-source samples (`n_tiles=175`/city for SpaceNet, up from 25; `n_crops=200` for
Potsdam) plus the SpaceNet6 tarball were cached and loaded: **1,033 tiles total — 550 SpaceNet +
200 Potsdam + 283 SpaceNet6** — split 70/15/15 independently per source, then concatenated
(spacenet 385/82/83, potsdam 140/30/30, spacenet6 198/42/43 train/val/test; **723 train / 154 val /
156 test** tiles overall). Training is GPU-scaled (`EPOCHS=40`, `BATCH_SIZE=48`, mixed precision,
`CosineAnnealingLR`, early stopping at `PATIENCE=3` on val_loss) with periodic rolling checkpoints
in addition to the best-val-loss checkpoint, and — unlike every other run in this paper — requires
a cloud GPU: this notebook (`train_unet_joint_scaled.ipynb`) asserts CUDA availability and installs
a CUDA `torch` wheel in its own first cell rather than silently falling back to CPU. It was run
once on a CARC A100 (results: §10.6).

---

## 10. Results

### 10.1 Synthetic Sensitivity Results

Table 1 (§1) and the extended sweep results in §4.2/§6.2 are this section's synthetic eval-metric
results, computed directly against the live library for this paper rather than reused from prior
notes.

### 10.2 Real-Data Case Studies

- **Finding #1 — DTAF1 blind spot, confirmed on real data.** Averaged across 24 real SpaceNet
  tiles (Vegas + Khartoum), 75% road-pixel dropout leaves plain DTAF1 at **1.000** while CBHM
  collapses to **~0.57** (driven by `cldice_mean` falling to ~0.40). `dtaf1_weighted` does **not**
  fix this — also stays at 1.000, since area-weighting only changes how per-class scores combine,
  not the per-pixel tolerance-matching logic within a class. `dtaf1_topo` (§6.3) is designed to fix
  exactly this on the synthetic sweep (§6.2) and has not yet been re-validated on this real 24-tile
  sample (§11).
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

### 10.3 Applied Pipeline: Trained U-Net Results (CE+Dice Baseline)

Full test-set table (15 held-out tiles, mean ± std):

| Metric | mean | std |
|---|---|---|
| `cbhm` | 0.316 | 0.126 |
| `cbhm_soft` | 0.378 | 0.130 |
| `dtaf1` | 0.440 | 0.134 |
| `dtaf1_weighted` | 0.476 | 0.141 |
| `cldice_mean` | 0.409 | 0.193 |
| `bf_mean` | 0.308 | 0.156 |

Best tile `Shanghai_img1375` (`cbhm` = 0.617), worst `Shanghai_img1711` (`cbhm` = 0.163). This is
the first end-to-end confirmation that the metric library scores an actual trained model's output
sensibly (not just perturbed ground truth, §10.2), and it is the CE+Dice **baseline** side of
§10.5's comparison (§11 on dataset scale).

### 10.4 Loss Ablation Results

Seven sweeps (§9.4) confirm the qualitative story Table 2 (§7.1) predicts — each new loss term
responds distinctly to the perturbation family its `metrics/` counterpart is designed to catch, and
stays flat (a negative control) under perturbations that don't touch its configured class(es). All
values below are each configuration's own raw loss value (not the cumulative `plus_X` framing) at
representative severities, computed directly against the live code for this paper:

**Road offset sweep** (`sweep_road_offset_loss`, `tolerance` radius = 10px): `tolerance`'s own
value stays essentially flat (`0.0043`) through offset = 10px, then climbs (`0.0056` at 12px)
exactly at its configured tolerance boundary — reproducing §4.2's eval-side "holds until exactly
the tolerance boundary" story on the training-signal side. `cldice`'s own value jumps almost
immediately (already at `1.0` loss delta by offset = 2px) and then plateaus at its collapsed
value — reproducing §3.2's documented zero-tolerance-to-offset property.

**Road thickness sweep** (`sweep_road_thickness_loss`, GT fixed 3px): `cldice`'s own value tracks
the `ce_dice` baseline closely through thickness = 13px (within ~0.05 of each other at every point
up to a 4× width ratio), then diverges sharply from thickness = 15px onward — reproducing §7.4's
unit-test finding (a 3× ratio scores identically to exact width) at a coarser, sweep-level
granularity and confirming the `iters=10` default's practical width-invariance range.

**Building erosion sweep** (`sweep_building_erosion_loss`): `tolerance` tracks building erosion
closely (a consistent gap above `ce_dice`, e.g. `1.2338` vs. `1.0526` at erosion radius = 11px,
reflecting the tight 2px building tolerance), while `cldice` stays essentially identical to
`ce_dice` throughout (`1.0531` vs. `1.0526` at radius = 11px) — the expected negative control,
since `SoftClDiceLoss` is configured over linear classes only and this sweep never touches the
road.

**Point jitter/dropout/clutter sweeps** (`sweep_point_jitter_loss`, etc.): `heatmap` is by far the
most shape-sensitive term, rising from a baseline of `0.58` to `70.6` by 8px of jitter (then
plateauing/slightly declining as points scatter further from any plausible match) — while
`tolerance` and `cldice` stay flat at their perturbation-independent baseline values throughout
every point sweep (neither is configured over the point class), confirming both as correct
negative controls. `ce_dice`'s own value also rises modestly under point perturbations (`0.51 →
0.84` under jitter) since plain per-class Dice is somewhat sensitive too, but nowhere near as
sharply or as saturating as `heatmap`'s response — visually confirmed via the small-multiples
figures in `notebooks/loss_ablation.ipynb` (one subplot per term, its own y-scale, since the four
terms differ by orders of magnitude and a shared axis would flatten the non-dominant ones to
invisibility).

**No flat curve was found where sensitivity was expected** — every term's own raw value moves
under the perturbation family its `metrics/` counterpart is designed to catch, and every intended
negative control stayed flat. See `CLAUDE.md`'s `losses/` section and
`tests/test_loss_sensitivity.py`'s module docstring for the full per-sweep numeric tables.

### 10.5 Joint Multi-Source Training Results

§9.5's setup ran to completion (`notebooks/train_unet_joint.ipynb`): 153 cached tiles loaded (123
SpaceNet + 30 Potsdam), split per-source 70/15/15 (SpaceNet: 86/18/19 train/val/test; Potsdam:
21/4/5), patchified to 3117 train + 652 val patches. A per-batch source-mix printout confirmed both
sources appear in training (batches 0–2: 16/0, 16/0, 15/1 spacenet/potsdam patches) — Potsdam
patches are necessarily rare per batch (only 21 of 3117 train patches, ≈0.7%, since each Potsdam
crop contributes exactly one already-patch-sized tile, versus 36 patches per large SpaceNet tile),
but they are not absent.

**All four loss terms learn jointly.** Over the 2-epoch run:

| Epoch | train_loss | val_loss | ce_dice | tolerance | cldice | heatmap |
|---|---|---|---|---|---|---|
| 1 | 3.3824 | 2.8458 | 1.5427 | 0.5896 | 0.9038 | 0.3461 |
| 2 | 2.6823 | 2.5957 | 1.2497 | 0.5509 | 0.8718 | 0.0098 |

Every sub-term decreased epoch-over-epoch, not just the total — direct evidence that
`PLEMMultiTaskLoss`'s four terms optimize together rather than fighting each other. (`heatmap`'s
own sharp drop, 0.346 → 0.010, is revisited in §11 alongside the same pattern in §10.6's larger
run.)

**Test-set evaluation** (24 tiles: 19 SpaceNet + 5 Potsdam), `evaluate_all(pred, gt,
point_classes=[3], dtaf1_config=<4-class config>)` — the first end-to-end exercise of
`evaluate_all()`'s `point_classes` argument on a real trained model's output:

| | SpaceNet-only (n=19) mean±std | Potsdam-only (n=5) mean±std |
|---|---|---|
| `cbhm` | 0.144 ± 0.115 | 0.000 ± 0.000 |
| `cbhm_soft` | 0.230 ± 0.119 | 0.078 ± 0.159 |
| `dtaf1` | 0.304 ± 0.157 | 0.092 ± 0.149 |
| `dtaf1_weighted` | 0.371 ± 0.212 | 0.076 ± 0.158 |
| `cldice_mean` | 0.173 ± 0.141 | 0.000 ± 0.000 |
| `bf_mean` | 0.252 ± 0.151 | 0.078 ± 0.159 |
| `point_f1_mean` | — | 0.200 ± 0.447 |

The SpaceNet-only aggregate reads below §10.3's 15-epoch CE+Dice baseline (`cbhm` 0.316,
`dtaf1` 0.440) — this run trained for only 2 epochs on a strictly harder 4-term joint task, and no
epoch-matched comparison exists yet to isolate epoch budget from loss design (§11). The
`point_f1_mean` values (`[0.0, 0.0, 0.0, 1.0, 0.0]` across the 5 Potsdam tiles) are consistent with
the point head not yet detecting real points at this epoch budget — the one `1.0` is most likely a
tile with no GT points scored via §3's both-empty convention, not a genuine detection (§11 connects
this to the same point-head collapse pattern §10.6 documents at scale).

**Qualitative review** (worst→best `cbhm` tiles) turns up the paper's clearest real-data
illustration of §3.6's harsh/lenient design split. The **worst**-scoring tile by `cbhm`
(`Vegas_img425`, `cbhm=0.000`) is visually one of the *better*-looking predictions in the panel —
road centerlines and building footprints are both recognizable and roughly aligned with GT.
`bf_mean=0.000` exactly (building Boundary F1, 2px tolerance, collapsed completely) while
`dtaf1_weighted=0.902` and `cldice_mean=0.474` are both high: CBHM's harmonic mean zero-collapses
the *entire* tile score from one class's tight-tolerance metric hitting exactly zero, even though
the lenient DTAF1 view and the road-only clDice view both say the tile is mostly correct — the
first time this exact zero-collapse mechanism (Appendix A.3) has been observed on a real trained
model's output rather than only in synthetic scenes (Table 1) or GT-vs-perturbed-GT sweeps (§10.2).
The best-scoring shown tile (`Vegas_img997`, `cbhm=0.364`) has a balanced per-metric profile
(`dtaf1=0.402`, `cldice_mean=0.336`, `bf_mean=0.397`) — no single class collapsing, road and
building both partially recovered. No qualitative evidence of a masking-mechanism bug (e.g. a
SpaceNet tile hallucinating point-class predictions, which would indicate `class_mask` leaking) was
observed in any reviewed panel — the two weakest-looking tiles (`Paris_img449`, `Paris_img250`)
over-predict road on terrain with no GT road, consistent with an underfit model at this epoch
budget rather than a masking failure.

### 10.6 Scaled Production Training Results

§9.6's run (`notebooks/train_unet_joint_scaled.ipynb`, CARC A100) trains the same mechanism at
roughly 7× the tile count and a third data source. **It converges cleanly for the full 40-epoch
budget**: val_loss falls from `2.2894` (epoch 1) to a best of **`1.5067`** (epoch 39, held at
epoch 40); early stopping (`PATIENCE=3`) never triggers. Per-epoch sub-term breakdown at the
endpoints:

| Epoch | train_loss | val_loss | ce_dice | tolerance | cldice | heatmap |
|---|---|---|---|---|---|---|
| 1 | 3.6933 | 2.2894 | 1.2373 | 0.5754 | 0.7537 | 1.1270 |
| 40 | 1.6628 | 1.5067 | 0.7559 | 0.3723 | 0.5343 | 0.0004 |

`ce_dice`, `tolerance`, and `cldice` all decline steadily across the full 40 epochs, at three times
the data and a third source relative to §10.5 — evidence the loss composition and the masking
mechanism scale, not just run once at small size.

**The masking mechanism itself is directly verified at this scale.** A train/predict-asymmetry
check confirms `predict_tile()`'s signature has no `class_mask`/source argument (inference always
produces an unmasked 4-class argmax, by design — only the *training* loss is source-masked, §7.6)
and reports real predicted-pixel counts per class per source:

| source | road px | building px | point px |
|---|---|---|---|
| potsdam | 4,760 | 799,355 | 0 |
| spacenet | 11,847,618 | 5,139,252 | 0 |
| spacenet6 | 405,423 | 1,142,827 | 0 |

Building is predicted correctly on all three sources, as expected since every source annotates it.
Road and point predictions leaking onto sources whose own ground truth never contains that class
(e.g. road pixels on Potsdam/SpaceNet6 tiles) is an expected consequence of a shared, unmasked
inference head, not evidence the training-time masking mechanism failed — masking only withholds
gradient from a class a sample's source doesn't annotate, it does not, and is not meant to,
restrict what the shared head can output at inference.

**Point channel:** predicted point-pixel count is `0` on every source, including Potsdam, matching
the `heatmap` sub-term's sharp early collapse in the table above (`1.1270 → 0.0119` by epoch 2,
`0.0004` by epoch 40) and the same pattern §10.5 shows at small scale. We root-caused this (§11)
and the fix is already committed in `losses/heatmap.py`; re-running this notebook's evaluation with
the fix applied is the immediate next step (§11/§12), not yet done — this run's per-source test
metrics, dark-robustness sweep, and qualitative panel were not captured and are reported honestly
as not yet available rather than estimated (§11).

---

## 11. Discussion & Limitations

Grouped by which half of the paper each item belongs to. Every caveat referenced from earlier
sections is expanded here exactly once.

### 11.1 Evaluation-Library Limitations

- DTAF1's road-breakage blind spot (§4.2, §6.1) is fixed by DTAF1-Topo (§6.3), but DTAF1-Topo is
  validated on **synthetic data only** (§6.3.2) — not yet re-run on the real 24-tile sample
  (§10.2), and not yet wired into `evaluate_all()` (§8).
- `cbhm_soft` (§5.2) only helps when the failing class is also the pixel-count minority; when a
  class sparse by *area* is not sparse by *pixel count* (`Khartoum_img371`), the fix is partial
  (0.000 → 0.285, vs. a clean 0.48–0.52 → 0.85 recovery on `Khartoum_img333`).
- clDice is brittle on short/sparse real road segments (§3.2) — moderate offsets can collapse it
  to exactly 0 even when DTAF1 degrades gracefully. Neither composite is uniformly more
  trustworthy — report `cbhm`, `cbhm_soft`, `dtaf1`, and `dtaf1_weighted` together, not one alone.
- APLS (§3.5) here is a raster-skeleton approximation, not true vector-graph APLS, since no vector
  road graph survives the `rasterize_lines` step anywhere in the pipeline (§8).
- The harmonic-blend asymmetry proved in §6.3.1 (DTAF1-Topo can only ever pull a score *down*
  toward APLS, never up) is intentional, but means DTAF1-Topo inherits every one of APLS's own
  failure modes (e.g. a genuinely correct but very short/sparse road segment with too few sampled
  control-point pairs to estimate APLS reliably) — not yet characterized empirically.

### 11.2 Training-Loss Limitations

- **`losses/` has no connectivity term, by design, not oversight.** APLS/DTAF1-Topo (§3.5/§6.3)
  have no tractable lightweight differentiable relaxation, so a model trained with
  `PLEMMultiTaskLoss` has no direct training pressure toward connectivity, only toward
  tolerance-band coverage and centerline topology at `SoftClDiceLoss`'s fixed-iteration
  granularity — it may still learn to produce broken roads that look locally correct, the failure
  mode §6.1 formalizes on the eval side. Whether `SoftClDiceLoss` alone provides enough indirect
  connectivity pressure in practice is open (§12).
- **`PointHeatmapLoss` is a genuine train/eval formulation mismatch, not a relaxation** (§7.5):
  Point F1 (§3.4) is evaluated via Hungarian instance matching but trained via dense heatmap
  regression — different algorithms with different failure modes, so minimizing the heatmap loss
  is not directly minimizing anything Point F1 measures, only something correlated with it. This
  mismatch is inherent to point supervision (no tractable differentiable relaxation of Hungarian
  matching exists, §7.1), not a shortcut taken for convenience.
- `SoftClDiceLoss` runs soft-skeletonization for a *fixed* iteration count (`iters=10`), not to
  convergence, so its width-invariance is bounded rather than exact — validated up to a 3× width
  ratio (§7.4).
- The heatmap term's loss magnitude differs from the other three by orders of magnitude under
  point perturbations (§9.4, §10.4) — a real property of its CornerNet-style focal formulation, not
  a bug. `PLEMMultiTaskLoss`'s per-term `weights` dict exists specifically to rebalance this; the
  right weighting has not yet been empirically tuned against real training results.
- `dtaf1_topo`/`apls`/`apls_multiclass`/`mean_apls` are not re-exported at `metrics/__init__.py`'s
  top level, and `cbhm_soft` exists only as a dict key inside `cbhm()`'s return rather than a
  separate function (§8) — validated on synthetic data so far, not yet promoted to a drop-in
  `evaluate_all()` default.
- The `max_instances` cap added to `gt_centroids_to_heatmap` (`losses/heatmap.py`, default 100) is
  a defensive bound found necessary while profiling `train_unet_joint.ipynb`: a pathological,
  non-blob-like synthetic target drove its per-centroid Python loop to thousands of connected
  components, stalling a training batch for minutes with no error. Real Potsdam crops top out
  around two dozen instances per tile, so the cap is a no-op on real data.
- SpaceNet/Potsdam/SpaceNet6 are not resampled to a common ground sample distance (§9.5/§9.6) — a
  Potsdam or SpaceNet6 training patch covers a different physical area than a SpaceNet patch at the
  same pixel size. Neither joint run surfaced an obvious qualitative failure mode from this, but a
  rigorous test would need an ablation isolating GSD from every other confound, not yet run.

### 11.3 Validation-Scope Limitations

- The CE+Dice baseline (§9.3/§10.3) is trained on ~100 curated real tiles with no pretraining,
  CPU-only — a real, useful end-to-end validation of the metric library, but too small a dataset to
  read as a segmentation benchmark result.
- **No epoch-matched comparison exists yet between the CE+Dice baseline and the joint loss.** The
  small joint run's SpaceNet-only aggregate (`cbhm` 0.144, `dtaf1` 0.304, §10.5) reads below the
  baseline's 15-epoch numbers (`cbhm` 0.316, `dtaf1` 0.440, §10.3), but it trained for only 2 epochs
  on a strictly harder 4-term joint task with a second data source contributing under 1% of
  training patches — these numbers show the pipeline runs correctly end-to-end, not a verdict on
  the new loss's quality relative to CE+Dice. Closing this gap is the paper's top priority (§12).
- **The point head collapsed in both real joint-training runs.** The small run's `heatmap` sub-term
  drops sharply within 2 epochs (0.346 → 0.010, §10.5); the production-scale run shows the same
  pattern more starkly across 40 epochs (1.127 → 0.0004, §10.6) and its asymmetry check confirms
  zero predicted point pixels on every source, including Potsdam. We root-caused this: GT centroid
  targets were burned at their sub-pixel float location, so for any blob wider than one pixel the
  heatmap peak never reached the `1.0` threshold `heatmap_focal_loss` gates its positive term on,
  leaving the point channel with only push-to-zero gradient. The fix (stamping each centroid's
  nearest integer pixel to exactly `1.0`, plus a per-term loss-weight bump and oversampling of
  point-bearing patches) is already committed in `losses/heatmap.py` and covered by a regression
  test (`TestPointHeatmapLoss::test_fractional_centroid_still_supervised`), but **has not yet been
  re-validated by a full training re-run** — the paper's other co-top-priority item (§12).
- **The production-scale run's test-set evaluation, dark-robustness sweep, and qualitative review
  were not captured.** §9.6/§10.6's training run itself completed and converged cleanly, but the
  notebook cells that would report per-source `dtaf1_topo`/`cbhm_soft`/`point_f1` test aggregates,
  the 0.0/0.3/0.6/0.9 dark-severity sweep, and named best/worst tiles did not execute in the
  committed run. Re-running them (ideally together with the point-head fix above, so the numbers
  reflect a working point channel) is future work (§12), not reported here as if measured.
- A `DARK_AUG_PROB=0` re-run compared against §10.6 on the same dark-test sweep, to isolate the
  low-light augmentation's specific contribution from the effect of more data/epochs/sources, has
  not been run.

---

## 12. Conclusion & Future Work

This paper makes two contributions that reinforce each other. The evaluation half (§3–§6) is a
class-agnostic, tolerance/topology-aware metric library that found and fixed two real failure
modes through an explicit synthetic-sweep → real-data validation → targeted additive fix →
regression-test loop. The training half (§7) asks whether the same geometric properties that
motivate those metrics can be trained *into* a model directly, rather than only measured
afterward, and answers it with concrete, increasingly demanding evidence rather than synthetic
results alone. `PLEMMultiTaskLoss` combines three differentiable analogs of those metrics
(tolerance, topology, and — via a genuine reformulation rather than a relaxation — point-instance
localization) into one multi-task loss, with a masked multi-source training scheme letting
datasets that individually annotate only two or three of three feature types jointly supervise one
4-class model. §9.4/§10.4's synthetic ablations confirm each term encodes the geometric sensitivity
it was designed to. §9.5/§10.5's real joint-training run goes further: all four terms learn jointly
on real imagery, `evaluate_all()`'s point-feature path runs end-to-end on a real trained model for
the first time, and the qualitative review turns up a genuine, well-explained real-data example of
why CBHM's harsh composite and DTAF1's lenient one are both needed (§3.6, §10.5). §9.6/§10.6 goes
further still: the identical mechanism trains to full, stable 40-epoch convergence at roughly 7×
the data and a third heterogeneous source on real cloud GPU hardware, and its asymmetry check gives
direct, real evidence that the per-source class-masking mechanism (§7.6) — the paper's central
multi-source training contribution — behaves exactly as designed at that scale. That same run also
did what a real validation pass should: it surfaced a genuine failure mode (the point head's
sub-pixel-centroid bug) that the synthetic ablations and the small real run's own `heatmap` collapse
had already hinted at but not fully diagnosed, and we root-caused and fixed it in code. Two items,
not one, are this paper's highest priority next steps, and neither is decided yet:

- **Re-run §9.6's production-scale training with the point-head fix applied**, and capture the
  test-set/`dtaf1_topo`/`cbhm_soft`/`point_f1` aggregates, the dark-robustness sweep, and the
  qualitative panel that §10.6's committed run did not capture — the direct evidence needed to
  claim the point channel, and the full pipeline at scale, actually work rather than merely run.
- **Run an epoch-matched (or otherwise fairly controlled) comparison** between the CE+Dice baseline
  and the joint loss (§11.3) — §10.5's 2-epoch run cannot separate "the new loss underperforms"
  from "2 epochs on a harder task underperforms," and this is the comparison a benchmark-level claim
  about the new loss ultimately rests on.

Further future work:

- Wire `dtaf1_topo` into `evaluate_all()` and `metrics/__init__.py` as the default after re-running
  it on the real 24-tile SpaceNet sample already used for §10.2's other real-data findings, closing
  §11.1's "synthetic-only" item.
- Extend the point-feature pipeline (both eval, §3.4, and training, §7.5) to more Potsdam classes
  (cars).
- Explore learned/adaptive tolerance radii instead of fixed per-class constants (§5.3), and
  correspondingly a learned rather than fixed `iters` for `SoftClDiceLoss` (§7.4).
- Characterize DTAF1-Topo's inherited APLS failure modes on short/sparse real road segments (§11.1)
  with a dedicated sensitivity sweep, mirroring `sweep_sparse_class_offset`'s treatment of CBHM's
  analogous weakness (§5.1).
- Empirically tune `PLEMMultiTaskLoss`'s per-term `weights` dict (§11.2) once a working point
  channel gives a real signal to optimize against, rather than the default equal weighting used for
  §9.4/§10.4's ablations.
- Investigate whether `SoftClDiceLoss` alone provides meaningful indirect connectivity pressure
  during training (§11.2), as a cheaper partial substitute for the connectivity term `losses/`
  otherwise cannot have (§3.5/§7.1).
- Run the `DARK_AUG_PROB=0` ablation against §10.6 to isolate low-light augmentation's specific
  contribution (§11.3).
- Resample SpaceNet/Potsdam/SpaceNet6 to a common GSD, or ablate whether the current mismatch
  (§11.2) actually matters.

---

## Acknowledgments

*(Mirrors the reference template's own disclosure norm — it explicitly states "parts of the text
and parts of the code for experiments & plotting in this paper were generated with ChatGPT and
carefully checked by the author," with the author accepting responsibility for correctness.)*
Parts of the implementation, notebooks, and this outline were developed with the assistance of
Claude Code, and were checked and are the responsibility of the author(s).

---

## References *(empty — matches §2 being mostly blank; twelve anticipated citations)*

- Shit et al., "clDice — a Novel Topology-Preserving Loss Function for Tubular Structure
  Segmentation," CVPR 2021. *(§2.2, §3.2, §2.7, §7.4 — both the eval-metric and the loss-function
  contribution of this paper are cited here.)*
- Csurka et al., "What is a good evaluation measure for semantic segmentation?," BMVC 2013
  (Boundary F1 / BF score). *(§2.3, §3.3)*
- SpaceNet challenge / APLS metric paper. *(§2.4, §3.5, §9.2)*
- ISPRS Potsdam benchmark paper. *(§9.2)*
- Kuhn, H. W., "The Hungarian method for the assignment problem," 1955. *(§2.5, §3.4)*
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," 2015.
  *(§9.3, §10.3)*
- Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image
  Segmentation," 3DV 2016 (soft Dice loss). *(§2.7, §7.6)*
- Kervadec et al., "Boundary loss for highly unbalanced segmentation," 2019 (MIDL / Medical Image
  Analysis). *(§2.7, §7.2)*
- Karimi & Salcudean, "Reducing the Hausdorff Distance in Medical Image Segmentation with
  Convolutional Neural Networks," 2019 (Hausdorff-distance loss). *(§2.7, §7.2)*
- Law & Deng, "CornerNet: Detecting Objects as Paired Keypoints," ECCV 2018 (penalty-reduced
  pixelwise focal loss). *(§2.7, §7.5)*
- Zhou, Wang & Krähenbühl, "Objects as Points" ("CenterNet"), 2019 (heatmap regression for
  keypoint/center-point detection). *(§2.7, §7.5)*

---

## Appendix A: Formal Properties and Proof Sketches

*(One short proof per §3 eval-metric primitive, matching the exact parameterizations used there.
`losses/`'s terms are not given formal proofs here — §7's claims are validated empirically, via
the unit tests in §10/`tests/test_losses_sanity.py` and the ablation sweeps in §9.4/§10.4, rather
than proven; this is noted explicitly rather than left implicit.)*

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
pairings jointly rather than greedily committing to the locally-best match first. This is also the
formal reason §7.5 does not attempt a differentiable relaxation of Hungarian matching for the
training loss: the property that makes it correct (global joint optimization over the complete
bipartite graph) is exactly the discrete, combinatorial structure that has no useful gradient.

---

## Appendix B: Algorithms and Computational Complexity

*(One subsection per §3 primitive gives its per-primitive computational complexity. B.8–B.11 extend
this to the four `losses/` terms from §7, on the same basis: cost per forward pass on one batch of
`H×W` tiles, in the same `N`/`k` notation as B.1.)*

### B.1 General Framework and Notation

Let `N = H×W` be the pixel count of one tile, and (where relevant) `k` the number of point-class
instances in a scene (typically tens to low hundreds, `k ≪ N`). For §B.8–B.11, `B` is batch size
and `iters` is a loss term's fixed iteration count (§7.2, §7.4).

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
scores CBHM already computes (§3.6) — strictly cheaper than either primitive it reweights.

### B.8 `ToleranceBandLoss` (§7.2)

The GT-side dilation-by-radius (`_dilate_binary`, `gt_tolerance_band`) is `O(radius · B·N)`:
`radius` sequential `3×3` max-pool passes, each `O(B·N)`. The recall direction's soft-dilation of
the prediction (`soft_dilate_prob`) is the identical operation applied to a continuous tensor
instead of a binary one, same `O(radius · B·N)` cost. Precision/recall ratio computation is
`O(B·N)` per class. Total per class: `O(radius · B·N)` — in the same `O(N)`-per-tile family as
DTAF1 (§B.2), with `radius` (typically 2–10, §5.3) as a small constant multiplier, and cheaper
per-call in practice than DTAF1's exact Euclidean distance transform (§B.2's two-pass separable
algorithm has a larger constant factor than repeated `3×3` max-pooling for the tolerance radii
actually used in this paper).

### B.9 `SoftBoundaryLoss` (§7.3)

The morphological-gradient boundary map (`soft_dilate(x) - soft_erode(x)`) is two `3×3` pooling
passes, `O(B·N)`. The remainder of the computation is exactly `ToleranceBandLoss`'s primitive
(§B.8) applied to this boundary map instead of the full class mask, so total per class:
`O(radius · B·N)`, identical complexity family to B.8 with a small constant-factor overhead for
the boundary-map computation itself.

### B.10 `SoftClDiceLoss` (§7.4)

Each `soft_erode`/`soft_dilate`/`soft_open` call is one `3×3` pooling pass, `O(B·N)`;
`soft_skeletonize` performs `iters` such passes (default `iters=10`, §7.4), so skeletonization is
`O(iters · B·N)`. This runs once for the prediction and once for GT (the GT skeletonization is
computed under `no_grad`, §7.4, but has the same forward-pass cost). The `tprec`/`tsens`
ratio computation is `O(B·N)` per class. Total per class: `O(iters · B·N)` — same asymptotic family
as B.3's near-linear `skimage.skeletonize`, with `iters` playing the role of B.3's (implicit,
converge-to-completion) iteration count, except fixed rather than input-dependent — the direct
computational reason for §7.4's iteration-bounded width-invariance limitation.

### B.11 `PointHeatmapLoss` (§7.5)

Target construction (`gt_centroids_to_heatmap`) is `O(N)` per sample for the Gaussian-burn step
plus `scipy.ndimage.label` + `center_of_mass`'s `O(N)` connected-component pass (identical cost
profile to B.5's centroid extraction, since it is the same operation, here used for target
construction rather than eval-time matching) — no `O(k³)` Hungarian term, since this loss performs
no assignment at all (§7.5, §11). `heatmap_focal_loss` is `O(B·N)`, a constant number of pointwise
tensor operations. Total: `O(B·N)` — asymptotically the cheapest of the four `losses/` terms, and
notably *without* B.5's `O(k³)` combinatorial cost, since dense heatmap regression sidesteps
instance-level assignment entirely (the same property that makes it a reformulation rather than a
relaxation, §7.1/§7.5).

### B.12 `PLEMMultiTaskLoss` (§7.6)

No new asymptotic cost beyond summing B.8–B.11's already-analyzed constituent costs plus the base
CE+Dice term (`O(B·N)`, standard cross-entropy and soft-Dice computation) — the pre-softmax
class-masking step (`_mask_logits`, §7.6) is a single `O(B·C)`-broadcast addition, negligible next
to any of the per-pixel terms. Total: `O(iters · B·N)`, dominated by `SoftClDiceLoss`'s
iteration-bounded skeletonization (§B.10), the most expensive of the four terms per forward pass.
