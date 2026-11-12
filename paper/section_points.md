# PLEM paper — main points per section

The points each section of the paper must make, in the order of `skeleton.tex` (same directory).
It lists claims and the numbers that support them, not prose. The full text is in `sections/*.tex`.
The code and the recorded results (`results/scaled/`, notebook outputs) are the authority for
every number.

Items marked **(gap)** were found in the section-by-section review and are not yet in the paper
text.

---

## Abstract

- Pixel IoU fails on linear features. A small centerline offset or a width error collapses the
  score even when the road network is correct.
- **Metrics.**
  - DTAF1: distance-tolerant per-class F1, using a single formula for every class.
  - CBHM: harmonic mean of clDice and boundary F1, the harsh foil to DTAF1.
- **Fixes from real data.** Two failure modes found on SpaceNet/Potsdam, each fixed by an
  additive companion:
  - `cbhm_soft`: area-weighted, does not collapse.
  - DTAF1-Topo: adds raster-skeleton APLS, closing the road-breakage blind spot.
- **Training question.** Can one model learn points, lines and polygons jointly, with a loss
  shaped by the same tolerance, topology and instance considerations as the metrics?
- **The loss.** `PLEMMultiTaskLoss` combines:
  - a soft tolerance band (the DTAF1 analog);
  - soft-clDice;
  - a CenterNet-style heatmap for points;
  - masked multi-source supervision. SpaceNet has road, building and intersections; Potsdam and
    SpaceNet6 have buildings only.
- **The point feature** is the road intersection, derived from the same vector labels as the
  roads and predicted by an independent sigmoid head.
- **Headline results.** 1,033 tiles from three sources: SpaceNet `cbhm_soft` 0.685;
  intersection F1 0.279 at a validation-calibrated threshold.
- **Matched baseline** (CE+Dice, mIoU-selected): mIoU ties (0.570 vs 0.572), while every PLEM
  metric favours the joint loss (`cbhm_soft` +0.038; intersection F1 0.279 vs 0.218).
- **Four development lessons:**
  - a point channel in the softmax takes over road and building pixels;
  - the Potsdam tree/car point class could not be learned;
  - the heatmap negative-term normalization swings between 30× over-detection and almost none
    until the threshold is calibrated;
  - 36% of SpaceNet road-tile area has no public building labels.

## 1 Introduction

- **The three types.** Overhead maps mix three geometric feature types: linear (roads),
  polygonal (buildings) and point. Here the point is the road intersection, the network vertex
  that routing and conflation depend on and that a road mask does not make explicit.
- **Why IoU fails on roads.** For a feature a few pixels wide, the IoU denominator is dominated
  by pixels sensitive to registration noise, jitter and width error.
- **Same problem in training.** CE and soft Dice are as geometry-agnostic as IoU: no tolerance,
  no centerline topology, no instance identity.
- **Table `tab:failure-modes`** (synthetic 128×128 scene) previews five scenarios. Key cases:
  - IoU fails a 5 px shift;
  - DTAF1 scores 1.000 with 33% of road pixels deleted, which motivates DTAF1-Topo;
  - CBHM collapses on a sparse offset road.
- **Data representation.**
  - Labels are H×W uint8 maps: 0 = background, 1 = road, 2 = building, optional 3 = point.
  - Intersections lie on road pixels, so they are a separate (N, 2) point layer.
  - Tolerances are in pixels, tied to GSD: d = metres / (metres per pixel).
  - One `class_config` is shared by `dtaf1()` and `ToleranceBandLoss`.
- **Eight contributions:**
  1. the multi-task loss, masked multi-source training and the sigmoid point head;
  2. DTAF1;
  3. CBHM;
  4. `dtaf1_weighted` / `cbhm_soft`;
  5. raster APLS and DTAF1-Topo;
  6. Hungarian point F1 on road intersections derived from vector labels;
  7. synthetic, real-data and trained-model validation;
  8. the production run with its six-run development record.

  Plus the matched baseline: mIoU ties, PLEM metrics separate the models, and the two disagree
  in sign on 30 of 83 tiles.
- **Roadmap** of sections 2–12.

## 2 Related Work

### 2.1 IoU-family segmentation metrics
- IoU is the default in PASCAL VOC and Cityscapes, neither of which has a thin linear class.
- **Motivational baseline:** Potsdam mIoU reaches about 86% and F1 about 92%, so region overlap
  is near-saturated on polygons and hides failure modes.

### 2.2 Topology-/skeleton-aware metrics
- clDice (Shit et al., CVPR 2021). **(gap)** Needs real prose: what clDice measures and what it
  misses (offset brittleness).

### 2.3 Boundary-based metrics
- BF score (Csurka et al., BMVC 2013). **(gap)** Needs real prose.

### 2.4 Road-network extraction and connectivity metrics
- APLS (SpaceNet) and TOPO (Biagioni & Eriksson) are the graph-connectivity metrics.
- **Leaderboards plateau well below 1.0:**
  - SpaceNet 3 winner APLS 0.666;
  - CRESI 0.69 ± 0.02;
  - D-LinkNet pixel IoU 0.634 (test);
  - SpaceNet building F1 0.26 (challenge 1) and 0.693 (SpaceNet 2).
- These numbers are motivating context only, not comparable with PLEM's curated test tiles. A
  single scalar cannot tell scattered pixel noise from a fragmented network.

### 2.5 Point/instance detection metrics
- COCO AP; Hungarian matching (Kuhn).
- Graph-based road extraction (RoadTracer, Sat2Graph) predicts intersections as graph vertices.
- PLEM instead predicts them as a separate point output next to a raster mask and scores them as
  instances.

### 2.6 Composite / multi-task evaluation
- Prior work scores linear and polygonal types with separate, non-comparable metrics.
- The gap filled here: one class-agnostic formula (DTAF1) plus a harsh cross-type composite
  (CBHM). **(gap)** Needs citations.

### 2.7 Differentiable segmentation losses
- CE and soft Dice (V-Net) are geometry-agnostic.
- Three prior lines each address one property:
  1. distance/boundary losses (Kervadec; Karimi & Salcudean) → the tolerance term;
  2. soft-clDice (Shit) → the clDice term;
  3. heatmap regression (CornerNet, CenterNet) → the point term.
- The contribution is combining all three plus masked multi-source training.

## 3 Evaluation Protocol

- **Overview.** One sentence naming the metrics, and that three of them are the targets of the
  §7 loss terms.
- **Shared notation.** P and G are label maps; per-class masks are P_c and G_c; scores are
  reduced across the configured classes, macro or GT-pixel weighted.
- **Shared empty-mask rule.** Both empty → 1.0; exactly one empty → 0.0. Stated once, for every
  metric.

### 3.1 DTAF1
- **Definition.** Precision and recall by EDT tolerance radius d_c, combined as per-class F1.
- **Reductions.** Macro and GT-pixel-weighted; both are always computed.
- **Class-agnostic.** A new class is a config entry, not code.
- **Default tolerances:** road 10 px, building 2 px. **(gap)** Give these in metres: about 3 m
  and 0.6 m on SpaceNet at 0.3 m/px.

### 3.2 clDice
- **Definition.** Each skeleton is compared against the other mask.
- **Width-insensitive:** a 3.3× thick road scores 1.000.
- **Offset-brittle:** a 5 px shift scores 0, because the shifted skeleton falls outside the true
  mask. That brittleness motivates tolerance matching.

### 3.3 Boundary F1 and IoU
- **Definition.** Boundary precision and recall within a tolerance.
- IoU is introduced as the pixel-overlap comparator used throughout.

### 3.4 Point F1
- **Method.** Connected-component centroids, then tolerance-restricted Hungarian matching, not
  greedy.
- **Why not greedy.** Clustered points: greedy finds TP = 1 where Hungarian finds TP = 2
  (Appendix A.6).
- No differentiable analog exists; §7 reformulates point supervision instead.

### 3.5 APLS (raster approximation)
- **Why raster.** The model predicts a mask, not a graph. Vector labels are used only to derive
  intersections.
- **Procedure:**
  - skeletonize;
  - build an 8-connected graph;
  - sample GT control-point pairs in one component;
  - snap them to the prediction graph;
  - score each pair as max(0, 1 − |L_pred − L_gt| / L_gt), and 0 if unmatched.
- **Sensitivity.** Fragmentation lengthens or removes paths, so APLS catches what DTAF1 misses.
- **Eval-only by design:** no tractable differentiable relaxation.

### 3.6 CBHM
- **Definition.** Harmonic mean of mean clDice (linear classes) and mean BF (polygon classes).
- **Deliberately harsh.** One feature type scoring 0 collapses the composite. Example: a 5 px
  shift gives DTAF1 1.000 but CBHM 0.000.
- **(gap)** If a source has no linear class (Potsdam, SpaceNet6), CBHM equals the BF mean rather
  than 0, which is why `cbhm = bf` for those sources in the results tables.

## 4 Why These Metrics? Empirical Motivation

### 4.1 Setup
- **Scene.** 128×128, a 3 px road stripe and a 40×40 building.
- **Sweeps.** Nine single-axis sweeps (`tab:sweeps`): road offset, breakage and thickness;
  building erosion; class imbalance; sparse offset; point jitter, dropout and clutter.

### 4.2 Results
- **Offset.** IoU and clDice reach 0 by 4 px. DTAF1 holds 1.000 to exactly 10 px, then falls to
  0.667 at 12 px and 0.500 from 16 px.
- **Breakage.** IoU falls about linearly. DTAF1 stays at 1.000 from 0 to 90% dropout. This is the
  blind spot.
- **Thickness.** IoU drops from 1.000 to 0.157 at 21 px. clDice and DTAF1 stay at 1.000 through
  13 px.
- **Erosion (control).** All metrics degrade together; IoU was never wrong here.

### 4.3 Interpretation
- **Why width and offset are absorbed.** The tolerance disc absorbs them: every pixel still has a
  same-class pixel within d_c.
- **Why breakage is missed.** The same locality makes DTAF1 blind to breakage: it never asks
  which pixels are connected.
- **Why not just shrink d_c.** Any d_c wide enough to absorb noise is blind to connectivity at
  that scale, so a separate connectivity term is needed. The same trade-off reappears in the loss.

## 5 Analysis and Design for Composite Robustness

### 5.1 Class-imbalance tasks
- **The sweeps.** Vary the road's GT share (0.4–3.1%) or offset a sparse road.
- **Result.** `cbhm` is 0 at every non-trivial level, while `cbhm_soft` stays at 0.758–1.000.
  The harsh composite has no tolerance once one class fails.

### 5.2 Weighted reduction (`cbhm_soft`)
- **Definition.** A GT-pixel-share-weighted arithmetic mean of the type means. It reweights
  scores already computed, at no extra cost.
- **Why arithmetic.** Arithmetic, not geometric: a geometric mean also collapses at 0
  (Appendix A.3–A.4).
- **Case where it only partly helps.** `Khartoum_img371`: a 12 px offset gives `cbhm` 0.000
  against `dtaf1` 0.934. `cbhm_soft` reaches only 0.285, because the road has more GT pixels than
  the building.
- **Contrast.** `Khartoum_img333`: 0.48–0.52 → 0.85.
- **The limit.** Weighting helps only when the failing class is also the pixel minority.

### 5.3 Choosing tolerance radii
- **Validation.** Road 10 px and building 2 px were validated by the offset and erosion sweeps:
  DTAF1 holds exactly to the boundary.
- **Fixed, not learned.** They are physically motivated constants; learned radii are future
  work.
- The same values feed `ToleranceBandLoss`.

## 6 Analysis and Design for Topological Robustness

### 6.1 The connectivity blind spot
- **Formal property.** There are predictions with an arbitrarily large fraction of pixels deleted
  and the skeleton fully disconnected that still score DTAF1 = 1.0 (proof sketch in Appendix A).
- **Empirical witness:** the breakage sweep.

### 6.2 APLS as connectivity recovery
- **Two regimes.** APLS is nearly flat while the skeleton stays connected, then falls sharply.
- **Table `tab:breakage-topo`.** APLS goes from 0.883 to 0.247 between 20% and 30% dropout, while
  DTAF1 shows no transition.

### 6.3 DTAF1-Topo
- **Design.** Per linear class, the harmonic mean of F1 and APLS. Non-linear classes keep F1.
- **Closed-form boundaries:**
  - APLS = 0 gives a score of 0;
  - APLS = 1 and F1 = 1 gives 1;
  - with F1 = 1, the score is 2A / (1 + A) ≥ A.

  **(gap)** The current text says "≤ APLS", which is wrong: 2A / (1 + A) ≥ A. The intended claim
  is that the blend never exceeds F1 and is pulled toward the lower of F1 and APLS.
- **Numerical check.** DTAF1-Topo tracks APLS to near 0 by 50–70% dropout. Validated on
  synthetic data only.
- **Bridge to §7.** Tolerance, width insensitivity and point localization can be trained;
  connectivity cannot, so the loss leaves it out by design.

## 7 A Differentiable Multi-Task Loss (main contribution)

- **Why the metrics cannot be losses.** EDT, skeletonization and Hungarian assignment break
  autograd.
- **What `losses/` is.** Torch analogs of each metric, matching its behaviour without sharing
  code.

### 7.1 Design principle
- **Conventions.** Torch convention: (B, C, H, W) logits, channel = class id.
- **Table `tab:correspondence`:** each metric, its loss term, and what changed to make it
  differentiable. APLS has no analog.

### 7.2 Tolerance-band loss (DTAF1)
- **Precision** needs only a GT-side dilation, precomputed with no gradient.
- **Recall** soft-dilates the probability map with iterated max-pool, which is differentiable,
  so no stop-gradient is needed.
- **Loss** = w(1 − precision) + (1 − w)(1 − recall), with epsilon smoothing.

### 7.3 Soft boundary loss (BF)
- **Boundary map.** The soft boundary is the morphological gradient dilate − erode, which reaches
  1.0 at a hard edge.
- An avg-pool formulation peaked at about 0.56 and capped recall, so it was rejected.
- **Loss.** The tolerance primitive applied to boundary maps.

### 7.4 Soft-clDice (clDice)
- **Method.** Shit et al.'s soft skeletonization, run for a fixed 10 iterations.
- **Limitation.** Width invariance is bounded by the iteration count.
- **Evidence.** 3 px and 9 px predictions give an identical loss (4.77e-7), while plain soft Dice
  gives 0.500.

### 7.5 Point heatmap loss (reformulated, not relaxed)
- **Target.** Gaussian targets at the centroids, with the nearest pixel stamped to 1.0. Without
  the stamp a sub-pixel centroid never fires the positive term (run 1).
- **Loss.** CenterNet focal loss; N_pos counts only point-annotating samples.
- **The negative normalizer Z_neg matters most:**
  - the per-pixel mean makes false positives nearly free;
  - CenterNet's N_pos at full resolution, about 100k background pixels per intersection (40× its
    design ratio), makes the head under-fire;
  - either way, calibrate the threshold on validation tiles.
- **Independent sigmoid head.** It cannot take pixels from road or building, and a point may lie
  on a road. Decoding is NMS peaks over the stitched tile.
- **Mismatch.** The train/eval mismatch with Hungarian point F1 is stated openly.

### 7.6 `PLEMMultiTaskLoss` and multi-source masking
- **Combination.** A weighted sum of CE+Dice, tolerance, clDice and heatmap.
- **Masking.** Before the softmax, masked classes get −1e4 added to their logits, which removes
  them from all terms through one mechanism.
- **Unlabelled areas.** The same mechanism masks unlabelled building areas per patch.
- **Point head.** The sigmoid point head is excluded per sample instead.
- **Return value.** A dict of the named sub-terms, mirroring the metrics' convention.

## 8 Implementation

- **Interfaces.**
  - `metrics/` and `datasets/` take NumPy H×W uint8 maps and return dicts of named scalars.
  - `losses/` is pure torch, kept separate because the metrics' contract is non-differentiable.
- **Tech stack (`tab:techstack`):** numpy/scipy, scikit-image, networkx/cKDTree,
  `linear_sum_assignment`, rasterio/shapely, requests/kaggle, torch, pytest/pandas/matplotlib.
- **Candidate to shrink.** Could shrink to a paragraph or move to an appendix; package-export
  notes are not paper content.

## 9 Experimental Setup

### 9.1 Synthetic benchmark
- The 128×128 scene and the nine sweeps.

### 9.2 Real-data sample
- **SpaceNet.** SN2 buildings and SN3 roads; four cities; about 100 curated tiles. Each 1300 px
  road tile spans 2×2 building tiles.
- **Potsdam.** 15 crops at 6 cm/px, with trees as points.
- **Predictions.** "Pred" is GT perturbed at fixed severities (offset, dropout, thickening,
  erosion, point jitter/dropout/clutter).

### 9.3 First trained model
- **Model.** 3-level U-Net, 483,475 parameters, softmax over bg/road/building.
- **Data.** 100 tiles split 70/15/15 by tile.
- **Training.** Class-weighted CE + Dice, 15 CPU epochs.
- **Status.** A pipeline sanity check, not a benchmark.

### 9.4 Loss ablation setup
- **Scenes.** Same scenes and perturbations as §4.
- **Measurement.** Loss value of a perturbed prediction against GT, encoded as moderately
  confident logits.
- **Configurations.** Five: `ce_dice_only`, `+tolerance`, `+cldice`, `+heatmap`, `full`.
- **Sweeps.** Seven; class imbalance and sparse offset have no loss analog.
- **Confidence-scale note.** Full confidence saturates the focal log term. Its magnitude differs
  from the other terms, which motivates per-term weights.

### 9.5 Joint multi-source setup (historical proof of concept)
- **Point class.** Potsdam trees/cars inside the softmax; predates intersections.
- **Data.** 153 tiles (123 SpaceNet + 30 Potsdam), split per source.
- **No common GSD.** No GSD resampling (0.3–0.5 m vs 6 cm), stated as a limitation.
- **Status.** A pipeline sanity check.

### 9.6 Scaled production setup
- **Sources.**
  - SpaceNet6 SAR is added, as pseudo-RGB. It is illumination-invariant: the real half of the
    night-robustness approach.
  - The synthetic half is low-light augmentation on 40% of patches.
  - In total 1,033 tiles: 550 SpaceNet, 200 Potsdam, 283 SpaceNet6. Split 723/154/156 tiles.
- **Intersections.**
  - Vertices with three or more arms in SN3 vectors; overpasses are not counted.
  - Merged within 20 px.
  - Agrees with geometric noding on 11 of 12 tiles.
  - 8,907 intersections, 16.2 per tile.
  - Scored with point F1 at 10 px (3 m), micro-averaged.
- **Building labels.**
  - An exact tile-extent index; some building tiles are absent from the public set.
  - Coverage is 64% of tile area; building pixels went from 65.7M to 96.6M after the repair.
  - 43% of SpaceNet training patches have building supervision switched off.
  - Evaluation drops predicted buildings in unlabelled areas.
- **Training.**
  - 40 epochs, batch size 48, AMP, cosine schedule, gradient clipping at 5.0, patience 8, A100.
  - The point bias starts at −4.6.
  - The point threshold is chosen on validation tiles only.
- **Evaluation.** Per-source scoring on the annotated classes; a dark test at four severities.

### 9.7 Conventional baseline setup
- **Held fixed:** everything except the loss and the selection rule.
- **Loss.** CE+Dice (parity-tested identical) plus BCE (pos_weight capped at 100, raw ratio
  5,770) + Dice on 5×5 intersection squares. No geometric terms.
- **Selection.** Validation mIoU.
- **Decoding.** Connected-component centroids, at 0.5 and at a validation-calibrated threshold.
- **Scoring.** Conventional metrics and PLEM metrics, paired by tile: bootstrap 95% CI (10k
  resamples) and Wilcoxon test.
- **Two claims tested.** The metric claim (the two kinds of metric disagree) and the loss claim
  (the joint loss wins on PLEM metrics).

## 10 Results

### 10.1 Synthetic sensitivity
- Points to §4 and §6, which hold these results. Could be dropped or merged.

### 10.2 Real-data case studies
- **Blind spot on real data.** On 24 real tiles, 75% road dropout leaves DTAF1 at 1.000 while
  CBHM falls to about 0.57. `dtaf1_weighted` does not fix it.
- **Sparse-class collapse.** `Khartoum_img371` and `Khartoum_img333` (see §5.2).
- **Aggregate.** `cbhm` → `cbhm_soft`: offset at 16 px 0.669 → 0.712; dropout at 75%
  0.568 → 0.646.
- **Points (Potsdam).**
  - dropout: precision stays at 1, recall falls linearly;
  - clutter: precision falls to 0.13 at 20 spurious blobs;
  - jitter: about 0.38 at 6 px.

### 10.3 First trained U-Net
- **Scores.** `cbhm` 0.316, `cbhm_soft` 0.378, `dtaf1` 0.440 (15 tiles).
- **Status.** Shows the library scores real model output sensibly; not a benchmark.

### 10.4 Loss ablation results
- **Tolerance term.** Flat to 10 px of offset, then rises.
- **clDice term.** Jumps by 2 px of offset. Matches CE+Dice up to a 4× width ratio, then
  diverges.
- **Erosion.** The tolerance term tracks it; clDice is a flat negative control.
- **Points.** The heatmap term goes from 0.58 to 70.6 by 8 px of jitter; the other terms stay
  flat.
- **Summary.** Every expected sensitivity moved, and every negative control stayed flat.

### 10.5 Joint training results (proof of concept)
- **Run.** 2 epochs.
- **Loss terms** all decrease; the heatmap term drops 35×.
- **Point F1** is computed end to end for the first time.
- **Below the baseline.** SpaceNet `cbhm` 0.144 against the baseline's 0.316, but at 2 epochs
  against 15, so it is not evidence about the loss.
- **`Vegas_img425`.** CBHM is 0 while the tile looks good: `bf` = 0 while `dtaf1_weighted` is
  0.902. This is the zero-collapse seen on real model output.

### 10.6 Scaled production results
- **Final model.** Run 6, best epoch 38. All loss terms decline.
- **Table `tab:scaled-test`.** SpaceNet `cbhm_soft` 0.685, `dtaf1_topo` 0.482 against `dtaf1`
  0.771 (APLS sees fragmentation). Potsdam 0.568; SpaceNet6 0.350.
- **Intersections.**
  - 547 predicted against 1,389 real, 270 matched; precision 0.494, recall 0.194, F1 0.279;
    threshold 0.1.
  - Separability: median probability 0.046 at junctions against 0.013 on plain road.
  - Recall ceiling: 62% of junctions have a peak even at threshold 0.002.
- **Asymmetry table.** SpaceNet6 gets 544K road pixels along building rows, unsupervised.
- **Dark test.**
  - SpaceNet `cbhm_soft` 0.685 → 0.434 at severity 0.9;
  - SpaceNet6 is flat;
  - intersection F1 0.279 → 0.071.
- **Development record (`tab:scaled-history`).** Six runs, one failure and fix each. A
  development record, not an ablation.

### 10.7 Baseline vs joint loss
- **Metric claim: supported.**
  - SpaceNet ΔmIoU is +0.002 (p = 0.64).
  - Every PLEM metric favours the joint loss with CIs excluding 0: `dtaf1` +0.069, clDice
    +0.043, `cbhm` +0.043, `dtaf1_topo`/`cbhm_soft` +0.038, APLS/BF +0.020.
  - Road IoU +0.036 but building IoU −0.026, while BF is +0.020.
  - Opposite signs on 30 of 83 tiles.
  - Kendall τ between mIoU and the PLEM metrics is 0.47–0.80; for intersections 0.30.
  - Intersection pixel IoU is 0.01–0.03 regardless of F1.
- **SpaceNet6.** mIoU ties, but BF/`cbhm` is 0.071 against 0.350. The baseline predicts almost
  no SAR buildings. SAR layover is suspected but unverified.
- **Loss claim: supported on SpaceNet and SpaceNet6, not on Potsdam.**
  - Intersections: 0.279 against 0.218 (p = 0.015).
  - Potsdam: PLEM metrics tie; building IoU favours the baseline (−0.055).
- **Dark test.** The joint model degrades less on PLEM metrics (0.434 vs 0.302 at 0.9); the mIoU
  gap is small.
- **Caveats.** One seed; Bonferroni (APLS and intersection F1 do not survive).

## 11 Discussion & Limitations

The single home for every caveat; other sections point here.

### 11.1 Evaluation library
- DTAF1-Topo has been run on real model output but not re-run on the 24-tile perturbation sample.
- `cbhm_soft` gives only a partial fix when the sparse-area class is not the pixel minority.
- clDice is brittle on short or sparse roads. Report all four composites together.
- APLS is a raster approximation, not vector APLS.
- Intersection ground truth inherits SpaceNet's vertex-snapping conventions; it was not audited
  at scale.
- DTAF1-Topo inherits APLS's failure modes, which are not yet characterized.

### 11.2 Training loss
- No connectivity term, by design. This is consistent with the `dtaf1_topo` < `dtaf1` gap.
- The heatmap negative normalization has no safe default at full resolution, and recall is
  capped at 62%.
- Point training and evaluation use different formulations.
- Soft-clDice width invariance is bounded (3× validated).
- The heatmap term's magnitude differs from the others, and the weights are untuned.
- SpaceNet6 road-on-buildings confusion: masking leaves it unpenalized.
- The `max_instances` cap is defensive and a no-op on real data.
- There is no common GSD across the sources.

### 11.3 Validation scope
- The first baseline is small and CPU-trained.
- The matched comparison uses one seed per model; gaps of +0.02 to +0.07 could include seed
  variance.
- The point head is still recall-limited (0.194).
- Run 6's building scores mix label repair with model improvement.
- The production runs are a development sequence, not an ablation.
- There is no run without augmentation to isolate its effect on the dark test.

## 12 Conclusion & Future Work

- **Summary of the two halves.**
  - The metric library was built through a synthetic → real → fix → regression-test loop.
  - The loss trains those properties directly.
- **Headline numbers.** `cbhm_soft` 0.685; intersection F1 0.279. The development lessons are a
  result in themselves.
- **Baseline.** mIoU ties while PLEM metrics separate the models; seed variance is the open
  question.
- **Top next steps:**
  1. raise intersection recall (an intermediate negative weight, or a lower-resolution head);
  2. more seeds for the baseline comparison.
- **Further work:**
  - re-run `dtaf1_topo` on the 24-tile sample;
  - fix SpaceNet6 road confusion;
  - separate the label repair from model improvement (re-score run 5);
  - learned tolerances and `iters`;
  - a sweep for DTAF1-Topo on sparse roads;
  - tune the loss weights;
  - test clDice as indirect connectivity pressure;
  - common-GSD resampling.

---

## Appendix A — Formal properties

- **A.1** Shared empty-mask rule holds for every metric, so composites inherit a well-defined 1.0.
- **A.2** DTAF1 macro and weighted values lie in [0, 1]; they are equal when GT counts are equal.
- **A.3** CBHM = 0 if and only if either mean is 0.
- **A.4** `cbhm_soft` = 0 only if every positively weighted type mean is 0.
- **A.5** APLS per-pair and aggregate scores lie in [0, 1].
- **A.6** Hungarian matching is optimal where greedy is not (the clustered-points example). The
  same joint optimization is why it has no useful gradient.
- Loss terms are validated empirically, not proven; say so explicitly.

## Appendix B — Complexity

- **B.1 Notation.** N = H×W, k = instances, B = batch size, iters.
- **B.2–B.4** DTAF1, clDice and BF are O(N) per class.
- **B.5** Point F1 is O(N) plus O(k³) Hungarian.
- **B.6** APLS: Dijkstra grouped by source reduces the work from n_pairs to √n_pairs runs, each
  O((V+E) log V).
- **B.7** Composites add no asymptotic cost.
- **B.8–B.9** Tolerance and boundary losses are O(radius · BN).
- **B.10** Soft-clDice is O(iters · BN).
- **B.11** Heatmap target construction is O(k′N), the only term that scales with instance count.
- **B.12** The total is O(iters · BN + k′N).
