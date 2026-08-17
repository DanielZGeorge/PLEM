"""
Sanity / unit tests for the losses/ package (differentiable training losses).

Each test class follows the same triad: (1) perfect prediction -> loss ~= 0;
(2) degenerate empty-mask case -> well-defined, finite, no NaN/Inf; (3) finite
gradients + loss decreases after one optimizer step on a toy logits tensor.
Mirrors tests/test_metrics_sanity.py's pattern/fixtures, adapted to torch
tensors. Run with: pytest tests/test_losses_sanity.py -v
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from losses.tolerance import ToleranceBandLoss, soft_tolerance_pair_loss
from losses.boundary import SoftBoundaryLoss
from losses.cldice import SoftClDiceLoss, soft_cldice
from losses.heatmap import PointHeatmapLoss, gt_centroids_to_heatmap
from losses.multitask import PLEMMultiTaskLoss


# ---------------------------------------------------------------------------
# Fixtures: synthetic 32x32 label maps -> torch tensors
# ---------------------------------------------------------------------------

NUM_CLASSES = 4  # background, road, building, point
SHAPE = (32, 32)


def make_road_label(shape=SHAPE, col=16, thickness=3):
    m = np.zeros(shape, dtype=np.int64)
    half = thickness // 2
    m[:, col - half: col + half + 1] = 1
    return m


def make_building_label(shape=SHAPE, top=4, left=4, height=8, width=8):
    m = np.zeros(shape, dtype=np.int64)
    m[top: top + height, left: left + width] = 2
    return m


def make_point_label(shape=SHAPE, coords=((6, 6), (6, 24), (24, 14)), radius=1):
    m = np.zeros(shape, dtype=np.int64)
    for r, c in coords:
        m[max(0, r - radius): r + radius + 1, max(0, c - radius): c + radius + 1] = 3
    return m


def combine(*masks):
    out = np.zeros_like(masks[0])
    for m in masks:
        out[m > 0] = m[m > 0]
    return out


def to_batch(label: np.ndarray) -> torch.Tensor:
    """(H, W) int64 numpy -> (1, H, W) int64 tensor."""
    return torch.from_numpy(label).long().unsqueeze(0)


def perfect_logits(label: np.ndarray, num_classes: int = NUM_CLASSES, scale: float = 12.0) -> torch.Tensor:
    """(H, W) label -> (1, C, H, W) logits whose softmax is ~= one-hot(label)."""
    label_t = torch.from_numpy(label).long()
    onehot = F.one_hot(label_t, num_classes).permute(2, 0, 1).float()
    logits = onehot * scale - (1.0 - onehot) * scale
    return logits.unsqueeze(0)


def uniform_logits(shape=SHAPE, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """All-zero logits -> uniform softmax probability everywhere."""
    return torch.zeros((1, num_classes) + shape)


def background_only_logits(shape=SHAPE, num_classes: int = NUM_CLASSES, scale: float = 12.0) -> torch.Tensor:
    """A confident all-background ('null') prediction."""
    label = np.zeros(shape, dtype=np.int64)
    return perfect_logits(label, num_classes, scale)


ROAD = make_road_label()
BUILDING = make_building_label()
POINTS = make_point_label()
GT3 = combine(ROAD, BUILDING, POINTS)
GT3_BATCH = to_batch(GT3)

ROAD_CONFIG = {1: {"name": "road", "tolerance": 6}}
BUILDING_CONFIG = {2: {"name": "building", "tolerance": 2}}
ROAD_BUILDING_CONFIG = {**ROAD_CONFIG, **BUILDING_CONFIG}


def assert_finite(t: torch.Tensor):
    assert torch.isfinite(t).all(), f"expected finite tensor, got {t}"


def one_step_decreases(module_fn, logits: torch.Tensor, target: torch.Tensor, steps: int = 5):
    """Optimize a toy logits Parameter against `module_fn` for a few Adam
    steps and assert the loss decreased."""
    param = torch.nn.Parameter(logits.clone())
    opt = torch.optim.Adam([param], lr=0.5)
    first = None
    last = None
    for i in range(steps):
        opt.zero_grad()
        loss = module_fn(param, target)
        if i == 0:
            first = loss.item()
        loss.backward()
        assert_finite(param.grad)
        opt.step()
        last = loss.item()
    return first, last


# ---------------------------------------------------------------------------
# ToleranceBandLoss (DTAF1 analog)
# ---------------------------------------------------------------------------

class TestToleranceBandLoss:
    def test_perfect_prediction_near_zero(self):
        logits = perfect_logits(GT3)
        loss_fn = ToleranceBandLoss(ROAD_BUILDING_CONFIG)
        loss = loss_fn(logits, GT3_BATCH)
        assert_finite(loss)
        assert loss.item() < 0.05, f"perfect prediction should have near-zero loss, got {loss.item():.4f}"

    def test_empty_mask_well_defined(self):
        empty_label = np.zeros(SHAPE, dtype=np.int64)
        logits = perfect_logits(empty_label)
        target = to_batch(empty_label)
        loss_fn = ToleranceBandLoss(ROAD_BUILDING_CONFIG)
        loss = loss_fn(logits, target)
        assert_finite(loss)
        assert loss.item() < 0.05, "both pred and GT empty for these classes should be near-zero loss"

    def test_gradient_flow_and_decrease(self):
        logits = uniform_logits()
        loss_fn = ToleranceBandLoss(ROAD_BUILDING_CONFIG)
        first, last = one_step_decreases(loss_fn, logits, GT3_BATCH)
        assert last < first, f"loss should decrease after optimizer steps: {first:.4f} -> {last:.4f}"

    def test_null_prediction_high_loss(self):
        logits = background_only_logits()
        loss_fn = ToleranceBandLoss(ROAD_BUILDING_CONFIG)
        loss = loss_fn(logits, GT3_BATCH)
        assert_finite(loss)
        assert loss.item() > 0.5, f"confidently-wrong (all background) prediction should have high loss, got {loss.item():.4f}"


# ---------------------------------------------------------------------------
# SoftBoundaryLoss (Boundary F1 analog)
# ---------------------------------------------------------------------------

class TestSoftBoundaryLoss:
    def test_perfect_prediction_near_zero(self):
        logits = perfect_logits(GT3)
        loss_fn = SoftBoundaryLoss(BUILDING_CONFIG)
        loss = loss_fn(logits, GT3_BATCH)
        assert_finite(loss)
        assert loss.item() < 0.1, f"perfect prediction should have near-zero loss, got {loss.item():.4f}"

    def test_empty_mask_well_defined(self):
        empty_label = np.zeros(SHAPE, dtype=np.int64)
        logits = perfect_logits(empty_label)
        target = to_batch(empty_label)
        loss_fn = SoftBoundaryLoss(BUILDING_CONFIG)
        loss = loss_fn(logits, target)
        assert_finite(loss)

    def test_gradient_flow_and_decrease(self):
        # SoftBoundaryLoss responds to LOCAL spatial variation (it's a
        # morphological-gradient/edge detector), so a perfectly flat/uniform
        # starting point is a genuine degenerate case: max-pool/min-pool have
        # zero subgradient at an exactly-tied, spatially-constant input,
        # unlike a real CNN's logits which always have some spatial variation
        # even at initialization. A small fixed random perturbation gives the
        # optimizer something to work with, matching realistic init.
        torch.manual_seed(0)
        logits = uniform_logits() + torch.randn(uniform_logits().shape) * 0.05
        loss_fn = SoftBoundaryLoss(BUILDING_CONFIG)
        first, last = one_step_decreases(loss_fn, logits, GT3_BATCH)
        assert last < first, f"loss should decrease after optimizer steps: {first:.4f} -> {last:.4f}"


# ---------------------------------------------------------------------------
# SoftClDiceLoss (clDice analog) -- includes the width-insensitivity check
# ---------------------------------------------------------------------------

class TestSoftClDiceLoss:
    def test_perfect_prediction_near_zero(self):
        logits = perfect_logits(GT3)
        loss_fn = SoftClDiceLoss(linear_classes=[1])
        loss = loss_fn(logits, GT3_BATCH)
        assert_finite(loss)
        assert loss.item() < 0.1, f"perfect prediction should have near-zero loss, got {loss.item():.4f}"

    def test_empty_mask_well_defined(self):
        empty_label = np.zeros(SHAPE, dtype=np.int64)
        logits = perfect_logits(empty_label)
        target = to_batch(empty_label)
        loss_fn = SoftClDiceLoss(linear_classes=[1])
        loss = loss_fn(logits, target)
        assert_finite(loss)
        assert loss.item() < 0.1, "no road anywhere in pred or GT should be near-zero loss"

    def test_gradient_flow_and_decrease(self):
        logits = uniform_logits()
        loss_fn = SoftClDiceLoss(linear_classes=[1])
        first, last = one_step_decreases(loss_fn, logits, GT3_BATCH)
        assert last < first, f"loss should decrease after optimizer steps: {first:.4f} -> {last:.4f}"

    def test_width_insensitivity(self):
        """
        A thickened prediction along the SAME centerline should not increase
        soft-clDice loss much -- the single most important regression test
        here, since soft-skeletonization is easy to get numerically wrong
        (e.g. degenerating toward plain Dice, which very much IS
        width-sensitive). Mirrors
        test_metrics_sanity.py::TestRoadWidthInsensitivity, but on the
        differentiable loss instead of the eval metric.
        """
        gt_label = make_road_label(thickness=3)
        thin_pred_label = make_road_label(thickness=3)
        thick_pred_label = make_road_label(thickness=9)

        target = to_batch(gt_label)
        loss_fn = SoftClDiceLoss(linear_classes=[1])

        thin_loss = loss_fn(perfect_logits(thin_pred_label), target).item()
        thick_loss = loss_fn(perfect_logits(thick_pred_label), target).item()

        assert thin_loss < 0.1, f"exact-width match should be near-zero loss, got {thin_loss:.4f}"
        assert thick_loss < 0.25, (
            f"a 3x-thicker prediction on the same centerline should still score a low "
            f"soft-clDice loss (width-insensitivity), got {thick_loss:.4f}"
        )

        # Contrast: plain Dice (IoU-like) IS width-sensitive -- this confirms
        # soft-clDice is doing something meaningfully different, not just
        # numerically saturating to ~0 for every input.
        thick_probs = torch.softmax(perfect_logits(thick_pred_label), dim=1)[:, 1]
        gt_onehot = F.one_hot(torch.from_numpy(gt_label).long(), NUM_CLASSES).permute(2, 0, 1).float()[1].unsqueeze(0)
        dice = (2 * (thick_probs * gt_onehot).sum() + 1e-6) / (thick_probs.sum() + gt_onehot.sum() + 1e-6)
        assert dice.item() < 0.6, "sanity check: a 3x-too-thick road really should hurt plain Dice"


# ---------------------------------------------------------------------------
# PointHeatmapLoss (Point F1 analog)
# ---------------------------------------------------------------------------

class TestPointHeatmapLoss:
    def test_perfect_prediction_near_minimum(self):
        logits = perfect_logits(GT3)
        loss_fn = PointHeatmapLoss(point_classes=[3])
        loss = loss_fn(logits, GT3_BATCH)
        assert_finite(loss)
        assert loss.item() < 0.2, f"perfect centroid alignment should have low loss, got {loss.item():.4f}"

    def test_empty_gt_well_defined(self):
        empty_label = np.zeros(SHAPE, dtype=np.int64)
        logits = perfect_logits(empty_label)
        target = to_batch(empty_label)
        loss_fn = PointHeatmapLoss(point_classes=[3])
        loss = loss_fn(logits, target)
        assert_finite(loss)
        assert loss.item() < 0.1, "no GT points and no predicted points should be low loss"

    def test_gradient_flow_and_decrease(self):
        logits = uniform_logits()
        loss_fn = PointHeatmapLoss(point_classes=[3])
        first, last = one_step_decreases(loss_fn, logits, GT3_BATCH)
        assert last < first, f"loss should decrease after optimizer steps: {first:.4f} -> {last:.4f}"

    def test_heatmap_normalized_and_clipped(self):
        point_mask = torch.from_numpy((POINTS > 0).astype(np.float32)).unsqueeze(0)
        heatmap = gt_centroids_to_heatmap(point_mask, sigma=2.0)
        assert_finite(heatmap)
        assert heatmap.min().item() >= 0.0
        assert heatmap.max().item() <= 1.0 + 1e-6

    def test_nearby_points_do_not_exceed_one(self):
        """Two GT points close together should combine via elementwise max,
        not sum -- otherwise overlapping Gaussians could inflate past 1.0."""
        close_points = np.zeros(SHAPE, dtype=np.uint8)
        close_points[15, 15] = 1
        close_points[15, 17] = 1  # 2px away -- Gaussian tails will overlap
        mask = torch.from_numpy((close_points > 0).astype(np.float32)).unsqueeze(0)
        heatmap = gt_centroids_to_heatmap(mask, sigma=2.0)
        assert heatmap.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# PLEMMultiTaskLoss (combined wrapper + class masking)
# ---------------------------------------------------------------------------

class TestPLEMMultiTaskLoss:
    def _make_loss(self):
        return PLEMMultiTaskLoss(
            class_config={1: {"name": "road", "tolerance": 6}, 2: {"name": "building", "tolerance": 2}},
            linear_classes=[1],
            polygon_classes=[2],
            point_classes=[3],
        )

    def test_perfect_prediction_low_loss_and_all_terms_present(self):
        logits = perfect_logits(GT3)
        loss_fn = self._make_loss()
        full_mask = torch.ones((1, NUM_CLASSES))
        out = loss_fn(logits, GT3_BATCH, full_mask)
        assert_finite(out["loss"])
        for key in ("ce_dice", "tolerance", "cldice", "heatmap"):
            assert key in out, f"missing sub-term '{key}' in returned dict"
        assert out["loss"].item() < 0.5, f"perfect prediction should have low total loss, got {out['loss'].item():.4f}"

    def test_gradient_flow_and_decrease(self):
        logits = uniform_logits()
        full_mask = torch.ones((1, NUM_CLASSES))
        loss_fn = self._make_loss()
        param = torch.nn.Parameter(logits.clone())
        opt = torch.optim.Adam([param], lr=0.5)
        first = loss_fn(param, GT3_BATCH, full_mask)["loss"].item()
        for _ in range(5):
            opt.zero_grad()
            out = loss_fn(param, GT3_BATCH, full_mask)
            out["loss"].backward()
            assert_finite(param.grad)
            opt.step()
        last = loss_fn(param, GT3_BATCH, full_mask)["loss"].item()
        assert last < first, f"total loss should decrease after optimizer steps: {first:.4f} -> {last:.4f}"

    def test_class_masking_zeroes_gradient_for_masked_class(self):
        """
        A SpaceNet-like sample (road+building GT only, no point pixels) with
        the point class masked out: perturbing ONLY the point-class logit
        channel must not change the total loss, since PLEMMultiTaskLoss
        biases masked channels to a large negative logit before any softmax
        is taken (see losses/multitask.py's _mask_logits docstring).
        """
        road_building_label = combine(ROAD, BUILDING)  # no point pixels, matches a real SpaceNet tile
        target = to_batch(road_building_label)
        base_logits = perfect_logits(road_building_label)

        class_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # point class (id 3) masked out
        loss_fn = self._make_loss()

        out_base = loss_fn(base_logits, target, class_mask)

        perturbed_logits = base_logits.clone()
        perturbed_logits[:, 3] += 50.0  # large perturbation to the masked-out point channel
        out_perturbed = loss_fn(perturbed_logits, target, class_mask)

        assert out_base["loss"].item() == pytest.approx(out_perturbed["loss"].item(), abs=1e-4), (
            "perturbing a masked-out class's logits should not change the total loss: "
            f"{out_base['loss'].item():.6f} vs {out_perturbed['loss'].item():.6f}"
        )

    def test_class_masking_gradient_is_near_zero(self):
        """Same scenario as above, checked via .grad directly: the masked
        point channel's gradient should be near-zero after backward(). Starts
        from uniform (not perfect) logits -- at a perfect/near-optimal
        prediction ALL channels' gradients are near-zero simply because the
        loss is near its minimum everywhere, which would defeat the point of
        this test (it needs the unmasked channels to have a real, nonzero
        gradient to contrast against)."""
        road_building_label = combine(ROAD, BUILDING)
        target = to_batch(road_building_label)
        logits = uniform_logits().clone().requires_grad_(True)
        class_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        loss_fn = self._make_loss()
        out = loss_fn(logits, target, class_mask)
        out["loss"].backward()

        assert_finite(logits.grad)
        point_channel_grad = logits.grad[:, 3].abs().max().item()
        other_channel_grad = logits.grad[:, :3].abs().max().item()
        assert point_channel_grad < 1e-3, f"masked point-channel gradient should be ~0, got {point_channel_grad:.6f}"
        assert other_channel_grad > 1e-6, "unmasked channels should still receive a real gradient"
