"""
Sanity / unit tests for all PLEM metrics.

Each test verifies a fundamental behavioural property that any valid extraction
metric must satisfy. Run with:  pytest tests/test_metrics_sanity.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics.dtaf1 import dtaf1, dtaf1_road_building
from metrics.cldice import cldice
from metrics.boundary_f1 import boundary_f1, iou
from metrics.unified import cbhm, evaluate_all


# ---------------------------------------------------------------------------
# Fixtures: synthetic 64×64 label maps
# ---------------------------------------------------------------------------

@pytest.fixture
def canvas():
    """Return a blank 64×64 canvas."""
    return np.zeros((64, 64), dtype=np.uint8)


def make_road(shape=(64, 64), col=32, thickness=2):
    """Vertical road stripe through the centre of the image."""
    m = np.zeros(shape, dtype=np.uint8)
    half = thickness // 2
    m[:, col - half: col + half + 1] = 1
    return m


def make_building(shape=(64, 64), top=10, left=10, height=20, width=20):
    """Rectangular building block."""
    m = np.zeros(shape, dtype=np.uint8)
    m[top: top + height, left: left + width] = 2
    return m


def make_combined(road_mask, building_mask):
    """Merge road (class 1) and building (class 2) into one label map."""
    out = np.zeros_like(road_mask)
    out[road_mask > 0] = 1
    out[building_mask > 0] = 2
    return out


ROAD = make_road()
BUILDING = make_building()
GT = make_combined(ROAD, BUILDING)

ROAD_CONFIG = {
    1: {"name": "road",     "tolerance": 10},
    2: {"name": "building", "tolerance":  2},
}


# ---------------------------------------------------------------------------
# 1. Perfect prediction → score == 1.0
# ---------------------------------------------------------------------------

class TestPerfectPrediction:
    def test_dtaf1_perfect(self):
        result = dtaf1(GT, GT, ROAD_CONFIG)
        assert result["dtaf1"] == pytest.approx(1.0), \
            "DTAF1 should be 1.0 when pred == GT"

    def test_cldice_perfect(self):
        result = cldice(ROAD, ROAD)
        assert result["cldice"] == pytest.approx(1.0), \
            "clDice should be 1.0 when pred == GT"

    def test_bf_perfect(self):
        result = boundary_f1(BUILDING, BUILDING, tolerance=2)
        assert result["bf"] == pytest.approx(1.0), \
            "BF should be 1.0 when pred == GT"

    def test_cbhm_perfect(self):
        result = cbhm(GT, GT, linear_classes=[1], polygon_classes=[2])
        assert result["cbhm"] == pytest.approx(1.0), \
            "CBHM should be 1.0 when pred == GT"

    def test_iou_perfect(self):
        building_bin = (BUILDING == 2).astype(np.uint8)
        assert iou(building_bin, building_bin) == pytest.approx(1.0), \
            "IoU should be 1.0 when pred == GT"


# ---------------------------------------------------------------------------
# 2. Null prediction (all zeros) → score == 0.0
# ---------------------------------------------------------------------------

class TestNullPrediction:
    def test_dtaf1_null(self):
        null = np.zeros_like(GT)
        result = dtaf1(null, GT, ROAD_CONFIG)
        assert result["dtaf1"] == pytest.approx(0.0), \
            "DTAF1 should be 0.0 for null prediction"

    def test_cldice_null(self):
        null = np.zeros_like(ROAD)
        result = cldice(null, ROAD)
        assert result["cldice"] == pytest.approx(0.0), \
            "clDice should be 0.0 for null prediction"

    def test_bf_null(self):
        null = np.zeros_like(BUILDING)
        result = boundary_f1(null, BUILDING, tolerance=2)
        assert result["bf"] == pytest.approx(0.0), \
            "BF should be 0.0 for null prediction"

    def test_cbhm_null(self):
        null = np.zeros_like(GT)
        result = cbhm(null, GT, linear_classes=[1], polygon_classes=[2])
        assert result["cbhm"] == pytest.approx(0.0), \
            "CBHM should be 0.0 for null prediction"

    def test_iou_null(self):
        null = np.zeros((64, 64), dtype=np.uint8)
        building_gt = (BUILDING == 2).astype(np.uint8)
        assert iou(null, building_gt) == pytest.approx(0.0), \
            "IoU should be 0.0 for null prediction"


# ---------------------------------------------------------------------------
# 3. Both GT and pred are empty → score == 1.0 (vacuously correct)
# ---------------------------------------------------------------------------

class TestBothEmpty:
    def test_cldice_both_empty(self):
        empty = np.zeros((32, 32), dtype=np.uint8)
        result = cldice(empty, empty)
        assert result["cldice"] == pytest.approx(1.0)

    def test_bf_both_empty(self):
        empty = np.zeros((32, 32), dtype=np.uint8)
        result = boundary_f1(empty, empty)
        assert result["bf"] == pytest.approx(1.0)

    def test_iou_both_empty(self):
        empty = np.zeros((32, 32), dtype=np.uint8)
        assert iou(empty, empty) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Class-specific failure isolation
#    Perfect road, wrong buildings → metric penalises buildings
#    Perfect buildings, broken road → metric penalises roads
# ---------------------------------------------------------------------------

class TestClassIsolation:
    def test_perfect_road_wrong_building(self):
        # Correct road, completely missing building
        pred_no_building = np.where(GT == 2, 0, GT)
        result = dtaf1(pred_no_building, GT, ROAD_CONFIG)
        road_f1 = result["per_class"][1]["f1"]
        building_f1 = result["per_class"][2]["f1"]
        assert road_f1 == pytest.approx(1.0), \
            "Road F1 should be 1.0 when road is correct"
        assert building_f1 == pytest.approx(0.0), \
            "Building F1 should be 0.0 when building is missing"
        assert result["dtaf1"] < 1.0, \
            "Unified score should be <1 when a class fails"

    def test_perfect_building_broken_road(self):
        # Correct building, completely missing road
        pred_no_road = np.where(GT == 1, 0, GT)
        result = dtaf1(pred_no_road, GT, ROAD_CONFIG)
        road_f1 = result["per_class"][1]["f1"]
        building_f1 = result["per_class"][2]["f1"]
        assert building_f1 == pytest.approx(1.0, abs=0.05), \
            "Building F1 should be ~1 when building is correct"
        assert road_f1 == pytest.approx(0.0), \
            "Road F1 should be 0.0 when road is missing"
        assert result["dtaf1"] < 1.0, \
            "Unified score should be <1 when a class fails"

    def test_cbhm_penalises_missing_road(self):
        pred_no_road = np.where(GT == 1, 0, GT)
        result = cbhm(pred_no_road, GT, linear_classes=[1], polygon_classes=[2])
        assert result["cldice_mean"] == pytest.approx(0.0), \
            "clDice should be 0 when road is missing"
        assert result["cbhm"] == pytest.approx(0.0), \
            "CBHM harmonic mean collapses to 0 when either component is 0"

    def test_cbhm_penalises_missing_building(self):
        pred_no_building = np.where(GT == 2, 0, GT)
        result = cbhm(pred_no_building, GT, linear_classes=[1], polygon_classes=[2])
        assert result["bf_mean"] == pytest.approx(0.0), \
            "BF should be 0 when building is missing"
        assert result["cbhm"] == pytest.approx(0.0), \
            "CBHM harmonic mean collapses to 0 when either component is 0"


# ---------------------------------------------------------------------------
# 5. Road width insensitivity — DTAF1 and clDice vs IoU
#    A thickened road should score well on DTAF1/clDice but poorly on IoU
# ---------------------------------------------------------------------------

class TestRoadWidthInsensitivity:
    def _make_road_mask(self, thickness):
        return make_road(thickness=thickness)

    def test_thick_road_iou_low(self):
        gt_road = self._make_road_mask(thickness=2)
        pred_road = self._make_road_mask(thickness=10)   # 5× too thick
        score = iou(pred_road, gt_road)
        # Thick road union >> intersection → low IoU
        assert score < 0.5, f"IoU={score:.3f} expected <0.5 for over-thick road"

    def test_thick_road_cldice_high(self):
        gt_road = self._make_road_mask(thickness=2)
        pred_road = self._make_road_mask(thickness=10)
        result = cldice(pred_road, gt_road)
        # clDice: skeleton of thick road still lies within GT mask → high score
        assert result["cldice"] > 0.8, \
            f"clDice={result['cldice']:.3f} expected >0.8 for over-thick road (correct centerline)"

    def test_thick_road_dtaf1_high(self):
        gt_road = self._make_road_mask(thickness=2)
        pred_road = self._make_road_mask(thickness=10)
        # Use single-class config for this test
        cfg = {1: {"name": "road", "tolerance": 10}}
        gt_map = np.where(gt_road > 0, 1, 0).astype(np.uint8)
        pred_map = np.where(pred_road > 0, 1, 0).astype(np.uint8)
        result = dtaf1(pred_map, gt_map, cfg)
        assert result["dtaf1"] > 0.8, \
            f"DTAF1={result['dtaf1']:.3f} expected >0.8 for over-thick road (centerline correct)"


# ---------------------------------------------------------------------------
# 6. Road offset sensitivity — small shifts should affect IoU more than DTAF1
# ---------------------------------------------------------------------------

class TestRoadOffsetTolerance:
    def test_small_offset_dtaf1_higher_than_iou(self):
        gt_road = make_road(col=32, thickness=2)
        pred_road = make_road(col=35, thickness=2)   # 3-pixel offset

        # IoU
        iou_score = iou(pred_road, gt_road)

        # DTAF1 with 10px road tolerance
        cfg = {1: {"name": "road", "tolerance": 10}}
        gt_map = np.where(gt_road > 0, 1, 0).astype(np.uint8)
        pred_map = np.where(pred_road > 0, 1, 0).astype(np.uint8)
        dtaf1_score = dtaf1(pred_map, gt_map, cfg)["dtaf1"]

        assert dtaf1_score > iou_score, (
            f"DTAF1={dtaf1_score:.3f} should exceed IoU={iou_score:.3f} "
            "for a small road offset within tolerance"
        )

    def test_large_offset_dtaf1_penalised(self):
        gt_road = make_road(col=10, thickness=2)
        pred_road = make_road(col=50, thickness=2)  # 40-pixel offset > tolerance

        cfg = {1: {"name": "road", "tolerance": 10}}
        gt_map = np.where(gt_road > 0, 1, 0).astype(np.uint8)
        pred_map = np.where(pred_road > 0, 1, 0).astype(np.uint8)
        result = dtaf1(pred_map, gt_map, cfg)
        assert result["dtaf1"] < 0.3, \
            f"DTAF1={result['dtaf1']:.3f} should be low when offset >> tolerance"


# ---------------------------------------------------------------------------
# 7. Building erosion — proportional degradation
# ---------------------------------------------------------------------------

class TestBuildingErosion:
    def test_eroded_building_lower_score(self):
        from scipy.ndimage import binary_erosion
        gt_b = (BUILDING == 2).astype(np.uint8)
        pred_b = binary_erosion(gt_b, iterations=3).astype(np.uint8)
        result_full = boundary_f1(gt_b, gt_b, tolerance=2)
        result_eroded = boundary_f1(pred_b, gt_b, tolerance=2)
        assert result_eroded["bf"] < result_full["bf"], \
            "Eroded building should score lower than perfect prediction"
        assert result_eroded["bf"] > 0.0, \
            "Eroded building should still score > 0 (partially correct)"


# ---------------------------------------------------------------------------
# 8. Monotonicity — evaluate_all score decreases as degradation increases
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_dtaf1_decreases_with_more_road_pixels_removed(self):
        rng = np.random.default_rng(42)
        road_pixels = np.argwhere(GT == 1)
        scores = []
        for frac in [0.0, 0.25, 0.5, 0.75]:
            pred = GT.copy()
            if frac > 0:
                n_remove = int(frac * len(road_pixels))
                idx = rng.choice(len(road_pixels), n_remove, replace=False)
                for r, c in road_pixels[idx]:
                    pred[r, c] = 0
            result = dtaf1(pred, GT, ROAD_CONFIG)
            scores.append(result["dtaf1"])

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"DTAF1 not monotonically decreasing: scores={scores}"
            )
