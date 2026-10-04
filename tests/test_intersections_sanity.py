"""
Sanity tests for datasets/intersections.py and the intersection (`points`)
plumbing in datasets/joint.py. Network-free: pure functions and synthetic
on-disk caches only, consistent with the repo convention that dataset
acquisition itself is not unit-tested.
"""

import os
import sys

import numpy as np
from affine import Affine
from shapely.geometry import LineString, MultiLineString

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.intersections import (
    road_intersections, merge_close_points, intersections_to_pixels, points_to_mask,
)
from datasets.joint import load_joint_tiles, class_mask_for_source, SOURCE_CLASSES


def roads(*lines):
    return [(ln, {}) for ln in lines]


class TestRoadIntersections:
    def test_x_junction_of_four_segments(self):
        c = (5.0, 5.0)
        pts = road_intersections(roads(
            LineString([(0, 5), c]), LineString([c, (10, 5)]),
            LineString([(5, 0), c]), LineString([c, (5, 10)]),
        ))
        assert pts == [c]

    def test_t_junction_with_unsplit_through_road(self):
        """Through road has the junction as an INTERIOR vertex (2 arms) and the
        side road ends there (1 arm) -> 3 arms."""
        pts = road_intersections(roads(
            LineString([(0, 0), (5, 0), (10, 0)]), LineString([(5, 0), (5, 8)]),
        ))
        assert pts == [(5.0, 0.0)]

    def test_bend_and_dead_end_are_not_intersections(self):
        assert road_intersections(roads(LineString([(0, 0), (5, 0), (5, 5)]))) == []
        assert road_intersections(roads(
            LineString([(0, 0), (5, 0)]), LineString([(5, 0), (9, 3)]),
        )) == []

    def test_overpass_without_shared_vertex_is_not_counted(self):
        pts = road_intersections(roads(
            LineString([(0, 5), (10, 5)]), LineString([(5, 0), (5, 10)]),
        ))
        assert pts == []

    def test_multilinestring_parts_are_counted(self):
        pts = road_intersections(roads(
            MultiLineString([[(0, 0), (5, 0)], [(5, 0), (10, 0)]]), LineString([(5, 0), (5, 6)]),
        ))
        assert pts == [(5.0, 0.0)]

    def test_empty_input(self):
        assert road_intersections([]) == []


class TestPixelConversion:
    # Identity transform + same CRS on both sides: x == col, y == row.
    CRS = "EPSG:32611"

    def _to_px(self, pts, shape=(100, 100), merge_px=8.0):
        return intersections_to_pixels(pts, self.CRS, Affine.identity(), shape,
                                       merge_px=merge_px, src_crs=self.CRS)

    def test_row_col_order_and_out_of_tile_dropped(self):
        out = self._to_px([(30.0, 10.0), (150.0, 10.0), (-4.0, 2.0)])
        assert out.shape == (1, 2) and out.dtype == np.float32
        np.testing.assert_allclose(out[0], [10.0, 30.0], atol=1e-4)  # (row, col) = (y, x)

    def test_close_nodes_merge_far_nodes_do_not(self):
        out = self._to_px([(20.0, 20.0), (23.0, 20.0), (70.0, 70.0)])
        assert len(out) == 2
        merged = out[np.argmin(out[:, 0])]
        np.testing.assert_allclose(merged, [20.0, 21.5], atol=1e-4)

    def test_merge_is_transitive_along_a_chain(self):
        chain = np.array([[10, 10], [10, 16], [10, 22]], dtype=np.float32)  # 6 px apart each
        assert len(merge_close_points(chain, merge_px=8)) == 1

    def test_empty(self):
        assert self._to_px([]).shape == (0, 2)


class TestPointsToMask:
    def test_stamp_size_and_centroid(self):
        mask = points_to_mask(np.array([[10.2, 20.7]]), (40, 40), radius=2)
        assert mask.dtype == np.uint8 and mask.sum() == 25
        rr, cc = np.nonzero(mask)
        assert (rr.mean(), cc.mean()) == (10.0, 21.0)

    def test_edge_point_is_clipped_not_dropped(self):
        mask = points_to_mask(np.array([[0.0, 0.0]]), (40, 40), radius=2)
        assert mask.sum() == 9

    def test_no_points(self):
        assert points_to_mask(np.zeros((0, 2)), (8, 8)).sum() == 0


class TestJointPointPlumbing:
    def test_source_classes(self):
        assert SOURCE_CLASSES["spacenet"] == [1, 2, 3]
        assert SOURCE_CLASSES["potsdam"] == [2]
        np.testing.assert_array_equal(class_mask_for_source("spacenet"), [1, 1, 1, 1])
        np.testing.assert_array_equal(class_mask_for_source("potsdam"), [1, 0, 1, 0])

    def test_load_joint_tiles_points_and_potsdam_point_strip(self, tmp_path, capsys):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        sn = tmp_path / "spacenet" / "Vegas"
        sn.mkdir(parents=True)
        lab = np.zeros((16, 16), dtype=np.uint8)
        lab[8, :] = 1
        np.savez_compressed(sn / "Vegas_img1.npz", image=img, label=lab,
                            points=np.array([[8.0, 4.0]], dtype=np.float32))
        np.savez_compressed(sn / "Vegas_img2.npz", image=img, label=lab)  # pre-intersection cache

        pot = tmp_path / "potsdam"
        pot.mkdir()
        plab = np.zeros((16, 16), dtype=np.uint8)
        plab[:4, :4] = 2
        plab[10:12, 10:12] = 3  # old tree/car point class
        np.savez_compressed(pot / "crop1.npz", image=img, label=plab)

        tiles = {t["tile"]: t for t in load_joint_tiles(sn.parent, pot, tmp_path / "none")}
        assert tiles["Vegas_img1"]["points"].shape == (1, 2)
        assert tiles["Vegas_img2"]["points"].shape == (0, 2)
        assert "add_intersections_to_cache" in capsys.readouterr().out
        p = tiles["crop1"]
        assert p["points"].shape == (0, 2)
        assert not (p["label"] == 3).any() and (p["label"] == 2).sum() == 16


class TestBuildingCoverage:
    """datasets/spacenet.py: which building tiles overlap a road tile, and the
    `building_valid` mask marking where building labels are actually known."""

    # A road tile covering [0, 2] x [0, 2]; four building tiles form its 2x2 block.
    ROAD = (0.0, 0.0, 2.0, 2.0)
    INDEX = {
        1: (0.0, 0.0, 1.0, 1.0), 2: (1.0, 0.0, 2.0, 1.0), 3: (0.0, 1.0, 1.0, 2.0),
        10: (2.0, 0.0, 3.0, 1.0),   # shares only an edge with the road tile
        11: (5.0, 5.0, 6.0, 6.0),   # far away
    }

    def test_overlapping_tiles_exclude_edge_touching_and_distant(self):
        from datasets.spacenet import overlapping_building_tiles
        assert overlapping_building_tiles(self.ROAD, self.INDEX) == [1, 2, 3]

    def test_sliver_overlap_is_excluded(self):
        from datasets.spacenet import overlapping_building_tiles
        index = {7: (1.995, 0.0, 2.995, 1.0)}  # 0.5% of its area inside the road tile
        assert overlapping_building_tiles(self.ROAD, index) == []

    def test_building_valid_mask_marks_only_covered_area(self):
        from datasets.spacenet import building_valid_mask
        crs = "EPSG:4326"
        # 0.01 deg per px, origin at lon 0 / lat 2 (north-up): a 200x200 px tile = [0,2]x[0,2].
        transform = Affine(0.01, 0, 0.0, 0, -0.01, 2.0)
        covered = [self.INDEX[i] for i in (1, 2, 3)]  # everything but the north-east quarter
        valid = building_valid_mask((200, 200), crs, transform, covered)
        assert valid.dtype == np.uint8
        assert valid[:100, 100:].sum() == 0, "north-east quarter has no building tile -> unknown"
        assert valid[100:, :].all() and valid[:100, :100].all()
        assert building_valid_mask((200, 200), crs, transform, []).sum() == 0

    def test_load_joint_tiles_carries_building_valid(self, tmp_path, capsys):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        lab = np.zeros((16, 16), dtype=np.uint8)
        pts = np.zeros((0, 2), dtype=np.float32)
        sn = tmp_path / "spacenet" / "Vegas"
        sn.mkdir(parents=True)
        valid = np.ones((16, 16), dtype=np.uint8)
        valid[:, 8:] = 0
        np.savez_compressed(sn / "Vegas_img1.npz", image=img, label=lab, points=pts, building_valid=valid)
        np.savez_compressed(sn / "Vegas_img2.npz", image=img, label=lab, points=pts)  # old cache
        pot = tmp_path / "potsdam"
        pot.mkdir()
        np.savez_compressed(pot / "crop1.npz", image=img, label=lab)

        tiles = {t["tile"]: t for t in load_joint_tiles(sn.parent, pot, tmp_path / "none")}
        np.testing.assert_array_equal(tiles["Vegas_img1"]["building_valid"], valid)
        assert tiles["Vegas_img2"]["building_valid"] is None
        assert "add_building_coverage_to_cache" in capsys.readouterr().out
        assert tiles["crop1"]["building_valid"] is None
