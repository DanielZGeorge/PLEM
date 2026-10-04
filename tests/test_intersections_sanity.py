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
