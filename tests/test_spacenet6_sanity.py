"""
Sanity / unit tests for datasets/spacenet6.py (real SAR building-footprint
data -- the illumination-invariant half of PLEM's night/dark-image
robustness approach) and datasets/joint.py's 3-source extension.

Deliberately network-free, pure-function tests only -- consistent with the
existing repo convention that datasets/spacenet.py's/datasets/potsdam.py's
actual data *acquisition* isn't unit-tested (that's exercised by running the
real data-prep notebooks against the live services instead).

Run with: pytest tests/test_spacenet6_sanity.py -v
"""

import os
import sys

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import tarfile
import tempfile

from datasets.spacenet6 import (
    sar_bands_to_pseudo_rgb,
    rasterize_sn6_tile_labels,
    extract_members_streaming,
)
from datasets.joint import class_mask_for_source, load_joint_tiles, SOURCE_CLASSES

SHAPE = (32, 32)
IDENTITY_TRANSFORM = Affine.identity()
CRS = "EPSG:4326"


class TestSarBandsToPseudoRgb:
    def test_replicate_mode_shape_dtype(self):
        rng = np.random.default_rng(0)
        sar = rng.uniform(0, 1000, size=(6, 32, 32))
        out = sar_bands_to_pseudo_rgb(sar, mode="replicate")
        assert out.shape == (32, 32, 3)
        assert out.dtype == np.uint8

    def test_replicate_mode_channels_identical(self):
        rng = np.random.default_rng(0)
        sar = rng.uniform(0, 1000, size=(6, 32, 32))
        out = sar_bands_to_pseudo_rgb(sar, mode="replicate")
        np.testing.assert_array_equal(out[..., 0], out[..., 1])
        np.testing.assert_array_equal(out[..., 1], out[..., 2])

    def test_pauli_mode_raises_until_verified(self):
        rng = np.random.default_rng(0)
        sar = rng.uniform(0, 1000, size=(6, 32, 32))
        with pytest.raises(NotImplementedError):
            sar_bands_to_pseudo_rgb(sar, mode="pauli")

    def test_unknown_mode_raises(self):
        rng = np.random.default_rng(0)
        sar = rng.uniform(0, 1000, size=(6, 32, 32))
        with pytest.raises(ValueError):
            sar_bands_to_pseudo_rgb(sar, mode="bogus")

    def test_constant_intensity_no_nan(self):
        sar = np.full((1, 16, 16), 5.0)
        out = sar_bands_to_pseudo_rgb(sar, mode="replicate")
        assert np.isfinite(out).all()


class TestRasterizeSn6TileLabels:
    def test_building_only(self):
        building_geom = Polygon([(4, 4), (4, 12), (12, 12), (12, 4)])
        building_geoms = [(building_geom, {})]
        label = rasterize_sn6_tile_labels(SHAPE, CRS, IDENTITY_TRANSFORM, building_geoms)
        assert (label == 2).any()
        assert not (label == 1).any()
        assert not (label == 3).any()

    def test_no_geoms_gives_empty_label(self):
        label = rasterize_sn6_tile_labels(SHAPE, CRS, IDENTITY_TRANSFORM, [])
        assert not (label > 0).any()

    def test_utm_geom_crs_reprojects_without_proj_error(self):
        # Regression: real SN6 footprints are in EPSG:32631 (UTM 31N), same
        # CRS as the raster -- not lon/lat. Feeding UTM eastings/northings
        # through common.py's lon/lat default made PROJ raise
        # "utm: Invalid latitude". rasterize_sn6_tile_labels must pass the
        # geometry CRS through as src_crs so the reprojection is identity.
        transform = Affine(0.5, 0.0, 593556.98, 0.0, -0.5, 5752109.14)
        utm_poly = Polygon([
            (593600, 5752000), (593600, 5752050),
            (593650, 5752050), (593650, 5752000),
        ])
        label = rasterize_sn6_tile_labels(
            (900, 900), "EPSG:32631", transform, [(utm_poly, {})],
            geom_crs="EPSG:32631",
        )
        assert (label == 2).any()
        assert not (label == 1).any()

    def test_geom_crs_defaults_to_raster_crs(self):
        # geom_crs=None must fall back to the raster CRS, not lon/lat.
        transform = Affine(0.5, 0.0, 593556.98, 0.0, -0.5, 5752109.14)
        utm_poly = Polygon([
            (593600, 5752000), (593600, 5752050),
            (593650, 5752050), (593650, 5752000),
        ])
        label = rasterize_sn6_tile_labels(
            (900, 900), "EPSG:32631", transform, [(utm_poly, {})],
        )
        assert (label == 2).any()


class TestExtractMembersStreaming:
    def _make_tarball(self, path, names):
        with tarfile.open(path, "w:gz") as tf:
            for name in names:
                data = f"contents of {name}".encode()
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    def test_extracts_only_requested_members_one_pass(self):
        with tempfile.TemporaryDirectory() as d:
            tb = os.path.join(d, "t.tar.gz")
            self._make_tarball(tb, [f"dir/f{i}.txt" for i in range(10)])
            out = os.path.join(d, "out")
            extract_members_streaming(tb, ["dir/f2.txt", "dir/f7.txt"], out)
            assert os.path.exists(os.path.join(out, "dir", "f2.txt"))
            assert os.path.exists(os.path.join(out, "dir", "f7.txt"))
            assert not os.path.exists(os.path.join(out, "dir", "f0.txt"))
            assert not os.path.exists(os.path.join(out, "dir", "f9.txt"))

    def test_missing_member_name_is_silently_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            tb = os.path.join(d, "t.tar.gz")
            self._make_tarball(tb, ["a.txt", "b.txt"])
            out = os.path.join(d, "out")
            extract_members_streaming(tb, ["a.txt", "nope.txt"], out)
            assert os.path.exists(os.path.join(out, "a.txt"))


class TestJointSpacenet6Integration:
    def test_source_classes_has_spacenet6(self):
        assert "spacenet6" in SOURCE_CLASSES
        assert SOURCE_CLASSES["spacenet6"] == [2]

    def test_class_mask_for_source_spacenet6(self):
        mask = class_mask_for_source("spacenet6")
        assert mask.shape == (4,)
        np.testing.assert_array_equal(mask, np.array([1, 0, 1, 0], dtype=np.float32))

    def test_load_joint_tiles_skips_missing_spacenet6_dir(self):
        tiles = load_joint_tiles(
            spacenet_dir="__nonexistent_spacenet__",
            potsdam_dir="__nonexistent_potsdam__",
            spacenet6_dir="__nonexistent_spacenet6__",
        )
        assert tiles == []
