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
import json
import tarfile
import tempfile

import rasterio
from affine import Affine as _Affine

from datasets.spacenet6 import (
    sar_bands_to_pseudo_rgb,
    rasterize_sn6_tile_labels,
    extract_members_streaming,
    build_spacenet6_sample,
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


class TestBuildSpacenet6Robustness:
    """A2/A3: a bad tile must be skipped (not raised), and every tile's
    extracted raw files must be cleaned whether it was cached or skipped."""

    _T = _Affine(0.5, 0.0, 593556.98, 0.0, -0.5, 5752109.14)
    _SAR_DIR = "train/AOI_11_Rotterdam/SAR-Intensity"
    _BLD_DIR = "train/AOI_11_Rotterdam/geojson_buildings"

    def _sar_bytes(self):
        buf = io.BytesIO()
        with rasterio.MemoryFile() as mem:
            with mem.open(driver="GTiff", height=32, width=32, count=1,
                          dtype="uint8", crs="EPSG:32631", transform=self._T) as ds:
                ds.write((np.ones((32, 32)) * 40).astype("uint8"), 1)
            buf.write(mem.read())
        return buf.getvalue()

    def _good_geojson(self):
        # A building polygon well inside the 32x32 @ 0.5 m tile (16 m across).
        x0, y0 = 593560.0, 5752100.0
        poly = [[x0, y0], [x0, y0 - 6], [x0 + 6, y0 - 6], [x0 + 6, y0], [x0, y0]]
        return json.dumps({
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32631"}},
            "features": [{"type": "Feature", "properties": {},
                         "geometry": {"type": "Polygon", "coordinates": [poly]}}],
        }).encode()

    def _add(self, tf, name, data):
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    def _make_tarball(self, path):
        keys = ["20190822_tile_good", "20190822_tile_bad"]
        with tarfile.open(path, "w:gz") as tf:
            for k in keys:
                self._add(tf, f"{self._SAR_DIR}/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{k}.tif",
                          self._sar_bytes())
            self._add(tf, f"{self._BLD_DIR}/SN6_Train_AOI_11_Rotterdam_Buildings_20190822_tile_good.geojson",
                      self._good_geojson())
            self._add(tf, f"{self._BLD_DIR}/SN6_Train_AOI_11_Rotterdam_Buildings_20190822_tile_bad.geojson",
                      b"{ this is not valid json ")

    def test_bad_tile_skipped_and_tmp_cleaned(self, tmp_path):
        raw = tmp_path / "spacenet6_raw"
        raw.mkdir()
        tarball = raw / "SN6_buildings_AOI_11_Rotterdam_train.tar.gz"
        self._make_tarball(tarball)

        samples = build_spacenet6_sample(
            n_tiles=10, cache_dir=str(tmp_path / "spacenet6"),
            tarball_path=str(tarball), seed=0,
        )

        keys = {s["tile_key"] for s in samples}
        assert "20190822_tile_good" in keys
        assert "20190822_tile_bad" not in keys  # malformed geojson -> skipped, not raised

        tmp_dir = raw / "_extract_tmp"
        leaked = list(tmp_dir.rglob("*.tif")) + list(tmp_dir.rglob("*.geojson")) if tmp_dir.exists() else []
        assert leaked == [], f"extracted raw files not cleaned up: {leaked}"


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
