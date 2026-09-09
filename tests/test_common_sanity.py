"""
Sanity / unit tests for datasets/common.py's download + geojson-CRS helpers.

Network-free -- the HTTP layer is stubbed so we can exercise the retry /
truncation-detection logic added to `download_file` (a cleanly-closed-early
multi-GB download must NOT be cached as complete, or every later run dies in
tarfile.open / rasterio.open with the skip-guard preventing re-download).

Run with: pytest tests/test_common_sanity.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import common
from datasets.common import download_file, _TruncatedDownload, read_geojson_crs


class _FakeResponse:
    def __init__(self, body: bytes, content_length):
        self._body = body
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


def _patch_get(monkeypatch, response_factory):
    calls = {"n": 0}

    def fake_get(url, stream=True, timeout=60):
        calls["n"] += 1
        return response_factory(calls["n"])

    monkeypatch.setattr(common._SESSION, "get", fake_get)
    monkeypatch.setattr(common, "_DOWNLOAD_RETRIES", 3)
    return calls


class TestDownloadFile:
    def test_complete_download_succeeds(self, tmp_path, monkeypatch):
        body = b"x" * 5000
        _patch_get(monkeypatch, lambda n: _FakeResponse(body, len(body)))
        dest = tmp_path / "sub" / "file.bin"
        out = download_file("http://x/file.bin", dest)
        assert out == dest
        assert dest.read_bytes() == body
        assert not list(tmp_path.rglob("*.part"))

    def test_truncated_download_raises_and_cleans_part(self, tmp_path, monkeypatch):
        # Server promises 10000 bytes but the stream ends at 4000 with no error.
        calls = _patch_get(
            monkeypatch, lambda n: _FakeResponse(b"y" * 4000, 10000)
        )
        dest = tmp_path / "file.bin"
        with pytest.raises(_TruncatedDownload):
            download_file("http://x/file.bin", dest)
        assert not dest.exists()
        assert not list(tmp_path.rglob("*.part"))
        assert calls["n"] == 3  # retried, not accepted on the first short read

    def test_retry_then_success(self, tmp_path, monkeypatch):
        body = b"z" * 3000
        _patch_get(
            monkeypatch,
            lambda n: _FakeResponse(b"z" * 10, 3000) if n == 1 else _FakeResponse(body, 3000),
        )
        dest = tmp_path / "file.bin"
        out = download_file("http://x/file.bin", dest)
        assert out.read_bytes() == body

    def test_missing_content_length_is_accepted(self, tmp_path, monkeypatch):
        body = b"w" * 128
        _patch_get(monkeypatch, lambda n: _FakeResponse(body, None))
        dest = tmp_path / "file.bin"
        assert download_file("http://x/file.bin", dest).read_bytes() == body

    def test_existing_file_not_redownloaded(self, tmp_path, monkeypatch):
        dest = tmp_path / "file.bin"
        dest.write_bytes(b"old")

        def boom(*a, **k):
            raise AssertionError("should not hit the network")

        monkeypatch.setattr(common._SESSION, "get", boom)
        assert download_file("http://x/file.bin", dest).read_bytes() == b"old"


class TestReadGeojsonCrs:
    def test_reads_named_crs_member(self, tmp_path):
        p = tmp_path / "x.geojson"
        p.write_text('{"type":"FeatureCollection","crs":{"type":"name",'
                     '"properties":{"name":"urn:ogc:def:crs:EPSG::32631"}},"features":[]}')
        assert read_geojson_crs(p) == "urn:ogc:def:crs:EPSG::32631"

    def test_defaults_when_absent(self, tmp_path):
        p = tmp_path / "x.geojson"
        p.write_text('{"type":"FeatureCollection","features":[]}')
        assert read_geojson_crs(p, default="EPSG:4326") == "EPSG:4326"
