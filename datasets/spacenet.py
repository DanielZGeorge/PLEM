"""
SpaceNet (SN2 buildings + SN3 roads) data acquisition and rasterization.

Both datasets are pulled from the public, unauthenticated `spacenet-dataset`
S3 bucket via plain anonymous HTTPS GET (verified: no AWS credentials,
boto3, or --no-sign-request needed -- individual per-tile files are
directly fetchable, not just multi-GB tarballs).

Important caveat (verified): SN2 building tiles (650x650px) and SN3 road
tiles (1300x1300px) use different tiling grids even for the same city, so
`img{N}` does NOT mean the same geographic tile between the two datasets.
This module resolves that by treating each SN3 road tile's own raster as the
canonical pixel grid, and spatially matching SN2 building tiles whose
lon/lat bounding box (computed straight from their own geojson coordinates,
no image download needed) overlaps that road tile's geographic extent.
"""

import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
import requests
from rasterio.warp import transform_bounds

from datasets.common import (
    download_file,
    load_geojson_features,
    rasterize_polygons,
    rasterize_lines,
    read_image_rgb,
)

BASE_URL = "https://spacenet-dataset.s3.amazonaws.com"
AOI_INDEX = {"Vegas": 2, "Paris": 3, "Shanghai": 4, "Khartoum": 5}

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_IMG_ID_RE = re.compile(r"_img(\d+)\.(?:tif|geojson)$")


def _list_keys(prefix: str) -> list:
    """List all object keys under `prefix` in the public bucket (paginated, anonymous)."""
    keys = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(f"{BASE_URL}/", params=params, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall("s3:Contents", _S3_NS):
            keys.append(c.find("s3:Key", _S3_NS).text)
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=_S3_NS) == "true"
        if not truncated:
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
        if not token:
            break
    return keys


def list_tile_ids(city: str, dataset: str = "roads") -> list:
    """
    List all available img{N} tile ids for a city's SN3 roads (geojson_roads)
    or SN2 buildings (geojson_buildings) via anonymous S3 listing.
    """
    aoi = AOI_INDEX[city]
    if dataset == "roads":
        prefix = f"spacenet/SN3_roads/train/AOI_{aoi}_{city}/geojson_roads/"
    elif dataset == "buildings":
        prefix = f"spacenet/SN2_buildings/train/AOI_{aoi}_{city}/geojson_buildings/"
    else:
        raise ValueError(f"Unknown dataset {dataset!r}, use 'roads' or 'buildings'")

    keys = _list_keys(prefix)
    ids = set()
    for k in keys:
        m = _IMG_ID_RE.search(k)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def _road_tif_url(city, aoi, img_id):
    return (f"{BASE_URL}/spacenet/SN3_roads/train/AOI_{aoi}_{city}/PS-RGB/"
            f"SN3_roads_train_AOI_{aoi}_{city}_PS-RGB_img{img_id}.tif")


def _road_geojson_url(city, aoi, img_id):
    return (f"{BASE_URL}/spacenet/SN3_roads/train/AOI_{aoi}_{city}/geojson_roads/"
            f"SN3_roads_train_AOI_{aoi}_{city}_geojson_roads_img{img_id}.geojson")


def _building_geojson_url(city, aoi, img_id):
    return (f"{BASE_URL}/spacenet/SN2_buildings/train/AOI_{aoi}_{city}/geojson_buildings/"
            f"SN2_buildings_train_AOI_{aoi}_{city}_geojson_buildings_img{img_id}.geojson")


def get_tile_geometry(tif_path) -> dict:
    """Read a GeoTIFF's CRS, affine transform, geographic bounds, and pixel shape."""
    with rasterio.open(tif_path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "shape": (src.height, src.width),
        }


def _bounds_lonlat(bounds, crs) -> tuple:
    return tuple(transform_bounds(crs, "EPSG:4326", *bounds))


def _geojson_bbox(geoms_props: list):
    """Bounding box (minx, miny, maxx, maxy) of a list of (geometry, properties) tuples."""
    xs, ys = [], []
    for geom, _ in geoms_props:
        minx, miny, maxx, maxy = geom.bounds
        xs += [minx, maxx]
        ys += [miny, maxy]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_intersects(a, b) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0


def _build_coarse_spatial_index(city, aoi, tile_ids, geojson_url_fn, raw_dir, n_coarse=100):
    """
    Sparse, evenly-spaced sample of a dataset's tile bboxes across the whole
    city, used as a nearest-neighbor spatial index.

    Verified empirically: SpaceNet tile img{N} ids are assigned in roughly a
    raster-scan order over the city (id increases ~monotonically with
    longitude, with a sawtooth in latitude), so nearby ids are usually nearby
    geographically. That means we can find which tiles overlap an arbitrary
    bounding box without downloading every tile in the city -- take a coarse
    evenly-spaced sample, find the id(s) geographically nearest the target,
    then densely search only the neighborhood of those ids.
    """
    step = max(1, len(tile_ids) // n_coarse)
    coarse_ids = tile_ids[::step]
    index = []
    for tid in coarse_ids:
        dest = Path(raw_dir) / f"img{tid}.geojson"
        try:
            download_file(geojson_url_fn(city, aoi, tid), dest)
        except requests.RequestException:
            continue
        geoms = load_geojson_features(dest)
        bbox = _geojson_bbox(geoms)
        if bbox is not None:
            index.append({
                "id": tid,
                "bbox": bbox,
                "centroid": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            })
    return index


def _nearby_tile_ids(target_bbox, coarse_index, tile_ids, neighborhood=20, top_k=3) -> list:
    """
    Given a target bbox, find the nearest tiles in a coarse spatial index (by
    centroid distance), then expand to a dense neighborhood of ids around
    each match in the full sorted tile_ids list.
    """
    if not coarse_index:
        return []
    tcx = (target_bbox[0] + target_bbox[2]) / 2
    tcy = (target_bbox[1] + target_bbox[3]) / 2
    dists = sorted(
        coarse_index,
        key=lambda e: (tcx - e["centroid"][0]) ** 2 + (tcy - e["centroid"][1]) ** 2,
    )
    candidates = set()
    for entry in dists[:top_k]:
        pos = tile_ids.index(entry["id"])
        lo = max(0, pos - neighborhood)
        hi = min(len(tile_ids), pos + neighborhood + 1)
        candidates.update(tile_ids[lo:hi])
    return sorted(candidates)


def rasterize_tile_labels(
    shape, crs, transform,
    road_geoms: list, building_geoms: list,
    lane_width_m: float = 3.0, gsd_m: float = 0.3,
) -> np.ndarray:
    """
    Rasterize road (class 1) and building (class 2) vector labels onto a
    tile's pixel grid. Buildings are drawn after (on top of) roads, since
    building footprints are more precisely surveyed than lane-buffered road
    polygons -- on overlap, the building class wins.
    """
    label = np.zeros(shape, dtype=np.uint8)
    if road_geoms:
        road_mask = rasterize_lines(
            road_geoms, shape, crs, transform,
            lane_width_m=lane_width_m, gsd_m=gsd_m, value=1,
        )
        label[road_mask > 0] = 1
    if building_geoms:
        building_mask = rasterize_polygons(building_geoms, shape, crs, transform, value=2)
        label[building_mask > 0] = 2
    return label


def build_spacenet_sample(
    city: str = "Vegas",
    n_tiles: int = 8,
    cache_dir="data/spacenet",
    seed: int = 0,
    n_road_candidates: int = 30,
    n_building_coarse: int = 100,
    building_neighborhood: int = 20,
    building_top_k: int = 3,
) -> list:
    """
    Build a curated local sample of up to `n_tiles` real SpaceNet scenes for
    `city`, each with an RGB image and a rasterized 0/1/2 (bg/road/building)
    label map, cached under `cache_dir/<city>/`.

    Each SN3 road tile defines the pixel grid. Matching SN2 building tiles
    are found via a coarse-then-dense nearest-neighbor spatial search (see
    `_build_coarse_spatial_index`/`_nearby_tile_ids`) rather than downloading
    every building tile in the city -- a purely random building sample was
    tried first and (verified) essentially never overlaps a random road
    tile's small footprint, since each tile covers only a fraction of a
    percent of the whole city's area. Only road tiles with at least one
    spatially-overlapping building tile are kept, so every returned sample
    has real building coverage, not just roads.
    """
    if city not in AOI_INDEX:
        raise ValueError(f"Unknown city {city!r}, choose from {list(AOI_INDEX)}")
    aoi = AOI_INDEX[city]

    cache_dir = Path(cache_dir) / city
    raw_dir = cache_dir / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    road_ids_all = list_tile_ids(city, "roads")
    building_ids_all = list_tile_ids(city, "buildings")
    if not road_ids_all or not building_ids_all:
        raise RuntimeError(f"No SpaceNet tiles found for city={city!r} -- check bucket layout")

    road_candidate_ids = rng.choice(
        road_ids_all, size=min(n_road_candidates, len(road_ids_all)), replace=False
    ).tolist()

    building_geojson_dir = raw_dir / "buildings_geojson"
    coarse_index = _build_coarse_spatial_index(
        city, aoi, building_ids_all, _building_geojson_url, building_geojson_dir,
        n_coarse=n_building_coarse,
    )

    samples = []
    for rid in road_candidate_ids:
        if len(samples) >= n_tiles:
            break

        tif_dest = raw_dir / "roads_tif" / f"img{rid}.tif"
        geojson_dest = raw_dir / "roads_geojson" / f"img{rid}.geojson"
        try:
            download_file(_road_tif_url(city, aoi, rid), tif_dest)
            download_file(_road_geojson_url(city, aoi, rid), geojson_dest)
        except requests.RequestException:
            continue

        geom_info = get_tile_geometry(tif_dest)
        road_bbox_lonlat = _bounds_lonlat(geom_info["bounds"], geom_info["crs"])

        nearby_ids = _nearby_tile_ids(
            road_bbox_lonlat, coarse_index, building_ids_all,
            neighborhood=building_neighborhood, top_k=building_top_k,
        )

        building_geoms = []
        for bid in nearby_ids:
            dest = building_geojson_dir / f"img{bid}.geojson"
            try:
                download_file(_building_geojson_url(city, aoi, bid), dest)
            except requests.RequestException:
                continue
            geoms = load_geojson_features(dest)
            bbox = _geojson_bbox(geoms)
            if bbox is not None and _bbox_intersects(bbox, road_bbox_lonlat):
                building_geoms.extend(geoms)

        if not building_geoms:
            continue  # no matched building coverage for this road tile -- skip

        road_geoms = load_geojson_features(geojson_dest)

        label = rasterize_tile_labels(
            geom_info["shape"], geom_info["crs"], geom_info["transform"],
            road_geoms, building_geoms,
        )
        if not (label == 2).any():
            # A lon/lat bbox "hit" can still be a false positive: two tiles that
            # merely touch at a shared edge overlap in bbox terms but not in
            # actual footprint, so nothing lands inside this tile's pixel grid
            # once reprojected. The rasterized pixel count is the real test.
            continue
        image = read_image_rgb(tif_dest)

        cache_path = cache_dir / f"{city}_img{rid}.npz"
        np.savez_compressed(cache_path, image=image, label=label)
        samples.append({
            "tile_id": rid, "city": city, "path": str(cache_path),
            "image": image, "label": label,
        })

    return samples
