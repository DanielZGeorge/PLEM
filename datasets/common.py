"""
Shared data-acquisition and rasterization helpers for the datasets/ package.

Keeps metrics/ fully decoupled from data-loading concerns: everything in
datasets/ only ever needs to produce plain H×W integer label maps (PLEM's
universal contract) -- never instance/point representations. Instance
extraction (connected components -> centroids) is point_f1()'s job, not
this package's.
"""

import json
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio import features
from rasterio.warp import transform_geom
from shapely.affinity import affine_transform
from shapely.geometry import shape as shapely_shape


def download_file(url: str, dest, overwrite: bool = False) -> Path:
    """Download `url` to `dest` via a plain anonymous HTTPS GET, unless it already exists."""
    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return dest


def load_geojson_features(geojson_path) -> list:
    """
    Load a SpaceNet-style geojson (lon/lat, CRS84) and return a list of
    (shapely_geometry, properties_dict) tuples. Geometries are left in their
    original lon/lat CRS -- reprojection to a raster's pixel grid happens in
    `to_pixel_geometries`.
    """
    with open(geojson_path, "r") as f:
        data = json.load(f)
    out = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if geom is None:
            continue
        out.append((shapely_shape(geom), feat.get("properties", {})))
    return out


def to_pixel_geometries(geoms_props: list, raster_crs, transform, src_crs="EPSG:4326") -> list:
    """
    Reproject a list of (geometry, properties) tuples from `src_crs` (default:
    WGS84 lon/lat, as used by SpaceNet geojsons) into the pixel (col, row)
    space of a raster's own CRS + affine transform.

    Returns a list of (pixel_geometry, properties) tuples whose coordinates
    are directly usable with rasterio.features.rasterize's default identity
    transform (x=col, y=row).
    """
    inv = ~transform
    out = []
    for geom, props in geoms_props:
        mapping = transform_geom(src_crs, raster_crs, geom.__geo_interface__)
        geom_proj = shapely_shape(mapping)
        # Apply the inverse affine (projected x,y -> pixel col,row) via shapely's
        # 6-parameter matrix: x' = a*x + b*y + xoff; y' = d*x + e*y + yoff.
        geom_px = affine_transform(geom_proj, [inv.a, inv.b, inv.d, inv.e, inv.c, inv.f])
        out.append((geom_px, props))
    return out


def rasterize_polygons(geoms_props: list, out_shape, raster_crs, transform, value: int = 1) -> np.ndarray:
    """Rasterize a list of (polygon, properties) tuples (in the source geojson's
    lon/lat CRS) onto a raster's pixel grid, filled with `value`."""
    pixel_geoms = to_pixel_geometries(geoms_props, raster_crs, transform)
    shapes = [(g, value) for g, _ in pixel_geoms if not g.is_empty]
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)
    return features.rasterize(shapes, out_shape=out_shape, fill=0, dtype=np.uint8)


def rasterize_lines(
    geoms_props: list,
    out_shape,
    raster_crs,
    transform,
    lane_width_m: float = 3.0,
    gsd_m: float = 0.3,
    default_lanes: float = 2,
    value: int = 1,
) -> np.ndarray:
    """
    Rasterize a list of (LineString, properties) road-segment tuples by
    buffering each line to a physically-motivated pixel width before filling,
    consistent with PLEM's `d = physical_metres / GSD_metres_per_pixel`
    convention. Width is derived from each segment's own `lane_number`
    property when present (SpaceNet SN3/SN5 roads carry this), falling back
    to `default_lanes`.
    """
    pixel_geoms = to_pixel_geometries(geoms_props, raster_crs, transform)
    px_per_m = 1.0 / gsd_m
    shapes = []
    for geom, props in pixel_geoms:
        if geom.is_empty:
            continue
        lanes = props.get("lane_number", props.get("lane_numbe", default_lanes))
        try:
            lanes = float(lanes)
            if lanes <= 0:
                lanes = default_lanes
        except (TypeError, ValueError):
            lanes = default_lanes
        radius_px = max(1.0, lanes * lane_width_m * px_per_m / 2.0)
        buffered = geom.buffer(radius_px)
        if not buffered.is_empty:
            shapes.append((buffered, value))
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)
    return features.rasterize(shapes, out_shape=out_shape, fill=0, dtype=np.uint8)


def read_image_rgb(tif_path) -> np.ndarray:
    """Read a GeoTIFF's first 3 bands as an H×W×3 uint8 array for display/caching."""
    with rasterio.open(tif_path) as src:
        bands = src.read(list(range(1, min(3, src.count) + 1)))
    img = np.moveaxis(bands, 0, -1)
    if img.shape[-1] < 3:
        img = np.repeat(img, 3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img.astype(np.float64) / max(img.max(), 1) * 255, 0, 255).astype(np.uint8)
    return img
