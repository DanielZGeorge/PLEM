"""
Road-intersection point labels derived from vector road centrelines.

PLEM's point (0D) feature is the intersection of two or more roads. No
separate annotation is needed: SpaceNet 3's road labels are vector
LineStrings that share a vertex wherever roads meet (verified on real SN3
geojsons -- counting shared vertices and geometric noding via
`shapely.ops.unary_union` agree on almost every tile), so intersections fall
straight out of the road graph. Point and road labels therefore come from
the same annotation of the same image and cannot disagree, unlike the
earlier Potsdam tree/car point class.

Intersections lie ON road pixels, so they cannot be a class id in the single
`H x W` label map without punching holes in the road class. They are kept as
a separate per-tile point layer: an `(N, 2)` float32 array of (row, col)
pixel coordinates, rasterized on demand by `points_to_mask`.

Pure shapely/numpy, no torch import, consistent with `datasets/`'s contract.
"""

from collections import Counter

import numpy as np
from shapely.geometry import Point

from datasets.common import to_pixel_geometries

# ~1 cm in lon/lat degrees: merges float noise, never distinct vertices.
_COORD_DECIMALS = 7


def _line_parts(geom) -> list:
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    return []


def road_intersections(road_geoms: list, min_arms: int = 3) -> list:
    """
    (geometry, properties) road tuples -> list of (x, y) intersection
    coordinates, in the geometries' own CRS (lon/lat for SN3).

    Counts road "arms" at every vertex across all lines: a line's endpoint
    contributes 1 arm, an interior vertex 2. A vertex with >= `min_arms`
    (default 3) is an intersection -- this covers X-junctions, T-junctions,
    and a T whose through road is not split at the junction (interior vertex
    of one line + endpoint of another = 3 arms). A plain bend (2 arms) and a
    dead end (1 arm) are not intersections.

    Deliberately vertex-based rather than geometric: two lines that cross
    with no shared vertex (an overpass) are not counted.
    """
    arms = Counter()
    for geom, _ in road_geoms:
        for line in _line_parts(geom):
            coords = [(round(c[0], _COORD_DECIMALS), round(c[1], _COORD_DECIMALS)) for c in line.coords]
            if len(coords) < 2:
                continue
            closed = coords[0] == coords[-1]
            last = len(coords) - 1
            for i, pt in enumerate(coords):
                if closed and i == last:
                    continue  # ring: start/end are one vertex, counted once below
                is_end = i in (0, last) and not closed
                arms[pt] += 1 if is_end else 2
    return [pt for pt, n in arms.items() if n >= min_arms]


def merge_close_points(points_rc: np.ndarray, merge_px: float) -> np.ndarray:
    """
    (N, 2) -> (M, 2): greedily replaces groups of points closer than
    `merge_px` with their mean. A dual carriageway or a complex junction
    yields several graph nodes within a few metres; they are one intersection
    at image scale, and separate targets would overlap anyway.
    """
    points_rc = np.asarray(points_rc, dtype=np.float32).reshape(-1, 2)
    if len(points_rc) < 2 or merge_px <= 0:
        return points_rc
    remaining = list(range(len(points_rc)))
    merged = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        # Grow the group transitively so chains of near nodes collapse together.
        frontier = [seed]
        while frontier:
            cur = frontier.pop()
            near = [j for j in remaining
                    if np.hypot(*(points_rc[j] - points_rc[cur])) < merge_px]
            for j in near:
                remaining.remove(j)
            group.extend(near)
            frontier.extend(near)
        merged.append(points_rc[group].mean(axis=0))
    return np.asarray(merged, dtype=np.float32)


def intersections_to_pixels(
    points_xy: list, raster_crs, transform, shape, merge_px: float = 20.0, src_crs="EPSG:4326",
) -> np.ndarray:
    """
    Intersection coordinates in `src_crs` -> `(N, 2)` float32 (row, col) pixel
    coordinates on a raster's grid. Points outside the `shape` (H, W) tile are
    dropped; points closer than `merge_px` are merged (`merge_close_points`).
    """
    if not points_xy:
        return np.zeros((0, 2), dtype=np.float32)
    pixel = to_pixel_geometries(
        [(Point(x, y), {}) for x, y in points_xy], raster_crs, transform, src_crs=src_crs,
    )
    h, w = shape
    rc = [(g.y, g.x) for g, _ in pixel if 0 <= g.y < h and 0 <= g.x < w]
    if not rc:
        return np.zeros((0, 2), dtype=np.float32)
    return merge_close_points(np.asarray(rc, dtype=np.float32), merge_px)


def points_to_mask(points_rc, shape, radius: int = 2) -> np.ndarray:
    """
    `(N, 2)` (row, col) points -> `H x W` uint8 0/1 mask with a
    `(2*radius+1)^2` square stamped at each point. This is the form both
    `losses/heatmap.py::gt_centroids_to_heatmap` and `metrics/point_f1.py`
    consume (each reduces a blob back to its centroid).
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for r, c in np.asarray(points_rc, dtype=np.float32).reshape(-1, 2):
        r, c = int(round(float(r))), int(round(float(c)))
        if not (0 <= r < h and 0 <= c < w):
            continue
        mask[max(0, r - radius):r + radius + 1, max(0, c - radius):c + radius + 1] = 1
    return mask
