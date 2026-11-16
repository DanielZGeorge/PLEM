"""Cache the SpaceNet test tiles used in the qualitative figure (image, label,
building_valid, intersection points) under data/spacenet_cases/<city>/, with
the same functions build_spacenet_sample uses."""
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from datasets import spacenet as sn  # noqa: E402
from datasets.common import download_file, load_geojson_features  # noqa: E402

TILES = {"Vegas": [1407, 1505], "Shanghai": [1166]}
for city, ids in TILES.items():
    aoi = sn.AOI_INDEX[city]
    raw_dir = root / "data" / "spacenet" / city / "raw"
    out_dir = root / "data" / "spacenet_cases" / city
    out_dir.mkdir(parents=True, exist_ok=True)
    index = sn.build_building_tile_index(city, raw_dir)
    for rid in ids:
        tif = raw_dir / "roads_tif" / f"img{rid}.tif"
        gj = raw_dir / "roads_geojson" / f"img{rid}.geojson"
        download_file(sn._road_tif_url(city, aoi, rid), tif)
        download_file(sn._road_geojson_url(city, aoi, rid), gj)
        info = sn.get_tile_geometry(tif)
        roads = load_geojson_features(gj)
        bgeoms, bvalid, _ = sn.tile_building_labels(city, info, index, raw_dir / "buildings_geojson")
        label = sn.rasterize_tile_labels(info["shape"], info["crs"], info["transform"], roads, bgeoms)
        points = sn.tile_intersections(roads, info)
        np.savez_compressed(out_dir / f"{city}_img{rid}.npz", image=sn.read_image_rgb(tif),
                            label=label, building_valid=bvalid, points=points)
        print(f"{city}_img{rid}: {len(points)} intersections, building known {bvalid.mean():.2f}",
              flush=True)
