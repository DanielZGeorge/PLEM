"""Cache specific SpaceNet road tiles (the paper's sparse-class case studies)
under data/spacenet_cases/<city>/, with the same rasterization as
datasets/spacenet.py::build_spacenet_sample, but outside the random sample."""
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from datasets import spacenet as sn  # noqa: E402
from datasets.common import download_file, load_geojson_features  # noqa: E402

CASES = {"Khartoum": [371, 333]}
for city, ids in CASES.items():
    aoi = sn.AOI_INDEX[city]
    raw_dir = root / "data" / "spacenet" / city / "raw"  # reuse the existing raw cache
    out_dir = root / "data" / "spacenet_cases" / city
    out_dir.mkdir(parents=True, exist_ok=True)
    index = sn.build_building_tile_index(city, raw_dir)
    for rid in ids:
        tif = raw_dir / "roads_tif" / f"img{rid}.tif"
        gj = raw_dir / "roads_geojson" / f"img{rid}.geojson"
        download_file(sn._road_tif_url(city, aoi, rid), tif)
        download_file(sn._road_geojson_url(city, aoi, rid), gj)
        info = sn.get_tile_geometry(tif)
        bgeoms, bvalid, _ = sn.tile_building_labels(city, info, index, raw_dir / "buildings_geojson")
        label = sn.rasterize_tile_labels(info["shape"], info["crs"], info["transform"],
                                         load_geojson_features(gj), bgeoms)
        np.savez_compressed(out_dir / f"{city}_img{rid}.npz", image=sn.read_image_rgb(tif),
                            label=label, building_valid=bvalid)
        print(f"{city}_img{rid}: road {(label == 1).mean():.4f}, building {(label == 2).mean():.4f}, "
              f"road px {(label == 1).sum()}, building px {(label == 2).sum()}, "
              f"building labels known on {bvalid.mean():.2f} of tile")
