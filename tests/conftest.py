"""Shared fixtures for testing the shelterbelt segmentation pipeline."""

import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping


@pytest.fixture
def tmp_geoai_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary mock GEOAI_ROOT directory structure with synthetic data."""
    root = tmp_path / "mock_geoai"
    root.mkdir()

    # Delineation
    delin_dir = root / "delineation" / "all_3857_2021"
    delin_dir.mkdir(parents=True)
    poly = box(4_000_000, 6_000_000, 4_001_000, 6_001_000)
    feat = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {"STNDTYPE": 1, "STNDCOND": 1, "WIDTH": 20},
            }
        ],
    }
    geojson_path = delin_dir / "1223-1328.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(feat, f)

    # Images
    img_dir = root / "images" / "2021" / "train" / "global_monthly_2021_07_mosaic"
    img_dir.mkdir(parents=True)
    tif_path = img_dir / "1223-1328.tif"

    transform = from_origin(4_000_000, 6_002_000, 4.77, 4.77)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": 512,
        "height": 512,
        "count": 3,
        "crs": "EPSG:3857",
        "transform": transform,
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(np.ones((3, 512, 512), dtype=np.uint8) * 128)

    # Tile splits
    splits = {
        "train": ["1223-1328"],
        "val": [],
        "test": [],
    }
    splits_path = root / "tile_splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    monkeypatch.setenv("GEOAI_ROOT", str(root))
    return root
