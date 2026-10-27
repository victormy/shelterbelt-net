"""Unit and integration tests for pipeline.py stages."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure notebooks/ is on sys.path
_notebooks_dir = Path(__file__).resolve().parent.parent / "notebooks"
if str(_notebooks_dir) not in sys.path:
    sys.path.insert(0, str(_notebooks_dir))

import pipeline
import config


def test_gaussian_weight_2d():
    size = 256
    weight = pipeline._gaussian_weight_2d(size)
    assert weight.shape == (size, size)
    assert np.isclose(weight.max(), 1.0)
    assert weight[0, 0] < weight[size // 2, size // 2]
    assert np.allclose(weight, weight.T)


def test_reclassify_masks(tmp_geoai_root: Path):
    pipeline.reclassify_masks(
        year="2021",
        stnd_codes=[1],
        splits_path=tmp_geoai_root / "tile_splits.json",
    )
    out_mask = tmp_geoai_root / "masks" / "2021" / "train" / "1223-1328.geojson"
    assert out_mask.exists()


def test_chip_training_dataset(tmp_geoai_root: Path):
    pipeline.reclassify_masks(
        year="2021",
        stnd_codes=[1],
        splits_path=tmp_geoai_root / "tile_splits.json",
    )
    pipeline.chip_training_dataset(
        year="2021",
        data_type="global_monthly_2021_07_mosaic",
        chip_size=256,
        train_stride=256,
        splits_path=tmp_geoai_root / "tile_splits.json",
    )
    chips_dir = tmp_geoai_root / "models" / "2021" / "train" / "global_monthly_2021_07_mosaic"
    assert (chips_dir / "images").exists()
    assert (chips_dir / "masks").exists()
