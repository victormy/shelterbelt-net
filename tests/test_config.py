"""Unit tests for config.py."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure notebooks/ is on sys.path
_notebooks_dir = Path(__file__).resolve().parent.parent / "notebooks"
if str(_notebooks_dir) not in sys.path:
    sys.path.insert(0, str(_notebooks_dir))

import config


def test_num_channels():
    assert config.num_channels("global_monthly_2021_07_mosaic") == 3
    assert config.num_channels("global_multiseasonal_2021_mosaic") == 9
    assert config.num_channels("ps_scenes_2021_07") == 4


def test_window_positions_coverage():
    # 512 x 512 with window 256 and stride 128
    positions = list(config.window_positions(512, 512, 256, 128))
    assert (0, 0) in positions
    assert (256, 256) in positions
    # Final positions must cover the trailing edge
    xs = [x for x, y in positions]
    ys = [y for x, y in positions]
    assert max(xs) == 512 - 256
    assert max(ys) == 512 - 256


def test_normalize_minmax():
    arr = np.array([[[0, 50], [100, 200]]], dtype=np.float32)
    norm = config.normalize_minmax(arr)
    assert norm.min() == 0.0
    assert norm.max() == 1.0


def test_normalize_minmax_constant():
    arr = np.ones((3, 64, 64), dtype=np.float32) * 5
    norm = config.normalize_minmax(arr)
    assert np.all(norm == 0.0)


def test_normalize_imagenet():
    arr = np.zeros((3, 32, 32), dtype=np.uint8)
    norm = config.normalize_imagenet(arr)
    assert norm.shape == (3, 32, 32)
    assert np.isclose(norm[0, 0, 0], -0.485 / 0.229, atol=1e-3)


def test_tile_splits_load():
    splits = config.load_tile_splits()
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert len(splits["train"]) > 0
