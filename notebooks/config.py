"""Single source of truth for paths, constants, seeds, and shared helpers.

All user-configurable parameters live here.  Notebooks and the Streamlit UI
read these values -- no parameter duplication.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterator

logging.getLogger("rasterio").setLevel(logging.ERROR)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if "GDAL_DATA" not in os.environ:
    _conda_gdal = Path(sys.prefix) / "Library" / "share" / "gdal"
    if _conda_gdal.exists():
        os.environ["GDAL_DATA"] = str(_conda_gdal)

import numpy as np

# ===================================================================
# User-configurable parameters  (edit ONLY this block)
# ===================================================================
_script_dir = Path(__file__).resolve().parent
_DEFAULT_ROOT = _script_dir if (_script_dir / "delineation").exists() else _script_dir.parent
GEOAI_ROOT = Path(os.environ.get("GEOAI_ROOT", _DEFAULT_ROOT))

YEAR = "2021"
DATA_TYPE = "global_monthly_2021_07_mosaic"
ARCHITECTURE = "_conv"          # "_conv" (U-Net / ResNet34) or "_trans" (SegFormer / MiT-B0)
NORMALIZE = "minmax"            # "minmax" (geoai dataset) or "imagenet"
STND_CODES = [1, 2, 3]          # 1=shelterbelts; [1,2,3]=all linear plantations
PREDICTION_MONTHS = ["07"]              # list of months to include in voting (1-12); default [6] for June-only

# ===================================================================
# Derived & fixed constants  (do NOT edit unless you know the impact)
# ===================================================================
SEED = 42

# -- Chip / window geometry ----------------------------------------
CHIP_SIZE = 256
TRAIN_STRIDE = 128              # 50% overlap for training chips
EVAL_STRIDE = 256               # no overlap for val / test chips
INFERENCE_WINDOW = 256
INFERENCE_OVERLAP = 128         # 50% overlap for smooth Gaussian blending
INFERENCE_STRIDE = INFERENCE_WINDOW - INFERENCE_OVERLAP  # 128

# -- Training hyper-parameters -------------------------------------
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_INFERENCE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
THRESHOLD = 0.5
MAX_GRAD_NORM = 1.0

# -- ImageNet normalisation (smp pretrained encoders) ---------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# -- Ukraine Albers Equal Area CRS (WKT) ---------------------------
UKRAINE_ALBERS_WKT = (
    'PROJCS["Ukraine_Albers_East",'
    'GEOGCS["WGS 84",'
    'DATUM["World Geodetic System 1984",'
    'SPHEROID["WGS 84",6378137.0,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0.0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.017453292519943295],'
    'AXIS["Geodetic longitude",EAST],'
    'AXIS["Geodetic latitude",NORTH],'
    'AUTHORITY["EPSG","4326"]],'
    'PROJECTION["Albers_Conic_Equal_Area"],'
    'PARAMETER["standard_parallel_1",45.7],'
    'PARAMETER["standard_parallel_2",51.0],'
    'PARAMETER["latitude_of_center",44.0],'
    'PARAMETER["longitude_of_center",31.15],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["X",EAST],'
    'AXIS["Y",NORTH]]'
)

# ===================================================================
# Channel count  (derived from DATA_TYPE)
# ===================================================================

def num_channels(data_type: str | None = None) -> int:
    """Return the band count implied by *data_type*.

    * ``"multiseasonal"`` in the name  → 9  (3 months x 3 RGB bands)
    * ``"ps_scenes"`` in the name      → 4  (PlanetScope 4-band)
    * otherwise                        → 3  (single month, RGB)
    """
    dt = (data_type or DATA_TYPE).lower()

    if "multiseasonal" in dt:
        return 9
    if "ps_scene" in dt:
        return 4
    return 3


# ===================================================================
# Tile splits
# ===================================================================
_TILE_SPLITS_PATH = Path(__file__).resolve().parent / "tile_splits.json"


def load_tile_splits(path: Path | None = None) -> dict[str, list[str]]:
    """Return ``{"train": [...], "val": [...], "test": [...]}``."""
    p = path or _TILE_SPLITS_PATH
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ===================================================================
# Path helpers  (all accept overrides; default to module-level constants)
# ===================================================================

def images_dir(year: str | None = None, split: str = "", data_type: str | None = None) -> Path:
    dt = data_type or DATA_TYPE
    p = GEOAI_ROOT / "images" / (year or YEAR) / split / dt
    if not p.exists() and "_cl" in dt:
        import re
        base_dt = re.sub(r"_cl\d+$", "", dt)
        alt = GEOAI_ROOT / "images" / (year or YEAR) / split / base_dt
        if alt.exists():
            return alt
    return p


def masks_dir(year: str | None = None, split: str = "") -> Path:
    return GEOAI_ROOT / "masks" / (year or YEAR) / split


def chips_dir(year: str | None = None, split: str = "", data_type: str | None = None) -> Path:
    return GEOAI_ROOT / "models" / (year or YEAR) / split / (data_type or DATA_TYPE)


def delineation_dir(year: str | None = None) -> Path:
    return GEOAI_ROOT / "delineation" / f"all_3857_{year}"


def checkpoint_path(data_type: str | None = None, arch: str | None = None) -> Path:
    dt = data_type or DATA_TYPE
    ar = arch or ARCHITECTURE
    p = GEOAI_ROOT / "models" / "unet_models" / f"{dt}{ar}_best_model.pth"
    if not p.exists() and not dt.endswith("_cl123"):
        alt = GEOAI_ROOT / "models" / "unet_models" / f"{dt}_cl123{ar}_best_model.pth"
        if alt.exists():
            return alt
    return p


def history_path(data_type: str | None = None, arch: str | None = None) -> Path:
    dt = data_type or DATA_TYPE
    ar = arch or ARCHITECTURE
    p = GEOAI_ROOT / "models" / "unet_models" / f"{dt}{ar}_training_history.pth"
    if not p.exists() and not dt.endswith("_cl123"):
        alt = GEOAI_ROOT / "models" / "unet_models" / f"{dt}_cl123{ar}_training_history.pth"
        if alt.exists():
            return alt
    return p


def output_dir(year: str | None = None, data_type: str | None = None) -> Path:
    return GEOAI_ROOT / "outputs" / (year or YEAR) / (data_type or DATA_TYPE)


def mosaic_path(year: str | None = None, data_type: str | None = None) -> Path:
    return GEOAI_ROOT / "outputs" / (year or YEAR) / f"{(data_type or DATA_TYPE)}_albers.tif"


# ===================================================================
# Seed helper
# ===================================================================

def seed_everything(seed: int = SEED) -> None:
    """Fix all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ===================================================================
# Sliding-window position generator
# ===================================================================

def window_positions(
    width: int, height: int, window: int, stride: int,
) -> Iterator[tuple[int, int]]:
    """Yield ``(x, y)`` positions covering the full raster including edges.

    The final position on each axis is forced to ``total - window`` so that
    the trailing strip is always covered, even when
    ``(total - window) % stride != 0``.
    """
    def _axis_positions(total: int) -> list[int]:
        if total < window:
            return [0] if total > 0 else []
        positions = list(range(0, total - window + 1, stride))
        if positions[-1] != total - window:
            positions.append(total - window)
        return positions

    for y in _axis_positions(height):
        for x in _axis_positions(width):
            yield x, y


# ===================================================================
# Normalisation helpers
# ===================================================================

def normalize_imagenet(chip: np.ndarray) -> np.ndarray:
    """Normalize a ``(C, H, W)`` chip with ImageNet mean/std.

    Accepts uint8 (0-255) or float (0-1 / arbitrary range).
    For 9-channel multi-seasonal inputs the 3-channel mean/std is tiled.

    Raises ``ValueError`` for inputs that are not 3-D arrays with a channel
    count that is a multiple of 3.
    """
    if chip.ndim != 3:
        raise ValueError(f"Expected 3-D array (C, H, W), got {chip.ndim}-D")
    c = chip.shape[0]
    if c % 3 != 0:
        raise ValueError(f"Channel count must be a multiple of 3, got {c}")

    out = chip.astype(np.float32, copy=True)
    if out.max() > 1.0:
        out /= 255.0

    mean = np.tile(IMAGENET_MEAN, c // 3)[:, None, None]
    std = np.tile(IMAGENET_STD, c // 3)[:, None, None]
    return (out - mean) / std


def normalize_minmax(chip: np.ndarray) -> np.ndarray:
    """Per-chip min-max normalization to [0, 1].

    This matches the normalization used by ``geoai.train.SemanticSegmentationDataset``.
    Use this in inference when the model was trained with the geoai dataset
    class (which applies per-chip min-max internally).
    """
    if chip.ndim != 3:
        raise ValueError(f"Expected 3-D array (C, H, W), got {chip.ndim}-D")
    out = chip.astype(np.float32, copy=True)
    cmin = out.min()
    cmax = out.max()
    if cmax - cmin < 1e-8:
        return np.zeros_like(out)
    return (out - cmin) / (cmax - cmin)
