"""Pipeline stages for the shelterbelt segmentation workflow.

Each public function corresponds to one notebook and accepts an optional
``progress`` callback with signature ``(fraction: float, message: str) -> None``
where *fraction* is in [0, 1].

All parameters default to the values defined in ``config.py`` -- the single
source of truth.  Notebooks call these functions with **no arguments**;
the Streamlit UI may override via keyword arguments.
"""

from __future__ import annotations

import contextlib
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable

# Silence noisy GDAL CPLE_FileIO / rasterio warnings (e.g., gdalvrt.xsd)
logging.getLogger("rasterio").setLevel(logging.ERROR)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if "GDAL_DATA" not in os.environ:
    _conda_gdal = Path(sys.prefix) / "Library" / "share" / "gdal"
    if _conda_gdal.exists():
        os.environ["GDAL_DATA"] = str(_conda_gdal)

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box, shape
from tqdm import tqdm

import config

_NOOP: Callable[[float, str], None] = lambda _f, _m: None


def _gaussian_weight_2d(size: int, sigma: float | None = None) -> np.ndarray:
    """Create a 2D Gaussian weight matrix for smooth tile blending.

    Center pixels get weight ~1.0, edges taper toward 0.  This eliminates
    the hard transition at tile boundaries that causes grid artefacts.
    """
    if sigma is None:
        sigma = size / 4.0
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    weight = np.outer(gauss_1d, gauss_1d)
    weight /= weight.max()
    return weight


def _clean_output_dir(directory: Path, extension: str, progress: Callable) -> None:
    """Remove existing output files and warn via progress callback."""
    existing = [f for f in directory.iterdir() if f.suffix.lower() == extension]
    if existing:
        progress(0.0, f"Cleaning {len(existing)} old files from {directory.name}/")
        for f in existing:
            f.unlink()


# ======================================================================
# Stage 0 -- Multi-seasonal Planet stack
# ======================================================================

def stack_multiseasonal(
    year: str | None = None,
    months: list[str] | None = None,
    tile_types: list[str] | None = None,
    progress: Callable[[float, str], None] = _NOOP,
) -> None:
    """Stack monthly RGB basemaps into a single multi-temporal raster per tile.

    For every tile, reads bands 1-3 from each month and writes a
    ``3 * len(months)``-band GeoTIFF.
    """
    year = year or config.YEAR
    months = months or ["04", "07", "10"]
    tile_types = tile_types or ["train", "val", "test"]

    work_items: list[tuple[str, str]] = []
    for tt in tile_types:
        first_dir = config.images_dir(year, tt, f"global_monthly_{year}_{months[0]}_mosaic")
        if not first_dir.exists():
            continue
        for p in first_dir.iterdir():
            if p.suffix.lower() in (".tif", ".tiff"):
                work_items.append((tt, p.name))

    total = len(work_items)
    for idx, (tt, tile_name) in enumerate(work_items):
        bands: list[np.ndarray] = []
        profile = None

        for month in months:
            src_path = config.images_dir(year, tt, f"global_monthly_{year}_{month}_mosaic") / tile_name
            if not src_path.exists():
                progress((idx + 1) / total, f"Skipping {tile_name} -- missing {month}")
                bands = []
                break
            with rasterio.open(src_path) as src:
                bands.append(src.read([1, 2, 3]))
                if profile is None:
                    profile = src.profile

        if not bands:
            continue

        stacked = np.vstack(bands)
        profile.update(count=stacked.shape[0])

        out_dir = config.images_dir(year, tt, f"global_multiseasonal_{year}_mosaic")
        out_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_dir / tile_name, "w", **profile) as dst:
            dst.write(stacked)

        progress((idx + 1) / total, f"Stacked {tt}/{tile_name} -> {stacked.shape[0]} bands")

    progress(1.0, "Multi-seasonal stacking complete")


# ======================================================================
# Stage 1 -- Reclassify by stand type
# ======================================================================

def reclassify_masks(
    year: str | None = None,
    stnd_codes: list[int] | None = None,
    progress: Callable[[float, str], None] = _NOOP,
) -> None:
    """Filter reference polygons by stand type and write binary masks."""
    import geopandas as gpd
    from shapely.validation import make_valid

    year = year or config.YEAR
    stnd_codes = stnd_codes or config.STND_CODES
    splits = config.load_tile_splits()
    input_folder = config.delineation_dir(year)

    work: list[tuple[str, str]] = []
    for split_name in ("train", "val", "test"):
        for tile in splits[split_name]:
            work.append((split_name, tile))

    total = len(work)
    for idx, (split_name, tile) in enumerate(work):
        poly_in = input_folder / f"{tile}.geojson"
        out_folder = config.masks_dir(year, split_name)
        out_folder.mkdir(parents=True, exist_ok=True)
        poly_out = out_folder / f"{tile}.geojson"

        poly = gpd.read_file(poly_in)
        poly["class"] = np.where(np.isin(poly["STNDTYPE"], stnd_codes), 1, 0)
        poly_flt = poly[poly["class"] == 1].copy()

        invalid = (~poly_flt.geometry.is_valid).sum()
        if invalid > 0:
            poly_flt["geometry"] = poly_flt["geometry"].apply(make_valid)
            poly_flt = poly_flt[poly_flt.geometry.is_valid & ~poly_flt.geometry.is_empty]

        cols = ["STNDCOND", "STNDTYPE", "WIDTH", "class", "geometry"]
        poly_flt[cols].to_file(poly_out)

        progress((idx + 1) / total, f"Reclassified {split_name}/{tile}")

    progress(1.0, "Mask reclassification complete")


# ======================================================================
# Stage 2 -- Create training dataset (chipping)
# ======================================================================

def _list_tifs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff"))


def _list_geojsons(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".geojson", ".json"))


def _match_pairs(images_folder: Path, masks_folder: Path):
    images = _list_tifs(images_folder)
    masks = _list_geojsons(masks_folder)
    mask_map = {p.stem: p for p in masks}
    pairs, missing = [], []
    for img in images:
        if img.stem in mask_map:
            pairs.append((img, mask_map[img.stem]))
        else:
            missing.append(img.stem)
    return pairs, missing


def _geojson_geometries(geojson_path: Path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") == "FeatureCollection":
        feats = data.get("features", [])
    elif data.get("type") == "Feature":
        feats = [data]
    else:
        feats = [{"geometry": data}]
    geoms = []
    for feat in feats:
        geom = feat.get("geometry", feat)
        if geom:
            geoms.append(shape(geom))
    return geoms


def chip_training_dataset(
    year: str | None = None,
    data_type: str | None = None,
    progress: Callable[[float, str], None] = _NOOP,
) -> None:
    """Chip full-resolution tiles + vector masks into 256x256 GeoTIFF pairs."""
    year = year or config.YEAR
    data_type = data_type or config.DATA_TYPE

    # tile_types = ["train", "val", "test"]
    tile_types = ["test"]

    all_logs: list[pd.DataFrame] = []
    all_missing: dict[str, list[str]] = {}

    total_types = len(tile_types)
    for type_idx, tile_type in enumerate(tile_types):
        images_folder = config.images_dir(year, tile_type, data_type)
        masks_folder = config.masks_dir(year, tile_type)
        out_root = config.chips_dir(year, tile_type, data_type)
        out_images = out_root / "images"
        out_masks = out_root / "masks"
        out_images.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)

        stride = config.TRAIN_STRIDE if tile_type == "train" else config.EVAL_STRIDE
        pairs, missing = _match_pairs(images_folder, masks_folder)
        all_missing[tile_type] = missing
        rows: list[dict] = []

        for pair_idx, (img_path, geojson_path) in enumerate(pairs):
            scene_id = img_path.stem
            geoms = _geojson_geometries(geojson_path)

            with rasterio.open(img_path) as src:
                width, height = src.width, src.height

                for x, y in config.window_positions(width, height, config.CHIP_SIZE, stride):
                    win = Window(x, y, config.CHIP_SIZE, config.CHIP_SIZE)
                    img = src.read(window=win)
                    if img.shape[1] != config.CHIP_SIZE or img.shape[2] != config.CHIP_SIZE:
                        continue

                    transform = src.window_transform(win)
                    tile_bounds = rasterio.windows.bounds(win, src.transform)
                    tile_box = box(*tile_bounds)
                    selected = [g for g in geoms if g.intersects(tile_box)]

                    if selected:
                        mask = rasterize(
                            [(geom, 1) for geom in selected],
                            out_shape=(config.CHIP_SIZE, config.CHIP_SIZE),
                            transform=transform,
                            fill=0,
                            dtype=np.uint8,
                            all_touched=False,
                        )
                    else:
                        mask = np.zeros((config.CHIP_SIZE, config.CHIP_SIZE), dtype=np.uint8)

                    empty = bool(np.all(mask == 0))
                    if empty:
                        continue

                    tile_name = f"{scene_id}_{y:05d}_{x:05d}.tif"

                    img_profile = src.profile.copy()
                    img_profile.update(height=config.CHIP_SIZE, width=config.CHIP_SIZE, transform=transform)

                    mask_profile = src.profile.copy()
                    mask_profile.update(
                        count=1, height=config.CHIP_SIZE, width=config.CHIP_SIZE,
                        dtype="uint8", nodata=0, transform=transform,
                    )

                    with rasterio.open(out_images / tile_name, "w", **img_profile) as dst:
                        dst.write(img)
                    with rasterio.open(out_masks / tile_name, "w", **mask_profile) as dst:
                        dst.write(mask, 1)

                    rows.append({
                        "scene_id": scene_id,
                        "tile_name": tile_name,
                        "x": x, "y": y,
                        "empty": empty,
                        "image": str(img_path),
                        "mask": str(geojson_path),
                        "tile_type": tile_type,
                    })

            frac = (type_idx + (pair_idx + 1) / max(len(pairs), 1)) / total_types
            progress(frac, f"{tile_type}: chipped {scene_id}")

        all_logs.append(pd.DataFrame(rows))

    log_dir = config.GEOAI_ROOT / "models" / year
    log_dir.mkdir(parents=True, exist_ok=True)
    log_df = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    log_df.to_csv(log_dir / "tile_log.csv", index=False)

    missing_rows = [
        {"tile_type": tt, "missing_stem": item}
        for tt, items in all_missing.items()
        for item in items
    ]
    pd.DataFrame(missing_rows).to_csv(log_dir / "missing_pairs.csv", index=False)

    progress(1.0, f"Chipping complete -- {len(log_df)} chip pairs")


class PipelineDataset:
    """Unified segmentation dataset for both 8-bit RGB and 16-bit 4-band imagery.

    Scaling strategy:
    - 16-bit PlanetScope (Surface Reflectance): scaled by 10,000 to [0.0, 1.0].
      PlanetScope SR digital numbers represent 0..100% surface reflectance via 0..10,000.
    - 8-bit optical RGB / Mosaics: scaled by 255.0 to [0.0, 1.0].
    Both types produce normalized tensors strictly in [0.0, 1.0], matching ImageNet
    expectations and allowing standard data augmentations to operate safely.
    """

    def __init__(
        self,
        image_paths: list[str],
        label_paths: list[str],
        transforms=None,
        num_channels: int | None = None,
        target_size: tuple[int, int] = (256, 256),
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms
        self.num_channels = num_channels
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        import torch

        img_path = self.image_paths[idx]
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
            # Physical scaling: 16-bit PlanetScope SR -> / 10000.0, 8-bit RGB -> / 255.0
            if src.dtypes[0] == "uint16" or image.max() > 255.0:
                image = np.clip(image / 10000.0, 0.0, 1.0)
            else:
                image = np.clip(image / 255.0, 0.0, 1.0)

        if self.num_channels is not None:
            if image.shape[0] > self.num_channels:
                image = image[: self.num_channels]
            elif image.shape[0] < self.num_channels:
                padded = np.zeros(
                    (self.num_channels, image.shape[1], image.shape[2]),
                    dtype=np.float32,
                )
                padded[: image.shape[0]] = image
                image = padded

        image_t = torch.as_tensor(image, dtype=torch.float32)

        lbl_path = self.label_paths[idx]
        with rasterio.open(lbl_path) as src:
            mask = src.read(1).astype(np.int64)

        if mask.max() > 1:
            mask = (mask > 0).astype(np.int64)

        mask_t = torch.as_tensor(mask, dtype=torch.long)

        if self.transforms is not None:
            image_t, mask_t = self.transforms(image_t, mask_t)

        return image_t, mask_t


# ======================================================================
# Stage 3 -- Train / validate / test
# ======================================================================

def train_model(
    year: str | None = None,
    data_type: str | None = None,
    architecture: str | None = None,
    progress: Callable[[float, str], None] = _NOOP,
) -> dict:
    """Train a segmentation model; return the training history dict."""
    import torch
    from torch.utils.data import DataLoader
    import segmentation_models_pytorch as smp
    try:
        from geoai.train import (
            SemanticRandomHorizontalFlip,
            SemanticRandomVerticalFlip,
            SemanticRandomRotation90,
            SemanticBrightnessAdjustment,
            SemanticContrastAdjustment,
        )
    except ImportError:
        import random as _py_random

        class SemanticRandomHorizontalFlip:
            def __init__(self, prob=0.5): self.p = prob
            def __call__(self, img, msk):
                return (torch.flip(img, [2]), torch.flip(msk, [1])) if _py_random.random() < self.p else (img, msk)

        class SemanticRandomVerticalFlip:
            def __init__(self, prob=0.5): self.p = prob
            def __call__(self, img, msk):
                return (torch.flip(img, [1]), torch.flip(msk, [0])) if _py_random.random() < self.p else (img, msk)

        class SemanticRandomRotation90:
            def __init__(self, prob=0.5): self.p = prob
            def __call__(self, img, msk):
                if _py_random.random() < self.p:
                    k = _py_random.choice([1, 2, 3])
                    return torch.rot90(img, k, [1, 2]), torch.rot90(msk, k, [0, 1])
                return img, msk

        class SemanticBrightnessAdjustment:
            def __init__(self, brightness_range=(0.8, 1.2), prob=0.5):
                self.rng, self.p = brightness_range, prob
            def __call__(self, img, msk):
                if _py_random.random() < self.p:
                    factor = self.rng[0] + _py_random.random() * (self.rng[1] - self.rng[0])
                    return torch.clamp(img * factor, 0.0, 1.0), msk
                return img, msk

        class SemanticContrastAdjustment:
            def __init__(self, contrast_range=(0.8, 1.2), prob=0.5):
                self.rng, self.p = contrast_range, prob
            def __call__(self, img, msk):
                if _py_random.random() < self.p:
                    factor = self.rng[0] + _py_random.random() * (self.rng[1] - self.rng[0])
                    mean = img.mean(dim=(1, 2), keepdim=True)
                    return torch.clamp((img - mean) * factor + mean, 0.0, 1.0), msk
                return img, msk

    year = year or config.YEAR
    data_type = data_type or config.DATA_TYPE
    architecture = architecture or config.ARCHITECTURE
    n_channels = 4 if "ps_scene" in data_type.lower() else config.num_channels(data_type)

    config.seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    progress(0.0, f"Training: {data_type}, arch={architecture}, channels={n_channels}")

    root = config.GEOAI_ROOT / "models" / year
    train_imgs = sorted(glob.glob(str(root / "train" / data_type / "images" / "*.tif")))
    train_masks = sorted(glob.glob(str(root / "train" / data_type / "masks" / "*.tif")))
    val_imgs = sorted(glob.glob(str(root / "val" / data_type / "images" / "*.tif")))
    val_masks = sorted(glob.glob(str(root / "val" / data_type / "masks" / "*.tif")))
    test_imgs = sorted(glob.glob(str(root / "test" / data_type / "images" / "*.tif")))
    test_masks = sorted(glob.glob(str(root / "test" / data_type / "masks" / "*.tif")))

    assert train_imgs, f"No training images found in {root / 'train' / data_type / 'images'}"
    assert val_imgs, f"No validation images found in {root / 'val' / data_type / 'images'}"
    assert test_imgs, f"No test images found in {root / 'test' / data_type / 'images'}"

    # -- Augmentations ---------------------------------------------------------
    class ApplyTransforms:
        def __init__(self, transforms):
            self.transforms = transforms
        def __call__(self, image, mask):
            for t in self.transforms:
                image, mask = t(image, mask)
            return image, mask

    train_transforms = ApplyTransforms([
        SemanticRandomHorizontalFlip(prob=0.5),
        SemanticRandomVerticalFlip(prob=0.5),
        SemanticRandomRotation90(prob=0.5),
        SemanticBrightnessAdjustment(brightness_range=(0.8, 1.2), prob=0.5),
        SemanticContrastAdjustment(contrast_range=(0.8, 1.2), prob=0.5),
    ])

    train_ds = PipelineDataset(
        image_paths=train_imgs, label_paths=train_masks,
        transforms=train_transforms, num_channels=n_channels,
        target_size=(config.CHIP_SIZE, config.CHIP_SIZE),
    )
    val_ds = PipelineDataset(
        image_paths=val_imgs, label_paths=val_masks,
        transforms=None, num_channels=n_channels,
        target_size=(config.CHIP_SIZE, config.CHIP_SIZE),
    )
    test_ds = PipelineDataset(
        image_paths=test_imgs, label_paths=test_masks,
        transforms=None, num_channels=n_channels,
        target_size=(config.CHIP_SIZE, config.CHIP_SIZE),
    )

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE_TRAIN, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE_TRAIN, shuffle=False, num_workers=0)

    # -- Model -----------------------------------------------------------------
    if architecture == "_conv":
        model = smp.create_model(
            "unet", encoder_name="resnet34", encoder_weights="imagenet",
            in_channels=n_channels, classes=1,
        ).to(device)
    else:
        model = smp.create_model(
            "segformer", encoder_name="mit_b0", encoder_weights="imagenet",
            in_channels=n_channels, classes=1,
        ).to(device)

    loss_fn = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    ckpt = config.checkpoint_path(data_type, architecture)
    hist_path = config.history_path(data_type, architecture)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    # -- Evaluate helper (single-pass: loss + metrics) -------------------------
    def evaluate(loader, desc="Val"):
        model.eval()
        tp = fp = fn = tn = 0.0
        running_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(loader, desc=desc, leave=False):
                preds = model(images.to(device)).squeeze(1)
                masks_d = masks.to(device)
                running_loss += loss_fn(preds, masks_d.float()).item()
                b_tp, b_fp, b_fn, b_tn = smp.metrics.get_stats(
                    preds.sigmoid(), masks_d.long(), mode="binary", threshold=config.THRESHOLD,
                )
                tp += b_tp.sum().item()
                fp += b_fp.sum().item()
                fn += b_fn.sum().item()
                tn += b_tn.sum().item()
        t = lambda x: torch.tensor(x)
        avg_loss = running_loss / max(len(loader), 1)
        iou = smp.metrics.iou_score(t(tp), t(fp), t(fn), t(tn), reduction="micro").item()
        f1 = smp.metrics.f1_score(t(tp), t(fp), t(fn), t(tn), reduction="micro").item()
        prec = smp.metrics.precision(t(tp), t(fp), t(fn), t(tn), reduction="micro").item()
        rec = smp.metrics.recall(t(tp), t(fp), t(fn), t(tn), reduction="micro").item()
        return avg_loss, iou, f1, prec, rec

    # -- Training loop ---------------------------------------------------------
    history = {
        "train_loss": [], "val_loss": [],
        "val_iou": [], "val_f1": [], "val_precision": [], "val_recall": [],
    }
    best_val_iou = -1.0
    best_val_f1 = -1.0
    patience_counter = 0
    eps = 1e-6

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            optimizer.zero_grad()
            preds = model(images.to(device)).squeeze(1)
            loss = loss_fn(preds, masks.to(device).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # Single-pass validation: loss + metrics together
        val_loss, val_iou, val_f1, val_prec, val_rec = evaluate(val_loader, f"Val {epoch+1}")

        # CosineAnnealingLR.step() takes NO arguments
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        torch.save(history, str(hist_path))

        is_better = (
            (val_iou > best_val_iou + eps)
            or (abs(val_iou - best_val_iou) <= eps and val_f1 > best_val_f1)
        )
        if is_better:
            best_val_iou = val_iou
            best_val_f1 = val_f1
            torch.save(model.state_dict(), str(ckpt))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                break

        progress(
            (epoch + 1) / config.NUM_EPOCHS,
            f"Epoch {epoch+1}: loss={train_loss:.4f} val_iou={val_iou:.4f} val_f1={val_f1:.4f}",
        )

    # -- Test evaluation -------------------------------------------------------
    if ckpt.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location=device))
    _, test_iou, test_f1, test_prec, test_rec = evaluate(test_loader, "Test")
    history["test_metrics"] = {
        "iou": test_iou, "f1": test_f1, "precision": test_prec, "recall": test_rec,
    }
    torch.save(history, str(hist_path))

    progress(1.0, f"Training complete -- test IoU={test_iou:.4f} F1={test_f1:.4f}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history


def evaluate_test_model(
    year: str | None = None,
    data_type: str | None = None,
    architecture: str | None = None,
    test_year: str | None = None,
    test_data_type: str | None = None,
    progress: Callable[[float, str], None] = _NOOP,
) -> dict:
    """Evaluate the saved best checkpoint on held-out test chips.

    Parameters
    ----------
    year : str
        Year used for default parameter resolution.
    data_type, architecture : str
        Identify the checkpoint to load.
    test_year : str or None
        Year of the test chips.  Defaults to *year*.  Set this to
        evaluate a model trained on one year against test chips from
        another (e.g. ``test_year="2025"`` with a 2021 checkpoint).

    Returns
    -------
    dict
        ``{"loss", "iou", "f1", "precision", "recall",
        "checkpoint", "n_test_images"}``
    """
    import torch
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader

    year = str(year or config.YEAR)
    data_type = data_type or config.DATA_TYPE
    architecture = architecture or config.ARCHITECTURE
    test_year = str(test_year or year)
    test_data_type = test_data_type or data_type

    config.seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load checkpoint and deduce channel count from weights
    ckpt = config.checkpoint_path(data_type, architecture)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    state_dict = torch.load(str(ckpt), map_location=device)
    first_weight_key = next((k for k in state_dict.keys() if "conv1.weight" in k or "patch_embed" in k), None)
    if first_weight_key and hasattr(state_dict[first_weight_key], "shape") and len(state_dict[first_weight_key].shape) >= 2:
        n_channels = state_dict[first_weight_key].shape[1]
    else:
        n_channels = 4 if "ps_scene" in data_type.lower() else config.num_channels(data_type)

    # Test chips — may come from a different year than the model
    root = config.GEOAI_ROOT / "models" / test_year
    test_imgs = sorted(glob.glob(str(root / "test" / test_data_type / "images" / "*.tif")))
    test_masks = sorted(glob.glob(str(root / "test" / test_data_type / "masks" / "*.tif")))

    assert test_imgs, f"No test images in {root / 'test' / test_data_type / 'images'}"
    assert test_masks, f"No test masks in {root / 'test' / test_data_type / 'masks'}"
    assert len(test_imgs) == len(test_masks), "Image/mask count mismatch"

    batch_size = getattr(config, "BATCH_SIZE_INFERENCE", 4)
    test_ds = PipelineDataset(
        image_paths=test_imgs, label_paths=test_masks,
        transforms=None, num_channels=n_channels,
        target_size=(config.CHIP_SIZE, config.CHIP_SIZE),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=0,
    )

    # Model — checkpoint is year-agnostic (keyed by data_type + arch)
    if architecture == "_conv":
        model = smp.create_model(
            "unet", encoder_name="resnet34", encoder_weights=None,
            in_channels=n_channels, classes=1,
        ).to(device)
    else:
        model = smp.create_model(
            "segformer", encoder_name="mit_b0", encoder_weights=None,
            in_channels=n_channels, classes=1,
        ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    loss_fn = smp.losses.DiceLoss(mode="binary")

    progress(
        0.0,
        f"Testing: model_dt={data_type}, test_dt={test_data_type}, "
        f"arch={architecture}, channels={n_channels}, test_year={test_year}"
    )

    tp = fp = fn = tn = 0.0
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Test", leave=False):
            preds = model(images.to(device)).squeeze(1)
            masks_d = masks.to(device)
            running_loss += loss_fn(preds, masks_d.float()).item()
            b_tp, b_fp, b_fn, b_tn = smp.metrics.get_stats(
                preds.sigmoid(), masks_d.long(), mode="binary", threshold=config.THRESHOLD,
            )
            tp += b_tp.sum().item()
            fp += b_fp.sum().item()
            fn += b_fn.sum().item()
            tn += b_tn.sum().item()

    t = lambda x: torch.tensor(x)
    avg_loss = running_loss / max(len(test_loader), 1)
    iou = smp.metrics.iou_score(t(tp), t(fp), t(fn), t(tn), reduction="micro").item()
    f1 = smp.metrics.f1_score(t(tp), t(fp), t(fn), t(tn), reduction="micro").item()
    prec = smp.metrics.precision(t(tp), t(fp), t(fn), t(tn), reduction="micro").item()
    rec = smp.metrics.recall(t(tp), t(fp), t(fn), t(tn), reduction="micro").item()

    metrics = {
        "loss": avg_loss,
        "iou": iou,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "checkpoint": str(ckpt),
        "n_test_images": len(test_imgs),
    }

    # Append to training history if it exists
    hist_path = config.history_path(data_type, architecture)
    if hist_path.exists():
        history = torch.load(str(hist_path), map_location="cpu")
        key = "test_metrics" if test_year == year else f"test_metrics_{test_year}"
        history[key] = metrics
        torch.save(history, str(hist_path))

    progress(1.0, f"Test complete -- IoU={iou:.4f} F1={f1:.4f} "
                   f"(n={len(test_imgs)}, year={test_year})")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def plot_performance_metrics(
    year: str | None = None,
    data_type: str | None = None,
    architecture: str | None = None,
    save: bool = True,
    show: bool = True,
):
    """Plot training history loss, IoU, Precision, Recall in a publication-quality 2x2 grid."""
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    year = year or config.YEAR
    data_type = data_type or config.DATA_TYPE
    architecture = architecture or config.ARCHITECTURE

    hist_path = config.history_path(data_type, architecture)
    if not hist_path.exists():
        raise FileNotFoundError(f"History file not found: {hist_path}")

    history = torch.load(str(hist_path), map_location="cpu")
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()

    panels = [
        ("Loss",      ["train_loss", "val_loss"],   ["Train loss", "Val loss"],   ["#4878CF", "#E8843A"], "Loss"),
        ("IoU",       ["val_iou"],                  ["Val IoU"],                  ["#4878CF"],            "IoU"),
        ("Precision", ["val_precision"],            ["Val Precision"],            ["#4878CF"],            "Precision"),
        ("Recall",    ["val_recall"],               ["Val Recall"],               ["#4878CF"],            "Recall"),
    ]

    metric_keys = ["val_iou", "val_precision", "val_recall"]
    all_vals = np.concatenate([history[k] for k in metric_keys])
    valid_vals = all_vals[~np.isnan(all_vals)]
    y_min = np.floor(valid_vals.min() * 10) / 10 if len(valid_vals) else 0.0
    y_max = np.ceil(valid_vals.max() * 10) / 10 if len(valid_vals) else 1.0
    common_ticks = np.arange(y_min, y_max + 0.01, 0.1)

    loss_vals = np.concatenate([history["train_loss"], history["val_loss"]])
    valid_loss = loss_vals[~np.isnan(loss_vals)]
    loss_min = np.floor(valid_loss.min() * 20) / 20 if len(valid_loss) else 0.0
    loss_max = np.ceil(valid_loss.max() * 20) / 20 if len(valid_loss) else 1.0
    loss_ticks = np.arange(loss_min, loss_max + 0.001, 0.05)

    for ax, (title, keys, labels, colors, ylabel) in zip(axes, panels):
        for key, label, color in zip(keys, labels, colors):
            raw = history[key]
            ax.plot(epochs, raw, color=color, alpha=1, linewidth=1, label=label)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=4)
        ax.set_xlim(1, len(epochs))
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10, framealpha=0.5)
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        if title == "Loss":
            ax.set_yticks(loss_ticks)
            ax.set_ylim(loss_min - 0.01, loss_max + 0.01)
        else:
            ax.set_yticks(common_ticks)
            ax.set_ylim(y_min - 0.02, y_max + 0.02)

    for ax, letter in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(0.05, 0.95, letter, transform=ax.transAxes, fontsize=10, va="top")

    plt.tight_layout(pad=1.5)
    if save:
        save_path = config.GEOAI_ROOT / f"models/unet_models/{data_type}{architecture}_performance_metrics.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ======================================================================
# Stage 4 -- Run inference
# ======================================================================

def run_inference(
    year: str | None = None,
    data_type: str | None = None,
    architecture: str | None = None,
    inference_year: str | None = None,
    inference_data_type: str | None = None,
    clean_existing: bool = True,
    normalize: str | None = None,
    progress: Callable[[float, str], None] = _NOOP,
) -> None:
    """Sliding-window inference over all tiles in the AOI.

    Parameters
    ----------
    year : str
        Year used for default parameter resolution of the trained model.
    data_type : str
        Data type used to identify the trained checkpoint.
    architecture : str
        Model architecture suffix ("_conv" or "_trans").
    inference_year : str or None
        Year of the imagery to predict on. Defaults to *year*.
    inference_data_type : str or None
        Data type of the imagery to predict on. Defaults to *data_type*.
    clean_existing : bool
        If True, remove old prediction TIFs from the output directory.
    normalize : str
        Chip normalization strategy. Must match training.
    """
    import torch
    import segmentation_models_pytorch as smp

    year = year or config.YEAR
    data_type = data_type or config.DATA_TYPE
    architecture = architecture or config.ARCHITECTURE
    inference_year = inference_year or year
    inference_data_type = inference_data_type or data_type
    normalize = normalize or config.NORMALIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint path (robust to ps_scene vs ps_scenes naming)
    ckpt = config.checkpoint_path(data_type, architecture)
    if not ckpt.exists():
        for candidate_dt in [
            data_type.replace("ps_scene_", "ps_scenes_") if "ps_scene_" in data_type else None,
            data_type.replace("ps_scenes_", "ps_scene_") if "ps_scenes_" in data_type else None,
        ]:
            if candidate_dt and config.checkpoint_path(candidate_dt, architecture).exists():
                ckpt = config.checkpoint_path(candidate_dt, architecture)
                data_type = candidate_dt
                break
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    state_dict = torch.load(str(ckpt), map_location=device)

    # Derive channel count dynamically from checkpoint weights (conv1 / patch_embed)
    first_weight_key = next((k for k in state_dict.keys() if "conv1.weight" in k or "patch_embed" in k), None)
    if first_weight_key and hasattr(state_dict[first_weight_key], "shape") and len(state_dict[first_weight_key].shape) >= 2:
        n_channels = state_dict[first_weight_key].shape[1]
    else:
        n_channels = 4 if "ps_scene" in data_type.lower() else config.num_channels(data_type)

    if architecture == "_conv":
        model = smp.create_model(
            "unet",
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=n_channels,
            classes=1,
        ).to(device)
    else:
        model = smp.create_model(
            "segformer",
            encoder_name="mit_b0",
            encoder_weights=None,
            in_channels=n_channels,
            classes=1,
        ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # Resolve inference input directory (robust to 'all' subfolder and naming)
    input_dir = config.images_dir(inference_year, "all", inference_data_type)
    if not input_dir.exists():
        candidates = [
            config.images_dir(inference_year, "", inference_data_type),
            config.GEOAI_ROOT / "images" / inference_year / inference_data_type,
            config.GEOAI_ROOT / "images" / inference_year / "test" / inference_data_type,
        ]
        alt_dt = (
            inference_data_type.replace("ps_scene_", "ps_scenes_")
            if "ps_scene_" in inference_data_type
            else inference_data_type.replace("ps_scenes_", "ps_scene_")
        )
        candidates.extend([
            config.images_dir(inference_year, "all", alt_dt),
            config.images_dir(inference_year, "", alt_dt),
            config.GEOAI_ROOT / "images" / inference_year / alt_dt,
        ])
        for c in candidates:
            if c.exists():
                input_dir = c
                break
    if not input_dir.exists():
        raise FileNotFoundError(f"Inference image folder not found: {input_dir}")

    out = config.output_dir(inference_year, inference_data_type)
    out.mkdir(parents=True, exist_ok=True)

    if clean_existing:
        _clean_output_dir(out, ".tif", progress)

    # Auto-detect if checkpoint was trained with legacy /255.0 or new physical /10000.0 scaling
    is_ps_scene = "ps_scene" in data_type.lower() or "ps_scene" in inference_data_type.lower()
    legacy_scale = False
    if is_ps_scene and "encoder.bn1.running_mean" in state_dict:
        legacy_scale = bool(state_dict["encoder.bn1.running_mean"].abs().max().item() > 10.0)

    if is_ps_scene:
        if legacy_scale:
            normalize_fn = lambda c: c.astype(np.float32, copy=True) / 255.0
            norm_label = "ps_legacy (/255.0)"
        else:
            normalize_fn = lambda c: np.clip(c.astype(np.float32, copy=True) / 10000.0, 0.0, 1.0)
            norm_label = "ps_reflectance (/10000.0 -> [0, 1])"
    elif normalize in ("dataset", "div255"):
        normalize_fn = lambda c: np.clip(c.astype(np.float32, copy=True) / 255.0, 0.0, 1.0)
        norm_label = "div255"
    elif normalize == "minmax":
        normalize_fn = config.normalize_minmax
        norm_label = "minmax"
    else:
        normalize_fn = config.normalize_imagenet
        norm_label = "imagenet"

    progress(
        0.0,
        f"Inference: model_dt={data_type}, infer_dt={inference_data_type}, "
        f"arch={architecture}, channels={n_channels}, norm={norm_label}, "
        f"inference_year={inference_year}"
    )

    tiles = [f for f in os.listdir(input_dir) if f.endswith(".tif")]
    total = len(tiles)
    if total == 0:
        raise FileNotFoundError(f"No .tif tiles found in {input_dir}")

    for tile_idx, tile_name in enumerate(tiles):
        src_path = input_dir / tile_name
        dst_path = out / tile_name

        with rasterio.open(src_path) as src:
            if src.count < n_channels:
                raise ValueError(
                    f"{tile_name}: expected >= {n_channels} bands, got {src.count}"
                )

            H, W = src.height, src.width
            profile = src.profile.copy()
            image = src.read(list(range(1, n_channels + 1))).astype(np.float32)
            is_16bit = src.dtypes[0] == "uint16" or image.max() > 255.0

        accum = np.zeros((H, W), dtype=np.float32)
        weights = np.zeros((H, W), dtype=np.float32)

        gauss = _gaussian_weight_2d(config.INFERENCE_WINDOW)

        chips, coords = [], []
        for x, y in config.window_positions(
            W, H, config.INFERENCE_WINDOW, config.INFERENCE_STRIDE
        ):
            chip = image[:, y:y + config.INFERENCE_WINDOW, x:x + config.INFERENCE_WINDOW]
            if is_16bit and not is_ps_scene:
                chips.append(np.clip(chip / 10000.0, 0.0, 1.0))
            else:
                chips.append(normalize_fn(chip))
            coords.append((y, x))

        with torch.no_grad():
            for i in range(0, len(chips), config.BATCH_SIZE_INFERENCE):
                batch_np = np.stack(chips[i:i + config.BATCH_SIZE_INFERENCE])
                batch_t = torch.from_numpy(batch_np).to(device)
                logits = model(batch_t)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

                for j, (y, x) in enumerate(coords[i:i + config.BATCH_SIZE_INFERENCE]):
                    accum[y:y + config.INFERENCE_WINDOW, x:x + config.INFERENCE_WINDOW] += probs[j] * gauss
                    weights[y:y + config.INFERENCE_WINDOW, x:x + config.INFERENCE_WINDOW] += gauss

        weights[weights == 0] = 1
        prob_map = accum / weights
        pred = np.where(prob_map >= config.THRESHOLD, np.uint8(1), np.uint8(255))

        profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=255)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(pred[np.newaxis, ...])

        progress((tile_idx + 1) / total, f"Predicted {tile_name}")

    progress(1.0, f"Inference complete -- {total} tiles written to {out}")


# ======================================================================
# Stage 5 -- Mosaic and reproject
# ======================================================================

def mosaic_predictions(
    year: str | None = None,
    data_type: str | None = None,
    clean_existing: bool = True,
    reference_mosaic: str | Path | None = None, # used to allign pixels
    progress: Callable[[float, str], None] = _NOOP,
) -> None:
    """Merge per-tile predictions and reproject to Ukraine Albers.

    Parameters
    ----------
    clean_existing : bool
        If *True* (default), remove the old mosaic file before writing
        a new one.
    """
    from rasterio.crs import CRS
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    year = year or config.YEAR
    data_type = data_type or config.DATA_TYPE

    src_crs = CRS.from_epsg(3857)
    dst_crs = CRS.from_wkt(config.UKRAINE_ALBERS_WKT)

    # Clean old mosaic if it exists
    mosaic_out = config.mosaic_path(year, data_type)
    if clean_existing and mosaic_out.exists():
        progress(0.0, f"Removing old mosaic: {mosaic_out.name}")
        mosaic_out.unlink()

    pred_dir = config.output_dir(year, data_type)
    pred_paths = sorted(glob.glob(str(pred_dir / "*.tif")))
    if not pred_paths:
        raise FileNotFoundError(f"No prediction tiles in {pred_dir}")

    progress(0.1, f"Merging {len(pred_paths)} tiles")

    # Use ExitStack to guarantee all files are closed even on error
    with contextlib.ExitStack() as stack:
        src_files = [stack.enter_context(rasterio.open(p)) for p in pred_paths]
        # nodata=255 ensures merge respects transparent background
        mosaic_arr, mosaic_transform = merge(src_files, nodata=255)

    progress(0.5, "Reprojecting to Ukraine Albers")

    mosaic_arr = mosaic_arr.astype("uint8")

    if reference_mosaic is not None:
        ref_path = Path(reference_mosaic)
        if not ref_path.is_absolute():
            candidates = [
                Path(config.GEOAI_ROOT) / "outputs" / ref_path,
                Path(config.GEOAI_ROOT) / "outputs" / (year or "2021") / ref_path,
                Path(config.GEOAI_ROOT) / "outputs" / "2021" / ref_path,
            ]
            for c in candidates:
                if c.exists():
                    ref_path = c
                    break
        with rasterio.open(ref_path) as ref:
            transform = ref.transform
            width     = ref.width
            height    = ref.height
    else:
        # First month: compute grid from bounds
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs,
            mosaic_arr.shape[2], mosaic_arr.shape[1],
            *rasterio.transform.array_bounds(
                mosaic_arr.shape[1], mosaic_arr.shape[2], mosaic_transform
            ),
        )

    # Output: class 1 = shelterbelt, nodata (255) = everything else.
    # GIS software renders nodata as transparent, so only shelterbelts are
    # visible — matching the original single-class approach.
    dst_profile = {
        "driver": "GTiff",
        "dtype": rasterio.uint8,
        "count": 1,
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "compress": "lzw",
        "nodata": 255,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    out_path = config.mosaic_path(year, data_type)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **dst_profile) as dst:
        reproject(
            source=mosaic_arr[0],
            destination=rasterio.band(dst, 1),
            src_transform=mosaic_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=255,
            dst_nodata=255,
        )

    progress(1.0, f"Mosaic saved: {out_path}")

# ======================================================================
# Stage 6 -- Temporal majority mask adjustment
# ======================================================================

def vote_predictions(
    year: str | None = None,
    months: list[str] | None = None,
    min_votes: int = 2,
    chunk_rows: int = 1000,
    progress: Callable[[float, str], None] = _NOOP,
) -> None:
    """Combine per-month prediction mosaics into a single voted mosaic.

    Processes the rasters in horizontal strips of *chunk_rows* rows to
    avoid loading all months into RAM simultaneously.

    Input values:  1 = belt,  255 = nodata
    Output values: 1 = belt confirmed by >= min_votes months,  255 = nodata
    """
    year   = year   or config.YEAR
    months = months or config.PREDICTION_MONTHS

    mosaic_dir   = Path(config.GEOAI_ROOT) / "outputs" / year
    mosaic_paths = [
        mosaic_dir / f"global_monthly_{year}_{mn}_mosaic_cl123_albers.tif"
        for mn in months
    ]
    out_path = mosaic_dir / f"global_monthly_{year}_voted_cl123_albers.tif"

    for p in mosaic_paths:
        if not p.exists():
            raise FileNotFoundError(f"Monthly mosaic not found: {p}")

    # Use first mosaic as reference grid
    with rasterio.open(mosaic_paths[0]) as ref:
        profile    = ref.profile.copy()
        ref_height = ref.height
        ref_width  = ref.width

    profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=255)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    progress(0.0, f"Voting {len(mosaic_paths)} months in chunks of {chunk_rows} rows...")

    src_files = [rasterio.open(p) for p in mosaic_paths]
    try:
        with rasterio.open(out_path, "w", **profile) as dst:
            n_chunks = (ref_height + chunk_rows - 1) // chunk_rows
            for chunk_idx in range(n_chunks):
                row_start = chunk_idx * chunk_rows
                row_end   = min(row_start + chunk_rows, ref_height)
                win       = Window(0, row_start, ref_width, row_end - row_start)

                # Read this strip from each month; fill missing pixels with 255
                layers = []
                for src in src_files:
                    # handle mosaics that differ by a few pixels in height/width
                    src_win = Window(
                        0,
                        min(row_start, src.height - 1),
                        min(ref_width,  src.width),
                        min(row_end - row_start, src.height - row_start),
                    )
                    strip = np.full((row_end - row_start, ref_width), 255, dtype=np.uint8)
                    if src_win.height > 0:
                        data = src.read(1, window=src_win)
                        strip[:data.shape[0], :data.shape[1]] = data
                    layers.append(strip)

                stack   = np.stack(layers, axis=0)       # (N, chunk_rows, W)
                n_votes = (stack == 1).sum(axis=0)       # (chunk_rows, W)
                pred    = np.where(
                    n_votes >= min_votes,
                    np.uint8(1),
                    np.uint8(255),
                )
                dst.write(pred[np.newaxis, ...], window=win)

                progress(
                    (chunk_idx + 1) / n_chunks,
                    f"Chunk {chunk_idx + 1}/{n_chunks} (rows {row_start}–{row_end})"
                )
    finally:
        for s in src_files:
            s.close()

    progress(1.0, f"Voted mosaic saved: {out_path.name}")

# ======================================================================
# Stage 7 -- Loss mask
# ======================================================================

def postprocess_year(
    target_year: str = "2025",
    ref_year: str = "2021",
    chunk_rows: int = 1000,
    progress: Callable[[float, str], None] = _NOOP,
) -> None:
    """Clean target-year belt mosaic using ref-year and compute loss layer.

    Assumes vote_predictions() has already written:

        outputs/{ref_year}/global_monthly_{ref_year}_voted_cl123_albers.tif
        outputs/{target_year}/global_monthly_{target_year}_voted_cl123_albers.tif

    Outputs (in outputs/{target_year}):

        global_monthly_{target_year}_clean_cl123_albers.tif
            1   = belt in both years (cleaned target_year)
            255 = nodata (or outside belt network)

        global_monthly_{ref_year}_{target_year}_loss_cl123_albers.tif
            1   = loss (belt in ref_year, not in cleaned target_year)
            255 = nodata (no loss)
    """
    base_dir = Path(config.GEOAI_ROOT) / "outputs"

    # Inputs: voted (unfiltered) mosaics
    ref_path = base_dir / ref_year     / f"global_monthly_{ref_year}_voted_cl123_albers.tif"
    tgt_path = base_dir / target_year  / f"global_monthly_{target_year}_voted_cl123_albers.tif"

    for p in (ref_path, tgt_path):
        if not p.exists():
            raise FileNotFoundError(f"Mosaic not found: {p}")

    # Outputs: cleaned target-year mosaic and loss layer
    out_dir    = base_dir / target_year
    out_dir.mkdir(parents=True, exist_ok=True)
    out_clean  = out_dir / f"global_monthly_{target_year}_clean_cl123_albers.tif"
    out_change = out_dir / f"global_monthly_{ref_year}_{target_year}_loss_cl123_albers.tif"

    # Use ref mosaic as grid reference (profiles are identical by design)
    with rasterio.open(ref_path) as ref_src:
        profile    = ref_src.profile.copy()
        ref_height = ref_src.height
        ref_width  = ref_src.width

    profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=255)

    progress(0.0, f"Postprocessing {target_year} using {ref_year} as reference ...")

    src_ref = rasterio.open(ref_path)
    src_tgt = rasterio.open(tgt_path)

    try:
        with rasterio.open(out_clean,  "w", **profile) as dst_clean, \
             rasterio.open(out_change, "w", **profile) as dst_change:

            n_chunks = (ref_height + chunk_rows - 1) // chunk_rows

            for chunk_idx in range(n_chunks):
                row_start = chunk_idx * chunk_rows
                row_end   = min(row_start + chunk_rows, ref_height)
                win       = Window(0, row_start, ref_width, row_end - row_start)
                h         = row_end - row_start

                # Read matching strips (same grid → same window)
                a_ref = src_ref.read(1, window=win)
                a_tgt = src_tgt.read(1, window=win)

                belt_ref = a_ref == 1
                belt_tgt = a_tgt == 1

                # 1) Cleaned target-year belt: intersection with reference (binary 1/255)
                cleaned = np.where(belt_ref & belt_tgt, np.uint8(1), np.uint8(255))

                # 2) Loss layer (binary 1/255) based on cleaned 2025:
                #    loss = belt in 2021 but not in cleaned target_year
                belt_clean = cleaned == 1
                loss = np.where(belt_ref & ~belt_clean, np.uint8(1), np.uint8(255))

                dst_clean.write(cleaned[np.newaxis, ...], window=win)
                dst_change.write(loss[np.newaxis, ...],   window=win)

                progress(
                    (chunk_idx + 1) / n_chunks,
                    f"Chunk {chunk_idx + 1}/{n_chunks} (rows {row_start}–{row_end})",
                )
    finally:
        src_ref.close()
        src_tgt.close()

    progress(1.0, f"Saved:\n  {out_clean.name}\n  {out_change.name}")


import subprocess
