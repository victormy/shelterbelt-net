# Shelterbelt Segmentation from Planet Imagery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-enabled-brightgreen.svg)](https://git-lfs.github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning pipeline for binary semantic segmentation of shelterbelts (field-protective forest strips, *polezakhysni smugy*) from high-resolution Planet satellite imagery (4.7 m Planet basemaps / PlanetScope). Built to monitor and quantify vegetated land-cover change and net shelterbelt area losses caused by the armed conflict in eastern Ukraine across five epochs (2021--2025).

---

## Table of Contents

- [Overview](#overview)
- [Planet Imagery & E&R License Compliance](#planet-imagery--er-license-compliance)
- [Repository Structure](#repository-structure)
- [Pre-Trained Models & Prediction Products](#pre-trained-models--prediction-products)
- [Git LFS Notice (Large File Storage)](#git-lfs-notice-large-file-storage)
- [Delineated Reference Perimeters](#delineated-reference-perimeters)
- [Automated Basemap Download (`download_basemaps`)](#automated-basemap-download-download_basemaps)
- [Pipeline Stages](#pipeline-stages)
- [Quick Start](#quick-start)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Option A: Interactive Jupyter Notebooks](#2-option-a-interactive-jupyter-notebooks)
  - [3. Option B: Streamlit Web UI](#3-option-b-streamlit-web-ui)
  - [4. Option C: Headless Python Script](#4-option-c-headless-python-script)
- [Statistical Area Estimation (Olofsson et al. 2014)](#statistical-area-estimation-olofsson-et-al-2014)
- [Configuration Reference](#configuration-reference)
- [Citation & Contact](#citation--contact)

---

## Overview

Shelterbelts (*polezakhysni smugy*) are narrow linear tree plantations established along agricultural field boundaries across Ukraine's steppe and forest-steppe zones. They mitigate wind erosion, retain winter snow cover, preserve soil moisture, and safeguard crop yields. Hostilities, shelling, defensive entrenchments, wildfires, and military movement since February 2022 have caused extensive degradation and fragmentation of these protective networks.

This repository provides the full, reproducible workflow to:
1. Reclassify multi-class reference polygons into binary masks.
2. Chip high-resolution satellite rasters and vector masks into georeferenced $256 \times 256$ px pairs with spatial striding.
3. Train and benchmark deep convolutional (U-Net / ResNet34) and transformer (SegFormer / MiT-B0) architectures using Dice loss.
4. Run wall-to-wall sliding-window inference with **2D Gaussian blending** to eliminate seam artifacts.
5. Merge tile predictions into seamless mosaics reprojected to the **Ukraine Albers Equal Area** projection.
6. Multi-temporal voting to synthesize robust annual baseline maps (2021 vs. 2025).
7. Quantify net area losses with **Olofsson et al. (2014) stratified error-adjusted area estimation** and spatial block-bootstrap 95% confidence intervals.

---

## Planet Imagery & E&R License Compliance

> [!IMPORTANT]
> **Data Distribution Policy (Planet Labs E&R License)**
>
> Planet Labs PBC distributes high-resolution satellite imagery under the [Planet Education and Research (E&R) Program](https://www.planet.com/markets/education-and-research/). Under Section 3 of Planet's Terms of Service:
> - Users are granted a personal, non-commercial research license.
> - **Public redistribution, sublicensing, or hosting of raw Planet GeoTIFF rasters on public repositories (including GitHub) is strictly prohibited.**
>
> In accordance with this legal restriction, **raw input Planet GeoTIFFs are not hosted in this repository**. The [`images/`](images/) directory provides the exact folder layout template preserved with `.gitkeep` files. All vector ground-truth delineations, quad lists, configuration files, automated download scripts, pre-trained model weights, and derived publication-ready prediction mosaics are included.

To download the input Planet basemaps for this study area using your own Planet API key, refer to [`download_basemaps/README.md`](download_basemaps/README.md).

---

## Repository Structure

```
shelterbelt-segmentation-planet/
├── config.py                         # Single source of truth (paths, parameters, seeds)
├── pipeline.py                       # Core processing engine (stages 000–007 + dataset classes)
├── app.py                            # Streamlit interactive web interface
├── tile_splits.json                  # Spatial train/val/test quad partition
├── requirements.txt                  # Python dependencies
├── .gitattributes                    # Configures Git LFS for GeoTIFF mosaics and PyTorch models
├── .gitignore                        # Ignores intermediate chips, caches, and system files
├── README.md                         # Main repository guide
│
├── notebooks/                        # Sequential Jupyter workflow notebooks
│   ├── 000_multiseasonal_planet_stack.ipynb   # Stage 0: 9-band composite stacking
│   ├── 001_reclassify_by_stndtype.ipynb       # Stage 1: Vector mask reclassification
│   ├── 002_create_training_dataset.ipynb      # Stage 2: 256x256 image/mask chipping
│   ├── 003_train_val_test_model.ipynb         # Stage 3: Model training and validation
│   ├── 004_run_inference.ipynb                # Stage 4: Sliding-window Gaussian inference
│   ├── 005_mosaic_predicted_tiles.ipynb       # Stage 5: Reprojection & Albers mosaicking
│   ├── 006_voted_mosaics.ipynb                # Stage 6: Multi-temporal ensemble voting
│   ├── 007_postprocessing.ipynb               # Stage 7: Olofsson error-adjusted net loss
│   ├── config.py                              # Local config reference
│   ├── pipeline.py                            # Local pipeline reference
│   ├── app.py                                 # Streamlit app
│   ├── tile_splits.json                       # Tile split definitions
│   ├── requirements.txt                       # Notebook environment dependencies
│   ├── FAQ.md                                 # Frequently asked questions
│   └── CODE_REVIEW.md                         # Architecture review notes
│
├── delineation/                      # Ground-truth reference perimeters (Vector GeoJSON)
│   ├── all_3857_2021/                # 23 pre-war tile delineations (EPSG:3857)
│   ├── all_3857_2025/                # 3 post-invasion held-out test delineations (EPSG:3857)
│   ├── all_quads.geojson             # Study area quad grid in EPSG:4326
│   └── README.md                     # Delineation documentation
│
├── download_basemaps/                # Automated tools for Planet Basemaps API
│   ├── create_quad_index.py          # Queries Planet API to index quads intersecting AOI
│   ├── download_by_tile_index.py     # Batch downloads missing GeoTIFFs via BasicAuth
│   ├── config.yaml                   # Download config referencing quad IDs and mosaics
│   ├── mosaic_ids.csv                # Lookup table of Planet monthly basemap UUIDs (2020-2025)
│   ├── aoi.geojson                   # Outer study area polygon in eastern Ukraine
│   ├── khar-done-zapo.geojson        # Oblast boundaries (Kharkiv, Donetsk, Zaporizhzhia)
│   ├── all_quads.geojson             # Complete quad grid (45 quads)
│   ├── sampled_quads.geojson         # Sampled quad subset used in experiments
│   ├── how_to_download_mosaics       # Quick command cheat-sheet
│   ├── how_to_update_mosaic_list     # API URL template for querying new mosaic IDs
│   └── README.md                     # Detailed step-by-step download guide
│
├── images/                           # Input satellite imagery directory tree template
│   ├── 2021/                         # Pre-war monthly basemap folders (train/val/test/all)
│   ├── 2025/                         # Post-invasion monthly basemap folders (06-09)
│   └── README.md                     # Directory mapping instructions
│
├── models/                           # Model checkpoints and training artifacts
│   └── unet_models/                  # Pre-trained U-Net weights & history curves
│       ├── global_monthly_2021_07_mosaic_cl123_conv_best_model.pth           # Best weights (~93 MB)
│       ├── global_monthly_2021_07_mosaic_cl123_conv_training_history.pth     # Training history
│       └── global_monthly_2021_07_mosaic_cl123_conv_performance_metrics.png # Metrics curve
│
└── outputs/                          # Prediction products & final Albers mosaics
    ├── 2021/
    │   └── global_monthly_2021_voted_cl123_albers.tif  # 2021 baseline voted mosaic (~127 MB)
    └── 2025/
        └── global_monthly_2025_clean_cl123_albers.tif  # 2025 post-invasion clean mosaic (~119 MB)
```

---

## Pre-Trained Models & Prediction Products

To enable immediate reproduction and analysis without re-running 50 training epochs or downloading gigabytes of satellite imagery:
- **Pre-Trained Weights**: [`models/unet_models/global_monthly_2021_07_mosaic_cl123_conv_best_model.pth`](models/unet_models/) is ready for inference via `pipeline.run_inference()` or Stage `004_run_inference.ipynb`.
- **Classification Mosaics**: The final annual prediction mosaics for **2021** (pre-war voted baseline) and **2025** (post-invasion clean map) are included in [`outputs/`](outputs/). You can run [`notebooks/007_postprocessing.ipynb`](notebooks/007_postprocessing.ipynb) immediately to reproduce the error-adjusted area estimates, confusion matrices, and wartime loss statistics.

---

## Git LFS Notice (Large File Storage)

The repository includes pre-computed GeoTIFF mosaics in `outputs/` (~120--128 MB) and a PyTorch checkpoint in `models/` (~93 MB). GitHub enforces a strict 100 MB per-file limit on standard Git commits.

This repository is pre-configured with **[Git LFS](https://git-lfs.github.com/)** via `.gitattributes`. Before cloning or pushing:

```bash
# Install Git LFS (run once on your machine)
git lfs install

# Clone the repository with large files
git clone https://github.com/your-username/shelterbelt-segmentation-planet.git

# Pull large files if not automatically downloaded
git lfs pull
```

---

## Delineated Reference Perimeters

Manual reference delineations were created by expert photointerpretation on Planet 4.7 m imagery:
- **Pre-war (2021)**: Delineations across 23 tiles in eastern Ukraine (`delineation/all_3857_2021/`).
- **Post-invasion (2025)**: Re-delineation of the three held-out test quads (`1222-1323`, `1228-1353`, `1240-1341`) on July 2025 imagery (`delineation/all_3857_2025/`).

### Stand Type Attribute (`STNDTYPE`)

| Code | Type | Meaning | Pipeline Inclusion |
|:---:|:---|:---|:---:|
| `1` | Field-protective shelterbelts | Tree belts buffering agricultural crop fields | Yes (primary) |
| `2` | Infrastructure strips | Railway and roadside protective tree lines | Yes (when `STND_CODES=[1,2,3]`) |
| `3` | Other linear plantations | Windbreaks, riparian strips, boundary lines | Yes (when `STND_CODES=[1,2,3]`) |
| `4` | Massive / areal forest | Continuous natural forest blocks | **Excluded** |

---

## Automated Basemap Download (`download_basemaps`)

The [`download_basemaps/`](download_basemaps/) directory contains complete tools to fetch Planet basemaps via Planet's Basemaps API:

```bash
cd download_basemaps

# 1. Set your Planet API key
export PL_API_KEY="your-api-key"

# 2. (Optional) Re-index quads for a custom AOI
python create_quad_index.py aoi.geojson all_quads.geojson

# 3. Download quads specified in config.yaml
python download_by_tile_index.py config.yaml 100
```

See the detailed guide in [`download_basemaps/README.md`](download_basemaps/README.md).

---

## Pipeline Stages

| Stage | Script / Notebook | Core Function | Description |
|:---:|:---|:---|:---|
| **0** | `000_multiseasonal_planet_stack` | `stack_multiseasonal()` | Stacks April, July, and October RGB basemaps into 9-band composites per tile. |
| **1** | `001_reclassify_by_stndtype` | `reclassify_masks()` | Filters reference GeoJSONs by `STND_CODES`, repairs geometries with `shapely`, and writes binary masks. |
| **2** | `002_create_training_dataset` | `chip_training_dataset()` | Chips tiles and masks into georeferenced $256 \times 256$ px pairs (stride 128 for train; 256 for val/test). |
| **3** | `003_train_val_test_model` | `train_model()` | Trains segmentation model (U-Net / SegFormer) with Dice loss, AdamW, and Cosine Annealing. Selects checkpoint by best validation IoU/F1. |
| **4** | `004_run_inference` | `run_inference()` | Sliding-window inference over AOI tiles with **2D Gaussian blending** (overlap 128 px) and sigmoid thresholding $\ge 0.5$. |
| **5** | `005_mosaic_predicted_tiles` | `mosaic_predictions()` | Merges tile predictions and reprojects to **Ukraine Albers Equal Area** with nearest-neighbor resampling. |
| **6** | `006_voted_mosaics` | `vote_predictions()` | Ensembles multiple monthly predictions (June--September) via pixel voting for temporal stability. |
| **7** | `007_postprocessing` | `postprocess_year()` / analysis | Difference mapping, skeletonization, and **Olofsson et al. (2014)** error-adjusted area estimation. |

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/your-username/shelterbelt-segmentation-planet.git
cd shelterbelt-segmentation-planet
pip install -r requirements.txt
```

*(Optional)* Set the `GEOAI_ROOT` environment variable if your data is located in an external directory:
```bash
export GEOAI_ROOT=/path/to/data
```
By default, `config.py` automatically detects the repository root.

### 2. Option A: Interactive Jupyter Notebooks

Launch Jupyter and open the notebooks in order:

```bash
cd notebooks
jupyter lab
```

Run notebooks `000` through `007`. Each notebook is a clean, parameter-free wrapper:
```python
import pipeline

def progress(frac, msg):
    print(f"[{frac:.0%}] {msg}")

pipeline.run_inference(progress=progress)
```

### 3. Option B: Streamlit Web UI

Run the interactive dashboard to control parameters, switch architectures, and monitor runs with visual progress bars:

```bash
streamlit run app.py
```

### 4. Option C: Headless Python Script

Execute the complete pipeline in a Python script or automated CI runner:

```python
import pipeline

# 1. Prepare masks
pipeline.reclassify_masks()

# 2. Chip dataset
pipeline.chip_training_dataset()

# 3. Train model
history = pipeline.train_model()

# 4. Predict
pipeline.run_inference()

# 5. Mosaic to Ukraine Albers
pipeline.mosaic_predictions()
```

---

## Statistical Area Estimation (Olofsson et al. 2014)

Pixel counting on raw satellite classification maps introduces systematic area bias due to misclassification. Notebook [`007_postprocessing.ipynb`](notebooks/007_postprocessing.ipynb) implements the good-practice stratified estimator of **Olofsson et al. (2014)** (*Remote Sensing of Environment* 148:42–57):

1. **Error Matrix**: Built across held-out test quads comparing map predictions to manual 2021 and 2025 reference delineations.
2. **Stratum Weights**: $W_i = \text{mapped area of stratum } i / \text{total study area}$.
3. **Error-Adjusted Area**:
   $$\hat{A}_k = A_{\text{tot}} \sum_{i} W_i \frac{n_{ik}}{n_{i\cdot}}$$
4. **Spatial Block-Bootstrap**: Because adjacent 4.7 m pixels within quads exhibit spatial autocorrelation, notebook 007 resamples $256 \times 256$ px blocks with replacement ($N=1{,}000$ iterations) to report robust 95% confidence intervals on headline net loss.

---

## Configuration Reference

Key parameters in `config.py`:

| Parameter | Default | Description |
|---|:---:|---|
| `YEAR` | `"2021"` | Active processing year (`"2021"`, `"2022"`, ..., `"2025"`) |
| `DATA_TYPE` | `"global_monthly_2021_07_mosaic"` | Input raster collection name |
| `ARCHITECTURE` | `"_conv"` | Model architecture: `"_conv"` (U-Net ResNet34) or `"_trans"` (SegFormer MiT-B0) |
| `NORMALIZE` | `"minmax"` | Normalization: `"minmax"` (0--1 per chip) or `"imagenet"` (ImageNet statistics) |
| `STND_CODES` | `[1, 2, 3]` | Included stand types (`[1]` = shelterbelts only; `[1,2,3]` = all linear) |
| `CHIP_SIZE` | `256` | Square chip dimension in pixels |
| `TRAIN_STRIDE` | `128` | Stride for training chips (50% spatial overlap) |
| `EVAL_STRIDE` | `256` | Stride for validation/test chips (0% overlap) |
| `INFERENCE_OVERLAP`| `128` | Overlap for Gaussian-weighted sliding-window blending |
| `BATCH_SIZE_TRAIN`| `8` | Training batch size |
| `NUM_EPOCHS` | `50` | Maximum training epochs |
| `PATIENCE` | `15` | Early stopping patience on validation IoU/F1 |
| `THRESHOLD` | `0.5` | Sigmoid classification threshold |

---

## Coordinate Reference Systems

- **Input Imagery & Delineations**: WGS 84 / Pseudo-Mercator (**EPSG:3857**).
- **Study Area Boundaries**: WGS 84 (**EPSG:4326**).
- **Mosaics & Area Statistics**: **Ukraine Albers Equal Area** (Custom WKT centered on Ukraine, eliminating area distortion across UTM zones).

---

## Citation & Contact

If you use this pipeline, pre-trained weights, or reference delineations in your research, please cite:

```bibtex
@article{shelterbelt_ukraine_2026,
  title={Quantifying Shelterbelt Loss from Armed Conflict in Eastern Ukraine Using Planet Satellite Imagery and Deep Learning},
  author={GEOAI Research Team},
  journal={Environmental Research Letters},
  year={2026}
}
```

For questions or issues, please open an issue in this repository or refer to [`notebooks/FAQ.md`](notebooks/FAQ.md).
