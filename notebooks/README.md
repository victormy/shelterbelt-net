# Shelterbelt Segmentation Notebooks & Workflow Guide

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Journal: Environmental Research Letters](https://img.shields.io/badge/Journal-Environmental%20Research%20Letters-008080.svg)](https://iopscience.iop.org/journal/1748-9326)

> [!NOTE]
> **Associated Publication**
>
> This repository provides the reproducible code, spatial partitions, reference delineations, pre-trained deep learning models, and regional prediction products accompanying the peer-reviewed research article:
>
> **"Deep learning reveals severe shelterbelt degradation in eastern Ukraine due to the war"**
>
> **Authors:** Viktor Myroniuk, Yevhenii Khan, Oleksandr Sovakov, Maksym Matsala, Andrii Lashchenko, Anatolii Makarevych, Andrii Terentiev
>
> *Environmental Research Letters* (2026).

---

## Study Area Map

![Study Area & Spatial Quad Partitions across Eastern Ukraine](../figures/study_area.png)

---

## Notebooks Sequence

The `notebooks/` directory contains the sequential stages of the shelterbelt segmentation and change detection pipeline:

| Notebook | Stage Name | Description |
|:---|:---|:---|
| [`000_multiseasonal_planet_stack.ipynb`](000_multiseasonal_planet_stack.ipynb) | Multi-Seasonal Composite Stacking | Stacks April, July, and October RGB basemaps per quad into a 9-band composite capturing seasonal phenology. |
| [`001_reclassify_by_stndtype.ipynb`](001_reclassify_by_stndtype.ipynb) | Stand Type Filtering & Masking | Filters vector ground truth by `STND_CODES` (`[1, 2, 3]`), repairs invalid geometries, and writes binary masks. |
| [`002_create_training_dataset.ipynb`](002_create_training_dataset.ipynb) | Image & Mask Chipping | Extracts georeferenced $256 \times 256$ px image/mask pairs with spatial striding (128 px for train; 256 px for val/test). |
| [`003_train_val_test_model.ipynb`](003_train_val_test_model.ipynb) | Model Training & Benchmarking | Trains U-Net (ResNet34) or SegFormer (MiT-B0) using Dice loss and AdamW with Cosine Annealing. |
| [`004_run_inference.ipynb`](004_run_inference.ipynb) | Artifact-Free Sliding Inference | Performs sliding-window inference with **2D Gaussian blending** (128 px overlap) to eliminate edge seams. |
| [`005_mosaic_predicted_tiles.ipynb`](005_mosaic_predicted_tiles.ipynb) | Reprojection & Mosaicking | Reprojects quad predictions to **Ukraine Albers Equal Area** and merges into regional mosaics. |
| [`006_voted_mosaics.ipynb`](006_voted_mosaics.ipynb) | Temporal Ensemble Voting | Combines monthly predictions (June–September) via majority pixel voting to filter out cloud/shadow noise. |
| [`007_postprocessing.ipynb`](007_postprocessing.ipynb) | Error-Adjusted Net Loss Analysis | Implements **Olofsson et al. (2014)** stratified area estimation and spatial block-bootstrap ($N=1{,}000$) confidence intervals. |

---

## Working with Split Regional Outputs & VRTs

To remain below GitHub's 100 MB per-file threshold, the prediction products in `../outputs/` are partitioned across three administrative regions:
- `*-done.tif` (Donetsk region)
- `*-khar.tif` (Kharkiv region)
- `*-zapo.tif` (Zaporizhzhia region)

Virtual Rasters (`.vrt`) stitch these regional files together seamlessly:
- `../outputs/2021/global_monthly_2021_voted_cl123_albers.vrt`
- `../outputs/2025/global_monthly_2025_clean_cl123_albers.vrt`

Notebook [`007_postprocessing.ipynb`](007_postprocessing.ipynb) is configured to automatically detect and load these `.vrt` files or individual regional `.tif` files.

---

## Running Notebook 007 Directly

You can reproduce the post-processing and statistical change estimation immediately without re-training:

```python
import config
from pathlib import Path
import rasterio

# Check detected GEOAI root
print(f"Using GEOAI_ROOT: {config.GEOAI_ROOT}")

# Check 2021 and 2025 output VRTs
vrt_2021 = config.GEOAI_ROOT / "outputs" / "2021" / "global_monthly_2021_voted_cl123_albers.vrt"
print(f"2021 VRT exists: {vrt_2021.exists()}")
```

---

## Citation

```bibtex
@article{myroniuk2026shelterbelts,
  title={Deep learning reveals severe shelterbelt degradation in eastern Ukraine due to the war},
  author={Myroniuk, Viktor and Khan, Yevhenii and Sovakov, Oleksandr and Matsala, Maksym and Lashchenko, Andrii and Makarevych, Anatolii and Terentiev, Andrii},
  journal={Environmental Research Letters},
  year={2026},
  publisher={IOP Publishing}
}
```
