# FAQ -- Shelterbelt Segmentation Pipeline

Frequently asked questions based on team discussions during pipeline testing
(April 2026).

---

## 1. How does Python import work when I only import one function from `pipeline.py`?

**Question:** In the notebook I write `from pipeline import chip_training_dataset`.
This function uses variables like `CHIP_SIZE` that come from `config.py`.
But I only imported *one function* -- how does it get access to the config?

**Answer:** When Python executes `from pipeline import chip_training_dataset`,
it runs the **entire** `pipeline.py` file top-to-bottom *once*.  This means
the line `import config` at the top of `pipeline.py` executes too, making
all `config.*` names available inside the module.  Python caches the module
in `sys.modules`, so subsequent imports of other functions do not re-execute
the file.

In short: you request *one name*, but Python loads the whole module.  The
`import config` at the top of `pipeline.py` gives every function in that
file access to `config.CHIP_SIZE`, `config.TRAIN_STRIDE`, etc.

---

## 2. What do `STND_CODES` values mean and how does reclassification work?

**Question:** Script 001 recodes stand types 1, 2, 3 into class 1.  How?

**Answer:** The reference polygons have a `STNDTYPE` field with codes:

| Code | Meaning |
|------|---------|
| 1 | Field-protective shelterbelts (polezakhysni) |
| 2 | Railway shelterbelts |
| 3 | Other linear stands |

In `config.py`, `STND_CODES = [1]` selects only field-protective
shelterbelts.  Change to `[1, 2, 3]` to include all linear plantations.
The function `reclassify_masks()` assigns `class = 1` to any polygon whose
`STNDTYPE` is in the list, drops everything else, and writes a binary
GeoJSON per tile.

---

## 3. Why did the mosaic contain 607 files instead of 117?

**Question:** After re-running inference (script 004), the mosaic script
picked up 607 files -- old predictions mixed with new ones.

**Answer:** The output directory was not cleaned between runs.  Old `.tif`
files from a previous inference remained alongside the new predictions.

**Fix applied:** `run_inference()` and `mosaic_predictions()` now accept a
`clean_existing=True` parameter (default).  Before writing, the function
removes old `.tif` files from the output directory and logs this action via
the progress callback.  This prevents stale results from contaminating new
runs.

---

## 4. What causes the grid artefacts (sieve pattern) in the prediction mosaic?

**Question:** The mosaic shows a grid of lines at ~128-pixel (600 m) spacing
with 2-pixel-wide artefacts at tile seams.

**Answer:** Two problems combined:

1. **Simple averaging in overlap zones.**  The original code accumulated raw
   probabilities and divided by a count.  At the boundary between "1
   prediction" and "2 overlapping predictions", the average changes
   abruptly, creating a visible edge.

2. **Insufficient overlap.**  With `INFERENCE_OVERLAP = 64` (25% of the
   256 px window), the blending zone was too narrow to mask the
   discontinuity.

**Fix applied:**

- **Gaussian-weighted blending:** Each 256x256 prediction is multiplied by a
  2-D Gaussian weight matrix (center ~1.0, edges ~0.0) before accumulation.
  This creates a smooth spatial transition -- center pixels dominate, edge
  pixels contribute little.
- **Increased overlap:** `INFERENCE_OVERLAP` raised from 64 to 128 (50%
  overlap), giving a wider blending zone.

Together, these eliminate the grid artefacts.

---

## 5. Why do training and inference normalization need to match?

**Question:** What are `image_mean` and `image_std`?  Why does the model
produce poor results if normalization differs?

**Answer:** Neural networks are sensitive to input value distributions.
There are two common normalization strategies:

| Strategy | How it works | When to use |
|----------|-------------|-------------|
| **Per-chip min-max** | Scale each chip independently to [0, 1] | When training dataset class (e.g. `geoai.train.SemanticSegmentationDataset`) uses this internally |
| **ImageNet mean/std** | Subtract `[0.485, 0.456, 0.406]`, divide by `[0.229, 0.224, 0.225]` | When the pretrained encoder expects ImageNet statistics |

The `geoai.train.SemanticSegmentationDataset` uses per-chip min-max.
If inference uses ImageNet normalization instead, the model receives input
in a different value range than what it was trained on, which degrades
predictions.

**Fix applied:** `config.NORMALIZE = "minmax"` by default.  All pipeline
functions read this from config -- no need to set it in notebooks.

---

## 6. What is the difference between the original output format and the refactored one?

**Question:** Original script used class 1 and NA/transparency for
background.  The refactored version wrote 0 (background) and 1
(shelterbelt).  Why does this matter?

**Answer:** When the mosaic step uses `nodata=0`, it cannot distinguish
between "predicted background" and "no data" (e.g. pixels outside the tile
footprint).  This causes:

- Nodata areas being treated as background (loss of transparency in GIS)
- `reproject()` filling gaps with 0, which looks like valid background

**Fix applied:** Predictions now write:
- `1` = shelterbelt
- `255` = everything else (nodata)

The mosaic uses `nodata=255` for both merge and reproject, so GIS software
(QGIS, ArcGIS) renders shelterbelts as a single class on a transparent
background -- matching the original single-class approach.

---

## 7. Why does the progress bar (tqdm) not appear immediately?

**Question:** When training starts, the progress bar takes a few seconds to
appear.

**Answer:** This is a Jupyter kernel initialization delay, not a code issue.
On first execution, Python loads PyTorch, segmentation-models-pytorch, and
CUDA drivers into memory.  The `tqdm` progress bar appears as soon as the
first batch starts processing.

If it consistently fails to appear, check that `tqdm` is installed:
```bash
pip install tqdm>=4.65
```

The training loop uses `tqdm` for both the training batches
(`desc="Train N"`) and validation (`desc="Val N"`), plus the `progress()`
callback reports epoch-level metrics including IoU and F1.

---

## 8. Why did inference crash with "size mismatch for encoder.conv1.weight"?

**Question:** Running inference gave `RuntimeError: size mismatch ...
shape torch.Size([64, 3, 7, 7]) from checkpoint, the shape in current model
is torch.Size([64, 9, 7, 7])`.

**Answer:** The model was trained on 9-channel multi-seasonal data but
inference was creating a model with 3 input channels (`NUM_CHANNELS = 3`
was hardcoded in config).

**Fix applied:** `NUM_CHANNELS` is no longer a constant.  It is now
`config.num_channels(data_type)` — a function that derives the channel
count from the data type name:

- `"multiseasonal"` in name → **9** channels (3 months x 3 RGB bands)
- otherwise → **3** channels (single month, RGB)

All pipeline functions call `config.num_channels(data_type)` automatically.
You just set `config.DATA_TYPE` once in `config.py` and everything
propagates.

---

## 9. Why were parameters duplicated in config and notebooks?

**Question:** It was confusing that `YEAR`, `DATA_TYPE`, `ARCHITECTURE` were
defined in `config.py` AND again at the top of each notebook.

**Fix applied:** `config.py` is now the **single source of truth** for all
parameters.  Notebooks contain zero parameter definitions — they just call:

```python
import pipeline
pipeline.run_inference(progress=progress)
```

To change any setting, edit `config.py` once:

```python
YEAR = "2021"
DATA_TYPE = "global_multiseasonal_2021_mosaic"
ARCHITECTURE = "_conv"
NORMALIZE = "minmax"
```

The Streamlit UI reads these defaults and lets users override them from
the sidebar.

---

## 10. How are chips (tiles) created and what is the overlap logic?

**Question:** How does chipping work?  Why is there 50% overlap for training
but none for val/test?

**Answer:**

| Split | Window | Stride | Overlap | Rationale |
|-------|--------|--------|---------|-----------|
| Train | 256 px | 128 px | 50% | Data augmentation -- more training samples from same tiles |
| Val | 256 px | 256 px | 0% | No augmentation needed; avoids counting the same pixel twice in metrics |
| Test | 256 px | 256 px | 0% | Same as val |
| Inference | 256 px | 128 px | 50% | Smooth blending with Gaussian weights |

During chipping (`chip_training_dataset`), only chips that contain at least
one shelterbelt pixel are saved -- fully empty chips are skipped.

---

## 11. How long does training take on different hardware?

Based on team testing with 3 epochs:

| Hardware | Epoch 1 | Epoch 2+ | 3 epochs total |
|----------|---------|----------|----------------|
| GPU (RTX-class) | ~2 min | ~1.5 min | ~5 min |
| Laptop GPU | ~5 min | ~4 min | ~15 min |
| CPU only | ~60 min | ~50 min | ~2.5 hours |

GPU acceleration is effectively required for practical use.  CPU-only
training is not recommended.

---

## 12. What IoU and F1 scores should I expect?

| Metric | Expected range | Notes |
|--------|---------------|-------|
| Val IoU | 0.59--0.69 | Depends on epochs, data quality |
| Val F1 | 0.74--0.82 | Generally higher than IoU |
| Test IoU | 0.63--0.79 | 3-epoch: ~0.63; full training: ~0.79 |

Reference: a comparable study on German hedgerows achieved IoU ~0.60--0.63
for similar linear vegetation features at 10 m resolution.  Our pipeline
at Planet 4.7 m achieves comparable or better results.

---

## 13. Can shelterbelts of different widths be detected?

Based on the reference dataset statistics:

| Type | Width range | Median width |
|------|-------------|-------------|
| Field-protective (polezakhysni) | 7--60 m | 30 m |
| Railway | 30--100 m | 60 m |
| Other | Variable | Variable |

At Planet's 4.7 m pixel size, a 30 m shelterbelt is ~6 pixels wide, which
is sufficient for the U-Net to detect.  Narrower shelterbelts (< 10 m)
become challenging -- the German hedgerow study noted similar limitations.

---

## 14. How do I use GitHub for this project with large raster data?

GitHub has file size limits (100 MB hard limit).  Options:

1. **GitHub Education Pack** (free for academics) -- provides unlimited
   private repos and Git LFS storage
2. **Git LFS** -- tracks large files (`.tif`, `.pth`) via pointers;
   actual data stored on LFS server
3. **`.gitignore`** -- the project already excludes `*.tif`, `*.tiff`,
   `*.pth`, `*.pt` from version control

Recommended workflow: track code on GitHub, store raster data locally or
on shared storage (e.g. Google Drive, university NAS).
