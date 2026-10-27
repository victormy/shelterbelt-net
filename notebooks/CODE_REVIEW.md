# Code Review: Shelterbelt Segmentation Pipeline

Reviewed: 2026-04-28.
Scope: notebooks `000`--`005`, `README.md`.

---

## 1. Correctness

### 1.1 Inference sliding window drops trailing pixels

**File:** `004_run_inference.ipynb`, cell `b5503b5e`, function `predict_tile`.

```python
for y in range(0, H - WINDOW_SIZE + 1, STRIDE):
    for x in range(0, W - WINDOW_SIZE + 1, STRIDE):
```

When `(H - WINDOW_SIZE) % STRIDE != 0`, the final strip of pixels is never
visited. Example: `H = 4100`, `WINDOW_SIZE = 256`, `STRIDE = 192` -- the last
`y` is 3840 which covers rows 3840..4095, leaving rows 4096..4099 unclassified.
Contrast with `002_create_training_dataset.ipynb` (cell `1c041580`), where
`tile_positions()` correctly appends `total - tile_size` as a final position.
Unpredicted pixels stay at zero in the `accum` and `counts` arrays; dividing
`0 / 1` yields 0, so they are silently labelled background -- a data-loss bug
for every tile whose dimensions are not an exact multiple of the stride.

**Severity:** :red_circle: bug.

### 1.2 `scheduler.step(val_iou)` misuses CosineAnnealingLR

**File:** `003_train_val_test_model.ipynb`, cell `a4774a13`.

```python
scheduler.step(val_iou)
```

[`CosineAnnealingLR.step()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
accepts only `epoch=None`. The `epoch` parameter has been **deprecated since
PyTorch 1.4**: *"If provided, sets `last_epoch` to `epoch` and uses
`_get_closed_form_lr()`. This is not universally supported. Use `step()` without
arguments instead."*

Passing `val_iou` (a float around 0.5--0.7) is interpreted as epoch number
0.65, which teleports the scheduler to that position on the cosine curve. The
learning rate schedule is silently destroyed every epoch.

(The `metrics` keyword belongs to `ReduceLROnPlateau`, not to
`CosineAnnealingLR`.)

**Severity:** :red_circle: bug.

### 1.3 Validation set is iterated twice per epoch

**File:** `003_train_val_test_model.ipynb`, cell `a4774a13`.

The training loop first iterates `val_loader` to accumulate `val_loss_epoch`,
then calls `evaluate(val_loader)` which iterates the entire loader a second time
to compute TP/FP/FN/TN. This doubles validation wall-time and is unnecessary --
loss can be accumulated in the same forward pass that collects confusion-matrix
statistics.

**Severity:** :red_circle: bug (performance; doubles epoch time for no benefit).

---

## 2. ML Pitfalls

### 2.1 Per-chip min-max normalization vs ImageNet mean/std

**Files:** `003` (via `SemanticSegmentationDataset`) and `004` (cell `b5503b5e`,
`normalize_chip`).

Training and inference both use per-chip min-max to [0, 1], so they are
**mutually consistent**. However, ImageNet-pretrained encoders (ResNet34,
MiT-B0) were trained on images normalized with `mean = [0.485, 0.456, 0.406]`,
`std = [0.229, 0.224, 0.225]`
([smp docs](https://smp.readthedocs.io/en/latest/quickstart.html)).
Per-chip min-max creates two problems:

1. **Uniform-value chips** (e.g. ocean, cloud-masked) produce `max - min ~ 0`,
   causing a near-division-by-zero (the `+ 1e-8` guard in `normalize_chip`
   helps numerically but still outputs ~0 for every pixel).
2. **Distribution mismatch** with the encoder's pre-training statistics,
   reducing the benefit of transfer learning.

The smp documentation notes that proper preprocessing *"is relevant only for
1-2-3-channel images and not necessary in case you train the whole model, not
only decoder"*. Since this pipeline fine-tunes encoder + decoder jointly for only
5 epochs, the mismatch is partially compensated but still sub-optimal.

**Severity:** :yellow_circle: concern.

### 2.2 No gradient clipping

**File:** `003_train_val_test_model.ipynb`, cell `a4774a13`.

The training loop calls `loss.backward()` and `optimizer.step()` without any
`torch.nn.utils.clip_grad_norm_()` guard. Dice loss can produce large gradients
when the predicted mask is far from the target, leading to training instability
-- especially in early epochs and with small batch sizes.

**Severity:** :yellow_circle: concern.

### 2.3 Both models instantiated but only one used

**Files:** `003` (cell `7d09f7f5`), `004` (cell `7482ca6f`).

```python
model_conv  = smp.create_model("unet", ...)      # ~24 M params
model_trans = smp.create_model("segformer", ...)  # ~3.7 M params
model = model_conv
```

Both architectures are loaded onto the GPU; the unused one wastes ~15 MB of VRAM
and can push a tight-memory training run out of memory.

**Severity:** :yellow_circle: concern.

---

## 3. Code Quality

### 3.1 Hardcoded absolute paths leak directory structure

| Notebook | Path |
|---|---|
| 000 | `G:\Мій диск\GEOAI` (Google Drive, Ukrainian locale) |
| 001 | `c:/GEOAI` |
| 002--005 | `c:/GEOAI/...` |

Two different roots, both OS-specific absolute paths. Anyone cloning the repo
must search-and-replace before running.

**Severity:** :yellow_circle: concern.

### 3.2 Hardcoded tile lists

**File:** `001_reclassify_by_stndtype.ipynb`, cell `af3943ad`.

```python
train_tiles = ["1223-1352", "1224-1333", ...]
val_tiles   = ["1228-1329", ...]
test_tiles  = ["1222-1323", ...]
```

These tile names also appear in the README. A single source-of-truth JSON file
would prevent silent split leakage if someone updates one list but forgets the
other.

**Severity:** :yellow_circle: concern.

### 3.3 Unclosed file handles in mosaic merge

**File:** `005_mosaic_predicted_tiles.ipynb`, cell `e4bac557`.

```python
src_files = [rasterio.open(p) for p in pred_paths]
mosaic, mosaic_transform = merge(src_files)
for f in src_files:
    f.close()
```

If `merge()` raises, the files are never closed. Should use
`contextlib.ExitStack`.

**Severity:** :yellow_circle: concern.

### 3.4 No band-count validation before reading

**File:** `004_run_inference.ipynb`, cell `b5503b5e`.

```python
image = src.read(list(range(1, NUM_CHANNELS + 1))).astype(np.float32)
```

No check that `src.count >= NUM_CHANNELS`. A 3-band tile read as 9-channel
would raise a cryptic rasterio error.

**Severity:** :yellow_circle: concern.

### 3.5 No checkpoint existence check

**Files:** `003` (cell `47590820`), `004` (cell `7482ca6f`).

`torch.load(BEST_CKPT)` without `os.path.exists()` guard.

**Severity:** :yellow_circle: concern.

### 3.6 Inline WKT for Ukraine Albers CRS

**File:** `005` (cell `90a6151e`).

A 20-line WKT string is embedded in a notebook code cell. It should live in
`config.py` where it can be validated once on import and reused.

**Severity:** :green_circle: nit.

### 3.7 Magic numbers

`256`, `128`, `192`, `64`, `0.5`, `1e-8`, `1e-6`, `15`, `5`, `8`, `4`, `512`
appear as bare literals across multiple notebooks with no named constants.

**Severity:** :green_circle: nit.

---

## 4. Security

### 4.1 OS-specific paths expose user directory structure

`G:\Мій диск\GEOAI` leaks the Google Drive mount point and locale. Not a
vulnerability per se, but reveals the author's directory tree to anyone reading
the committed notebook.

**Severity:** :green_circle: nit.

### 4.2 No committed secrets found

No API keys, tokens, or credentials were found in the codebase.

---

## 5. Reproducibility

### 5.1 Seeds are set -- good

`003` calls `seed_everything(42)` and sets `torch.backends.cudnn.deterministic`.

### 5.2 No pinned dependencies

No `requirements.txt` or `environment.yml`. The custom `geoai` package is
imported without version pinning or installation instructions.

**Severity:** :yellow_circle: concern.

### 5.3 No `.gitignore`

Checkpoints (`.pth`), GeoTIFFs, `__pycache__`, and `.ipynb_checkpoints` can be
committed accidentally.

**Severity:** :yellow_circle: concern.

---

## 6. Tests

**No tests exist.** The repository has zero test files.

**Severity:** :yellow_circle: concern.

---

## 7. Top-5 Highest-Impact Issues

| # | Finding | Severity | Impact |
|---|---|---|---|
| 1 | Inference window loop drops trailing pixels (004) | :red_circle: | Every tile loses up to `STRIDE - 1` rows/cols of predictions |
| 2 | `scheduler.step(val_iou)` breaks LR schedule (003) | :red_circle: | Cosine annealing never follows intended curve |
| 3 | Val loader iterated twice per epoch (003) | :red_circle: | 2x validation time, no benefit |
| 4 | Per-chip min-max instead of ImageNet mean/std (003, 004) | :yellow_circle: | Sub-optimal transfer learning; uniform-chip instability |
| 5 | Hardcoded absolute paths, no env-var config (all) | :yellow_circle: | Repo is non-portable without manual edits |

---

## 8. Sources

- [CosineAnnealingLR -- PyTorch 2.11 docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
- [smp Quick Start -- preprocessing](https://smp.readthedocs.io/en/latest/quickstart.html)
- [smp Insights -- encoder preprocessing params](https://segmentation-models-pytorch.readthedocs.io/en/latest/insights.html)
- [Prithvi-EO-2.0 -- NASA/IBM foundation model (arxiv 2412.02732)](https://arxiv.org/abs/2412.02732)
- [Clay Foundation Model -- open EO foundation model](https://clay-foundation.github.io/model/)
- [UNetFormer boundary-enhanced multi-scale segmentation (Nature, 2025)](https://www.nature.com/articles/s41598-025-99663-9)
- [SAM2-UNet -- SAM encoder for segmentation (Springer, 2025)](https://link.springer.com/article/10.1007/s44267-025-00106-w)
- [Satellite image deep learning techniques -- sliding window inference](https://github.com/satellite-image-deep-learning/techniques)
- [nb-clean -- notebook hygiene for version control](https://github.com/srstevenson/nb-clean)
- [Tiling and Stitching Segmentation Output for Remote Sensing (2021)](https://github.com/satellite-image-deep-learning/techniques)

### 2025--2026 trends in RS segmentation

- **Foundation models**: Prithvi-EO-2.0 (600 M, ViT + MAE, multi-temporal
  Landsat/Sentinel, 30 m) and Clay (open, dual-stage transformer) offer
  pre-trained backbones that can replace ImageNet encoders for EO tasks.
- **Boundary-aware architectures**: BEMS-UNetFormer adds a Boundary Awareness
  Module to UNetFormer for sharper edges in urban scene segmentation.
- **SAM-assisted segmentation**: SAM2-UNet uses the SAM 2 encoder as a strong
  backbone for both natural and medical image segmentation; SAM3 extends to
  text-prompted video object segmentation.
- **Scale-adaptive inference**: Gaussian-weighted sliding-window blending
  reduces tile-boundary artefacts; calibrated inter-patch blending (MICCAI
  2025) down-weights false positives via neighbour agreement.
- **Notebook hygiene**: `nb-clean` and `nbstripout` are now standard pre-commit
  hooks for stripping outputs and metadata before committing notebooks.
