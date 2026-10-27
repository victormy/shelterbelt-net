# Delineated Reference Perimeters

This directory contains the manual ground-truth reference delineations used for training, validating, and testing shelterbelt segmentation models across eastern Ukraine.

All vector datasets are provided as GeoJSON files projected in **Web Mercator (EPSG:3857)** to align with Planet basemap tiling coordinates.

---

## Directory Contents

### 1. `all_3857_2021/` (23 GeoJSON files)
Pre-war (2021) baseline reference annotations delineated on Planet 4.7 m imagery across the study quads in Kharkiv, Donetsk, and Zaporizhzhia oblasts.
These files are used by Stage 001 (`pipeline.reclassify_masks()`) to generate binary ground-truth masks for training, validation, and testing according to the spatial partition in `notebooks/tile_splits.json`.

**Tile List (23 tiles):**
- **Train (16 quads)**: `1223-1328`, `1223-1352`, `1224-1333`, `1227-1324`, `1227-1348`, `1229-1334`, `1232-1325`, `1232-1349`, `1233-1330`, `1233-1354`, `1235-1340`, `1236-1345`, `1238-1331`, `1238-1355`, `1239-1336`, `1244-1337`.
- **Val (4 quads)**: `1228-1329`, `1231-1344`, `1234-1335`, `1237-1350`.
- **Test (3 quads)**: `1222-1323`, `1228-1353`, `1240-1341`.

### 2. `all_3857_2025/` (3 GeoJSON files)
Post-invasion (2025) reference annotations re-delineated on July 2025 Planet imagery for the three held-out test quads:
- `1222-1323.geojson`
- `1228-1353.geojson`
- `1240-1341.geojson`

Used in Stage 007 (`007_postprocessing.ipynb`) for the Olofsson et al. (2014) stratified error-adjusted area estimation and wartime net change analysis.

---

## Stand Type Classification Scheme (`STNDTYPE`)

Each vector polygon carries a `STNDTYPE` attribute code:

| Code | Stand Type | Description | Included in Pipeline |
|:---:|---|---|:---:|
| **1** | Field-protective shelterbelts | Narrow linear tree plantations (*polezakhysni smugy*) protecting crops | Yes (default) |
| **2** | Infrastructure strips | Railway and roadside protective tree lines | Yes (when `STND_CODES=[1,2,3]`) |
| **3** | Other linear plantations | Miscellaneous windbreaks, riparian buffers, field borders | Yes (when `STND_CODES=[1,2,3]`) |
| **4** | Massive / areal forest | Continuous natural forest blocks | **Excluded** |

Configure `config.py` to select the target class:
- `STND_CODES = [1]` → Models shelterbelts only.
- `STND_CODES = [1, 2, 3]` → Models all linear tree plantations.

---

## Study Area Quad Footprints

The outer study area boundary and quad footprint polygons are located in [`../download_basemaps/`](../download_basemaps/):
- `download_basemaps/all_quads.geojson`: All quad boundaries intersecting the study area.
- `download_basemaps/sampled_quads.geojson`: The sampled quad grid used in this study.
- `download_basemaps/aoi.geojson`: The AOI envelope polygon.
