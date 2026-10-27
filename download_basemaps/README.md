# Planet Basemaps Automated Download Tools (`download_basemaps`)

Tools and scripts to automatically query Planet's Basemaps API, index spatial quad footprints, and download high-resolution satellite imagery tiles (GeoTIFF) covering the eastern Ukraine study area.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites & Authentication](#prerequisites--authentication)
- [Installation](#installation)
- [Directory Contents](#directory-contents)
- [Step-by-Step Workflow](#step-by-step-workflow)
  - [Step 1: Obtain or Update Mosaic IDs](#step-1-obtain-or-update-mosaic-ids)
  - [Step 2: Generate the Spatial Quad Index](#step-2-generate-the-spatial-quad-index)
  - [Step 3: Configure `config.yaml`](#step-3-configure-configyaml)
  - [Step 4: Download Basemap Quads](#step-4-download-basemap-quads)
  - [Step 5: Mapping Downloaded Tiles to the Segmentation Pipeline](#step-5-mapping-downloaded-tiles-to-the-segmentation-pipeline)
- [How the Scripts Work](#how-the-scripts-work)
  - [`create_quad_index.py`](#create_quad_indexpy)
  - [`download_by_tile_index.py`](#download_by_tile_indexpy)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Overview

Planet global monthly basemaps are tiled into regular cartographic grid units called **quads** (~12.5 × 12.5 km at equator, EPSG:3857, 4.77 m ground sample distance). Rather than manually searching Planet Explorer, these scripts automate:
1. Spatial intersection between your Area of Interest (AOI) boundary and Planet's global basemap tiling scheme.
2. Generating a local GeoJSON index of intersecting quads.
3. Batch downloading GeoTIFF quads using HTTP Basic Authentication.
4. Resuming interrupted downloads without re-downloading existing tiles.

---

## Prerequisites & Authentication

1. **Planet Account with Basemap Access**:
   You need access to Planet Basemaps (e.g. through a university license or the free [Planet Education and Research (E&R) Program](https://www.planet.com/markets/education-and-research/)).
2. **API Key**:
   Find your API key under your **Planet Account Settings** → **API Keys**.
3. **Set the Environment Variable**:
   The scripts look for the `PL_API_KEY` environment variable.

   - **Windows (Command Prompt / CMD)**:
     ```cmd
     set PL_API_KEY=your_actual_planet_api_key_here
     ```
   - **Windows (PowerShell)**:
     ```powershell
     $env:PL_API_KEY="your_actual_planet_api_key_here"
     ```
   - **Linux / macOS**:
     ```bash
     export PL_API_KEY="your_actual_planet_api_key_here"
     ```

   *(Optional test)*: Check that your key works:
   ```bash
   curl -u "%PL_API_KEY%:" "https://api.planet.com/basemaps/v1/mosaics?_page_size=1"
   ```

---

## Installation

Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install click geopandas pandas shapely requests pyyaml fiona
```

---

## Directory Contents

| File | Type | Description |
|---|---|---|
| **`create_quad_index.py`** | Python Script | Queries Planet API to find all quads intersecting an input AOI polygon and saves them as a GeoJSON layer. |
| **`download_by_tile_index.py`** | Python Script | Reads `config.yaml`, queries quads for each configured mosaic, and downloads missing GeoTIFFs. |
| **`config.yaml`** | Configuration | Defines target download directory, quad index file, and mosaic names/IDs to download. |
| **`mosaic_ids.csv`** | Reference Data | Lookup table of Planet monthly basemap names, date ranges, and mosaic UUIDs (2020--2025). |
| **`aoi.geojson`** | Vector Layer | Outer boundary polygon covering the study area in eastern Ukraine (WGS 84, EPSG:4326). |
| **`khar-done-zapo.geojson`** | Vector Layer | Regional boundary envelope covering Kharkiv, Donetsk, and Zaporizhzhia oblasts. |
| **`all_quads.geojson`** | Vector Layer | Pre-computed quad footprint grid covering the full eastern Ukraine study area (45 quads). |
| **`sampled_quads.geojson`** | Vector Layer | Subset of quads selected for the train/val/test experiments. |
| **`how_to_download_mosaics`** | Text Guide | Quick command cheat-sheet for running the download commands. |
| **`how_to_update_mosaic_list`** | Text Guide | API URL template for querying Planet for newly released monthly mosaic IDs. |

---

## Step-by-Step Workflow

### Step 1: Obtain or Update Mosaic IDs

Each Planet monthly basemap has a unique UUID. Check [`mosaic_ids.csv`](mosaic_ids.csv) for pre-indexed IDs:
- **July 2021**: `e0c03fb4-0820-4ff8-be26-06cbaef9a7a7` (`global_monthly_2021_07_mosaic`)
- **July 2025**: `86650aba-b553-4446-9665-59361946f980` (`global_monthly_2025_07_mosaic`)

To find newly published mosaic IDs (e.g. for other months or years), open the following URL in your browser (replacing `<your-planet-api-key>`):
```
https://api.planet.com/basemaps/v1/mosaics/?name__contains=global_monthly_2025&api_key=<your-planet-api-key>
```
Or query via terminal:
```bash
python -c "import os, requests; auth=(os.getenv('PL_API_KEY'),''); res=requests.get('https://api.planet.com/basemaps/v1/mosaics?name__contains=global_monthly_2021', auth=auth).json(); [print(m['name'], m['id']) for m in res.get('mosaics', [])]"
```

---

### Step 2: Generate the Spatial Quad Index

If you want to create a fresh quad grid for a custom AOI boundary:

```bash
python create_quad_index.py aoi.geojson all_quads.geojson
```

**What this does**:
1. Reads `aoi.geojson` (must be in WGS 84, EPSG:4326).
2. Queries Planet's global basemap grid within the bounding box of the AOI.
3. Performs an inner spatial intersection join (`gpd.sjoin(..., how="inner", predicate="intersects")`).
4. Outputs a clean GeoJSON (`all_quads.geojson`) containing only quads that intersect your AOI.

*(Note: `all_quads.geojson` and `sampled_quads.geojson` are already pre-generated and included in this folder, so you can skip this step if using the default study area.)*

---

### Step 3: Configure `config.yaml`

Edit [`config.yaml`](config.yaml) to specify:
1. `tile_directory`: Destination folder where downloaded imagery should be stored.
2. `tile_feature_filename`: Path to the quad index GeoJSON (`./sampled_quads.geojson` or `./all_quads.geojson`).
3. `mosaics`: List of mosaic collections to download (name and UUID).

Example:
```yaml
tile_directory: ../images/downloaded
tile_feature_filename: ./sampled_quads.geojson

mosaics:
  - name: global_monthly_2021_07_mosaic
    id: e0c03fb4-0820-4ff8-be26-06cbaef9a7a7
    output_status_filename: ./status/status_global_monthly_2021_07.geojson

  - name: global_monthly_2025_07_mosaic
    id: 86650aba-b553-4446-9665-59361946f980
    output_status_filename: ./status/status_global_monthly_2025_07.geojson
```

---

### Step 4: Download Basemap Quads

Run the download script with your config file and the maximum number of quads to download:

```bash
python download_by_tile_index.py config.yaml 100
```

Where:
- `config.yaml`: Path to your YAML configuration.
- `100`: Maximum number of quads to download in this run (use a large number like `500` for all quads).

**Features**:
- **Automatic Resume**: The script checks if `{tile_directory}/{mosaic_name}/{quad_id}.tif` already exists on disk. If so, it skips it and only downloads missing quads.
- **Progress Output**: Prints quad index, mosaic name, and download status to stdout.

---

### Step 5: Mapping Downloaded Tiles to the Segmentation Pipeline

Once downloaded, the GeoTIFF quads have file names matching their quad index (e.g. `1223-1328.tif`, `1235-1340.tif`).

To use them in the shelterbelt segmentation pipeline:
1. Refer to [`notebooks/tile_splits.json`](../notebooks/tile_splits.json) for the geographic partition:
   - **`train`** (16 tiles): `1223-1328`, `1223-1352`, `1224-1333`, `1227-1324`, `1227-1348`, `1229-1334`, `1232-1325`, `1232-1349`, `1233-1330`, `1233-1354`, `1235-1340`, `1236-1345`, `1238-1331`, `1238-1355`, `1239-1336`, `1244-1337`.
   - **`val`** (4 tiles): `1228-1329`, `1231-1344`, `1234-1335`, `1237-1350`.
   - **`test`** (3 tiles): `1222-1323`, `1228-1353`, `1240-1341`.

2. Move or copy the tiles into the corresponding split folders under `images/`:
   ```
   images/
   +-- 2021/
   |   +-- train/global_monthly_2021_07_mosaic/   <-- put training tiles here
   |   +-- val/global_monthly_2021_07_mosaic/     <-- put validation tiles here
   |   +-- test/global_monthly_2021_07_mosaic/    <-- put test tiles here
   |   +-- all/global_monthly_2021_07_mosaic/     <-- put all tiles for full inference
   +-- 2025/
       +-- global_monthly_2025_07_mosaic/
       +-- all/global_monthly_2025_07_mosaic/
   ```

---

## How the Scripts Work

### `create_quad_index.py`
```
AOI Shapefile/GeoJSON (EPSG:4326)
         │
         ▼
Compute total_bounds [minx, miny, maxx, maxy]
         │
         ▼
GET https://api.planet.com/basemaps/v1/mosaics?name__contains=global_monthly_2024
(Retrieves prototype mosaic UUID)
         │
         ▼
GET /basemaps/v1/mosaics/{mosaic_id}/quads?bbox={bbox}&_page_size=500
(Paginates through all pages)
         │
         ▼
Convert bounding boxes to Shapely box geometries
         │
         ▼
Spatial Join: gpd.sjoin(quads, aoi, predicate="intersects")
         │
         ▼
Save output GeoJSON (e.g. all_quads.geojson)
```

### `download_by_tile_index.py`
```
Read config.yaml & set HTTPBasicAuth(PL_API_KEY, "")
         │
         ▼
Load tile_feature_filename (e.g. sampled_quads.geojson)
         │
         ▼
For each mosaic in config.yaml:
   │
   ├─► Scan tile_directory: find which quads are already downloaded
   │
   ├─► GET /basemaps/v1/mosaics/{mosaic_id}/quads?bbox={aoi_bbox}
   │
   ├─► Filter: keep only quads present in AOI and NOT yet on disk
   │
   └─► For quad in needed_quads[:num_quads]:
          Download GeoTIFF via quad._links.download
          Save to {tile_directory}/{mosaic_name}/{quad_id}.tif
```

---

## Troubleshooting & FAQ

### 1. `HTTP 401 Unauthorized`
- Cause: `PL_API_KEY` is not set or invalid.
- Fix: Check `echo %PL_API_KEY%` (Windows) or `echo $PL_API_KEY` (Linux/macOS). Make sure there are no trailing spaces or quotes.

### 2. `HTTP 404 Not Found` for Mosaic ID
- Cause: The mosaic UUID in `config.yaml` has expired or changed.
- Fix: Check [`mosaic_ids.csv`](mosaic_ids.csv) or query `https://api.planet.com/basemaps/v1/mosaics?name__contains=<mosaic_name>&api_key=<your_key>`.

### 3. `KeyError: 'items'` when listing quads
- Cause: The API response returned an error message instead of the expected JSON list (e.g. quota exceeded or bad bounding box).
- Fix: Check your account quota and verify that the AOI polygon coordinates are in WGS 84 (longitude between 30°--40°E, latitude between 45°--52°N for Ukraine).

### 4. Planet E&R License Note
Planet Labs PBC provides satellite imagery under the Education and Research (E&R) agreement for academic research. **Do not upload downloaded GeoTIFFs to public repositories.** The `.gitignore` in the repository root automatically excludes `*.tif` files to prevent accidental commits.
