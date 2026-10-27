# Planet Imagery Directory

This directory contains the folder hierarchy for input Planet satellite imagery.

> [!IMPORTANT]
> **Planet Education and Research (E&R) License Notice**
> 
> Under Section 3 of the Planet Labs Education & Research (E&R) License Agreement, users are granted a personal, non-commercial license to access and process satellite imagery for scientific research. **Public redistribution, sublicensing, or hosting of raw or processed Planet raster imagery (GeoTIFFs) on third-party public platforms (such as GitHub) is strictly prohibited.**
> 
> In compliance with Planet's Terms of Service and data redistribution policy, **no raster imagery is distributed in this repository**. The directory structure below is provided with empty folders so that researchers can reproduce the exact study area by populating their own licensed Planet imagery.

---

## Expected Directory Layout

```
images/
+-- 2021/
|   +-- train/
|   |   +-- global_monthly_2021_04_mosaic/    # April RGB basemap tiles (train split)
|   |   +-- global_monthly_2021_06_mosaic/    # June RGB basemap tiles
|   |   +-- global_monthly_2021_07_mosaic/    # July RGB basemap tiles (primary single-season)
|   |   +-- global_monthly_2021_10_mosaic/    # October RGB basemap tiles
|   +-- val/
|   |   +-- global_monthly_2021_04_mosaic/    # April RGB basemap tiles (val split)
|   |   +-- global_monthly_2021_06_mosaic/
|   |   +-- global_monthly_2021_07_mosaic/
|   |   +-- global_monthly_2021_10_mosaic/
|   +-- test/
|   |   +-- global_monthly_2021_04_mosaic/    # April RGB basemap tiles (test split)
|   |   +-- global_monthly_2021_06_mosaic/
|   |   +-- global_monthly_2021_07_mosaic/
|   |   +-- global_monthly_2021_10_mosaic/
|   +-- all/
|       +-- global_monthly_2021_07_mosaic/    # All AOI tiles for full inference
|
+-- 2025/
    +-- global_monthly_2025_06_mosaic/        # June 2025 RGB basemap tiles
    +-- global_monthly_2025_07_mosaic/        # July 2025 RGB basemap tiles
    +-- global_monthly_2025_08_mosaic/        # August 2025 RGB basemap tiles
    +-- global_monthly_2025_09_mosaic/        # September 2025 RGB basemap tiles
    +-- all/
        +-- global_monthly_2025_07_mosaic/    # All 2025 AOI tiles for wall-to-wall inference
```

*Note: Multi-seasonal stacked directories (such as `global_multiseasonal_2021_mosaic/`) are generated automatically by Stage 000 (`pipeline.stack_multiseasonal()`) and do not need to be manually created.*

---

## How to Acquire the Planet Imagery

1. **Apply for a Planet E&R Account**:
   University students, faculty, and researchers can apply for free access through Planet's [Education and Research Program](https://www.planet.com/markets/education-and-research/).

2. **Retrieve your API Key**:
   Obtain your personal API key from your Planet Account Settings page and export it in your environment:
   ```bash
   export PL_API_KEY="your-planet-api-key"
   ```

3. **Download Basemap Tiles**:
   Use the provided automated scripts in [`download_basemaps/`](../download_basemaps/):
   ```bash
   cd download_basemaps
   python download_by_tile_index.py config.yaml 100
   ```
   The configuration references the study area quad boundaries in [`download_basemaps/sampled_quads.geojson`](../download_basemaps/sampled_quads.geojson) and Planet mosaic IDs in [`download_basemaps/mosaic_ids.csv`](../download_basemaps/mosaic_ids.csv).

4. **Sort into Spatial Splits**:
   Distribute the downloaded GeoTIFFs into `train/`, `val/`, and `test/` based on [`notebooks/tile_splits.json`](../notebooks/tile_splits.json).
