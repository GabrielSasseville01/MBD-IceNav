# Ridge density costmap (AUTO-IceNav)

Ridge density maps are used in the planning costmap to avoid ice ridges during path planning. Based on [Lensu & Similä (2022)](https://tc.copernicus.org/articles/16/4363/2022/), "Ice ridge density signatures in high-resolution SAR images".

## Getting the data

You need two files:

1. **Segmented PNG** (`MEDEA_fram_20100629.png`) — pre-segmented ice floes, one unique color per floe
2. **Satellite TIF** (`MEDEA_fram_20100629_satellite.tif`) — high-resolution original imagery (optional but better for ridge detection)

### Segmented PNG

From the MEDEA dataset [Hwang et al. 2022, UK Polar Data Centre](https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=7dc6e19d-79fa-41f1-99a7-da408592382f):

```bash
# Direct download from MEDEA repository (~167 KB)
curl -L -o data/MEDEA_fram_20100629.png \
  "https://ramadda.data.bas.ac.uk/repository/entry/get/MEDEA_fram_20100629.png?entryid=synth%3A7dc6e19d-79fa-41f1-99a7-da408592382f%3AL0ZyYWN0dXJpbmcvZnNkL01FREVBX2ZyYW1fMjAxMDA2MjkucG5n"
```

Or via AUTO-IceNav gdown (if included in their data bundle):

```bash
uv add gdown
uv run gdown 1Bk2QQaUft7j8kSz088lGYwHEAMRWR_cf --output data/
uv run gdown 1SUFHewABXArSkEn3cs8bqU-5OuViSlQY --output data/  # second file if needed
```

### Satellite TIF

```bash
uv add gdown
uv run gdown 1jV6xLCtFoe4qCWu_pOy-OnXI583lJ3Q3 --output data/
# Rename to MEDEA_fram_20100629_satellite.tif if the downloaded filename differs
```

## Generating the ridge costmap

```bash
uv run python scripts/generate_ridge_costmap.py \
  --input data/MEDEA_fram_20100629.png \
  --satellite data/MEDEA_fram_20100629_satellite.tif \
  --output data/ridge_costmap_MEDEA_fram_20100629_scale05_fixed.pkl \
  --crop_size 600 \
  --costmap_scale 0.5 \
  --method combined \
  --window_size 25 \
  --visualize
```

- **--input**: Segmented PNG (extent/cropping)
- **--satellite**: Original satellite TIF (optional; better ridge detection than segmented image)
- **--output**: Output pickle path
- **--costmap_scale**: Meters per pixel (0.5 = coarser, matching planning resolution)
- **--visualize**: Save visualization PNG next to the pickle
- **--polygon**: Optional path to ice floe polygon pickle for overlay (AUTO-IceNav style, cyan outlines)

## Testing without real data

```bash
uv run python scripts/generate_ridge_costmap.py --synthetic
```

Creates a synthetic test image and ridge costmap in `outputs/ridge_density_test/`.

## Using the ridge costmap in planning

The ridge costmap is integrated into the planning costmap. Paths avoid high-ridge regions.

**CLI (run_medea_ship_fracture.py):**

```bash
uv run python run_medea_ship_fracture.py --full-map --plan \
  --ridge-costmap data/ridge_costmap_MEDEA_fram_20100629_scale05_fixed.pkl \
  --ridge-weight 2.0
```

**Config (medea_ice_ship/config.py RunConfig):**

```python
planning_ridge_costmap_file: "data/ridge_costmap_MEDEA_fram_20100629_scale05_fixed.pkl"
planning_ridge_cost_weight: 2.0
```

Ridge density is added only to ice cells (AUTO-IceNav style).

The domain must be aligned with the ridge costmap extent (0–600 m for MEDEA crop).
