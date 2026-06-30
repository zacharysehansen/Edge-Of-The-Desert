# Edge of the Desert

A planning-stage environmental model for an eight-county southern Arizona region
(Pima, Pinal, Santa Cruz, Cochise, Graham, Greenlee, Yuma, La Paz). It takes human
and climate pressures as inputs — population, irrigation, public-supply withdrawal,
urbanization, reservoir operations, temperature, precipitation, drought — and predicts
six environmental responses: vegetation health (NDVI), groundwater storage anomaly
(GRACE), groundwater well depth, surface-water discharge, wildfire risk, and wildlife
abundance.

Each response gets its own independent regression model trained on the same shared
input set. A browser frontend loads the exported models and lets a user move the input
sliders and watch the predicted responses update live.

The project is organized into three phases:

| Phase | What it does | Code | Produces |
|-------|--------------|------|----------|
| **Phase 1** | Pull/clean raw sources into clean monthly & annual CSVs | `scripts/phase1/` | `data/Final/*.csv` |
| **Phase 2** | Merge, engineer features, train & export models | `scripts/phase2/` | `model/*.onnx` + JSON sidecars |
| **Phase 3** | Compute slider/output ranges + serve the frontend | `scripts/phase3/`, `frontend/` | `frontend/computed_stats.json` + web app |

See [PHASE1_SETUP.md](PHASE1_SETUP.md) for the dataset inventory and
[PHASE2_REPORT.md](PHASE2_REPORT.md) for full model-training results.

---

## Repository layout

```
config/            phase1.example.json — study-area bbox, counties, date range
data/
  raw/             source files (HDF/nc4/CSV/TIF) — inputs to Phase 1
  processed/       intermediate panels written by Phase 2 (monthly_panel, annual_panel)
  Final/           clean per-variable CSVs written by Phase 1 — inputs to Phase 2
scripts/
  phase1/          one script per source variable + run_phase1.py orchestrator
  phase2/          merge, features, baselines, model_*, export, run_phase2.py
  phase3/          generate_stats.py
model/             exported ONNX models + feature/CV JSON sidecars (Phase 2 output)
frontend/          static web app (index.html, ui.js, models.js, state.js, style.css)
requirements.txt   Python dependencies
```

---

## Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All commands below are run **from the repository root**. The geospatial Phase 1
scripts need GDAL/rasterio system libraries; on most systems the pip wheels suffice,
otherwise install GDAL via your package manager or conda
(`conda install -c conda-forge gdal rasterio`).

The study area, bounding box, and date range are defined once in
[scripts/phase1/region.py](scripts/phase1/region.py) and mirrored in
[config/phase1.example.json](config/phase1.example.json). Every Phase 1 script imports
from `region.py`, so there is a single source of truth for the boundary.

---

## Phase 1 — Data processing (`scripts/phase1/`)

Each script reads raw source data from `data/raw/<source>/`, clips it to the
eight-county region, aggregates to a consistent grain (monthly or annual), and writes
one clean CSV to `data/Final/`. Every script exposes a `main()` and can be run on its
own as a module, or all of them can be run together.

### Run everything

```bash
python scripts/phase1/run_phase1.py
```

This imports and runs `main()` for every script in `scripts/phase1/` (skipping
`region.py` and `run_phase1.py`) and prints a completion line per file.

### Run one variable at a time

```bash
python -m scripts.phase1.ndvi
python -m scripts.phase1.precipitation
python -m scripts.phase1.wildfire_monthly
# ...etc
```

### Scripts and their outputs

| Script | Raw input (under `data/raw/`) | Output in `data/Final/` |
|--------|-------------------------------|-------------------------|
| `population.py` | Census PEP API / CSVs | `azpop_monthly.csv` |
| `irrigation.py` | HUC12 withdrawal CSV | `irrigation_monthly.csv` |
| `public_supply.py` | NWAA HUC12 CSV | `public_supply_monthly.csv` |
| `lake_mead.py` | Reclamation HydroData CSVs | `lake_mead_monthly.csv` |
| `urbanization.py` | NLCD impervious TIFs (`NLCD/`) | `urbanization_monthly.csv` |
| `water_stress.py` | USDM DSCI API | `water_stress_monthly.csv` |
| `temperature.py` | MERRA-2 `.nc4` (`merra_temperature_2m/`) | `temperature_monthly.csv` |
| `precipitation.py` | MERRA-2 `.nc4` (`merra_precipitation/`) | `precipitation_monthly.csv` |
| `grace_groundwater.py` | GRACE `.nc4` (`grace_groundwater_anomaly/`) | `grace_monthly.csv` |
| `ndvi.py` | MODIS MOD13A3 `.hdf` (`modis_ndvi/`) | `ndvi_monthly.csv` |
| `groundwater_levels.py` | Well depth records | `groundwater_levels_monthly.csv` |
| `water_surface.py` | Stream discharge records | `water_surface_monthly.csv` |
| `wildfire_monthly.py` | InterAgency fire perimeter CSV (`wildfire/`) | `wildfire_monthly.csv` |
| `wildfire.py` | Same source, annual grain | `wildfire_annual.csv` |
| `wildlife.py` | BBS routes + counts (`bbs/`) | `wildlife_annual.csv` |
| `region.py` | — (shared boundary helpers, not runnable) | — |

> The scripts assume the raw files already exist in `data/raw/`. Several sources
> require network access or credentials (NASA Earthdata via `earthaccess` for
> MODIS/MERRA-2/GRACE; the Census and USDM APIs; ScienceBase for BBS). See
> [PHASE1_SETUP.md](PHASE1_SETUP.md) and [data/Final/DATA.md](data/Final/DATA.md)
> for endpoints and re-pull instructions.

---

## Phase 2 — Modeling (`scripts/phase2/`)

Phase 2 merges the `data/Final/` CSVs into panels, engineers lag/rolling/anomaly/
seasonal features, computes persistence baselines, trains one model per response, and
exports each to ONNX with JSON sidecars. The full pipeline runs in ~60 seconds.

### Run the full pipeline

```bash
python -m scripts.phase2.run_phase2
```

Stages, in order:

1. **merge.py** — builds `data/processed/monthly_panel.csv` (288×) and
   `data/processed/annual_panel.csv` from `data/Final/`.
2. **features.py** — engineers per-model feature matrices.
3. **baselines.py** — computes lag-1 persistence baselines for comparison →
   `model/baselines.json`.
4. **model_*.py** — trains, evaluates (TimeSeriesSplit / LOO), and exports each model.
5. **export.py** — writes the side-by-side `model/model_comparison.json` and prints
   the leaderboard.

### Useful flags

```bash
# Train only specific models
python -m scripts.phase2.run_phase2 --models ndvi grace

# Run merge/features/baselines/export but skip model training
python -m scripts.phase2.run_phase2 --skip-models
```

Valid `--models` values: `ndvi`, `grace`, `groundwater`, `surface_water`,
`wildfire_monthly`, `wildlife`.

### Exported artifacts (per model, written to `model/`)

- `{id}.onnx` — deployable model
- `{id}_feature_names.json` — ordered feature list (maps to ONNX input positions)
- `{id}_feature_stats.json` — per-feature mean/std for normalization
- `{id}_feature_importance.json` — importance / coefficients
- `{id}_cv_results.json` — CV scores, best params, fold details
- `historical_{id}.csv` — actual vs. predicted over the training window

Plus `model/model_comparison.json` and `model/baselines.json` summaries. The model IDs
and their results are documented in [PHASE2_REPORT.md](PHASE2_REPORT.md).

---

## Phase 3 — Stats + Frontend

### Step 1: Generate slider/output ranges

The frontend needs the realistic range (5th/50th/95th percentiles) of every input and
output to set slider bounds and output baselines. Regenerate it whenever the
`data/Final/` CSVs change:

```bash
python scripts/phase3/generate_stats.py
```

This reads `data/Final/` and writes `frontend/computed_stats.json` (`SLIDER_STATS` and
`OUTPUT_STATS`), which `state.js` imports directly.

### Step 2: Serve the frontend

The frontend is a static, dependency-free web app. It loads the ONNX models in the
browser via `onnxruntime-web` (from a CDN, declared in the importmap in
[frontend/index.html](frontend/index.html)) and runs inference client-side — no backend.

It fetches models from the absolute path `/model/<id>.onnx`, so the server root must be
the **repository root** (not the `frontend/` directory):

```bash
# from the repo root
python -m http.server 8000
```

Then open <http://localhost:8000/frontend/index.html>.

### How it fits together

- `index.html` builds three panels (human inputs, climate inputs, model outputs) and
  boots `ui.js`.
- `state.js` defines the sliders/outputs and loads ranges from `computed_stats.json`.
- `models.js` loads the six ONNX models from `/model/` along with their
  `_feature_names`/`_feature_stats` sidecars, then runs inference whenever a slider
  moves. Residual models (NDVI, GRACE, groundwater, surface water) predict the change
  from the previous month and reconstruct the level; surface water additionally uses a
  `log1p` target transform.
- `ui.js` renders the controls and updates the output cards.

Because the models are loaded straight from the Phase 2 `model/` directory, re-running
Phase 2 and refreshing the browser is all that's needed to deploy updated models.

---

## End-to-end, from scratch

```bash
# 0. setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. build clean CSVs from raw sources (requires data/raw/ to be populated)
python scripts/phase1/run_phase1.py

# 2. train and export all models
python -m scripts.phase2.run_phase2

# 3. compute frontend ranges, then serve
python scripts/phase3/generate_stats.py
python -m http.server 8000
#    → open http://localhost:8000/frontend/index.html
```
