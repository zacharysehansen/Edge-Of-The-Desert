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
  phase2/          merge, features, baselines, model_*, export, run_phase2.py,
                   experiment_human_block.py, experiment_grace_nested.py
  phase3/          generate_stats.py, top_inputs.py, slider_sensitivity.py,
                   check_catalog_parity.py
model/             exported ONNX models + feature/CV JSON sidecars (Phase 2 output)
frontend/          static web app (index.html, ui.js, models.js, catalog.js,
                   state.js, style.css)
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

### Experiments (not part of the pipeline)

```bash
python -m scripts.phase2.experiment_human_block
```

Measures whether the panel can identify a human-lever effect at all: forces the human
feature block into every monthly model, deseasonalizes it, constrains the signs to what
water balance requires, and scores every variant on identical test rows. Exports nothing
and changes no shipped artifact — it writes `model/experiment_human_block.json` and a
report. The result and what follows from it are in [PHASE3_PLAN.md](PHASE3_PLAN.md) §10.

```bash
python -m scripts.phase2.experiment_grace_nested
```

Follows that one up for the single model where it mattered. The experiment above uses one
fixed XGBoost config for every variant, so its "shipped" column is not the shipped model's
real score — and GRACE was the one model the human block appeared to *improve*. This re-runs
both arms under `model_grace.py`'s own nested tuner on identical rows, against a decision
rule fixed in the docstring before the run. Verdict: **null** — the gain is +0.0566 target R²
at t = +1.21, and 83% of it comes from a single 28-training-row fold. Writes
`model/experiment_grace_nested.json`, exports nothing, retrains nothing.
[PHASE3_PLAN.md](PHASE3_PLAN.md) §11.5.

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

Regenerate these whenever the `data/Final/` CSVs or the `model/` exports change:

```bash
python scripts/phase3/generate_stats.py     # -> frontend/computed_stats.json
python scripts/phase3/top_inputs.py         # -> frontend/top_inputs.json
python scripts/phase3/structural_params.py  # -> frontend/structural_params.json
```

`generate_stats.py` reads `data/Final/` and writes `SLIDER_STATS` / `OUTPUT_STATS`. Each
slider gets both the raw 5th/50th/95th percentiles **and** a `policy` block: a
deseasonalized baseline, a 12-month seasonal shape, and a delta range in policy units
(people added, % of annual withdrawal, points of impervious cover, feet of elevation).
The frontend drives the models from the policy delta, because a raw percentile range is
not a policy axis — irrigation's raw maximum only ever meant "June". See
[PHASE3_PLAN.md](PHASE3_PLAN.md) D5.

`top_inputs.py` reads the `model/*_feature_importance.json` sidecars and writes the
"responds mainly to" line for each output card, plus how many of that model's inputs are
human levers at all. It is generated rather than hand-written so it cannot drift away
from the deployed models.

`structural_params.py` writes the **Layer 2** coefficients — the human-lever response the
learned models cannot carry, because four of the six contain no human feature at all.
Signs and magnitudes come from water balance, land-cover arithmetic and the published
Colorado River shortage tiers, not from fitting; each carries a `value`, `band`, `source`
and status tag. It reads `model/aquifer_calibration.json` and `model/transfer_calibration.json`, so run
`aquifer_calibration.py` and `transfer_calibration.py` first if the panel has changed.

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
- The app is three layers (see [PHASE3_PLAN.md](PHASE3_PLAN.md)):
  `output = ML_climate(climate, season, lag1) + Σ β·(lever − baseline)`, with the second
  term integrated over the scenario duration against an empirical mean-reversion rate.
  The learned models see **climate only**; the human sliders are answered entirely by the
  structural layer.
- `catalog.js` turns a scenario (policy deltas + month + scenario duration) into the raw
  model inputs, reconstructing each one as `delta + that month's climatology` and
  building the lag/rolling/anomaly families the same way `scripts/phase2/features.py`
  did at training time. It runs under plain Node (`node frontend/catalog.js --dump`) so
  it can be checked against the Python mirror.
- `models.js` loads the six ONNX models from `/model/` along with their
  `_feature_names`/`_feature_stats` sidecars, then runs inference whenever a slider
  moves. Residual models (NDVI, GRACE, groundwater, surface water) predict the change
  from the previous month and reconstruct the level.
- `structural.js` is Layer 2 and Layer 3: it converts each human slider's policy delta
  into a physical response (pumping → storage balance → well depth; Lake Mead elevation →
  DCP shortage tier → substituted pumping; impervious cover → land-cover NDVI) and
  integrates the rate levers in closed form. It reads every number from
  `structural_params.json` and runs under Node (`node frontend/structural.js --dump`).
- `ui.js` renders the controls and updates the output cards.

Because the models are loaded straight from the Phase 2 `model/` directory, re-running
Phase 2 and refreshing the browser is all that's needed to deploy updated models. Re-run
`top_inputs.py` too, so the output cards describe the models that are actually deployed.

### Step 3: Checks

```bash
# Do the sliders move the outputs, and in which direction?
python scripts/phase3/slider_sensitivity.py --mode sweep

# frontend/catalog.js and the Python mirror must agree feature-for-feature
python scripts/phase3/check_catalog_parity.py

# Layer 2: estimate the coefficients the structural levers run on
python scripts/phase3/aquifer_calibration.py    # aquifer storage + the spreading-cone horizon sweep
python scripts/phase3/transfer_calibration.py   # output-to-output transfer edges

# every human lever must hold its declared sign in all 12 months
python scripts/phase3/slider_sensitivity.py --mode acceptance

# Layer 1 must stay climate-only: no human lever may move the learned output
python scripts/phase3/slider_sensitivity.py --mode no-double-count
```

The last two exit nonzero on failure, so they work as build gates. `--mode
no-double-count` asserts that swinging any human lever leaves all six learned outputs
**bit-identical** — if one moves, its response is being counted twice, once by the fitted
coefficient and once by the structural β
([PHASE3_PLAN.md §11.3](PHASE3_PLAN.md)).

`slider_sensitivity.py` runs the exported ONNX files headlessly and reports how far each
slider moves each output across its full policy range. It is the only check in the repo
that scores the question the interface asks — *move a lever, does the output respond?* —
rather than forecast accuracy. What it currently measures, and why, is
[PHASE3_PLAN.md](PHASE3_PLAN.md).

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

# 3. compute frontend ranges + output-card provenance, then serve
python scripts/phase3/generate_stats.py
python scripts/phase3/top_inputs.py
python scripts/phase3/structural_params.py
python -m http.server 8000
#    → open http://localhost:8000/frontend/index.html
```
