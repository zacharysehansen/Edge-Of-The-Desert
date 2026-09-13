# Edge of the Desert

A planning-stage environmental model for an eight-county central and southern Arizona
region: the three counties the Central Arizona Project serves (Maricopa, Pinal, Pima) plus
Santa Cruz, Cochise, Greenlee, Yuma and Gila. Until 2026-09-12 every document here named
Graham and La Paz in place of Maricopa and Gila; the code never did, and the region was kept
deliberately — see [PROBLEMS.md P8](PROBLEMS.md). It takes human
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

**[DISCUSSION.md](DISCUSSION.md) is the place to start if you have the app open and are asking
why a slider did what it did** — it walks through every input-to-output path, what is measured
versus assumed, and the three results that look wrong at first and are not.

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
                   check_catalog_parity.py, check_frontend.mjs,
                   aquifer_calibration.py, ndvi_endpoints.py,
                   streamflow_calibration.py, structural_params.py, structural.py
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
| `cap_deliveries.py` | CAP delivery-report PDFs (`cap/`, fetched if absent) | `cap_deliveries_monthly.csv` |
| `gldas.py` | GLDAS-2.1 Noah monthly subsets (`gldas/`, fetched via Earthdata token) | `gldas_monthly.csv` |
| `humidity.py` | MERRA-2 M2TMNXSLV subsets (`merra_humidity/`, fetched via Earthdata token) | `humidity_monthly.csv` (VPD) |
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
exports each to ONNX with JSON sidecars. The full pipeline runs in a few minutes. Four models
are gradient-boosted or in-fold-selected ensembles; **GRACE and groundwater are ridge regressions
on four climate inputs** led by GLDAS land-surface storage change
([PHASE3_PLAN.md §26–§27, §29](PHASE3_PLAN.md)), which is what gave each of them skill. The
groundwater target is the **Cochise County** well index, not the eight-county blend (§28–§29).

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

```bash
python -m scripts.phase2.experiment_grace_cap
```

The last monthly pumping proxy: CAP deliveries (`data/Final/cap_deliveries_monthly.csv`) as a
deseasonalized feature block for GRACE, same nested design and the same pre-declared rule.
Verdict: **null** — +0.0665 target R², 5 of 5 folds, t = +1.52, 72% of the gain in the 28-row
first fold. Writes `model/experiment_grace_cap.json`. [PHASE3_PLAN.md](PHASE3_PLAN.md) §23.
`--window full` re-runs it on GRACE's own 2002–2023 window (204 rows) as the declared last CAP
run: the gain shrinks to +0.0151, t = +1.27, **null**; writes `model/experiment_grace_cap_full.json`.
§24. CAP is closed for GRACE.

```bash
python -m scripts.phase2.experiment_grace_gldas
```

GLDAS land-surface state (`data/Final/gldas_monthly.csv`, needs an Earthdata token in
`~/.config/earthdata/token`) as an 8-column block for GRACE, same design and rule. Verdict:
**null** as a feature block (+0.0682, 5 of 5 folds, t = +1.34) — but the physics check in the
same run shows a one-coefficient linear model on GLDAS storage change scoring +0.244 out of fold
against the shipped +0.021. Writes `model/experiment_grace_gldas.json`. §25.

```bash
python -m scripts.phase2.experiment_grace_linear
```

Step 2b: the model class, not a feature. A ridge residual model on four physical inputs (GLDAS
storage change, rain, last month's rain, temperature anomaly) against the shipped XGBoost, same
folds, same rule. Verdict: **REAL** — +0.2845 target R² and skill +0.2015 against +0.0210 and
−0.0277, 4 of 5 folds, t = +2.52. GRACE's first positive result. Not deployed: that is an
architecture change, listed in §26. Writes `model/experiment_grace_linear.json`.

```bash
python scripts/phase3/groundwater_diagnosis.py
python -m scripts.phase2.experiment_groundwater_linear
```

The same two steps for groundwater (§28). The linear model is **null** (t = 0.83): the estimator
is not that model's floor. The diagnosis is: the index averages 44 Cochise wells with 14 Pima
wells that are uncorrelated with them, and the Cochise half alone forecasts at +0.35 target R²
where the blend forecasts at +0.01. Writes `model/groundwater_diagnosis.json` and
`model/experiment_groundwater_linear.json`. Re-run on the Cochise target (§29) it writes
`model/experiment_groundwater_linear_cochise.json`; the output became the Cochise index and ships
as a four-input ridge (skill +0.23). `scripts/phase3/cochise_share.py` measures Cochise's share of
regional pumping for the Layer 2 municipal levers.

```bash
python -m scripts.phase2.experiment_feature_blocks
```

VPD, GLDAS runoff and root-zone soil moisture offered to wildfire, NDVI and streamflow under the
same rule, with the shipped arm keeping its full history (§33). **Six nulls**; NDVI with soil
moisture is the near miss (+0.060, 4 of 5, t = 1.83). Writes `model/experiment_feature_blocks.json`.

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
[PHASE3_PLAN.md](PHASE3_PLAN.md) D5. It also writes a `DERIVED` block: the OLS that turns the
rain and temperature sliders into the GLDAS storage change the GRACE model takes as input
(R² 0.61, stated in the JSON), the way the drought index is derived from PDSI.

`top_inputs.py` reads the `model/*_feature_importance.json` sidecars and writes the
"responds mainly to" line for each output card, plus how many of that model's inputs are
human levers at all. It is generated rather than hand-written so it cannot drift away
from the deployed models.

`structural_params.py` writes the **Layer 2** coefficients — the human-lever response the
learned models cannot carry, because four of the six contain no human feature at all.
Signs and magnitudes come from water balance, land-cover arithmetic and the published
Colorado River shortage tiers, not from fitting; each carries a `value`, `band`, `source`
and status tag. It reads `model/aquifer_calibration.json`, `model/transfer_calibration.json`,
`model/ndvi_endpoints.json` and `model/cap_calibration.json`, so run those calibrations first if
the panel has changed.

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
- `ui.js` renders the controls and updates the output cards. Each lever row carries its
  **range over the declared parameter bands** alongside the point estimate, and its tooltip
  names the constants that drive that width — twelve of Layer 2's constants ship with a
  band — ten of which still feed a lever path, six of those still `UNTESTED`
  ([PHASE3_PLAN.md](PHASE3_PLAN.md) §15a). The range is Layer 2 parameter uncertainty only;
  the learned climate term on the same card carries its own error, which is not in it.

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
python scripts/phase3/ndvi_endpoints.py         # both NDVI endpoints, from MOD13A3 + NLCD + HUC12
python scripts/phase3/streamflow_calibration.py # the four streamflow constants — returns nulls
python scripts/phase3/cap_calibration.py        # lost CAP delivery per declared shortage cut
python scripts/phase3/transfer_calibration.py   # output-to-output transfer edges

# every human lever must hold its declared sign in all 12 months
python scripts/phase3/slider_sensitivity.py --mode acceptance

# Layer 1 must stay climate-only: no human lever may move the learned output
python scripts/phase3/slider_sensitivity.py --mode no-double-count

# (§32: the §13 switch that would integrate the learned residual over the scenario was
# re-tested under declared criteria by scripts/phase3/experiment_integrate_learned.py and
# stays OFF — a positive multiplier cannot fix a sign, and it sends slow outputs off the scale.)
# the learned CLIMATE responses must carry their physical sign in every month
# (rain raises storage, greenness and flow and lowers depth and fire; heat the reverse).
# Added 2026-09-12 after §27 and §29 found four wrong-signed responses by hand. It exits
# nonzero on the app's default 12-month scenario, and prints the same table as a one-month
# pulse beside it. It currently FAILS 5 of 18 sustained pairs and 5 of 18 pulse pairs — see
# PHASE3_PLAN.md §30–§31 before treating that as a regression: rain → streamflow fails
# sustained and passes the pulse, which is a display limit of the residual architecture.
python scripts/phase3/slider_sensitivity.py --mode climate-signs

# the frontend must load and render as a BROWSER sees it, not as Node sees it
node scripts/phase3/check_frontend.mjs
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
