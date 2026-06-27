# Phase 2 Setup — Southern Arizona Environmental Model

## Goal

Build six independent XGBoost regression models, one per environmental output, using the Phase 1 data collected in `data/Final/`. Each model receives the same shared set of human pressure and environmental condition inputs and predicts one outcome. 
---

## Data Inventory

All inputs are in `data/Final/`. Column names are exact.

### Inputs (Human Pressures + Environmental Conditions)

| File | Key Columns | Coverage | Notes |
|------|-------------|----------|-------|
| `azpop_monthly.csv` | `year_month, population` | 2000-01 – 2023-12 | Full |
| `irrigation_monthly.csv` | `year_month, irrigation_total_withdrawal_mgd` | 2000-01 – 2020-12 | Gap: 2021-2023 |
| `public_supply_monthly.csv` | `year_month, public_supply_groundwater_mgd` | 2000-01 – 2020-12 | Gap: 2021-2023 |
| `lake_mead_monthly.csv` | `year_month, mead_pool_elevation, mead_storage, mead_total_release, mead_release_volume` | 2000-01 – 2023-12 | Full |
| `water_stress_monthly.csv` | `year_month, usdm_dsci, water_stress_score` | 2000-01 – 2023-12 | Full |
| `temperature_monthly.csv` | `year_month, temperature_2m_c` | 2000-01 – 2023-12 | Full |
| `precipitation_monthly.csv` | `year_month, precipitation_mm_day` | 2000-01 – 2023-12 | Full |
| `urbanization_monthly.csv` | `year_month, impervious_pct` | 2000-01 – 2023-12 | Full |
| `wildfire_monthly.csv` | `year_month, fire_count, total_acres, log_acres, wildfire_risk_index` | 2000-01 – 2023-12 | Full |

### Outputs (Environmental Response Targets)

| File | Key Columns | Coverage | Model | Notes |
|------|-------------|----------|-------|-------|
| `ndvi_monthly.csv` | `year_month, ndvi` | 2000-01 – 2023-12 | 1 | Full |
| `grace_monthly.csv` | `year_month, grace_groundwater_anomaly, grace_available` | 2000-01 – 2023-12 | 2 | Pre-2002-04 filled with 0.0, inter-mission gap interpolated |
| `groundwater_levels_monthly.csv` | `year_month, depth_to_water_ft_mean` | 2000-01 – 2020-12 | 3 | Full |
| `water_surface_monthly.csv` | `year_month, discharge_cfs_mean, gage_height_ft_mean` | 2000-01 – 2020-12 | 4 | Full |
| `wildfire_annual.csv` | `year, fire_count, log_acres_total, wildfire_risk_index` | 1972 – 2023 (~50 rows) | 5a | Annual grain |
| `wildfire_monthly.csv` | `year_month, fire_count, total_acres, log_acres, wildfire_risk_index` | 2000-01 – 2023-12 | 5b | Monthly grain (288 rows) |
| `wildlife_annual.csv` | `year, route_count, total_abundance, species_richness, abundance_index` | 2000 – 2024, no 2020 | 6 | Annual grain, 24 rows |

All model data files are now available. Phase 2 scripts will be written for all models (1–6).

---

## Model Specifications

### Model 1 — NDVI (Vegetation Health)

| Field | Detail |
|-------|--------|
| **Target** | `ndvi` from `ndvi_monthly.csv` |
| **Grain** | Monthly |
| **Modeling window** | `2002-10` through `2020-12` (219 rows) |
| **Window rationale** | Starts 6 months into GRACE real data so all GRACE lag/rolling features are observed; ends where irrigation and public supply data end |
| **CV strategy** | `TimeSeriesSplit(n_splits=5)` |
| **Baseline** | `lag1_persistence` (predict next month = this month's NDVI) |
| **Target formulation** | Residual over lag1: train on `ndvi - ndvi_lag1`, add `ndvi_lag1` back at predict time |
| **GRACE as input?** | Yes — GRACE anomaly is an environmental condition that correlates with soil moisture and vegetation |
| **Export target** | ONNX, same scaled-input contract as the prior model |

### Model 2 — Groundwater Storage Anomaly (GRACE)

| Field | Detail |
|-------|--------|
| **Target** | `grace_groundwater_anomaly` from `grace_monthly.csv` |
| **Grain** | Monthly |
| **Modeling window** | `2002-10` through `2020-12` (219 rows) |
| **Window rationale** | 2002-04 is first real GRACE measurement; starting at 2002-10 allows 6-month lag/rolling features to be real observed values |
| **CV strategy** | `TimeSeriesSplit(n_splits=5)` |
| **Baseline** | `lag1_persistence` |
| **Target formulation** | Residual over lag1: train on `grace_groundwater_anomaly - grace_anomaly_lag1` |
| **GRACE as input?** | No — GRACE is the target; it must not appear as an input feature |
| **`grace_available` flag** | Retained as a feature — the inter-mission gap (2017-07 to 2018-05) is flagged and the model can learn the uncertainty |
| **Export target** | ONNX |

### Model 3 — Groundwater Well Levels

| Field | Detail |
|-------|--------|
| **Target** | `depth_to_water_ft_mean` from `groundwater_levels_monthly.csv` |
| **Grain** | Monthly |
| **Modeling window** | `2002-10` through `2020-12` (219 rows) |
| **Window rationale** | Same as NDVI/GRACE — starts after GRACE warmup, ends where irrigation data ends |
| **CV strategy** | `TimeSeriesSplit(n_splits=5)` |
| **Baseline** | `lag1_persistence` |
| **Target formulation** | Residual over lag1: train on `depth_to_water - depth_to_water_lag1` |
| **GRACE as input?** | Yes — satellite storage anomaly complements in-situ well readings |
| **Export target** | ONNX |

### Model 4 — Surface Water Conditions

| Field | Detail |
|-------|--------|
| **Target** | `discharge_cfs_mean` from `water_surface_monthly.csv` (secondary: `gage_height_ft_mean`) |
| **Grain** | Monthly |
| **Modeling window** | `2002-10` through `2020-12` (219 rows) |
| **Window rationale** | Same as NDVI/GRACE — starts after GRACE warmup, ends where irrigation data ends |
| **CV strategy** | `TimeSeriesSplit(n_splits=5)` |
| **Baseline** | `lag1_persistence` |
| **Target formulation** | Residual over lag1: train on `discharge - discharge_lag1` |
| **GRACE as input?** | Yes — satellite storage anomaly is informative for surface flow |
| **Export target** | ONNX |

### Model 5a — Wildfire Risk (Annual)

| Field | Detail |
|-------|--------|
| **Target** | `wildfire_risk_index` from `wildfire_annual.csv` |
| **Grain** | Annual |
| **Modeling window** | `2000` through `2021` (22 rows; climate feature set extends beyond irrigation end) |
| **Window rationale** | Climate-only feature set drops irrigation/supply dependency so window can extend to 2021 |
| **CV strategy** | `TimeSeriesSplit(n_splits=4)` — fewer splits due to small sample |
| **Baseline** | `lag1_persistence` (last year's wildfire risk index) |
| **Target formulation** | Direct (no residual, sample too small to reliably estimate a lag-based residual) |
| **Model selection** | Multi-model × multi-feature-set competition: Ridge, ElasticNet, XGBoost × full/climate/minimal (9 candidates) |
| **Input aggregation** | Monthly inputs aggregated to annual: mean for rates (temperature, water stress, NDVI), sum for volumes (precipitation total, irrigation total, public supply total), June value for population and lake mead level (end-of-spring state) |
| **Extra features** | Annual lag features (lag1, lag2), 3-year rolling mean, log-transformed annual precipitation total, peak summer temperature (JJA mean), fire-ecology features (wet-then-dry interaction, consecutive dry years, prior 2-year precip sum, JJA temperature anomaly, max drought severity) |
| **Export target** | ONNX |

### Model 5b — Wildfire Risk (Monthly)

| Field | Detail |
|-------|--------|
| **Target** | `wildfire_risk_index` from `wildfire_monthly.csv` |
| **Grain** | Monthly |
| **Modeling window** | `2002-10` through `2020-12` (219 rows) |
| **Window rationale** | Same as NDVI/GRACE — starts after GRACE warmup, ends where irrigation data ends |
| **CV strategy** | `TimeSeriesSplit(n_splits=5)` |
| **Baseline** | `lag1_persistence` (wildfire is episodic — baseline is very negative at -0.97) |
| **Target formulation** | Direct (residual formulation fails for episodic signals that are not autoregressive) |
| **Model selection** | Multi-model × multi-formulation competition: Ridge, ElasticNet, XGBoost × direct/residual (6 candidates) |
| **Wildfire columns excluded from input** | `wildfire_risk_index`, `fire_count`, `log_acres`, `total_acres` (they are the target) |
| **Export target** | ONNX |

### Model 6 — Wildlife Abundance

| Field | Detail |
|-------|--------|
| **Target** | `abundance_index` from `wildlife_annual.csv` |
| **Grain** | Annual |
| **Modeling window** | `2000` through `2020` (21 rows; 2020 is excluded from source data, 2021-2023 irrigation gap) |
| **CV strategy** | `TimeSeriesSplit(n_splits=3)` — minimum viable splits for 21 rows |
| **Baseline** | `lag1_persistence` |
| **Target formulation** | Direct |
| **Input aggregation** | Same annual aggregation as Model 5; additionally use the June-specific monthly values since BBS surveys occur in June |
| **Extra features** | NDVI annual mean (vegetation health directly relevant to bird abundance), annual lag and rolling features |
| **Special note** | 24 rows is below the 50-row reliability threshold. CV scores will be noisy. Report standard deviation alongside mean R² and interpret with caution. |
| **Export target** | ONNX |

---

## Feature Engineering Plan

### Monthly Features (Models 1 and 2)

Applied to each continuous input column:

```
{var}_lag1      — 1-month lag
{var}_lag3      — 3-month lag
{var}_lag6      — 6-month lag
{var}_roll3     — 3-month trailing mean
{var}_roll6     — 6-month trailing mean
{var}_roll12    — 12-month trailing mean
```

Seasonal encoding (deterministic, no lag needed):
```
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

Temperature anomaly (standardized departure from rolling 12-month mean):
```
temperature_2m_c_anomaly = temperature_2m_c - temperature_2m_c_roll12
```

Precipitation anomaly (same pattern):
```
precipitation_mm_day_anomaly = precipitation_mm_day - precipitation_mm_day_roll12
```

Key base inputs for monthly models (before lag/rolling expansion):
- `population`
- `irrigation_total_withdrawal_mgd`
- `public_supply_groundwater_mgd`
- `mead_pool_elevation` (primary Lake Mead signal; storage is collinear with elevation)
- `mead_total_release`
- `usdm_dsci`
- `temperature_2m_c`
- `precipitation_mm_day`
- `impervious_pct` (urbanization pressure — NLCD annual fractional impervious, interpolated monthly)
- For Model 1 (NDVI): also `grace_groundwater_anomaly`, `grace_available`
- For Model 2 (GRACE): `grace_groundwater_anomaly` excluded (it is the target)
- For Model 5b (Wildfire Monthly): `wildfire_risk_index`, `fire_count`, `log_acres` excluded (they are the target)

`mead_storage` and `mead_release_volume` are redundant with elevation/total-release and will be excluded, following the same Powell cleanup rationale from the prior model.

### Annual Features (Models 5 and 6)

Monthly → annual aggregation:
```
{var}_annual_mean     — mean over Jan-Dec
{var}_annual_sum      — sum over Jan-Dec (for volume variables)
{var}_june            — June value (spring/pre-monsoon state)
{var}_jja_mean        — June-July-August mean (peak stress season)
```

Derived annual features:
```
log_precip_annual     — log1p of annual precipitation total
year_linear           — year index (captures long-term trend)
```

Lag and rolling:
```
{var}_lag1_year       — prior year's value
{var}_lag2_year       — two years prior
{var}_roll3_year      — 3-year trailing mean
```

Model 6 (Wildlife) adds:
```
ndvi_annual_mean      — vegetation health signal most directly linked to bird habitat
```

---

## Data Coverage Windows Per Model

The modeling window is constrained by the intersection of target availability and complete input availability.

| Model | Target | Input constraint | Window | Rows |
|-------|--------|-----------------|--------|------|
| 1 (NDVI) | 2000-01 to 2023-12 | Irrigation/supply end 2020-12; lag features need 9 months warmup | **2002-10 to 2020-12** | ~219 |
| 2 (GRACE) | Real data 2002-04 to 2023-12 | Same input constraint as Model 1 | **2002-10 to 2020-12** | ~219 |
| 3 (Groundwater) | 2000-01 to 2020-12 | Same input constraint as Model 1 | **2002-10 to 2020-12** | ~219 |
| 4 (Surface Water) | 2000-01 to 2020-12 | Same input constraint as Model 1 | **2002-10 to 2020-12** | ~219 |
| 5 (Wildfire) | 1972 to 2023 (annual) | Project range starts 2000; irrigation ends 2020 | **2000 to 2020** | ~21 |
| 6 (Wildlife) | 2000 to 2024 minus 2020 | Irrigation ends 2020 | **2000 to 2019** | ~20 |

For Models 5 and 6, using only years 2000-2020 (or 2000-2019 for wildlife) is conservative but avoids imputing the 2021-2023 input gap. If enough signal is present in the smaller window, the models are still valid.

---

## Scripts To Write

All scripts live under `scripts/phase2/`.

### `scripts/phase2/merge.py`

**Purpose:** Load all `data/Final/` CSVs, join them on `year_month` (for monthly) or `year` (for annual), produce two clean DataFrames: one monthly panel and one annual panel.

**Responsibilities:**
- Load each CSV
- Parse `year_month` as a proper period index
- Outer join on `year_month`, then report any unexpected gaps
- For the annual panel: aggregate monthly inputs to annual using the aggregation rules defined above
- Join annual inputs to the `wildfire_annual.csv` and `wildlife_annual.csv` targets
- Emit `data/processed/monthly_panel.csv` and `data/processed/annual_panel.csv` (intermediate, not Final)
- Print a coverage summary table showing which columns have NaN in which date ranges

### `scripts/phase2/features.py`

**Purpose:** Take the merged panel DataFrames and produce fully-engineered feature matrices, one per model.

**Responsibilities:**
- Apply lag, rolling, and seasonal features to monthly panel
- Apply annual lag/rolling/aggregation features to annual panel
- Define and return the explicit feature column lists for each model (no `select_dtypes` catch-alls)
- Apply window filtering (e.g., clip to `2002-10` to `2020-12` for Models 1 and 2)
- Drop rows with NaN in any feature or target column after feature engineering
- Return a dict of `{model_id: (X_df, y_series)}` so each model script gets a clean, ready-to-use pair

### `scripts/phase2/baselines.py`

**Purpose:** Compute lag1 persistence and rolling mean baselines for every model using the same TimeSeriesSplit used for the learned models. Saves results to `model/baselines.json`.

**Format of `baselines.json`:**
```json
{
  "ndvi": {
    "lag1_persistence": { "mean_r2": ..., "std_r2": ..., "mean_mae": ..., "std_mae": ... },
    "roll3_mean":       { "mean_r2": ..., "std_r2": ..., "mean_mae": ..., "std_mae": ... }
  },
  ...
}
```

### `scripts/phase2/model_ndvi.py`

**Purpose:** Build, tune, evaluate, and export Model 1 (NDVI).

**Steps:**
1. Load `(X, y)` from `features.py`
2. Run baselines via `baselines.py`
3. Fit standard XGBoost (residual-over-lag1 formulation) with `TimeSeriesSplit(n_splits=5)`
4. Run focused `RandomizedSearchCV` with the same hyperparameter space as the prior model
5. Re-evaluate best estimator on the same CV splits; save fold-level R² and MAE
6. Compare against baselines — fail loudly if best model does not beat `lag1_persistence`
7. Export to ONNX: `model/ndvi.onnx`
8. Save feature names: `model/ndvi_feature_names.json`
9. Save feature stats (mean/std for scaling): `model/ndvi_feature_stats.json`
10. Save historical predictions: `model/historical_ndvi.csv` with columns `year_month, ndvi_actual, ndvi_predicted`
11. Save CV results: `model/ndvi_cv_results.json`
12. Save feature importance plot: `model/ndvi_feature_importance.png`

### `scripts/phase2/model_grace.py`

**Purpose:** Build, tune, evaluate, and export Model 2 (GRACE groundwater anomaly). Same structure as `model_ndvi.py` with these differences:
- Target: `grace_groundwater_anomaly`
- GRACE columns excluded from input features
- `grace_available` flag retained as an input feature
- Residual-over-lag1 uses `grace_groundwater_anomaly_lag1`
- Export artifacts: `model/grace.onnx`, `model/grace_feature_names.json`, etc.

### `scripts/phase2/model_groundwater_wells.py`

**Purpose:** Build, tune, evaluate, and export Model 3 (Groundwater Well Levels). Same structure as `model_ndvi.py` with these differences:
- Target: `depth_to_water_ft_mean`
- GRACE anomaly included as input feature (complementary signal)
- Residual-over-lag1 uses `depth_to_water_ft_mean_lag1`
- Export artifacts: `model/groundwater.onnx`, `model/groundwater_feature_names.json`, etc.

### `scripts/phase2/model_surface_water.py`

**Purpose:** Build, tune, evaluate, and export Model 4 (Surface Water Conditions). Same structure as `model_ndvi.py` with these differences:
- Target: `discharge_cfs_mean`
- GRACE anomaly included as input feature
- Residual-over-lag1 uses `discharge_cfs_mean_lag1`
- Export artifacts: `model/surface_water.onnx`, `model/surface_water_feature_names.json`, etc.

### `scripts/phase2/model_wildfire.py`

**Purpose:** Build, tune, evaluate, and export Model 5a (Wildfire Risk, Annual). Multi-model × multi-feature-set competition.

**Key differences from monthly models:**
- `TimeSeriesSplit(n_splits=4)` — fewer folds, small annual sample
- No residual formulation (too few rows; direct fit on `wildfire_risk_index`)
- Multi-model competition: Ridge, ElasticNet, XGBoost × 3 feature sets (full, climate, minimal) = 9 candidates
- `SimpleImputer(strategy="median")` + `StandardScaler` in sklearn Pipeline for linear models
- Fire-ecology features: `wet_then_dry`, `consecutive_dry_years`, `precip_prior_2yr_sum`, `jja_temp_anomaly`, `dsci_max_prior_2yr`
- Baseline: lag1 persistence of `wildfire_risk_index`
- Export: `model/wildfire.onnx` (via skl2onnx for sklearn pipelines, onnxmltools for XGBoost)

### `scripts/phase2/model_wildfire_monthly.py`

**Purpose:** Build, tune, evaluate, and export Model 5b (Wildfire Risk, Monthly). Multi-model × multi-formulation competition.

**Key differences:**
- `TimeSeriesSplit(n_splits=5)` — same as NDVI/GRACE
- Competes direct vs. residual formulations across Ridge, ElasticNet, XGBoost (6 candidates)
- Wildfire columns (`wildfire_risk_index`, `fire_count`, `log_acres`) excluded from input features
- Constrained XGBoost: `max_depth` 2–3, `min_child_weight` 3–10, strong regularization
- Export: `model/wildfire_monthly.onnx`

### `scripts/phase2/model_wildlife.py`

**Purpose:** Build, tune, evaluate, and export Model 6 (Wildlife Abundance). Annual grain.

**Key differences from wildfire model:**
- ~20 rows — LOO-CV (`LeaveOneOut`) used instead of `TimeSeriesSplit` to maximize fold count
- Target: `abundance_index`
- NDVI annual mean included as input feature
- Report both R² and Spearman correlation (R² is unstable for very small N; Spearman gives directional reliability)
- `max_depth = 2` hard ceiling to prevent single-row memorization
- Export: `model/wildlife.onnx`
- CV report includes explicit warning: "Wildlife model trained on <25 rows. LOO cross-validation R² will be volatile. Treat Spearman rank correlation as primary reliability indicator."

### `scripts/phase2/export.py`

**Purpose:** Collect all model artifacts into a single `model/` directory summary. Reads each `*_cv_results.json` and `baselines.json`, produces `model/model_comparison.json` and a printed leaderboard.

**`model_comparison.json` format:**
```json
{
  "ndvi":    { "model_r2": ..., "model_mae": ..., "baseline_lag1_r2": ..., "improvement_r2": ... },
  "grace":   { ... },
  "wildfire": { ... },
  "wildlife": { ... }
}
```

### `scripts/phase2/run_all.py`

**Purpose:** Top-level runner. Calls `merge.py`, `features.py`, then each model script in sequence. Accepts `--models ndvi grace wildfire wildlife` to run a subset.

---

## Evaluation Standards

Every model must clear these gates before being logged as a final result:

1. **Beats lag1 persistence** on mean CV R². If it doesn't, report the gap and do not export.
2. **CV R² std < 0.15** for monthly models (time-series stability). Annual models are exempt due to small N.
3. **No same-period leakage.** For monthly models: `usdm_dsci` and `water_stress_score` are both available at prediction time (they're inputs, not derived from the target) — they are allowed. For Model 2 (GRACE), the GRACE signal itself must be excluded. For Model 1 (NDVI), NDVI must be excluded except as a lag/rolling feature.
4. **Export compatibility.** Only models that can be serialized to ONNX with standard `sklearn`/`xgboost` pipelines are considered final exports.

---

## Output Artifacts

After Phase 2 completes, `model/` will contain:

```
model/
  ndvi.onnx
  ndvi_feature_names.json
  ndvi_feature_stats.json
  ndvi_feature_importance.json
  ndvi_cv_results.json
  historical_ndvi.csv

  grace.onnx
  grace_feature_names.json
  grace_feature_stats.json
  grace_feature_importance.json
  grace_cv_results.json
  historical_grace.csv

  groundwater.onnx
  groundwater_feature_names.json
  groundwater_feature_stats.json
  groundwater_feature_importance.json
  groundwater_cv_results.json
  historical_groundwater.csv

  surface_water.onnx
  surface_water_feature_names.json
  surface_water_feature_stats.json
  surface_water_feature_importance.json
  surface_water_cv_results.json
  historical_surface_water.csv

  wildfire.onnx
  wildfire_feature_names.json
  wildfire_feature_stats.json
  wildfire_feature_importance.json
  wildfire_cv_results.json
  historical_wildfire.csv

  wildfire_monthly.onnx
  wildfire_monthly_feature_names.json
  wildfire_monthly_feature_stats.json
  wildfire_monthly_feature_importance.json
  wildfire_monthly_cv_results.json
  historical_wildfire_monthly.csv

  wildlife.onnx
  wildlife_feature_names.json
  wildlife_feature_stats.json
  wildlife_feature_importance.json
  wildlife_cv_results.json
  historical_wildlife.csv

  baselines.json
  model_comparison.json
```

---

## Known Risks and Limitations Going In

1. **Annual model sample size.** Wildfire (~21 rows) and wildlife (~20 rows) are below the threshold for reliable ML. The models are built and exported but the CV scores should not be overinterpreted. They are more useful as scenario interpolation tools than as precision forecasters.

2. **Irrigation and public supply end in 2020.** Both monthly models are trained through 2020-12 only. When these models are used for scenario projection, irrigation and public supply inputs must be held at their 2020 values or extrapolated externally before being passed to the model.

3. **Urbanization — RESOLVED.** The NLCD impervious surface raster pipeline was completed in Phase 2. `urbanization_monthly.csv` (2000-01 to 2023-12) is included as input feature `impervious_pct` across all models. It appears in top-5 feature importance for GRACE and wildfire monthly models.

4. **GRACE pre-mission values (2000-01 to 2002-03) are filled with 0.0.** The modeling window starts at 2002-10 to avoid these fills appearing as lag/rolling inputs. For Model 2, the target window also starts at 2002-10 for the same reason.

5. **Selection optimism.** Multiple model families will be tried. The final reported CV R² for each model will be slightly optimistic relative to a one-shot never-retuned evaluation. This is expected and should be noted in any final report.
