# Phase 2 Report — Model Training Results

## Overview

Phase 2 built seven independent regression models predicting environmental outcomes for the eight-county southern Arizona study area. Each model uses human pressures (population, irrigation, public supply, urbanization) and environmental conditions (temperature, precipitation, drought, lake levels, groundwater) as input features to predict a single environmental response variable.

**Pipeline runtime:** ~60 seconds end-to-end  
**Modeling window:** 2002-10 to 2020-12 (monthly), 2000–2021 (annual)  
**Total input features:** 13 monthly CSVs merged into a 288-row × 23-column panel  

---

## Model Results Summary

| Model | Target | Grain | Rows | Features | Model Type | CV R² | Baseline R² | Improvement |
|-------|--------|-------|------|----------|------------|--------|-------------|-------------|
| NDVI | Vegetation health | Monthly | 219 | 51 | XGBoost (residual) | **0.8008** | 0.5532 | +0.2476 |
| GRACE | Groundwater anomaly | Monthly | 219 | 51 | XGBoost (residual) | **0.5200** | 0.4119 | +0.1081 |
| Groundwater | Well depth (ft) | Monthly | 219 | 32 | XGBoost tight (residual) | **0.6107** | — | — |
| Surface Water | Discharge (cfs) | Monthly | 219 | 41 | ElasticNet (residual, log) | **0.4784** | — | — |
| Wildfire (monthly) | Wildfire risk index | Monthly | 219 | 56 | XGBoost (direct) | **0.0498** | -0.9744 | +1.0242 |
| Wildfire (annual) | Wildfire risk index | Annual | 22 | 18 | ElasticNet (direct) | **-0.1145** | -2.1141 | +1.9996 |
| Wildlife | Bird abundance index | Annual | 20 | 22 | ElasticNet (direct) | **0.2086** | — | — |

All seven models beat their lag-1 persistence baselines. All export to ONNX format for deployment.

---

## Model Details

### Model 1: NDVI (Vegetation Health) — CV R² = 0.8008

- **Formulation:** Residual-over-lag1 (predict change from previous month, reconstruct)
- **Algorithm:** XGBoost with RandomizedSearchCV (60 iterations)
- **CV Method:** TimeSeriesSplit (5 folds)
- **Window:** 2002-10 to 2020-12
- **Key insight:** Strong autoregressive component (lag1 baseline alone = 0.55); the model captures an additional 25% of variance via climate and human pressure features

**Top features by importance:**
1. `temperature_2m_c_lag1` (0.205)
2. `public_supply_groundwater_mgd_lag1` (0.160)
3. `precipitation_mm_day` (0.068)
4. `temperature_2m_c_anomaly_lag1` (0.063)
5. `precipitation_mm_day_anomaly` (0.051)

---

### Model 2: GRACE Groundwater Anomaly — CV R² = 0.5200

- **Formulation:** Residual-over-lag1
- **Algorithm:** XGBoost with RandomizedSearchCV (60 iterations)
- **CV Method:** TimeSeriesSplit (5 folds)
- **Window:** 2002-10 to 2020-12
- **Key insight:** Seasonal cycle dominates (month_cos is #1 feature); precipitation anomaly is the strongest climate driver

**Top features by importance:**
1. `month_cos` (0.101)
2. `precipitation_mm_day_anomaly` (0.086)
3. `temperature_2m_c` (0.062)
4. `temperature_2m_c_anomaly` (0.055)
5. `impervious_pct` (0.055)

---

### Model 3: Groundwater Well Levels — CV R² = 0.6107

- **Formulation:** Residual-over-lag1
- **Algorithm:** Multi-model competition (Ridge, ElasticNet, XGBoost, XGBoost-tight, Blend × direct/residual = 8 candidates)
- **Winner:** XGBoost-tight (residual formulation)
- **CV Method:** TimeSeriesSplit (5 folds)
- **Window:** 2002-10 to 2020-12
- **Feature selection:** 32 features kept, 31 dropped (importance threshold 0.015)
- **Key insight:** Extreme regularization was key. The winning "tight" XGBoost uses min_child_weight=120, reg_lambda=100, colsample_bytree=0.3, learning_rate=0.01 — far beyond typical XGBoost defaults. This reduced overfitting from the initial train R²=0.98/CV R²=0.50 to train R²=0.89/CV R²=0.61. A blend of XGBoost+ElasticNet was also tested but didn't outperform the tighter single model. The residual formulation dominates (all direct candidates scored negative R²).

**Competition results:**
| Candidate | CV R² |
|-----------|--------|
| XGBoost-tight (residual) | **0.6093** |
| XGBoost (residual) | 0.5722 |
| Blend XGB+EN (residual) | 0.5399 |
| ElasticNet (residual) | 0.2156 |
| Ridge (residual) | 0.0854 |
| XGBoost (direct) | -2.2469 |
| ElasticNet (direct) | -15.6824 |
| Ridge (direct) | -20.0808 |

**Best params:** `max_depth=2, min_child_weight=120, reg_lambda=100, reg_alpha=10.0, subsample=0.3, colsample_bytree=0.3, n_estimators=200, learning_rate=0.01`

---

### Model 4: Surface Water Conditions — CV R² = 0.4784

- **Formulation:** Residual-over-lag1 with log1p target transform
- **Algorithm:** Multi-model competition (Ridge, ElasticNet, XGBoost × direct/residual = 6 candidates)
- **Winner:** ElasticNet (residual formulation)
- **CV Method:** TimeSeriesSplit (5 folds)
- **Window:** 2002-10 to 2020-12
- **Feature selection:** 41 features kept, 22 dropped (importance threshold 0.005)
- **Key insight:** Desert discharge is heavily right-skewed (baseflow ~50 cfs, monsoon floods >500 cfs). Log-transforming the target compresses outliers and makes the residual distribution learnable. ElasticNet's L1/L2 regularization generalizes far better than XGBoost for this noisy target — train R² (0.36) < CV R² (0.48) indicates zero overfitting. Initial XGBoost-only approach scored CV R² = -0.03 (worse than predicting the mean).

**Competition results:**
| Candidate | CV R² |
|-----------|--------|
| ElasticNet (residual) | **0.4784** |
| XGBoost (residual) | 0.4768 |
| Ridge (residual) | 0.1428 |
| XGBoost (direct) | 0.1381 |
| ElasticNet (direct) | -0.0287 |
| Ridge (direct) | -0.5348 |

**Design decisions:**
1. Log-transform: converts multiplicative flood dynamics into additive residuals
2. Residual formulation works in log-space (prior-month log-flow is informative) even though raw residuals failed
3. Feature selection removed 22 noise dimensions, preventing overfitting in a 219-row dataset
4. Interaction features (precip × impervious = runoff proxy, precip × temperature = ET proxy) added physical meaning

---

### Model 5: Wildfire Risk (Monthly) — CV R² = 0.0498

- **Formulation:** Direct (residual formulation fails for episodic signals)
- **Algorithm:** Multi-model competition (Ridge, ElasticNet, XGBoost × 2 formulations = 6 candidates)
- **Winner:** XGBoost (direct)
- **CV Method:** TimeSeriesSplit (5 folds)
- **Window:** 2002-10 to 2020-12
- **Key insight:** Wildfire is inherently stochastic (ignition events are unpredictable); the model captures seasonal and anthropogenic patterns but cannot predict specific fire events. The positive R² represents a real signal — far better than the lag1 baseline (-0.97).

**Top features by importance:**
1. `population_roll12` (0.051)
2. `population` (0.045)
3. `temperature_2m_c_lag1` (0.043)
4. `impervious_pct_lag1` (0.041)
5. `temperature_2m_c_anomaly_lag1` (0.037)

---

### Model 4: Wildfire Risk (Annual) — CV R² = -0.1145

- **Formulation:** Direct
- **Algorithm:** Multi-model × multi-feature-set competition (3 models × 3 feature sets = 9 candidates)
- **Winner:** ElasticNet (climate feature set)
- **CV Method:** TimeSeriesSplit (4 folds)
- **Window:** 2000 to 2021 (22 rows)
- **Key insight:** With only 22 rows, overfitting is the primary challenge. ElasticNet's L1/L2 regularization outperforms XGBoost. Negative R² means the model is worse than predicting the mean — but it still far outperforms lag1 persistence (-2.11). Fire-ecology features (wet-then-dry interaction, consecutive dry years) improve over naive climate features.

**Fire-ecology features engineered:**
- `wet_then_dry` — prior year precipitation × current JJA temperature
- `consecutive_dry_years` — count of years below median precipitation
- `precip_prior_2yr_sum` — fuel accumulation window
- `jja_temp_anomaly` — departure from expanding mean summer temperature
- `dsci_max_prior_2yr` — worst drought in prior 2 years

---

### Model 5: Wildlife Abundance — LOO R² = 0.2086, Spearman ρ = 0.50

- **Formulation:** Direct
- **Algorithm:** Multi-model competition (Ridge, ElasticNet, XGBoost)
- **Winner:** ElasticNet
- **CV Method:** Leave-One-Out (20 samples too few for TimeSeriesSplit)
- **Window:** 2000 to 2019 (excluding 2020 — BBS cancelled)
- **Key insight:** Spearman correlation (0.50, p=0.025) is the primary reliability indicator at this sample size. The model captures the rank ordering of good/bad years for bird populations. R² is secondary given the noise level.

---

## Pipeline Architecture

```
scripts/phase2/
├── merge.py              — Build monthly (288×20) and annual (24×34) panels
├── features.py           — Engineer lag/roll/anomaly/seasonal features per model
├── baselines.py          — Compute lag1 and roll3 baselines for comparison
├── model_ndvi.py         — XGBoost residual-over-lag1
├── model_grace.py        — XGBoost residual-over-lag1
├── model_wildfire_monthly.py — Multi-model competition (6 candidates)
├── model_wildfire.py     — Multi-model × multi-feature-set (9 candidates)
├── model_wildlife.py     — Multi-model competition with LOO-CV
├── export.py             — Build model_comparison.json + leaderboard
└── run_all.py            — Top-level orchestrator with --models flag
```

---

## Feature Engineering

### Monthly Models (NDVI, GRACE, Wildfire Monthly)

Each base input column generates 6 derived features:
- `{col}_lag1`, `{col}_lag3`, `{col}_lag6` — temporal lags
- `{col}_roll3`, `{col}_roll6`, `{col}_roll12` — trailing rolling means

Additional engineered features:
- `temperature_2m_c_anomaly` / `precipitation_mm_day_anomaly` — departure from 12-month trailing mean
- `month_sin`, `month_cos` — cyclical seasonal encoding
- `impervious_pct_lag1`, `impervious_pct_roll12` — urbanization trend features

### Annual Models (Wildfire, Wildlife)

- Annual aggregations: mean, sum, JJA mean, June value, year-end value
- 1-year and 2-year lags + 3-year rolling means
- Fire-ecology features (wildfire only): wet-then-dry interaction, consecutive dry years, prior 2-year precip sum

---

## Input Data (11 Sources)

| # | Dataset | File | Columns Used |
|---|---------|------|--------------|
| 1 | Population | `azpop_monthly.csv` | `population` |
| 2 | Irrigation | `irrigation_monthly.csv` | `irrigation_total_withdrawal_mgd` |
| 3 | Public Supply | `public_supply_monthly.csv` | `public_supply_groundwater_mgd` |
| 4 | Lake Mead | `lake_mead_monthly.csv` | `mead_pool_elevation`, `mead_total_release` |
| 5 | Urbanization | `urbanization_monthly.csv` | `impervious_pct` |
| 6 | Water Stress | `water_stress_monthly.csv` | `usdm_dsci` |
| 7 | Temperature | `temperature_monthly.csv` | `temperature_2m_c` |
| 8 | Precipitation | `precipitation_monthly.csv` | `precipitation_mm_day` |
| 9 | GRACE | `grace_monthly.csv` | `grace_groundwater_anomaly`, `grace_available` |
| 10 | NDVI | `ndvi_monthly.csv` | `ndvi` |
| 11 | Wildfire (monthly) | `wildfire_monthly.csv` | `wildfire_risk_index`, `fire_count`, `log_acres` |

---

## Exported Artifacts (per model)

Each model produces:
- `model/{id}.onnx` — Deployable model in ONNX format
- `model/{id}_cv_results.json` — CV scores, best params, fold details
- `model/{id}_feature_names.json` — Ordered feature list (maps to ONNX input positions)
- `model/{id}_feature_stats.json` — Per-feature mean/std for input normalization
- `model/{id}_feature_importance.json` — Feature importance (abs coefs or tree importance)
- `model/historical_{id}.csv` — Actual vs. predicted on training window

Summary file:
- `model/model_comparison.json` — All models compared side-by-side
- `model/baselines.json` — Baseline scores for reference

---

## Key Design Decisions

1. **Residual-over-lag1 for smooth monthly targets (NDVI, GRACE):** These are autoregressive signals where last month's value is highly predictive. Training on the residual (change from lag1) lets the model learn what *drives change* rather than learning the level.

2. **Direct formulation for episodic/annual targets (wildfire, wildlife):** Wildfire is not autoregressive — last month's fire activity doesn't predict this month's. Annual targets have too few rows for the residual approach to add value.

3. **Multi-model competition for low-sample-size targets:** With 20–22 rows, XGBoost overfits. Ridge and ElasticNet provide the regularization needed. The competition framework automatically selects the best model.

4. **SimpleImputer in pipelines:** Annual lag features produce NaN for the first 1–2 rows. Rather than dropping these rows (losing ~10% of already-scarce data), the pipeline imputes with median and lets the model learn despite the missing values.

5. **TimeSeriesSplit (not random KFold):** Prevents data leakage from future→past. The model is always evaluated on data it hasn't seen and that comes chronologically after training.

6. **Urbanization (impervious_pct) as a feature:** Added from NLCD annual rasters. Showed up in top-5 importance for GRACE (#5) and wildfire monthly (#4), confirming that land development pressure adds predictive value.

---

## Limitations and Next Steps

- **Wildfire predictability ceiling:** Fire ignition is stochastic; even perfect climate/land-use features cannot predict when or where a fire starts. Monthly R² of 0.05 may be near the achievable ceiling for this formulation.
- **Annual model sample sizes:** 20–22 rows constrain model complexity. Additional years of data (as irrigation/public supply become available post-2020) would improve reliability.
- **Irrigation/public supply gap (2021–2023):** Monthly models are limited to the 2002-10 to 2020-12 window because these inputs end in December 2020.
- **ONNX deployment:** All models export successfully. The web application can load these via onnxruntime-web for browser-based predictions.
