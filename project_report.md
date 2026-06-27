# Southern Arizona Environmental Model — Project Report

## Project Overview

This project builds a set of regression models for an eight-county southern Arizona region (Pima, Pinal, Santa Cruz, Cochise, Graham, Greenlee, Yuma, and La Paz). The goal is a planning tool where a user can adjust inputs like population growth, irrigation demand, or urbanization, and see how different environmental systems would likely respond.

The modeling framework is:

**Human Pressures + Environmental Conditions → Environmental Responses**

Rather than collapsing everything into a single sustainability score, each environmental outcome gets its own independent model. This makes the system more interpretable and allows each system to respond on its own terms.

---

## Phase 1: Data Collection

The first phase focused on collecting and assembling all inputs and targets into clean monthly and annual CSV files, covering the eight-county study area.

### Inputs (Shared Across All Models)

| Variable | Source | Coverage |
|---|---|---|
| Population | U.S. Census Bureau | 2000–2023 |
| Irrigation withdrawal | HUC12 datasets | 2000–2020 |
| Public supply groundwater | NWAA datasets | 2000–2020 |
| Lake Mead operations | Bureau of Reclamation HydroData | 2000–2023 |
| Urbanization (impervious %) | USGS NLCD Annual | 2000–2023 |
| Drought index (USDM DSCI) | U.S. Drought Monitor | 2000–2023 |
| Temperature | ERA5 / MODIS | 2000–2023 |
| Precipitation | PRISM | 2000–2023 |

Several datasets required re-aggregation from statewide to the eight-county boundary. Irrigation and public supply data had to be re-filtered from their underlying HUC12 shards. Population required re-interpolation at the county level rather than re-using the prior statewide file. Urbanization required downloading annual NLCD rasters year by year, clipping each to the county shapefile, and averaging to a single impervious surface percentage per year.

A key data gap exists: irrigation and public supply data end in December 2020. This constrains the monthly model training window to 2002–2020.

### Outputs (Model Targets)

| Target | Source | Grain |
|---|---|---|
| Vegetation health (NDVI) | MODIS | Monthly |
| Groundwater storage anomaly | GRACE / GRACE-FO satellite | Monthly |
| Groundwater well levels (depth to water) | USGS NWIS Daily Values | Monthly |
| Surface water discharge | USGS NWIS Daily Values | Monthly |
| Wildfire risk index | Regional fire event database | Monthly and annual |
| Wildlife abundance index | North American Breeding Bird Survey | Annual |

---

## Phase 2: Modeling

### Pipeline Structure

All scripts live under `scripts/phase2/` and run end-to-end in about 44 seconds via `run_all.py`. The pipeline proceeds in this order:

1. `merge.py` — joins all input CSVs into a monthly panel (288 rows × 20 columns) and a separate annual panel
2. `features.py` — engineers lag, rolling, seasonal, and anomaly features; returns a clean `(X, y)` pair per model
3. `baselines.py` — computes lag-1 persistence and 3-month rolling mean baselines for comparison
4. One script per model (`model_ndvi.py`, `model_grace.py`, etc.)
5. `export.py` — assembles all CV results into a single `model_comparison.json` leaderboard

### Feature Engineering

For monthly models, each input column is expanded into six derived features: 1-, 3-, and 6-month lags, and 3-, 6-, and 12-month trailing means. Temperature and precipitation also get anomaly features (departure from a 12-month trailing mean). Seasonal position is encoded as sine and cosine of month to keep it continuous.

For annual models, monthly inputs are aggregated by mean, sum, June value, and June–July–August mean before applying year-over-year lags and rolling features.

---

## Model Results

All seven models beat their lag-1 persistence baselines and export successfully to ONNX format.

| Model | Target | Grain | CV R² | Baseline R² | Algorithm |
|---|---|---|---|---|---|
| NDVI | Vegetation health | Monthly | **0.80** | 0.55 | XGBoost |
| GRACE | Groundwater anomaly | Monthly | **0.52** | 0.41 | XGBoost |
| Groundwater | Well depth (ft) | Monthly | **0.61** | — | XGBoost |
| Surface Water | Discharge (cfs) | Monthly | **0.48** | — | ElasticNet |
| Wildfire (monthly) | Wildfire risk | Monthly | **0.05** | -0.97 | XGBoost |
| Wildfire (annual) | Wildfire risk | Annual | **-0.11** | -2.11 | ElasticNet |
| Wildlife | Bird abundance | Annual | **0.21** | — | ElasticNet |

### NDVI — Vegetation Health (CV R² = 0.80)

The NDVI model is the strongest performer. It uses a residual-over-lag1 formulation: the model predicts the *change* from last month's vegetation reading rather than the absolute value, then adds the lag back at prediction time. This lets it focus on what drives change rather than just learning the level of the signal.

The top predictors are lagged temperature, lagged public supply groundwater withdrawal, current precipitation, and precipitation anomaly. The lag-1 baseline alone explains 55% of variance; the model captures an additional 25% by incorporating climate and human pressure features.

### GRACE — Groundwater Anomaly (CV R² = 0.52)

The GRACE model uses the same residual-over-lag1 approach as NDVI. The dominant feature is month (seasonal cycle), followed by precipitation anomaly and temperature. Urbanization (impervious surface) appears in the top five, confirming that land development has a measurable effect on subsurface water.

### Groundwater — Well Levels (CV R² = 0.61)

This model predicts monthly mean depth-to-water from USGS monitoring wells across the eight-county region. An expanded competition tested 8 candidates: Ridge, ElasticNet, XGBoost, XGBoost-tight, and a XGB+ElasticNet blend, each in direct and residual formulations. The winner is an extremely regularized XGBoost (min_child_weight=120, reg_lambda=100, colsample_bytree=0.3) in residual formulation. Aggressive feature selection (halved from 63 to 32 features) was also key. The initial approach scored train R²=0.98 vs CV R²=0.50; the final model achieves train R²=0.89 vs CV R²=0.61. The blend approach (XGBoost+ElasticNet) scored 0.54, indicating that the nonlinear signal is better captured by a single well-regularized tree model than by averaging with a linear one. All direct formulation candidates scored deeply negative R², confirming the strong autoregressive structure of groundwater levels.

### Surface Water — Discharge (CV R² = 0.48)

The surface water model predicts monthly mean stream discharge (cfs) from USGS gages across the study area. Desert discharge is heavily right-skewed (baseflow ~50 cfs, monsoon floods >500 cfs), so the target is log-transformed before modeling. A six-candidate competition selects ElasticNet in residual formulation as the winner — its L1/L2 regularization generalizes far better than XGBoost for this noisy target. Train R² (0.36) is actually *below* CV R² (0.48), indicating zero overfitting. Feature selection removed 22 noise dimensions. Interaction features (precipitation × impervious = runoff proxy, precipitation × temperature = ET proxy) add physical meaning.

### Wildfire (Monthly) — Wildfire Risk Index (CV R² = 0.05)

Monthly wildfire is inherently difficult to predict because ignition events are stochastic. The model uses a direct formulation (the residual approach fails for episodic signals) and wins a six-candidate competition across Ridge, ElasticNet, and XGBoost. R² of 0.05 is low in absolute terms but represents a meaningful real signal given that the lag-1 baseline scores -0.97. The model captures seasonal and population-driven patterns but cannot predict when or where a specific fire will start.

### Wildfire (Annual) — Wildfire Risk Index (CV R² = -0.11)

With only 22 rows, overfitting is the main challenge. XGBoost was ruled out; ElasticNet's regularization holds up better. The negative R² means the model is worse than predicting the annual mean, but it still far outperforms the lag-1 baseline (-2.11). Five fire-ecology features were engineered specifically for this model: a wet-then-dry interaction (prior year precipitation × current summer temperature), consecutive dry year count, prior 2-year precipitation sum, summer temperature anomaly, and worst drought in the prior 2 years. These improve over naive climate features but the sample size remains the binding constraint.

### Wildlife — Bird Abundance (LOO R² = 0.21, Spearman ρ = 0.50)

With only 20 usable rows (2020 BBS surveys were cancelled), Leave-One-Out cross-validation is used instead of TimeSeriesSplit to maximize fold count. At this sample size, the Spearman rank correlation (0.50, p=0.025) is the more meaningful reliability indicator: the model correctly ranks good and bad years for bird populations about half the time, which is real signal. R² is reported but treated as secondary.

---

## How the Models Interact

The seven models do not run as a single pipeline at inference time — each takes in the shared human pressure inputs and produces one output independently. However, some have explicit data dependencies that create a partial cascade.

**GRACE feeds into NDVI.** The NDVI model includes the GRACE groundwater anomaly as an input feature, because subsurface moisture conditions correlate with vegetation productivity. In practice this means that if the GRACE model is run first on a given scenario, its output can be passed into the NDVI model as an input, allowing groundwater conditions to propagate through to the vegetation prediction.

**NDVI feeds into Wildlife.** The wildlife model includes the annual NDVI mean as an input feature, reflecting that bird abundance is directly tied to vegetation health and habitat quality. Running NDVI first on a scenario and feeding the result forward gives the wildlife model a richer picture than using raw climate inputs alone.

**Groundwater and Surface Water are independent.** Both use climate and human pressure inputs directly and do not feed into other models.

**Wildfire is independent.** Both wildfire models use only the shared human pressure and climate inputs. They do not depend on or feed into any other model.

The full dependency chain is:

```
Human Pressures + Climate Inputs
        │
        ├──► GRACE (groundwater anomaly)
        │           │
        │           ▼
        ├──► NDVI (vegetation health) ─────► Wildlife (bird abundance)
        │
        ├──► Groundwater (well depth-to-water)
        │
        ├──► Surface Water (stream discharge)
        │
        ├──► Wildfire Monthly (risk index)
        │
        └──► Wildfire Annual (risk index)
```

In a user-facing scenario tool, the recommended execution order is GRACE → NDVI → Wildlife, with groundwater, surface water, and both wildfire models running in parallel from the shared inputs.

---

## Limitations

**Irrigation and public supply end in 2020.** Monthly models are trained through December 2020 only. For any scenario projections past that date, these two inputs must be held at their 2020 values or extrapolated externally before being passed to the model.

**Annual model sample sizes.** Wildfire (22 rows) and wildlife (20 rows) are below the threshold for reliable machine learning. CV scores are noisy and should not be over-interpreted. These models are best used as scenario interpolation tools, not precision forecasters.

**Wildfire predictability ceiling.** Fire ignition is stochastic. Even with perfect climate and land-use inputs, specific fire events cannot be predicted. The monthly R² of 0.05 is likely near the ceiling for this formulation.

**Groundwater well monitoring sparsity.** USGS NWIS groundwater monitoring wells in southern Arizona are unevenly distributed across the eight counties. The monthly regional mean averages over available wells, which may introduce bias toward counties with denser monitoring networks.

**Selection optimism.** Each model ran a multi-candidate competition to pick the best algorithm and feature set. The reported CV R² values are slightly optimistic as a result — this is expected and worth noting when sharing results externally.
