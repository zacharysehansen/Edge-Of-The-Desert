# Phase 2 Setup: Modeling

All code goes in `scripts/phase2.py`. Trained artifacts go in `model/`.

---

## Step 1: Load and Audit All Source Files

Load every CSV from `data/Final` into a dictionary of DataFrames keyed by filename. For each one, parse `year_month` as a period (or datetime) and print the shape, column names, date range, and null count per column. This audit runs first before any merging so coverage gaps are visible and explicit.

Files to load:
- `snotel_swe.csv`
- `irrigation_huc12_monthly_az_2000_2020.csv`
- `nwaa_public_supply_az_monthly.csv`
- `powell_combined.csv`
- `azpop_monthly.csv`
- `modis_ndvi.csv`
- `usgs_streamflow.csv`
- `usdm_sustainability.csv`

---

## Step 2: Decide on the Training Window

Coverage summary from the source files:

| Source | Start | End |
|---|---|---|
| `snotel_swe.csv` | 2000-10 | 2018-07 |
| `irrigation_huc12_monthly_az_2000_2020.csv` | 2000-01 | 2020-12 |
| `nwaa_public_supply_az_monthly.csv` | 2000-01 | 2020-12 |
| `powell_combined.csv` | 2000-01 | 2020-12 |
| `azpop_monthly.csv` | 2000-01 | 2020-12 |
| `modis_ndvi.csv` | 2000-02 | 2023-12 |
| `usgs_streamflow.csv` | 2000-01 | 2023-12 |
| `usdm_sustainability.csv` | 2000-01 | 2023-12 |

The shortest source is `snotel_swe.csv` (ends 2018-07). Dropping it entirely to extend the window is not the right move: snowpack is a physically meaningful predictor for Arizona water availability and we should keep it.

The plan is to use **2000-10 through 2020-12** as the training window. This gives roughly 243 monthly rows with full feature coverage, except for the SNOTEL gap between 2018-08 and 2020-12 which needs to be filled (see Step 4).

The four sources that end at 2020-12 (`irrigation`, `public_supply`, `powell`, `azpop`) define the upper bound. The USDM and streamflow data extending to 2023-12 are not used for training but their longer historical record is used to render the historical time series in the browser.

---

## Step 3: Merge All Sources on `year_month`

Build a base index spanning `2000-10` through `2020-12`. Left-join every source onto this index on `year_month`. After the merge, print null counts per column again to confirm which cells need imputation and which are structural (coverage gaps vs. genuinely missing values within coverage).

Keep only the model-relevant columns and drop any HUC12 count or unit-duplicate columns (e.g., `huc12_count`, `_gallons_per_day`, `_acre_feet_month` variants). The working feature set after merge:

- `snow_water_equivalent_in`
- `irrigation_total_withdrawal_mgd`
- `public_supply_groundwater_mgd`
- `powell_evaporation`, `powell_total_release`, `powell_inflow`, `powell_storage`, `powell_pool_elevation`
- `AZPOP`
- `ndvi`
- `streamflow_cfs`
- `usdm_dsci`, `usdm_sustainability` (target)

---

## Step 4: Impute Missing Values

**SNOTEL (2018-08 through 2020-12):** Compute the monthly climatological median for each calendar month from the observed 2000-10 to 2018-07 window. Fill the gap with those 12 median values keyed by month. This preserves the strong seasonal signal (near-zero in summer, peak in late winter/early spring) without extrapolating a trend into uncertain territory.

**Any other within-coverage nulls:** If a source has isolated nulls inside its stated coverage window, fill with linear interpolation (forward then backward at the edges). Check for these after Step 3 and log which cells were interpolated.

**Do not impute the target.** Any row where `usdm_sustainability` is null gets dropped.

---

## Step 5: Engineer Additional Features

These additions substantially improve accuracy on small tabular time series by giving the model explicit temporal structure it cannot recover from the raw features alone.

**Seasonal encoding:** Add `month_sin = sin(2π * month / 12)` and `month_cos = cos(2π * month / 12)`. This encodes month as a continuous cycle so the model sees December and January as adjacent rather than endpoints.

**Lag features:** Add one-month and three-month lags of the target (`usdm_sustainability_lag1`, `usdm_sustainability_lag3`) and one-month lags of the highest-signal predictors: `precipitation`, `streamflow_cfs`, `snow_water_equivalent_in`, `powell_storage`. Lag features require dropping the first few rows where the lag window is undefined.

**Rolling means:** Add three-month and six-month rolling means of `usdm_sustainability` (computed on rows prior to each point to avoid leakage). Also add a three-month rolling mean of `streamflow_cfs`.

**Population growth rate:** Replace raw `AZPOP` with its month-over-month percentage change (`AZPOP_pct_change`). The raw level is nearly collinear with the time index and less informative than the rate.

After engineering, drop any rows that have nulls introduced by the lag/rolling windows. Log the final row count.

---

## Step 6: Define Feature Matrix and Target

Separate the cleaned DataFrame into `X` (all engineered features excluding `usdm_dsci` and `usdm_sustainability`) and `y` (`usdm_sustainability`). Keep the `year_month` index attached to `X` for cross-validation fold logging but do not pass it into the model.

Print the final feature list and shape.

---

## Step 7: Build the Preprocessing Pipeline

Wrap preprocessing in a scikit-learn `Pipeline` so it is applied consistently inside each cross-validation fold without leakage.

Steps in the pipeline:
1. `SimpleImputer(strategy='median')` as a safety net for any residual nulls.
2. `StandardScaler()` to normalize feature magnitudes for the gradient boosting algorithm's internal regularization.

Attach the pipeline to an `XGBRegressor` as the final estimator.

---

## Step 8: Cross-Validate with TimeSeriesSplit

Use `TimeSeriesSplit(n_splits=5)` from scikit-learn. For each fold, fit the pipeline on the training slice and score on the test slice. Record R² and MAE per fold and print the mean and standard deviation across folds.

The expanding window design means later folds have more training data, which is the correct behavior for a time series that likely drifts. A fold-to-fold variance in R² above roughly 0.15 is a signal to revisit the feature set or check for data leakage.

---

## Step 9: Hyperparameter Search

Run a randomized search over the XGBoost hyperparameters using `RandomizedSearchCV` with the same `TimeSeriesSplitPhase2(n_splits=5)` as the CV strategy. Search over:

```
n_estimators: [100, 300, 500, 800]
max_depth: [3, 4, 5, 6]
learning_rate: [0.01, 0.05, 0.1, 0.2]
subsample: [0.6, 0.8, 1.0]
colsample_bytree: [0.6, 0.8, 1.0]
min_child_weight: [1, 3, 5]
reg_alpha: [0, 0.1, 0.5]
reg_lambda: [1, 2, 5]
```

Use `n_iter=80` and `scoring='r2'`. Set `early_stopping_rounds=20` on the XGBRegressor if using the native eval set API, otherwise rely on the grid search. Print the best parameters and best CV R² score.

---

## Step 10: Final Model Training

Refit the pipeline with the best hyperparameters found in Step 9 on the full training dataset (all 243 rows). Print the training R² and MAE as a sanity check. These will naturally be higher than the CV scores and are not the reported performance; the CV scores from Step 8 using the best parameters are.

Also generate and print the XGBoost feature importances (gain-based). Save a horizontal bar chart of the top 15 features to `model/feature_importance.png`.

---

## Step 11: Export to ONNX

Convert the fitted scikit-learn pipeline to ONNX using `skl2onnx`. The input type is a float array with shape `[None, n_features]`. Save to `model/water_sustainability.onnx`.

After export, run a round-trip validation: load the ONNX model using `onnxruntime`, pass the full `X` through it, and compare the outputs to the sklearn pipeline's predictions. Assert that the max absolute difference is below 1e-4. If the assertion fails, investigate the skl2onnx conversion options for XGBoost (sometimes requires specifying the `zipmap=False` output config).

---

## Step 12: Save Supporting Artifacts

Save the following to `model/`:

- `water_sustainability.onnx` (the exported model)
- `feature_names.json` (ordered list of feature column names matching the model input)
- `feature_stats.json` (per-feature min, max, median, and p5/p95 from the training data, used by the browser to set slider ranges and defaults)
- `historical_sustainability.csv` (the full `year_month` + `usdm_sustainability` series from 2000-01 through 2023-12, used by the D3 time series chart in the browser)
- `cv_results.json` (fold-level R² and MAE from Step 8, for the writeup)
- `feature_importance.png` (bar chart from Step 10)

The `feature_stats.json` structure should be:
```json
{
  "feature_name": {
    "min": float,
    "max": float,
    "median": float,
    "p5": float,
    "p95": float
  },
  ...
}
```

The browser sliders use `p5` and `p95` as range endpoints and `median` as the default value. Using the full `min`/`max` would expose extreme outliers as the slider range, which gives users a misleading sense of normal conditions.

---

## Step 13: Smoke Test

At the bottom of `phase2.py`, add a short smoke test that:

1. Loads `model/water_sustainability.onnx` fresh from disk.
2. Constructs a single input row using the median values from `feature_stats.json`.
3. Runs inference.
4. Asserts the output is a float between 0 and 100.
5. Prints "Smoke test passed: predicted sustainability = X.XX".

Run the full script end to end and confirm it completes without errors.