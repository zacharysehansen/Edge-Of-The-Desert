# Model Report Log

Date: 2026-04-01

## Goal

Build, evaluate, and export a monthly Arizona water sustainability regression model from the merged Phase 2 dataset in `scripts/phase2.py`, then improve it until it beat simple time-series baselines.

## Data And Target

- Training window: `2000-10` through `2020-12`
- Final modeled row count after lag/rolling features: `237`
- Original feature count: `22`
- Target: `usdm_sustainability`

## Step 1: Initial Phase 2 Model

The initial Phase 2 model used:

- All `22` engineered features
- `TimeSeriesSplit(n_splits=5)`
- An `XGBRegressor`
- Randomized hyperparameter search

Initial saved results:

- Pre-tuning mean CV `R² = 0.1574`
- Pre-tuning mean CV `MAE = 9.9454`
- Fold-to-fold `R²` standard deviation `= 0.6045`
- Final train `R² = 0.9999`
- Final train `MAE = 0.1260`

What we concluded:

- The model ran and exported successfully.
- Generalization was unstable.
- Train performance was unrealistically strong compared with CV performance.

## Step 2: Evaluation Fix

We found that the script was mixing:

- Pre-tuning fold-by-fold metrics
- Tuned search mean score

Changes made:

- Added reusable CV evaluation helpers.
- Saved fold-level tuned CV metrics.
- Saved fold date ranges and train/test sizes.
- Added baseline tracking directly into `cv_results.json`.

Why this mattered:

- It let us compare models fairly.
- It made it obvious whether the tuned model actually improved on a naive baseline.

## Step 3: Baseline Comparison

We added two baseline predictors:

- `lag1_persistence`
- `roll3_mean`

Most important baseline result:

- `lag1_persistence`: mean CV `R² = 0.7899`, mean CV `MAE = 5.1291`

This changed the modeling goal from:

- "make XGBoost look better"

to:

- "beat the lag-1 persistence forecast"

## Step 4: Multi-Model Experiment Runner

We converted `scripts/phase2.py` into a model comparison script that runs multiple candidates on the same folds.

Added experiments:

- `XGBoost (all features)`
- `XGBoost (compact features)`
- `Ridge (compact features)`
- `ElasticNet (compact features)`

We also added:

- A leaderboard
- `model/model_comparison.json`
- Automatic selection of the best export-compatible model

## Step 5: Compact Feature Set

We reduced the feature set from `22` to `14` features to improve generalization in early folds with small training windows.

Compact feature set:

- `month_sin`
- `month_cos`
- `usdm_sustainability_lag1`
- `usdm_sustainability_lag3`
- `usdm_sustainability_roll3`
- `usdm_sustainability_roll6`
- `streamflow_cfs`
- `streamflow_cfs_lag1`
- `streamflow_cfs_roll3`
- `snow_water_equivalent_in`
- `snow_water_equivalent_in_lag1`
- `powell_pool_elevation`
- `ndvi`
- `ndvi_lag1`

Results after the first experiment pass:

| Model | Mean CV R² | Mean CV MAE |
|---|---:|---:|
| `lag1_persistence` | `0.7899` | `5.1291` |
| `XGBoost (compact features)` | `0.6590` | `6.4389` |
| `ElasticNet (compact features)` | `0.6009` | `6.5589` |
| `Ridge (compact features)` | `0.5511` | `5.6795` |
| `XGBoost (all features)` | `0.5243` | `7.7273` |
| `roll3_mean` | `0.4657` | `8.7944` |

What we concluded:

- Reducing the feature set helped a lot.
- Compact XGBoost clearly beat full-feature XGBoost.
- Even after that improvement, the model still did not beat `lag1_persistence`.

## Step 6: Residual-Over-Lag1 Experiment

We added a new candidate:

- `XGBoost (compact, residual over lag1)`

Method:

- Train the model on `residual = usdm_sustainability - usdm_sustainability_lag1`
- At prediction time, output `lag1 + predicted_residual`

Why this change:

- The lag-1 baseline was already very strong.
- Predicting the residual forces the model to explain what changed beyond persistence.

## Step 7: Residual Model Results

Current best learned model:

- `XGBoost (compact, residual over lag1)`

Current performance:

- Mean CV `R² = 0.8226`
- Mean CV `MAE = 4.6977`
- CV `R²` standard deviation `= 0.0762`
- Train `R² = 0.9448`
- Train `MAE = 3.5885`

Comparison against the strongest baseline:

- Residual model mean CV `R²`: `0.8226`
- `lag1_persistence` mean CV `R²`: `0.7899`
- Improvement in `R²`: `+0.0326`

- Residual model mean CV `MAE`: `4.6977`
- `lag1_persistence` mean CV `MAE`: `5.1291`
- Improvement in `MAE`: `-0.4315`

Fold-by-fold residual model results:

- Fold 1: `R² = 0.8471`, `MAE = 5.2362`
- Fold 2: `R² = 0.7110`, `MAE = 5.1366`
- Fold 3: `R² = 0.7600`, `MAE = 3.2407`
- Fold 4: `R² = 0.9181`, `MAE = 2.9399`
- Fold 5: `R² = 0.8765`, `MAE = 6.9349`

What we concluded:

- The residual formulation was the first learned model to beat the lag-1 baseline.
- It improved both `R²` and `MAE`.
- It was much more stable across folds than the earlier full-feature model.

## Step 8: Export Improvements

Originally, the best learned residual model was marked comparison-only because the export path only handled plain estimator outputs.

We then updated export so the residual winner can also be saved:

- The saved ONNX model still uses the existing scaled-input contract.
- The ONNX graph now adds `lag1` back internally for residual models.
- This keeps the browser-facing input format consistent.

Current exported artifacts now represent the best-performing model path rather than a weaker fallback.

## Current Artifact Summary

Main files:

- `scripts/phase2.py`
- `model/cv_results.json`
- `model/model_comparison.json`
- `model/water_sustainability.onnx`
- `model/feature_names.json`
- `model/feature_stats.json`
- `model/historical_sustainability.csv`
- `model/feature_importance.png`

## Interpretation Note

`R² = 0.8226` does **not** mean the model is "82.26% accurate".

What it means:

- On the cross-validation folds, the model explains about `82.26%` of the variance in the target relative to a simple mean-only baseline.

Why that matters:

- `R²` is a goodness-of-fit metric, not a classification accuracy metric.
- For this project, `MAE` is also very important because it tells us the average size of the prediction error in target units.

## Current Best Recommendation

The current best model to carry forward is:

- `XGBoost (compact, residual over lag1)`

Reason:

- It is the top learned model on the leaderboard.
- It beats `lag1_persistence`.
- It is more stable than the earlier all-feature XGBoost model.
