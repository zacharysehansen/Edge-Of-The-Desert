# Model Report Log

Date: 2026-04-27

## Goal

Build, evaluate, improve, and export the strongest honest monthly Arizona water sustainability regression model from the Phase 2 dataset in `scripts/phase2.py`.

The project goal was not just to make one model score look good. It was to:

- beat simple time-series baselines
- keep the export path compatible with the browser ONNX workflow
- test whether the new precipitation, GRACE, and temperature signals materially improved performance

## Final Model

The finalized exported model is:

- `XGBoost (all features, residual over lag1, focused search, GRACE window)`

Final exported performance:

- Mean CV `R² = 0.8964`
- Mean CV `MAE = 3.7200`
- CV `R²` standard deviation `= 0.0238`
- CV `MAE` standard deviation `= 1.3363`

Final export window:

- `2002-10` through `2020-12`
- `219` modeled rows

Reason this became the final export:

- It is the highest-scoring model produced by the full set of experiments.
- It is exportable to ONNX.
- It beats the strongest same-window baseline by a meaningful margin.

Same-window baseline comparison:

- `lag1_persistence`: mean CV `R² = 0.7900`, mean CV `MAE = 5.1649`
- Final model improvement in `R²`: `+0.1064`
- Final model improvement in `MAE`: `-1.4449`

## Dataset Summary

Two modeling windows matter now:

1. Full engineered Phase 2 window
   - `2000-10` through `2020-12`
   - `237` rows after lag and rolling features
   - used for the standard leaderboard

2. Final export window
   - `2002-10` through `2020-12`
   - `219` rows
   - starts after GRACE has enough observed history to support lag and rolling features
   - used for the final exported model

Feature progression:

- Initial engineered feature count: `22`
- Early compact feature set: `14`
- Final full feature set: `37`

Target:

- `usdm_sustainability`
- derived from U.S. Drought Monitor `DSCI` as `100 - (usdm_dsci / 5)`
- higher `DSCI` means worse drought, so lower `usdm_sustainability` is expected
- this is an inverted drought-severity proxy on a `0-100` scale, not an independent field measurement
- the model predicts this transformed target directly, and same-month `usdm_dsci` is excluded from the input features to avoid direct leakage

## Step 1: Initial Phase 2 Model

The original Phase 2 setup used:

- all `22` engineered features
- `TimeSeriesSplit(n_splits=5)`
- `XGBRegressor`
- randomized hyperparameter search

Initial saved results:

- mean CV `R² = 0.1574`
- mean CV `MAE = 9.9454`
- fold-to-fold `R²` standard deviation `= 0.6045`
- train `R² = 0.9999`
- train `MAE = 0.1260`

What we concluded:

- the pipeline ran and exported
- generalization was poor
- train fit was unrealistically strong relative to CV fit

## Step 2: Evaluation Fixes And Baselines

We corrected the evaluation logic so the report stopped mixing:

- pre-tuning fold metrics
- post-tuning search averages

Changes made:

- added reusable CV evaluation helpers
- saved fold-level tuned CV metrics
- saved fold date ranges and train/test sizes
- added baseline tracking directly into `cv_results.json`
- added `model/model_comparison.json`

We also added two baselines:

- `lag1_persistence`
- `roll3_mean`

Most important baseline result on the full window:

- `lag1_persistence`: mean CV `R² = 0.7899`, mean CV `MAE = 5.1291`

This changed the modeling goal from:

- "make XGBoost look better"

to:

- "beat persistence honestly"

## Step 3: Compact Features And Residual-Over-Lag1

We converted `scripts/phase2.py` into a multi-model experiment runner and added:

- `XGBoost (all features)`
- `XGBoost (compact features)`
- `Ridge (compact features)`
- `ElasticNet (compact features)`

We then reduced the early compact feature set to `14` features and added the residual formulation:

- train on `usdm_sustainability - usdm_sustainability_lag1`
- add `lag1` back at prediction time

The first strong breakthrough was:

- `XGBoost (compact, residual over lag1)`
- mean CV `R² = 0.8226`
- mean CV `MAE = 4.6977`

This was the first learned model to beat `lag1_persistence`.

## Step 4: Export Improvements For Residual Models

Originally, the best residual model was comparison-only because the export path only handled plain estimator outputs.

We updated the export so residual winners could also be saved:

- the ONNX model still uses the scaled-input contract
- the ONNX graph adds `lag1` back internally for residual models
- the browser-facing input format stays consistent

This was a crucial infrastructure step because all later winning models used the residual formulation.

## Step 5: Precipitation And GRACE Endpoint Integration

We then extended Phase 2 to use the new Phase 1 endpoint files:

- `data/Final/merra_precipitation.csv`
- `data/Final/grace_groundwater_anomaly.csv`

Changes added to `scripts/phase2.py`:

- new source loading for precipitation and GRACE
- `precipitation_mm_day_lag1`
- `precipitation_mm_day_roll3`
- `precipitation_mm_day_roll6`
- `grace_groundwater_anomaly_lag1`
- `grace_groundwater_anomaly_roll3`
- `grace_groundwater_anomaly_roll6`
- `powell_pool_elevation_lag1`

We also added focused model families:

- `compact_plus_endpoints`
- `compact_plus_endpoint_dynamics`

Important intermediate result:

- `XGBoost (compact + precip/GRACE, residual over lag1)`: mean CV `R² = 0.8637`, mean CV `MAE = 4.0702`

And the stronger full-window winner after broader tuning became:

- `XGBoost (all features, residual over lag1, focused search)`: mean CV `R² = 0.8808`, mean CV `MAE = 3.8162`

What we concluded:

- precipitation clearly helped
- GRACE helped enough to keep
- the model was moving well beyond the compact-only setup

## Step 6: GRACE Cleanup And Observed-Window Experiment

To avoid overstating GRACE’s value before the mission actually begins, we cleaned up the GRACE treatment:

- added `grace_available`
- set pre-`2002-04` GRACE anomaly values to neutral `0.0`
- kept the full-window experiments for comparison
- added a separate observed-window experiment beginning at `2002-10`

Why `2002-10`:

- it starts after GRACE begins
- it leaves enough history for lag and rolling features to be real observed values instead of synthetic carryover

This experiment answered a specific question:

- does GRACE help more when the model only sees real GRACE-supported history?

Result before temperature:

- the GRACE-window winner outperformed the full-window winner
- this validated GRACE as a real signal rather than just a visual add-on

## Step 7: Temperature Endpoint Integration

We then added temperature as one more attainable hydrologic signal.

New Phase 1 artifact:

- `scripts/phase1_merra_temperature.py`
- generated `data/Final/merra_temperature_2m.csv`

New Phase 2 temperature features:

- `temperature_2m_c`
- `temperature_2m_c_lag1`
- `temperature_2m_c_roll3`
- `temperature_2m_c_roll6`
- `temperature_2m_c_anomaly`
- `temperature_2m_c_anomaly_lag1`
- `temperature_2m_c_anomaly_roll3`

This also added the `compact_plus_hydroclimate` experiment family.

Key result:

- temperature helped the best full model modestly
- temperature helped the GRACE-window models more clearly

Updated best full-window model:

- `XGBoost (all features, residual over lag1, focused search)`
- mean CV `R² = 0.8843`
- mean CV `MAE = 3.7769`

Updated best GRACE-window model:

- `XGBoost (all features, residual over lag1, focused search, GRACE window)`
- mean CV `R² = 0.8946`
- mean CV `MAE = 3.6383`

Powell cleanup refinement:

- we removed `powell_storage` and `powell_storage_lag1` from the modeled feature set
- `powell_pool_elevation` remains the retained Powell state signal
- this reduced redundancy between storage and elevation while keeping the more intuitive public-facing variable

## Step 8: Finalization To Highest-Scoring Export

At this point, the highest honest score came from the GRACE observed window rather than the full window.

We finalized the export path so the strongest exportable model is now selected even when it comes from the GRACE-window experiment set.

This changed the saved artifacts from:

- full-window winner at `R² = 0.8843`

to:

- GRACE-window winner at `R² = 0.8964`

The exported artifacts now reflect:

- `grace_window_xgb_full_residual_lag1_focused`
- `37` input features
- training/export window `2002-10` through `2020-12`

## Final Hyperparameters

Selected final model hyperparameters:

- `subsample = 1.0`
- `reg_lambda = 1`
- `reg_alpha = 0.05`
- `n_estimators = 800`
- `min_child_weight = 1`
- `max_depth = 3`
- `learning_rate = 0.05`
- `colsample_bytree = 0.7`

## Final Important Features

Top reported gain features in the exported model:

- `precipitation_mm_day_lag1`
- `usdm_sustainability_lag3`
- `precipitation_mm_day_roll3`
- `temperature_2m_c_lag1`
- `temperature_2m_c_anomaly`
- `ndvi`
- `temperature_2m_c_anomaly_lag1`
- `usdm_sustainability_roll6`
- `precipitation_mm_day`
- `month_cos`

Additional high-value features just below the top 10:

- `grace_groundwater_anomaly`
- `streamflow_cfs`
- `grace_groundwater_anomaly_roll6`
- `grace_groundwater_anomaly_lag1`

What this means:

- precipitation became one of the strongest short-memory signals
- temperature added real predictive value
- GRACE remained important in the final model

## Current Artifact Summary

Main model files:

- `scripts/phase2.py`
- `scripts/phase1_merra_temperature.py`
- `data/Final/merra_precipitation.csv`
- `data/Final/merra_temperature_2m.csv`
- `data/Final/grace_groundwater_anomaly.csv`
- `model/cv_results.json`
- `model/model_comparison.json`
- `model/water_sustainability.onnx`
- `model/feature_names.json`
- `model/feature_stats.json`
- `model/historical_sustainability.csv`
- `model/feature_importance.png`

Important note on `historical_sustainability.csv`:

- it now covers the finalized export window
- current row coverage is `2002-10` through `2020-12`

## Interpretation Note

`R² = 0.8964` does **not** mean the model is "89.64% accurate".

What it means:

- on the cross-validation folds, the model explains about `89.64%` of the variance in the target relative to a simple mean-only baseline

Why that matters:

- `R²` is a goodness-of-fit metric, not a classification accuracy metric
- `MAE` is also critical because it tells us the average size of the prediction error in target units

How to read the final MAE values:

- `mean_mae = 3.7200` means the model misses the true sustainability score by about `3.72` points on average on held-out cross-validation folds
- because the target is a `0-100` sustainability score, the error is in score points
- `std_mae = 1.2595` means that the fold-level average error changes by about `1.26` points across different train/test time splits
- in plain terms, if the true score were `60`, a typical prediction would often be somewhere around `56.4` to `63.6`, though some misses will be larger and some smaller

Overfitting note:

- the model does fit the training data extremely closely: train `R² = 1.00000`, train `MAE = 0.0081`
- that means the model has enough capacity to almost memorize the training window
- however, the held-out time-series CV results are still strong and fairly stable: mean CV `R² = 0.8964`, std `R² = 0.0238`, mean CV `MAE = 3.7200`
- so the honest conclusion is not "no overfitting at all"; it is "some training-set overfit is likely, but out-of-sample performance remains strong enough that the model still generalizes well for this project"
- the biggest remaining caution is selection optimism: many experiments were tried, so the final reported score is probably a little more optimistic than a never-retuned one-shot evaluation would be

## Final Recommendation

The model to carry forward into Phase 3 is:

- `XGBoost (all features, residual over lag1, focused search, GRACE window)`

Reason:

- it is the strongest model produced by the current experiment set
- it is exportable and already saved to ONNX
- it benefits from precipitation, GRACE, and temperature
- it is more stable than the earlier full-history models
- it gives the cleanest final score we reached without using questionable shortcuts
