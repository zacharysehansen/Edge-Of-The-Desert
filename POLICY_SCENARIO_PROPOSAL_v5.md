Policy Scenario Proposal (Version 5)

Date: 2026-06-03
Revised: 2026-06-03
Executive Summary

This proposal redesigns the public-facing scenario interface around five high-level attributes that each own a distinct set of model features. The goal is to create a system that is easier for residents to understand while preserving the existing ONNX model and projection engine.

The previous proposal allowed multiple attributes to modify the same underlying feature. While technically feasible, that approach created overlapping control paths that were difficult to explain. Version 5 adopts a one-to-one ownership model in which every model feature belongs to exactly one attribute.

This change improves:

    Transparency
    User comprehension
    Scenario explainability
    Implementation simplicity
    Future maintainability

The ONNX model remains unchanged.
Overview

Edge of the Desert is an interactive visualization that lets users manipulate environmental and human demand conditions across the American Southwest and see the predicted water sustainability score update in real time. The system combines satellite remote sensing data, hydrological measurements, and human consumption data to train a regression model, which is exported and run entirely in the browser via ONNX. The D3.js frontend visualizes both the user-controlled inputs and the model output simultaneously. Phase 1 data assembly is currently Arizona-focused within that broader Southwest framing, with the current monthly endpoint source files stored in data/Final.

Model summary (final exported model):

    Algorithm: XGBoost (all features, residual over lag1, focused search, GRACE window)
    Export format: ONNX (browser via onnxruntime-web / WASM)
    Final export window: 2002-10 through 2020-12 (219 modeled rows)
    Input features exported: 37
    Final hyperparameters: subsample=1.0, reg_lambda=1, reg_alpha=0.05, n_estimators=800, min_child_weight=1, max_depth=3, learning_rate=0.05, colsample_bytree=0.7
    Export artifact name: grace_window_xgb_full_residual_lag1_focused
    Key exported files: model/water_sustainability.onnx, model/feature_names.json, model/feature_stats.json, model/feature_importance.png, model/cv_results.json, model/model_comparison.json, model/historical_sustainability.csv

Performance (cross-validation, GRACE-window chosen export):

    Mean CV R² = 0.8964 (std R² = 0.0238)
    Mean CV MAE = 3.7200 (std MAE = 1.3363)
    Baseline (lag1_persistence) mean CV R² = 0.7900, MAE = 5.1649
    Final model improvement: +0.1064 R², -1.4449 MAE vs same-window lag1_persistence

Notes on interpretation:

    R² shows explained variance on CV folds; MAE indicates average error in the 0–100 sustainability score. Mean MAE ≈ 3.72 means typical predictions miss by ~3.7 points.

Data Sources
Feature	Source	Raw Resolution	Model Resolution
NDVI	MODIS MOD13A3	Monthly	Monthly
2m Temperature	MERRA-2	Daily	Monthly avg
Specific Humidity	MERRA-2	Daily	Monthly avg
Precipitation	MERRA-2	Daily	Monthly avg
Snow Water Equivalent	Arizona SNOTEL bundle -> `snotel_swe.csv	Daily	Monthly endpoint
Streamflow	USGS NWIS -> usgs_streamflow.csv	Daily	Monthly endpoint
GRACE Groundwater Anomaly	NASA GRACE	Monthly	Monthly
GRACE/GRACE-FO Derived Recharge Estimate	NASA GRACE / GRACE-FO	Monthly	Monthly
Lake Powell Operations	Manual Powell bundle -> powell_combined.csv	Daily / irregular	Monthly endpoint
Irrigation Total Withdrawal	USGS Arizona HUC12 -> irrigation_huc12_monthly_az_2000_2020.csv	Monthly	Monthly statewide endpoint
Public-Supply Total	NWAA Arizona HUC12 -> nwaa_public_supply_az_monthly.csv	Monthly	Monthly statewide endpoint
Population	Arizona AZPOP -> azpop_monthly.csv	Annual	Monthly endpoint via interpolation
Inverted USDM Score	USDM -> usdm_sustainability.csv	Weekly	Monthly endpoint (target)

Project target range: 2000–2023. Join key: year_month. Endpoint/source CSVs live in data/Final. Coverage notes: some sources shorter (e.g., snotel_swe.csv 2000-10 to 2018-07). GRACE available from ~2002, requiring the observed-window export to avoid synthetic pre-2002 values.
Target Variable

    Target: usdm_sustainability
    Derived from U.S. Drought Monitor DSCI as 100 - (usdm_dsci / 5) and inverted so higher = better sustainability on a 0–100 scale.
    Same-month usdm_dsci excluded from inputs to avoid leakage.

Model

    Algorithm: XGBoost Regressor via scikit-learn API.
    Cross-validation: TimeSeriesSplit (expanding-window) with fold-level tuned CV metrics saved.
    Residual formulation used in exportable model: train on residual = target - target_lag1, then add lag1 back at prediction time; ONNX graph re-adds lag1 internally for residual models.
    Export pipeline: pipeline exported to ONNX via skl2onnx; browser input format remains scaled-input contract; residual models handled in-export.

Key modeling decisions and results incorporated into the policy tool:

    Residual-over-lag1 formulation materially improved honest out-of-sample performance and enabled stable exports that maintain a simple browser contract.
    Precipitation, GRACE, and temperature features were tested and retained because they consistently improved CV performance (e.g., compact+precip/GRACE residual R²=0.8637; adding temperature and full tuning produced R² up to 0.8964 in the GRACE window).
    To avoid overstating GRACE before its mission, pre-2002 GRACE values were set to neutral 0.0 in full-window experiments and an observed-window (from 2002-10) was used to assess the true GRACE signal; the observed-window winner outperformed full-window options and thus became the final export.

Why keep the ONNX model unchanged for the public UI

    The existing ONNX export supports the residual formulation and the browser input contract.
    No model retraining required to implement the Policy Attribute translation layer; the translation only maps attribute sliders to the model's 37 input features and preserves reproducible, exportable inference behavior.

Purpose

The current application exposes low-level climate, hydrology, and demand variables directly.

Current controls include:

    Lake Powell elevation
    Snow water equivalent
    Temperature
    Precipitation
    Irrigation withdrawals
    Public-supply withdrawals
    Groundwater anomaly

These variables are useful for model development but do not match how most residents think about water policy.

The revised interface should expose recognizable real-world concepts rather than raw environmental measurements.

The public experience should help users explore questions such as:

    What happens if Arizona continues growing?
    What happens if landscaping becomes less water intensive?
    What happens if drought conditions worsen?
    How much do recharge programs matter?
    How dependent are outcomes on Colorado River conditions?

Audience

The primary audience is the general public.

The goal is not to create a technical planning tool.

The goal is to create an engaging educational experience that:

    Builds curiosity
    Encourages exploration
    Improves water-policy literacy
    Shows tradeoffs between growth and water management

Technical accuracy remains important, but public understanding takes priority over exposing model internals.
Core Design Principle

Every model feature must have exactly one attribute owner.

A user should always be able to answer:

    Which slider changed this feature?

with a single clear answer.

This avoids situations where multiple sliders simultaneously affect the same variable.
Five Policy Attributes

All attributes use a 0-100 scale.
1. Population And Development Pressure

User question:

    How quickly is Arizona growing?

Controls:

    AZPOP
    Public-supply demand

Interpretation:

    0 = slow growth
    50 = current trend
    100 = rapid growth

2. Irrigated Land Use

User question:

    How much water is being used to keep landscapes and crops green?

Controls:

    Irrigation withdrawals
    NDVI

Represents:

    Agriculture
    Turf
    Golf courses
    HOA landscaping
    Parks
    Irrigated open space

3. Water Management And Recharge

User question:

    How aggressively is groundwater being replenished?

Controls:

    GRACE-derived recharge estimate
    GRACE groundwater anomaly

Represents:

    Recharge projects
    Groundwater banking
    Reclaimed water
    Managed aquifer recharge

Notes:

    GRACE inputs include grace_groundwater_anomaly, grace_groundwater_anomaly_lag1, grace_groundwater_anomaly_roll3, and grace_groundwater_anomaly_roll6. A grace_available flag was added in modeling to avoid over-interpreting synthetic data before 2002.

4. Climate Stress

User question:

    How hot and dry are conditions?

Controls:

    Temperature
    Specific humidity
    Precipitation
    Snow water equivalent

Represents atmospheric and drought-related conditions.

Notes:

    Temperature features added from data/Final/merra_temperature_2m.csv include temperature_2m_c, temperature_2m_c_lag1, temperature_2m_c_roll3, temperature_2m_c_roll6, and anomaly variants. These improved GRACE-window performance modestly and helped full-window modestly.

5. Colorado River And Surface Water Conditions

User question:

    How healthy are the major reservoirs and river systems?

Controls:

    Lake Powell elevation
    Streamflow

Represents regional water-supply conditions visible to residents and policymakers.

Notes:

    powell_pool_elevation retained; powell_storage and its lag were removed from the modeled feature set to reduce redundancy.

Feature Ownership Map
Model Feature	Attribute Owner
AZPOP	Population And Development Pressure
Public Supply Total	Population And Development Pressure
Irrigation Withdrawal	Irrigated Land Use
NDVI	Irrigated Land Use
GRACE Recharge Estimate	Water Management And Recharge
GRACE Groundwater Anomaly	Water Management And Recharge
Temperature	Climate Stress
Specific Humidity	Climate Stress
Precipitation	Climate Stress
Snow Water Equivalent	Climate Stress
Lake Powell Elevation	Colorado River And Surface Water Conditions
Streamflow	Colorado River And Surface Water Conditions
precipitation_mm_day_lag1	Climate Stress
precipitation_mm_day_roll3	Climate Stress
precipitation_mm_day_roll6	Climate Stress
precipitation_mm_day	Climate Stress
temperature_2m_c_lag1	Climate Stress
temperature_2m_c_anomaly	Climate Stress
temperature_2m_c_anomaly_lag1	Climate Stress
usdm_sustainability_lag1	(internal lag feature; not user-controlled)
usdm_sustainability_lag3	(internal lag feature; not user-controlled)
usdm_sustainability_roll6	(internal rolling feature; not user-controlled)
grace_groundwater_anomaly_lag1	Water Management And Recharge
grace_groundwater_anomaly_roll3	Water Management And Recharge
grace_groundwater_anomaly_roll6	Water Management And Recharge
streamflow_cfs	Colorado River And Surface Water Conditions
ndvi	Irrigated Land Use
month_cos	(temporal feature; not user-controlled)

No feature belongs to more than one attribute. Internal lag and rolling features remain derived automatically by the translation layer and are not directly adjustable by the public sliders.
Why This Structure Is Better

The previous architecture required additive feature accounting.

Example:

Groundwater could be influenced by:

    Recharge
    Climate
    Agriculture
    Conservation

The application would then need to explain how much each slider contributed.

Version 5 eliminates that complexity.

Each feature has one owner.

Users can immediately understand cause and effect.
Scenario Mode Behavior

Scenario modes act as presets.

Selecting a scenario:

    Sets all five attributes.
    Updates the projection.
    Leaves sliders editable.
    Allows users to create custom scenarios.

The selected scenario label remains visible even after manual adjustments.
Scenario Modes
Current Trajectory
Attribute	Value
Population And Development Pressure	45
Irrigated Land Use	45
Water Management And Recharge	45
Climate Stress	45
Colorado River And Surface Water Conditions	45

Description:

Continuation of existing trends.
Growth-First Buildout
Attribute	Value
Population And Development Pressure	85
Irrigated Land Use	70
Water Management And Recharge	35
Climate Stress	50
Colorado River And Surface Water Conditions	45

Description:

Rapid growth with relatively limited investment in recharge and efficiency improvements.
Water-Wise Buildout
Attribute	Value
Population And Development Pressure	55
Irrigated Land Use	25
Water Management And Recharge	85
Climate Stress	55
Colorado River And Surface Water Conditions	60

Description:

Growth continues while water demand is reduced and recharge efforts expand.
Drought Emergency Response
Attribute	Value
Population And Development Pressure	35
Irrigated Land Use	15
Water Management And Recharge	90
Climate Stress	90
Colorado River And Surface Water Conditions	20

Description:

Extreme drought conditions combined with aggressive mitigation efforts.
Translation Layer

The ONNX model remains unchanged.

The frontend stores:

policyAttributeValues

The translation layer converts those values into model features.

Conceptually:

Policy Attributes
↓
Translation Layer
↓
Model Features
↓
ONNX Model
↓
Projection

Implementation notes:

    The translation maps each attribute's 0–100 slider to the scaled input expected by the ONNX model (preserving the scaled-input contract).
    For residual models, the translation must supply the lag1 feature value when calling the ONNX model; the ONNX graph will add lag1 internally for residual outputs.
    Historical-range warnings should be computed with the same observed-window ranges used in modeling (final export window 2002-10 through 2020-12 for historical_sustainability.csv).

Effects Panel

The effects panel should display:

    Active scenario
    Attribute values
    Translated model-feature values
    Historical-range warnings

The panel no longer needs per-attribute delta accounting.

Include model performance context:

    Display CV metrics (R², MAE) for the exported model and the primary same-window baseline (lag1_persistence) so users understand model skill and typical error magnitude (e.g., mean MAE ≈ 3.72 points).

Advanced Mode

An advanced mode should remain available.

Advanced mode exposes:

    Raw model inputs
    Historical feature ranges
    Debug information

This mode is intended for instructors, reviewers, and developers.
About Page

The About page should answer three questions.
Where Does The Data Come From?

Provide descriptions of:

    MODIS NDVI
    MERRA-2 climate data
    SNOTEL snowpack
    USGS streamflow
    NASA GRACE
    Lake Powell operations
    Arizona demand datasets
    U.S. Drought Monitor

Also reference the key endpoint files in the repo (data/Final/*) and the modeling scripts (scripts/phase2.py, scripts/phase1_merra_temperature.py).
How Does The Model Work?

Explain:

    Monthly observations
    Feature assembly (including lags and rolling features)
    Residual-over-lag1 training formulation
    XGBoost regression and cross-validation (TimeSeriesSplit)
    ONNX export and browser-based inference
    Final exported model performance (Mean CV R² = 0.8964, Mean CV MAE = 3.7200) and baseline comparison

Use plain language.
Why Does This Matter?

Explain:

    Growth pressures
    Drought risks
    Recharge strategies
    Colorado River uncertainty

Connect the model to local planning decisions.
Important Framing

The model predicts an inverted U.S. Drought Monitor score.

It does not directly predict long-term sustainability.

The strongest defensible statement is:

    This tool helps compare water-management scenarios and highlights which factors appear most influential under different growth and drought conditions.

Avoid claims that the model proves a specific policy causes a specific outcome.
