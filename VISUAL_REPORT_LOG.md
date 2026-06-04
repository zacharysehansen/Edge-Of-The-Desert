# Visual Report Log

Date: 2026-05-06

## Goal

Record the current Phase 3 frontend status in terms of what is actually implemented and verified in the browser app.

## Current Status

Implemented and verified:

1. Frontend scaffold and browser-ready model bundle
2. Bundle hydration and ONNX inference bridge
3. Metadata-driven control panel with custom knob inputs
4. Illustrated aquifer cross-section with live scene mappings
5. Historical time-series panel with D3 rendering
6. Brush zoom, persistent zoom window, reset button, and double-click reset
7. Runtime split into:
   - `frontend/src/runtime.js` for live app state and ONNX orchestration
   - `frontend/src/runtime-utils.js` for parsing, hydration, formatting, projection helpers, and autoregressive stepping support
8. Browser-side autoregressive projection engine
9. Projection-point storage that keeps generated feature rows, feature-source metadata, and scenario control values with each projected month
10. Compact point-detail legend for both historical and projected points
11. Coordinated point selection so a selected time-series point updates:
   - the cross-section scene
   - the score-driven ambient audio
   - the control-panel knob positions
12. Shared selected-point and default-point state so inspection has an explicit fallback when no point is selected
13. Background sonification layer synced to the same shared runtime score state after the first user gesture
14. Viewport-filling visualization layout with centered knob-card rows and responsive control cards

Not completed yet:

- the original gauge panel from the broader visualization plan is still not mounted in the current app shell
- the point-detail layer is functional but still intentionally compact rather than a full explanation drawer
- the current UI still does not expose a visible audio toggle or an explicit out-of-distribution warning for unrealistic knob combinations

## Verification

- `npm run build` succeeds in `frontend/`
- the build emits a static `frontend/dist/` bundle with copied model assets in `frontend/dist/model/`
- runtime hydration currently preserves the intended July 2020 historical baseline and starts projection from August 2020
- projected points remain inspectable after they are drawn because the runtime stores their feature rows and scenario control values rather than only their scores

## Live UI Summary

### Controls

User-facing controls currently live:

- `powell_pool_elevation`
- `snow_water_equivalent_in`
- `precipitation_mm_day`
- `temperature_2m_c`
- `irrigation_total_withdrawal_mgd`
- `public_supply_groundwater_mgd`
- `grace_groundwater_anomaly`

Notes:

- GRACE is the seventh visible control in the current UI
- controls use bundle metadata for labels, units, p5/p95 bounds, and defaults
- controls render as custom knob cards instead of horizontal sliders
- each knob card shows min/max values on either side of the dial, the live value below the knob, and a reset button beneath that value
- the knob deck is centered into a `4 + 3` row layout on desktop
- control changes restart the projection run from the shared runtime baseline
- selecting a historical or projected point repositions the knobs to that point's values when those values are available
- editing a knob clears point inspection and returns the app to live scenario authoring so the controls do not snap back to an inspected month

### Layout

The app shell treats the page as two stacked bands rather than a control column beside the visuals.

Current behavior:

- the `viz-panel` occupies the flexible top row and stretches to consume the remaining viewport height
- the control panel sits in a dedicated lower row so the knobs remain visible without scrolling on a typical desktop screen
- both visualization cards grow to fill the available height in that top band
- shorter or narrower viewports fall back to aspect-ratio-driven chart sizing instead of forcing the viewport-fit layout

### Cross-Section

The cross-section is the main finished visual.

Live mappings include:

- water table from GRACE + Lake Powell
- sky darkening and storm behavior from precipitation
- mountain and snow response from temperature and snowpack
- pipe fill and arrows from combined irrigation + public-supply demand
- vegetation / lighting mood from sustainability score

Current score / selection behavior:

- the cross-section score chip uses the newest projected forecast month when the app is in live scenario mode
- selecting a historical or projected point temporarily turns the cross-section into a view of that specific month
- if no point is selected, the cross-section falls back to the runtime's default live scenario behavior
- the sustainability score chip sits centered below the SVG instead of inside the upper-right corner of the scene
- the scene renders at full panel height and uses centered in-panel loading / error messages rather than a separate status strip below the illustration

### Time Series

The time-series panel is now both the historical/projection chart and the main inspection surface.

Current behavior:

- renders the historical line from `historical_sustainability.csv` on a shared `0-100` y-scale
- renders the projection line in amber with projected points added one month at a time
- animates the projection segment as the autoregressive forecast grows
- supports click selection and active-point highlighting
- supports a compact point legend for both historical and projected points
- supports brush zoom on a persistent lower overview rail
- supports widening the zoom window by dragging brush handles
- supports reset via button or double-click
- keeps the historical series visible when the projection restarts so the user always retains context
- places the historical / projection legend in the upper-right corner of the chart container instead of in a panel header
- uses larger, bolder chart typography so axes and chart labels stay readable at the larger panel size
- uses a compact footer row for reset, point-status text, and the unlimited-forecast toggle so more of the card height goes to the chart itself
- stretches to fill the available visualization band on desktop, with aspect-ratio fallback on smaller screens

Important interaction behavior:

- clicking a historical point opens a compact detail legend built from `historical_feature_vectors.csv`
- clicking a projected point opens a matching legend built from the stored projected `featureRow` and `scenarioControls`
- the selected point updates the cross-section, the audio layer, and the knob readouts to that month's conditions
- clicking empty chart space clears the selection and returns the app to live scenario mode
- the historical score artifact window runs through `2020-12`
- the projection engine intentionally anchors on the July 2020 baseline and begins forecasting at `2020-08`
- this means the amber projection is being compared against a longer historical record that continues beyond the forecast anchor

### Audio

The audio layer is now present as a background behavior rather than a visible control.

Current behavior:

- uses the Tone-based module in `frontend/src/components/components.js`
- enables ambient audio automatically after the first pointer or keyboard interaction so it stays within browser autoplay rules
- maps the sustainability score into three broad musical bands (`severe`, `moderate`, `healthy`)
- changes the drone chord set, melody scale/register, melody pacing, and filtered noise intensity based on score
- updates from the same shared runtime state used by the visual views
- follows the selected point while the user is inspecting history or projection
- falls back to the live scenario score when no point is selected
- does not expose a dedicated audio toggle in the current UI

## Historical Browser Assets

The two historical CSVs have clearly different jobs in the app.

`historical_sustainability.csv` is the skinny score-series artifact. It provides:

- `year_month`
- model-predicted sustainability score
- actual sustainability score
- residual

In the frontend, this file is used to build the plotted historical score series and the historical half of the comparison chart.

`historical_feature_vectors.csv` is the wide monthly model-state artifact. It provides:

- knob-facing raw inputs
- lagged model inputs
- calendar / derived fields
- visual-support fields used by the projection context
- row status such as `model_complete` vs `historical_only`

In the frontend, this file is used to:

- build the July 2020 historical baseline state when available
- derive historical exogenous histories for the autoregressive engine
- populate the compact detail legend for historical point inspection
- supply historical control values so the knob deck can mirror inspected months

## Runtime / Projection Engine

The runtime is no longer only prepared for projection. It now runs the projection and coordinates the interactive views.

Current projection behavior:

1. Load `display_metadata.json`, `feature_names.json`, `feature_stats.json`, `projection_seed.json`, `historical_sustainability.csv`, and `historical_feature_vectors.csv`
2. Hydrate shared state and derive control definitions, historical series, historical feature vectors, and projection context
3. Build `projectionConnections` describing control mappings, lag / rolling dependencies, fixed raw features, and history requirements
4. Build an initial `projectionPreparation` object with the next forecast month and a generated `pendingFeatureRow`
5. Run ONNX inference for one month ahead
6. Append a projected point to `projectionSeries`
7. Feed the predicted score back into target history and update lagged / rolling inputs for the next month
8. Repeat on a timer until the configured horizon is reached

Current state / interaction details:

- `projectionSeries` stores the live forecast months shown in the amber line
- each projected point stores:
  - `featureRow`
  - `featureSources`
  - `scenarioControls`
- `projectionPreparation` stores:
  - `nextProjectionMonth`
  - `targetHistory`
  - `exogenousHistory`
  - `pendingFeatureRow`
  - `pendingFeatureSources`
  - `latestScenarioScore`
- `selectedPoint` stores the user's current inspection target as a lightweight point reference
- `defaultPoint` stores the no-selection fallback reference used by coordinated views
- moving a control mid-run clears the current selection, restarts the forecast from the shared baseline, and rebuilds the live projection from the new scenario values

## Remaining Work

### Main Gaps

Still needed for a more finished public-facing app:

- a richer point explanation layer that can expand beyond the compact legend into actual vs predicted score, residual, and a broader slice of the model state
- the still-unmounted gauge panel from the broader design plan
- a visible audio toggle and better user-facing communication of what the sound means
- an out-of-distribution or scenario-validity cue for unrealistic knob combinations

## Summary

The frontend has moved well past the scaffold phase. It now has a working browser-side ONNX runtime, a live autoregressive projection engine, a historical-plus-forecast D3 time series, a cross-section that can operate in both live-scenario and inspected-point modes, a compact point-detail layer for both historical and projected months, a coordinated knob deck that mirrors inspected values, and a background sonification layer tied to the same shared runtime state. The main remaining gaps are richer explanation and packaging details, not the core interactive forecast loop itself.
