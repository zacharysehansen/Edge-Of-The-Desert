# Visual Report Log

Date: 2026-04-29

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
9. Projection-point storage that keeps generated feature rows and feature-source metadata with each projected month
10. Cross-section score behavior fixed so it now advances with each newly predicted forecast month instead of freezing on month 1
11. Viewport-filling visualization layout with centered knob-card rows and larger responsive control cards
12. Background sonification layer synced to the live scenario score after the first user gesture

Not completed yet:

- point tooltip / detail UI built from `historical_feature_vectors.csv`
- the original gauge panel from the broader visualization plan is still not mounted in the current app shell

## Verification

- `npm run build` succeeds in `frontend/`
- the build emits a static `frontend/dist/` bundle with copied model assets in `frontend/dist/model/`
- runtime hydration currently preserves the intended July 2020 historical baseline and starts projection from August 2020

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
- controls now render as custom knob cards instead of horizontal sliders
- each knob card shows min/max values on either side of the dial, the live value below the knob, and a reset button beneath that value
- the knob deck is now centered into a `4 + 3` row layout on desktop
- desktop knob cards currently render at `400px` wide, with a narrower tablet fallback
- control changes restart the projection run from the shared runtime state

### Layout

The app shell now treats the page as two stacked bands rather than a control column beside the visuals.

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

Current score behavior:

- the cross-section score chip uses the latest projected forecast month, not the first projected month
- the display stays pending until a real projected score exists
- the cross-section no longer falls back to a smoke-test score or to the historical July 2020 anchor as if that were the live scenario output
- the sustainability score chip now sits centered below the SVG instead of inside the upper-right corner of the scene
- the scene now renders at full panel height and uses centered in-panel loading / error messages rather than a separate status strip below the illustration

### Time Series

The time-series panel is now a live historical-plus-projection view rather than a scaffold.

Current behavior:

- renders the historical line from `historical_sustainability.csv` on a shared `0-100` y-scale
- renders the projection line in amber with projected points added one month at a time
- animates the projection segment as the autoregressive forecast grows
- supports click selection and active-point highlighting
- supports brush zoom on a persistent lower overview rail
- supports widening the zoom window by dragging brush handles
- supports reset via button or double-click
- keeps the historical series visible when the projection restarts so the user always retains context
- places the historical / projection legend in the upper-right corner of the `time-series-svg` container instead of in the panel header
- removes the old in-SVG `Historical` and `Projection` text labels now that the overlay legend carries that job
- uses larger, bolder chart typography so axes and chart labels stay readable at the larger panel size
- uses a compact footer row for reset, point-status text, and the unlimited-forecast toggle so more of the card height goes to the chart itself
- keeps the Reset Zoom and Unlimited Forecast buttons slightly enlarged for readability
- stretches to fill the available visualization band on desktop, with aspect-ratio fallback on smaller screens

Important current behavior:

- the historical score artifact window runs through `2020-12`
- the projection engine intentionally anchors on the July 2020 baseline and begins forecasting at `2020-08`
- this means the amber projection is being compared against a longer historical record that continues beyond the forecast anchor

### Audio

The audio layer is now present as a background behavior rather than a visible control.

Current behavior:

- uses the Tone-based module in `frontend/src/components/components.js`
- enables ambient audio automatically after the first pointer or keyboard interaction so it stays within browser autoplay rules
- keeps the audio state in shared runtime state and updates the sonification from the latest live scenario score
- does not expose a dedicated audio toggle in the current UI

## Historical Browser Assets

The two historical CSVs now have clearly different jobs in the app.

`historical_sustainability.csv` is the skinny score-series artifact. It provides:

- `year_month`
- model-predicted sustainability score
- actual sustainability score
- residual

In the frontend, this file is used to build the historical plotted score series and summary counts.

`historical_feature_vectors.csv` is the wide monthly model-state artifact. It provides:

- knob-facing raw inputs
- lagged model inputs
- calendar / derived fields
- visual-support fields used by the projection context
- row status such as `model_complete` vs `historical_only`

In the frontend, this file is used to:

- build the July 2020 historical baseline state when available
- derive historical exogenous histories for the autoregressive engine
- count model-complete rows in the time-series summary

What it does not do yet:

- it is not yet surfaced through a point-click tooltip or detail panel for historical months

## Runtime / Projection Engine

The runtime is no longer only prepared for projection. It now runs the projection.

Current projection behavior:

1. Load `display_metadata.json`, `feature_names.json`, `feature_stats.json`, `projection_seed.json`, `historical_sustainability.csv`, and `historical_feature_vectors.csv`
2. Hydrate shared state and derive control definitions, historical series, historical feature vectors, and projection context
3. Build `projectionConnections` describing control mappings, lag / rolling dependencies, fixed raw features, and history requirements
4. Build an initial `projectionPreparation` object with the next forecast month and a generated `pendingFeatureRow`
5. Run ONNX inference for one month ahead
6. Append a projected point to `projectionSeries`
7. Feed the predicted score back into target history and update lagged / rolling inputs for the next month
8. Repeat on a timer until the configured horizon is reached

Current state / data-model details:

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
- moving a control mid-run clears the current projection and starts a fresh forecast from the shared runtime baseline

## Remaining Work

### Step 11

Still needed for the time-series detail layer:

- click-to-open historical / projected point detail UI
- rendering of raw inputs, lag features, and source metadata per point
- visible distinction between score-only historical months and model-complete months when point details are shown

## Summary

The frontend has moved past the scaffold phase. It now has a working browser-side ONNX runtime, a live autoregressive projection engine, a historical-plus-forecast D3 time series, a cross-section that updates off the newest projected forecast month, a background sonification layer tied to the same shared runtime state, and a viewport-aware layout that gives the two main visuals as much screen space as possible while keeping the knob deck accessible. The main remaining gaps are richer point-detail interaction and the still-unmounted gauge panel, not the core forecast loop itself.
