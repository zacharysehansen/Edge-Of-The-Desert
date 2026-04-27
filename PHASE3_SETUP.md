# Phase 3 Setup: Frontend Visualization

This phase builds the public-facing single-page web application for the project: a D3.js interface with a control panel on the left and a coordinated visualization panel on the right. The final app must run entirely offline from static files, using the exported ONNX model in the browser with no backend server.

All frontend application code should go in `frontend/`. Repo-level data/export helpers should go in `scripts/`. Browser-ready artifacts should be copied into `frontend/public/model/`.

The target interaction model is:

- overview first: gauge + aquifer cross-section + historical time series visible at load
- zoom and filter: sliders, brush zoom, reset view
- details on demand: click tooltips on time-series points
- coordinated multiple views: every slider change updates all panels together

---

## Step 0: Align The Current Artifacts With The Proposal

Do not skip this step. The current Phase 2 export is strong, but it does **not** yet match the final proposal one-to-one.

Current exported model facts:

- `model/water_sustainability.onnx` now uses a full `39`-feature input contract
- exported inputs now include `streamflow_cfs`, `snow_water_equivalent_in`, `powell_pool_elevation`, `precipitation_mm_day`, `grace_groundwater_anomaly`, `temperature_2m_c`, seasonal terms, and lag/rolling features
- the repo now includes final endpoint CSVs for `merra_precipitation.csv`, `merra_temperature_2m.csv`, and `grace_groundwater_anomaly.csv` in `data/Final/`
- the current exported model already includes `precipitation_mm_day`
- the current exported model already includes `grace_groundwater_anomaly`
- the current exported model already includes temperature-derived features, even though temperature is not one of the four proposal sliders
- these new endpoint CSVs are still not part of the browser asset bundle yet
- `model/historical_sustainability.csv` now reflects the finalized export window and currently covers `2002-10` through `2020-12`, not the full `2000-01` through `2023-12` exhibit timeline
- `model/historical_sustainability.csv` still does not contain the full feature vector needed for time-series point tooltips

Because of that, Phase 3 needs a short artifact-alignment pass before the UI is built.

Required alignment tasks:

1. Copy `data/Final/merra_precipitation.csv`, `data/Final/merra_temperature_2m.csv`, and `data/Final/grace_groundwater_anomaly.csv` into the final browser asset bundle.
2. Freeze how the non-slider model inputs behave during browser projection.
   - The proposal exposes four sliders: Powell pool elevation, snow water equivalent, streamflow, and precipitation.
   - The exported model also consumes temperature, NDVI, irrigation, public supply, population change, GRACE availability, and several lag/rolling features.
   - Decide which non-slider exogenous inputs stay fixed, which are seeded from the latest historical row, and which are recomputed during the autoregressive loop.
3. Keep GRACE as both:
   - a model-facing input already required by the ONNX contract
   - a visual support signal for the aquifer cross-section water-table placement
4. Export a `historical_feature_vectors.csv` file so each historical point can open a tooltip with the raw inputs and lag features that produced that score.
5. Decide how to handle the historical coverage gap between the finalized export and the exhibit target timeline.
   - The current exported historical score series covers `2002-10` through `2020-12`.
   - The proposal still describes a visible historical record through `2023-12`.
   - If you want exact full-vector tooltips and model-aligned history through `2023-12`, extend or backfill the required feature sources and rebuild the model artifacts first.

If the proposal must be implemented exactly as written, do the artifact-alignment work first and then freeze the final feature contract before writing the frontend.

The biggest remaining mismatch is no longer precipitation or GRACE. It is that the current ONNX model depends on more covariates than the four planned sliders expose.

---

## Step 1: Create The Frontend Workspace

Use a local-bundled frontend rather than CDN-loaded scripts so the exhibit can run offline.

Recommended stack:

- `Vite` for local development and static production builds
- `D3.js` for all charting and SVG rendering
- `onnxruntime-web` for in-browser inference
- `Tone.js` or the native Web Audio API for sonification
- plain JavaScript modules unless the repo later chooses TypeScript

Create:

```text
frontend/
  package.json
  vite.config.js
  index.html
  public/
    model/
    audio/
  src/
    main.js
    styles/app.css
    app/store.js
    app/data-loader.js
    app/model-session.js
    app/projection-engine.js
    app/formatters.js
    components/control-panel.js
    components/gauge.js
    components/cross-section.js
    components/time-series.js
    components/tooltip.js
    components/audio.js
```

Add package scripts:

- `dev`
- `build`
- `preview`

The final `npm run build` output must be a static folder that can be copied to a laptop and opened locally through a simple static server with no network connection.

---

## Step 2: Prepare Browser-Ready Assets

Create a repo-level helper such as `scripts/phase3_prepare_assets.py` that assembles everything the browser needs and copies it into `frontend/public/model/`.

Minimum required browser assets:

- `water_sustainability.onnx`
- `feature_names.json`
- `feature_stats.json`
- `historical_sustainability.csv`
- `historical_feature_vectors.csv`
- `projection_seed.json`
- `display_metadata.json`
- `grace_groundwater_anomaly.csv`
- `merra_precipitation.csv`
- `merra_temperature_2m.csv`

`display_metadata.json` should include:

- display labels
- units
- decimal precision
- slider eligibility
- slider min/max/default values
- threshold band labels for the gauge
- color ramp constants for the gauge and aquifer

`historical_feature_vectors.csv` should include, at minimum:

- `year_month`
- `usdm_sustainability`
- all slider-facing raw inputs
- all lag features the model actually consumes at that point
- any visual-only fields needed by the cross-section
- a flag indicating whether the row is historical-only or model-complete

`projection_seed.json` should include the latest full model state needed to start the autoregressive projection:

- date
- latest observed score
- lag-1 target
- lag-3 target
- rolling target summaries
- lagged exogenous inputs used by the exported model

Do not read directly from `model/` at runtime. Copy browser assets into `frontend/public/model/` so the frontend build is self-contained.

---

## Step 3: Freeze The Frontend Data Contract

Define one shared application state shape before writing components.

Recommended state slices:

- `modelReady`
- `historicalSeries`
- `historicalFeatureVectors`
- `projectionSeries`
- `sliderValues`
- `currentPrediction`
- `brushDomain`
- `selectedPoint`
- `audioEnabled`
- `animationStatus`

Keep a strict distinction between:

- slider-facing environmental forcings
- model-derived lag features
- visual-only support fields such as GRACE anomaly

This matters because the sliders should update only the user-facing forcings, while the autoregressive engine owns the lagged values.

---

## Step 4: Build The ONNX Inference Bridge

Load the ONNX model once at app startup and keep a single shared inference session alive for the whole page.

Requirements:

1. Read `feature_names.json` and construct the input tensor in exactly that order.
2. Convert every input row to `Float32Array`.
3. Run one warm-up inference at startup using the default median row.
4. Return a single clipped sustainability score on `0-100`.
5. Expose one function such as `predictScore(featureRow)`.

Important implementation note:

- The current exported winner is a residual-over-lag1 model whose ONNX graph already adds `lag1` back internally.
- The browser should pass the ordered feature row only.
- Do **not** add `lag1` a second time in JavaScript.

Also add a small browser-side smoke test:

- build the median feature row from `feature_stats.json`
- run inference once
- assert the result is finite and within `0-100`
- log a readable startup message in development mode

---

## Step 5: Implement The Control Panel

The control panel sits on the left and owns the four main sliders from the proposal:

- `powell_pool_elevation`
- `snow_water_equivalent_in`
- `streamflow_cfs`
- `precipitation_mm_day` shown to users as `precipitation`

Each slider should use:

- default = historical `median`
- minimum = historical `p5`
- maximum = historical `p95`

Each slider row should display:

- human-readable label
- unit
- live value
- reset-to-median control

Behavior:

1. Moving a slider updates shared state immediately.
2. The gauge and cross-section update on the same interaction frame.
3. The forward projection restarts from the current seed state.
4. The time-series amber forecast redraws from the new scenario.

If the exact model retrain has not happened yet, keep the UI contract stable and gate the precipitation slider behind the Step 0 artifact-alignment work rather than silently substituting a different feature in the final build.

---

## Step 6: Build The Arc Gauge

The arc gauge is the primary overview panel.

Requirements:

- domain `0-100`
- diverging color scheme from red through yellow to blue
- labeled threshold bands for severe, moderate, and healthy ranges
- animated needle
- numeric score label

Use D3 to draw:

- a background arc
- threshold band segments
- a foreground fill or highlight
- a rotating needle
- static labels at major ticks

Initial threshold defaults can be:

- `0-33` severe
- `34-66` moderate
- `67-100` healthy

Keep these thresholds in `display_metadata.json` so they can be tuned later without rewriting the component.

Every model call should update:

- needle angle
- active score text
- band/fill color

---

## Step 7: Build The Aquifer Cross-Section

This sits below the gauge and is the spatial intuition panel.

Render it as a custom SVG scene rather than a generic chart. Include:

- sky/background
- desert surface silhouette
- layered soil bands
- aquifer fill region
- water table line

Encodings:

- water table vertical position = normalized blend of GRACE groundwater anomaly and `powell_pool_elevation`
- soil saturation color = sustainability score
- optional subtle motion = shimmer/ripple only, not constant distracting animation

Recommended first-pass mapping:

- normalize GRACE and Powell values independently to `0-1`
- use a weighted blend such as `0.6 * GRACE + 0.4 * Powell`
- map that blend to the water table Y position

Color behavior:

- low score -> dry ochre / dusty tan
- mid score -> muted neutral
- high score -> deeper blue-grey

If GRACE remains visual-only, document that clearly in the code and in the writeup. If GRACE becomes model-facing later, keep the visual mapping unchanged unless the retrained model suggests a better interpretation.

---

## Step 8: Build The Historical Time Series

The time-series panel should use one shared X/Y coordinate system with two visual zones:

- left zone: historical record from `2000-01` through `2023-12`
- right zone: forward projection in amber

Historical styling:

- muted blue-grey line
- small points or invisible hit targets for interactions

Projection styling:

- amber line
- animated point growth one month at a time

Use a configurable horizon, with `24` months as the default forward projection length.

The chart should support:

- axes
- line paths
- hover or click hit areas
- projection animation timer
- brush zoom
- reset zoom button

Keep the historical series loaded even when the projection restarts so the user always has context.

---

## Step 9: Implement The Autoregressive Projection Engine

This is the piece that makes the interface feel alive.

The projection engine should:

1. Start from the latest valid seed state from `projection_seed.json`.
2. Use the current slider values as the non-lag environmental forcing.
3. Predict one month ahead.
4. Feed the new predicted sustainability score back into the target lag features.
5. Advance the calendar month.
6. Repeat every `1000-2000` ms until the horizon is reached.

For the current exported compact model, the engine must also update these derived inputs across the sequence:

- `month_sin`
- `month_cos`
- `usdm_sustainability_lag1`
- `usdm_sustainability_lag3`
- `usdm_sustainability_roll3`
- `usdm_sustainability_roll6`
- any lagged exogenous predictors present in the final export

Recommended lag update policy:

- on the first projected month, use the historical seed values for lagged fields
- on later projected months, update target lags from previous predictions
- for slider-controlled exogenous features with lagged versions, transition the lagged values toward the fixed scenario values as the forecast advances

When the user moves a slider mid-animation:

- stop the current timer
- keep the historical series intact
- treat the latest displayed point as the restart seed if projection has already begun
- regenerate the amber projection from that point

This restart behavior matches the proposal better than clearing the full chart and beginning again from the original baseline every time.

---

## Step 10: Add Brush Zoom And Reset

The time series needs a brushable zoom for drought-year inspection.

Requirements:

- draggable brush on the X axis
- rescale the line chart to the selected date window
- keep both historical and projected layers on the same updated scale
- add a visible reset control

The reset behavior should:

- restore the full extent
- preserve the current slider scenario
- preserve the current projection state

The brush is a view transform only. It should not mutate the underlying series data.

---

## Step 11: Add Point Tooltips And Details On Demand

Clicking a time-series point should open a tooltip with the exact inputs used for that point.

Tooltip contents should include:

- date
- sustainability score
- raw values for the four slider-facing inputs
- target lag features
- any lagged exogenous features used by the model at that step
- a badge for `historical` or `projected`

Interaction behavior:

- click a point to open
- click another point to replace
- click off-chart or leave the point to close

This feature depends on `historical_feature_vectors.csv` and on storing each projected point's generated feature row as the forecast grows.

For any historical dates that lack full feature coverage, either:

- extend the supporting data first, which is the preferred final-state path
- or visibly mark those points as score-only and omit the full-vector tooltip

Do not silently fabricate missing feature vectors for display-only months.

---

## Step 12: Add Sonification

Treat sonification as a progressive enhancement layered on top of the finished visuals, not as a dependency that blocks the core UI.

Recommended audio mappings:

- sustainability score -> pitch register or harmonic consonance
- rate of score change -> tempo / pulse rate
- sustained healthy conditions -> slower, more consonant textures
- sustained decline -> lower register, more dissonance, faster pulse

Implementation requirements:

- audio must be opt-in because browsers block autoplay
- expose a clear play/mute toggle
- audio state should follow the same shared projection state as the visuals

Asset strategy:

- if custom composed tracks become available, place them in `frontend/public/audio/`
- if not, use public-domain ambient material or fully synthesized sound

The visual system should remain fully usable if audio is disabled.

---

## Step 13: Style The Layout For Exhibit Use

The page should read clearly on a laptop in a public exhibit environment.

Layout requirements:

- left control column
- right visualization column
- responsive collapse on smaller screens
- obvious labels and units
- large enough chart marks for non-expert users

Practical requirements:

- visible loading state while the ONNX model initializes
- visible error state if model assets fail to load
- no network dependency after the build is produced
- fonts, scripts, and audio all served locally

Because this is an exhibit-style tool, prioritize clarity and stability over flashy transitions.

---

## Step 14: Verification Checklist

Before calling Phase 3 complete, verify all of the following:

1. The page loads from a static build with the network disconnected.
2. The ONNX model initializes and passes the browser smoke test.
3. Each slider updates the gauge, cross-section, and projection together.
4. The gauge color and needle match the current score.
5. The water table line moves when Powell or GRACE-driven inputs change.
6. The time series shows historical data in muted blue-grey and forecast data in amber.
7. The forecast appends one new month every `1-2` seconds.
8. Moving a slider mid-animation restarts the projection from the current state.
9. Brush zoom rescales correctly and reset restores full extent.
10. Clicking a point opens the expected tooltip and closing behavior works.
11. Audio can be turned on and off without affecting the visual pipeline.

Also verify the browser bundle on the actual presentation hardware, not just on the development machine.

---

## Step 15: Final Deliverables

Phase 3 is complete when the repo contains:

- a working `frontend/` application
- a reproducible asset-prep step
- a static build that runs fully offline
- a D3 control panel with the final slider set
- a live ONNX-backed gauge
- a live aquifer cross-section
- a historical + projected time series with brush zoom
- click tooltips with full feature vectors where data exists
- optional sonification that can be enabled in the browser

At that point, the project is ready for Milestone 4 feedback and final exhibit polish.
