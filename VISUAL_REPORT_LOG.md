# Visual Report Log

Date: 2026-04-28

## Goal

Record the current Phase 3 frontend state after the control-panel and aquifer cross-section work, including the later modularization pass, the GRACE slider swap, and the most recent scene-tuning changes.

## Current Status

Implemented and verified:

1. Step 1: Frontend workspace scaffold
2. Step 2: Browser-ready asset preparation
3. Step 3: Frozen frontend data contract and bundle hydration
4. Step 4: ONNX inference bridge
5. Step 5: Control panel implementation
6. Step 7: Aquifer cross-section implementation
7. Frontend simplification pass to reduce file count and dead paths
8. Cross-section refactor into focused scene controllers
9. Population slider replacement with `grace_groundwater_anomaly`
10. Score-driven scene mood pass for vegetation and lighting

Not completed yet:

- Step 8: real historical D3 time-series chart
- Step 9: projection engine
- Step 10: brush zoom and reset
- Step 11: point tooltips
- Step 12: audio behavior

## Verification

- `npm run build` succeeds in `frontend/`
- the build emits a static `frontend/dist/` bundle with copied model assets in `frontend/dist/model/`

## Simplified Frontend Structure

The frontend is still intentionally compact, but the cross-section has now been split into focused scene helpers so tuning work stays manageable:

```text
frontend/
  index.html
  package.json
  vite.config.js
  public/
    model/
  src/
    main.js
    runtime.js
    styles/app.css
    components/
      control-panel.js
      cross-section-atmosphere.js
      cross-section-irrigation.js
      cross-section.js
      cross-section-pipe.js
      cross-section-snow.js
      cross-section-temperature.js
      cross-section-waterline.js
      time-series.js
```

Removed during the simplification pass:

- the old `src/app/` bucket
- the unused gauge path
- the projection stub file
- the audio placeholder
- the tooltip placeholder
- the unused `Tone.js` dependency

This leaves one runtime module, one stylesheet, and a small set of scene-specific component files instead of a single oversized cross-section script.

## Runtime Model

The biggest structural change is that the frontend no longer spreads current behavior across separate loader, store, formatter, model-session, and projection-stub files.

`frontend/src/runtime.js` now owns:

- bundle loading
- CSV/JSON parsing and normalization
- lightweight app state
- ONNX session initialization
- feature-row assembly
- current-score recomputation after control changes
- shared formatting helpers used by the panels

This means most behavior changes now happen in one place instead of across several support files.

## Browser Asset Prep

The asset-prep script at `scripts/phase3_prepare_assets.py` still assembles the browser bundle in `frontend/public/model/`.

Generated artifacts include:

- `water_sustainability.onnx`
- `feature_names.json`
- `feature_stats.json`
- `historical_sustainability.csv`
- `historical_feature_vectors.csv`
- `projection_seed.json`
- `display_metadata.json`

The browser bundle now carries a seven-knob contract with `grace_groundwater_anomaly` in the UI control set instead of `population`.

Population support data still exists in the seed/derived metadata because the model path still carries `AZPOP_pct_change` internally, but population is no longer a user-facing slider.

## Control Panel

`frontend/src/components/control-panel.js` is live and metadata-driven.

Implemented controls:

- `powell_pool_elevation`
- `snow_water_equivalent_in`
- `precipitation_mm_day`
- `temperature_2m_c`
- `irrigation_total_withdrawal_mgd`
- `public_supply_groundwater_mgd`
- `grace_groundwater_anomaly`

Current behavior:

- renders slider cards from metadata
- shows live values and units
- uses p5/p95 bounds from the prepared bundle
- resets to the metadata default
- calls one shared runtime action for scenario changes
- starts the GRACE slider at `-0.077` so the SVG opens at the intended groundwater state

## Aquifer Cross-Section

`frontend/src/components/cross-section.js` is now the main live visualization panel.

The panel now acts as a coordinator for a set of focused scene helpers:

- `cross-section-waterline.js`
- `cross-section-atmosphere.js`
- `cross-section-snow.js`
- `cross-section-temperature.js`
- `cross-section-pipe.js`
- `cross-section-irrigation.js`

Core required encodings from Step 7 are still implemented:

- sky/background
- desert surface silhouette
- layered soil bands
- aquifer fill region
- water table line
- water-table position from a live GRACE/Powell blend
- aquifer color from sustainability score

Additional knob-driven scene encodings were also added:

- snowpack recolors the distant mountain snow cap from dusty brown to bright white
- precipitation darkens the sky, scales up the cloud banks, and can push the clouds all the way to black under heavy storms
- cloned off-screen cloud elements create a continuous cloud-bank effect across the SVG
- temperature affects mountain hue after snow is applied, while precipitation keeps priority over the sky
- Lake Powell and GRACE together raise and lower the water table
- irrigation and public-supply groundwater jointly drive the pipe fill and flow-arrow intensity
- water arrows now grow larger as combined withdrawal demand rises
- the final sustainability score pass makes vegetation greener and lighting richer at healthy scores, or browner and more blown-out/whispy at poor scores

Recent cleanup/simplification in the scene:

- below-water recoloring was removed because it was not needed
- rain-line overlays were removed
- irrigation no longer directly recolors the vegetation group
- population is no longer part of the scene logic

This makes the scene carry more of the scenario story without relying on the removed gauge, while keeping the moving parts easier to reason about.

## Time Series

`frontend/src/components/time-series.js` is still a summary card, not the final chart yet.

Current behavior:

- reports historical score row count
- reports model-complete feature-vector row count
- shows the loaded historical window
- shows the first projected month from the seed state

Step 8 work is still ahead:

- historical line rendering
- forecast line rendering
- chart interactions

## Current Interaction Model

The current UI loop is intentionally simple:

1. load the prepared browser bundle
2. initialize the ONNX model
3. render the controls and aquifer scene
4. recompute the current score whenever a control changes
5. update the cross-section from the new state

No separate projection engine is active yet. That will return later when Step 9 is implemented for real rather than as placeholder scaffolding.

## Practical Summary

The frontend is now smaller and easier to change:

- fewer files
- fewer placeholder modules
- one runtime path for current behavior
- one live visual focus: the cross-section

The next major complexity jump should happen only when Step 8 and Step 9 are built, rather than being carried prematurely in the file structure.

## SVG Integration Update

The aquifer panel now uses the authored `Aquafer_cross_section.svg` scene instead of drawing the landscape procedurally in JavaScript.

Implemented in this pass:

- cleaned the SVG ids used as runtime hooks so the scene is easier to target from code
- loaded the repo-level SVG into the frontend through Vite as a bundled asset
- replaced the old procedural cross-section drawing path with SVG group and clip-mask updates
- kept the live environmental mappings by driving snow, sky/rain, aquifer level, vegetation, pipe water, and flow arrows from the existing runtime state

The production build now emits the illustrated scene as a bundled asset in `frontend/dist/assets/`.

## Cross-Section Tuning Update

The illustrated aquifer scene has been tuned to make a few of the environmental mappings read more clearly.

Implemented in this pass:

- increased the visual lift from high Lake Powell values so the water table rises farther into the upper groundwater layer
- darkened the sky much more aggressively as precipitation increases, with a stronger storm overlay and darker rain clouds
- replaced the old snow clip-and-shrink behavior with a color interpolation that shifts the snow pieces from dusty brown to bright white based on the snow control value
- moved the major visual behaviors into dedicated cross-section helper files for waterline, atmosphere, snow, temperature, pipe demand, and irrigation normalization
- replaced the population slider with a live `grace_groundwater_anomaly` slider and fed that directly into the waterline controller
- set the startup GRACE state to `-0.077`
- made pipe arrows scale up as combined irrigation and public-supply demand rises
- added a final score-based mood pass that shifts vegetation and overall lighting quality based on sustainability health
