# Final Report Structure: Southwest Water Sustainability Visualizer

---

## Abstract

A short paragraph (150-200 words) placed before the introduction. Cover:
- The problem: water scarcity in the American Southwest is a growing concern, and the data describing it is largely inaccessible to non-expert audiences
- What was built: an interactive browser-based visualization combining a trained XGBoost regression model exported to ONNX, a D3.js frontend, a real-time autoregressive projection engine, coordinated point inspection between the time series and cross-section, and ambient sonification tied to the same score state
- The core design premise: the audience is laypeople, and the goal is to build curiosity rather than deliver precise scientific findings
- Briefly name the larger interaction idea: the system works in both scenario-simulation mode and point-inspection mode, letting a user either author a hypothetical future or inspect a specific historical/projected month
- Brief mention of key quantitative results (R² = 0.896, MAE = 3.72) to anchor the reader before they reach the methods

---

## 1. Introduction

Rewrite the introduction to describe the project in its finished form rather than as a proposal. Structure it around three ideas:

### 1.1 Motivation

- Water sustainability in the American Southwest, particularly Arizona, is shaped by the interaction of climate signals (precipitation, temperature, snowpack) and human demand (irrigation, municipal supply, reservoir management). These dynamics are well-studied in hydrology literature but rarely communicated to general audiences in an interactive form.
- The driving question is not "can we predict a sustainability score" but "can we build something that invites a layperson to ask better questions about water." Establish that framing early because every design decision downstream flows from it.

### 1.2 Project Scope

- The system is Arizona-focused in its Phase 1 data assembly, though framed under a broader Southwest lens. Note here, early, that the cross-section illustration is stylistically Southern Arizona and Sonoran Desert in character. This is intentional for visual legibility but creates a mismatch with the statewide data that the model actually uses. Flag it honestly as a design choice that has audience implications.
- The project covers data from 2000 through 2020 (the final export window ends at 2020-12 due to the intersection of available source coverage windows).

### 1.3 Contribution

- The main contribution is not a new hydrological model. It is an interactive visualization system that embeds a pre-trained ML model directly in the browser via ONNX and couples it to a live autoregressive projection engine, an illustrated cross-section scene, a historical time series, a point-selection detail layer, and a score-driven audio layer. Users manipulate environmental and demand conditions and see both how those conditions propagate forward in time and where they fall relative to real historical conditions.
- A second contribution is the shared-state interaction model: the same selected month can drive the chart detail card, cross-section scene, control readouts, and sonification. This makes the project a coordinated multimodal system rather than a chart with decorative side panels.
- Distinguish this from prior work: existing systems in this space either run models as offline backend pipelines, display pre-computed results, or lack interactive visualization altogether.

---

## 2. Background and Related Work

This section should read as a short literature review, not a reading list. Use the citations and explain what each does and, critically, what gap it leaves that this project addresses.

### 2.1 Machine Learning for Drought and Water Sustainability Prediction

- **AGU 2022WR033847**: Uses ML for water-related prediction but runs as a backend practitioner pipeline with no interactive visual component. The model output is not accessible without running the code directly.
- **MDPI Remote Sensing 2072-4292/15/4/873**: Purely a remote sensing and hydrology contribution. No visualization tool is built. Results live in figures inside the paper.
- **ScienceDirect S0048969723041323**: Builds a reproducible and explainable ML pipeline to predict drought impacts, and analyzes feature importance for scientific interpretation. The audience is researchers. The output is not interactive.
- **MDPI Hydrology 2306-5338/11/5/66**: No visualization component at all. Establishes that this is a common pattern in the literature: strong modeling work with no path to public-facing engagement.

### 2.2 Interactive Environmental Visualization

- **Frontiers fenvs.2025.1564670**: The closest antecedent to this project in terms of building an app that shows environmental results. However, the model runs offline and the app displays pre-computed results. There is no ONNX export and no live browser inference. Users are viewing a static scenario, not exploring a live one.
- Identify the gap: none of the reviewed systems combine (a) real multi-source hydrological data, (b) a trained ML model running live in the browser, (c) an autoregressive projection engine responding to user input, and (d) an illustrated visual designed for a non-expert audience.

---

## 3. Methods

This is the largest section. Write it as a description of what was done, not what was planned. Organize it by phase.

### 3.1 Data Assembly

Describe each data source, its raw format, the temporal resolution it was acquired at, and how it was reduced to a monthly endpoint. Use a table matching the one in the README:

| Feature | Source | Raw Resolution | Model Resolution |
|---|---|---|---|
| NDVI | MODIS MOD13A3 | Monthly | Monthly |
| 2m Temperature | MERRA-2 | Daily | Monthly avg |
| Specific Humidity | MERRA-2 | Daily | Monthly avg |
| Precipitation | MERRA-2 | Daily | Monthly avg |
| Snow Water Equivalent | NRCS / WRCC SNOTEL | Daily | Monthly endpoint |
| Streamflow | USGS NWIS | Daily | Monthly endpoint |
| GRACE Groundwater Anomaly | NASA GRACE | Monthly | Monthly |
| GRACE/FO Recharge Estimate | NASA GRACE / GRACE-FO | Monthly | Monthly |
| Lake Powell Operations | Manual Powell bundle | Daily / irregular | Monthly endpoint |
| Irrigation Total Withdrawal | USGS Arizona HUC12 | Monthly | Monthly statewide endpoint |
| Public-Supply Total | NWAA Arizona HUC12 | Monthly | Monthly statewide endpoint |
| Population | Arizona AZPOP observations | Annual | Monthly endpoint via interpolation |
| Inverted USDM Score | USDM | Weekly | Monthly endpoint (target) |

For each source, note any coverage gaps:
- SNOTEL SWE ends at 2018-07
- Irrigation, public supply, Powell, and population end at 2020-12
- MODIS NDVI, USGS streamflow, and USDM extend to 2023-12

Explain how these differing coverage windows determined the final model export window of 2002-10 through 2020-12. The GRACE mission does not begin until 2002, and the lagged/rolling features built from GRACE need enough observed history before they become meaningful. Starting the model window at 2002-10 means every GRACE-derived feature in the training set is drawn from real observed values rather than imputed carryover.

Describe the join key (`year_month`) and the final flat CSV structure. Note that the target variable is `usdm_sustainability`, derived from the U.S. Drought Monitor DSCI as `100 - (dsci / 5)`, inverted so that 100 is maximum sustainability and 0 is crisis. Explain why this inversion is used: it makes the direction of the visual encodings intuitive without needing to explain to a viewer that a higher drought index means worse conditions.

### 3.2 Feature Engineering

Describe the engineered features added beyond the raw source columns:
- Lag 1, 3, 6 month features for USDM sustainability, precipitation, GRACE anomaly, and temperature
- 3-month and 6-month rolling means for the same variables
- Month sine/cosine encoding to give the model access to seasonal position without treating December and January as far apart numerically
- Temperature anomaly relative to a rolling baseline
- The residual target formulation: the model predicts `usdm_sustainability - usdm_sustainability_lag1` rather than the raw score. At prediction time, `lag1` is added back. This turned out to be the key structural choice that allowed the model to beat the lag-1 persistence baseline.

Total final feature count: 37.

### 3.3 Modeling

Describe the modeling choices and justify them relative to the alternatives:
- XGBoost over linear regression because the relationships between atmospheric, hydrological, and human demand signals are nonlinear and interact. A linear model would not capture the compounding effect of simultaneous drought in precipitation, snowpack, and groundwater.
- XGBoost over a neural network because the dataset is small (219 rows in the final export window). Tree-based ensembles outperform neural networks on small tabular datasets and train in seconds.
- `TimeSeriesSplit` with 5-6 folds and an expanding training window. This ensures every prediction is always forward in time, matching the actual deployment scenario.

Describe the iterative experiment structure at a high level:
- Initial 22-feature model had mean CV R² = 0.157, indicating data and structural issues had not yet been resolved
- Adding the residual formulation was the first step that produced a model beating the lag-1 persistence baseline
- Adding precipitation and GRACE as features pushed CV R² above 0.86
- Restricting the export window to the observed GRACE period (starting 2002-10) further improved stability and score to 0.8964
- Temperature added modest but real gain

Describe the final hyperparameters and how they were selected (randomized search followed by focused grid refinement):

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 800 |
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.7 |
| `min_child_weight` | 1 |
| `reg_alpha` | 0.05 |
| `reg_lambda` | 1 |

### 3.4 ONNX Export and Browser Inference

The Open Neural Network Exchange (ONNX) format allows a model trained in Python with scikit-learn or XGBoost to be serialized and then loaded in the browser using `onnxruntime-web` on WebAssembly. This removes the need for a backend server and makes the visualization fully self-contained as a static build. For a visualization intended to run in a classroom or be shared as a link, no-server deployment is a strong practical advantage over architectures that require a running inference service.

Describe the export pipeline:
- `skl2onnx` converts the fitted scikit-learn-compatible pipeline to an ONNX graph
- For residual models, the ONNX graph adds `lag1` back internally so the browser-facing input contract is the same as for non-residual models
- `feature_stats.json` stores per-feature medians and p5/p95 bounds used to initialize the sliders and define their ranges
- `feature_names.json` stores the ordered feature list for input tensor construction
- `onnxruntime-web` loads the ONNX graph in a WebAssembly runtime, meaning inference runs client-side with no server required

### 3.5 Autoregressive Projection Engine

The engine anchors on July 2020, the last month of the historical feature vector file, as its baseline state. Each projected month builds a new feature row by:

1. Reading the user-controlled knob values for exogenous inputs (precipitation, temperature, etc.)
2. Computing lag and rolling values from the growing history of projected scores
3. Assembling the full 37-feature input tensor
4. Running ONNX inference on that tensor
5. Appending the predicted score to the projection series and feeding it back into the target history
6. Repeating on a timer until the configured forecast horizon is reached

When a user moves a knob mid-run, the projection clears and restarts from the same July 2020 baseline, so the amber line always reflects the current knob configuration rather than a mix of old and new inputs.

Also note one important state-model detail: projected points store the generated feature row and the scenario control values used to create that month. This enables the frontend to inspect projected months after they are drawn, rather than treating the amber line as a purely visual animation with no recoverable underlying data.

One more important interaction detail belongs here because it is part of the runtime model, not only the UI: the runtime distinguishes between a live scenario state and a selected point state. When no point is selected, the app behaves as a scenario simulator. When a historical or projected point is selected, the cross-section, sonification, and control readouts temporarily become views onto that month instead.

### 3.6 Frontend Design

This subsection should carry the most detail because the course is a visualization course. All major design decisions connect back to the core audience premise: the user is a non-expert, and the goal is to generate curiosity and orient them toward asking better questions, not to deliver a precise hydrological forecast.

**Layout**

Two stacked bands: a visualization row at the top consuming most of the viewport height, and a control row below that keeps the knobs accessible without scrolling. Two visualization cards sit side by side in the top band: the cross-section on the left, the time series on the right. Shorter or narrower viewports fall back to aspect-ratio-driven sizing.

**Control Panel**

Sliders were replaced with custom rotary knob cards for the final UI. Rotary knobs communicate continuous analog control and are more visually engaging for an audience meant to feel like they are operating something. Seven user-facing controls cover the major conceptual categories: surface water (Powell elevation), snow (SWE), climate (precipitation, temperature), human demand (irrigation withdrawal, public supply), and groundwater state (GRACE anomaly). Non-intuitive variables are given plain-language labels. Each knob is initialized to the historical median for that feature so a first-time user starts in a "normal" conditions state.

The knobs also serve a second role beyond authoring scenarios: when a point is selected in the chart, the knobs reposition to that month's values when those values are available. This turns the control panel into a readout for inspection as well as an input surface for simulation. When the user edits a knob, the app exits inspection mode and returns to live scenario authoring.

**Cross-Section**

The cross-section is the centerpiece of the visual and is designed to communicate sustainability intuitively rather than precisely. Live scene mappings:
- Water table level from GRACE anomaly and Lake Powell elevation
- Sky darkening and storm behavior from precipitation
- Mountain snow cap from temperature and SWE
- Pipe fill and arrow direction from combined irrigation and public-supply demand
- Vegetation and lighting mood from the sustainability score

The illustration is visually Sonoran Desert in character, which matches Southern Arizona and Tucson. The data is statewide Arizona. This is a deliberate design choice made for visual legibility: a cross-section trying to represent all of Arizona at once would be visually incoherent. But it creates a real bias risk: a viewer in Flagstaff or the White Mountains would recognize that this scene does not look like their experience of Arizona, and a viewer anywhere might wrongly infer that the water levels shown correspond to a specific location. This Sonoran/Tucson character also means the visual resonates most naturally with the audience most likely to encounter it in this academic context.

**Time Series**

A D3 line chart of the historical sustainability score from 2002 through 2020, with an amber projection line growing from August 2020 forward as the autoregressive engine runs. The historical line provides grounding context: a user can see that their scenario of "no precipitation and high temperatures" produces scores comparable to the 2002 or 2012 drought years. Brush zoom on a lower overview rail lets users explore specific periods without losing the full-range context. Amber for the projection line distinguishes it visually from the historical record while keeping both readable on the same axis.

The final time-series design is not only a plot but also the primary inspection surface. Clicking either a historical or projected point opens a compact detail card that lists the score, the relevant control value at that point, and the model-linked features associated with that control. The selected point also updates the cross-section scene, the control readouts, and the audio layer to that month's conditions. This coordinated-view behavior matters for the report because it changes the interpretation of the illustration: the cross-section is not only a live "latest forecast" display, but also a way to visually inspect specific historical or projected months.

**Audio**

Ambient sonification tied to the sustainability score enables passively after the first user gesture, staying within browser autoplay rules. The purpose is attention retention: a visualization that produces sound as you move a knob creates a sense of agency and keeps a casual viewer engaged longer than a purely visual one.

It is worth describing this as a real design layer, not just a flourish. The sonification maps score bands into different harmonic palettes, melodic registers, pacing, and background noise intensity. Lower scores produce darker, noisier, more urgent textures; higher scores produce brighter, cleaner, more stable ones. Because the audio is driven by the same runtime state as the visuals, it also participates in coordinated inspection: selecting a point in the chart updates not only what the user sees, but what they hear.

**Coordinated Views and Interaction Modes**

One high-level concept worth naming explicitly in the report is that the interface operates in two modes. In scenario-simulation mode, the knobs are inputs and the cross-section/audio follow the newest projected month. In point-inspection mode, the chart becomes the driver and the other views temporarily reconfigure around the selected historical or projected point. This is an important design idea because it explains why the app feels like one system rather than a collection of independent widgets.

---

## 4. Results and Evaluation

### 4.1 Model Performance

Present the cross-validation results in a table. Rows should show the progression from the initial model through the final one, including the lag-1 persistence baseline:

| Model | Mean CV R² | Std R² | Mean CV MAE | Std CV MAE |
|---|---|---|---|---|
| lag1_persistence (baseline) | 0.7900 | — | 5.1649 | — |
| Initial XGBoost (22 features) | 0.1574 | 0.6045 | 9.9454 | — |
| XGBoost (compact, residual over lag1) | 0.8226 | — | 4.6977 | — |
| XGBoost (compact + precip/GRACE, residual) | 0.8637 | — | 4.0702 | — |
| XGBoost (all features, residual, focused search) | 0.8843 | — | 3.7769 | — |
| **XGBoost (all features, residual, GRACE window)** | **0.8964** | **0.0238** | **3.7200** | **1.3363** |

Include the feature importance plot from `model/feature_importance.png` and discuss what it reveals. Top features by gain:

1. `precipitation_mm_day_lag1`
2. `usdm_sustainability_lag3`
3. `precipitation_mm_day_roll3`
4. `temperature_2m_c_lag1`
5. `temperature_2m_c_anomaly`
6. `ndvi`
7. `temperature_2m_c_anomaly_lag1`
8. `usdm_sustainability_roll6`
9. `precipitation_mm_day`
10. `month_cos`

Also notable: `grace_groundwater_anomaly`, `streamflow_cfs`, `grace_groundwater_anomaly_roll6`, `grace_groundwater_anomaly_lag1`.

Add the interpretation note: R² = 0.896 does not mean the model is "89.6% accurate." It means the model explains about 89.6% of the variance in the target relative to a naive mean-only baseline. The MAE of 3.72 points (on a 0-100 scale) is the operationally meaningful error metric. If the true score were 60, a typical prediction would land somewhere around 56 to 64.

Discuss overfitting honestly: train R² is essentially 1.0. The held-out CV results remain strong and stable (R² std = 0.024 across folds), so the conclusion is not that there is no overfitting but that generalization remains good enough for the visualization use case. Also note selection optimism: many experiments were run, so the final reported score is likely a few tenths more optimistic than a never-retuned one-shot evaluation would produce.

### 4.2 Visualization Evaluation

Use screenshots or recorded interactions to walk through each visual element.

**Cross-Section**

Show how it responds to different knob combinations. A high-precipitation, low-temperature, low-demand scenario should produce a high water table, green vegetation, blue fill, and a strong sustainability score. A low-precipitation, high-temperature, high-demand scenario should produce a dry scene with a falling water table and a visibly weaker sustainability state. Describe what each of these visual transitions communicates to a viewer and why that mapping was chosen.

**Time Series**

Show a screenshot of the projection line against the historical record. Point out how the amber line for a drought scenario falls into the range of the 2002-2012 historical low period. This is the key evaluative moment: the user's hypothetical gets calibrated against reality.

**Point Inspection and Coordinated Views**

Add one screenshot or figure sequence showing a selected point in the time series and the cross-section updating to match that point's conditions. This is worth evaluating explicitly because it is one of the clearest places where the visualization behaves like a system rather than a set of separate charts. Explain that when no point is selected, the cross-section follows the newest projected month; when a point is selected, it temporarily becomes a view onto that historical or projected moment instead.

Also mention that the knob deck and the sonification follow that same selected point. This broadens the evaluation from "linked chart plus image" to "coordinated multimodal inspection."

**Sonification**

Evaluate the sound layer on its own terms. The goal is not precise quantitative decoding from audio alone. The goal is to reinforce directionality and sustain attention. Describe whether the score-band transitions are perceptible, whether the sound helps the app feel alive while the projection grows, and whether it supports the lay-audience goal better than silence would.

**Projection Engine**

Describe the autoregressive behavior qualitatively. A user who holds all knobs at the dry extreme will see the score drop progressively month over month as lagged drought signals compound. A user who moves back to wet conditions mid-projection will see the score recover over the next several months, not instantly, because the lag features still carry memory of the preceding dry period. This is the most ecologically honest behavior the system produces.

### 4.3 Limitations

**Geographic Representation**

The statewide Arizona data is summarized to a single monthly observation per feature. The cross-section illustration is visually Southern Arizona. The combination means the visualization speaks most directly to a Tucson audience and risks implying geographic specificity that the underlying data does not support. A user in northern Arizona, or a viewer unfamiliar with Arizona geography, may read the Saguaro cactus and flat desert floor as representing their local water conditions when the model is actually describing a statewide aggregate.

**Predictive Model Input Realism**

The knobs allow a user to set conditions the model was never trained on. A user can set monthly precipitation to its 95th percentile at the same time as temperature is at its 95th percentile. These conditions cannot co-occur in the real world the way they appear independently in the training data. The model will produce a number, but that number is extrapolating beyond the joint distribution of the training data. For a scientific tool, this would be a serious validity concern. For this visualization, it is a design risk: a user might interpret the output as a real prediction rather than an illustrative projection from a model that has not seen those conditions. Adding a visible out-of-distribution indicator would reduce this confusion.

**Projection Horizon and Anchoring**

The projection anchors on July 2020 regardless of when the user opens the visualization. Every forecast scenario projects forward from that fixed point. The amber line always begins in mid-2020, which is now more than five years in the past. A user may reasonably ask why this starts in 2020, and the answer requires understanding data coverage constraints that are not communicated in the UI.

**Detail-Layer Scope**

The point-inspection layer is now functional, but it is intentionally selective rather than exhaustive. It surfaces the control value and the model-linked features associated with that control, not every feature in the full 37-feature vector at once. This keeps the UI readable, but it means the detail view is a curated explanation layer rather than a complete model audit interface.

**Audio Accessibility and Legibility**

The sonification adds engagement, but it also raises accessibility and interpretation issues. Some users will browse with sound off, some will not be able to hear the distinctions clearly, and the current UI does not include a visible toggle or legend explaining what the audio encodes. This means the sound layer is valuable but optional, not a dependable primary communication channel.

---

## 5. Conclusions and Lessons Learned

### 5.1 What Was Accomplished

Summarize the system in its final form: a browser-deployed interactive visualization combining a trained XGBoost model (R² = 0.896, MAE = 3.72), a live autoregressive projection engine, a cross-section scene, a historical time-series chart, coordinated point inspection, and ambient audio. The full pipeline from raw data to browser inference is documented and reproducible.

### 5.2 What the Visualization Actually Does

Reflect critically on whether the system achieves its core premise. The goal was not to produce a scientifically authoritative hydrological forecast tool. It was to build something that captures a non-expert viewer's attention, gives them a sense of how inputs relate to water sustainability, and leaves them curious enough to look further. Evaluate against that standard rather than a scientific accuracy standard. The cross-section is effective at communicating the general direction of change. The time series grounds hypothetical scenarios in historical context. The coordinated inspection model makes the app easier to read as a system. The audio creates engagement. Whether this is enough to actually change how a viewer thinks about water is not testable from within the visualization itself, and that gap is worth acknowledging.

### 5.3 Design Lessons

Discuss what the iterative build process revealed:
- The knob layout replaced sliders after it became clear that sliders feel passive while rotary knobs create a sense of operating a system
- The amber projection line on the time series required the historical line to remain visible when the projection restarts. An earlier version cleared the historical context on restart, which broke the before/after comparison.
- Coordinating point selection between the time series and the cross-section made the app easier to read. Without that link, the chart and the scene feel adjacent; with it, the scene becomes a visual explanation of a specific month.
- Extending that same shared state to the knobs and the sonification mattered. Once every view responded to the same selected point, the interface stopped feeling like a dashboard and started feeling like a single instrument.
- The sonification layer nearly did not get built. It was added late and is not surfaced as a user control, but it meaningfully changes how long someone will stay with the visualization. Sound is often treated as a finishing detail in visualization projects and consistently underweighted in effort allocation.

### 5.4 Future Directions

**Still plausible next steps**

- The point-detail layer now exists, but it remains intentionally compact. A richer expansion could show actual vs predicted score, residual, major drought annotations, and a fuller slice of the 37-feature state for the selected month.
- The arc gauge panel described in the original design was never mounted in the final app shell. The cross-section and time series carry the visual load without it, but a proper gauge could serve as the primary visual on mobile where the cross-section is harder to render at small sizes.
- The audio layer should eventually gain a visible toggle, a short legend or onboarding cue explaining what it encodes, and a designed silent-mode fallback for classrooms or accessibility-sensitive contexts.

**Extensions that build on the results**

- Regional disaggregation: moving from a statewide Arizona aggregate to watershed-level or county-level estimates would let the cross-section scene change based on the user's selected region, resolving the Tucson-vs-Arizona mismatch
- Extending the historical window to 2023 is straightforward for sources that already reach that date (MODIS NDVI, USGS streamflow, USDM). The bottleneck is sources ending in 2020. Adding updated GRACE data and rerunning the export pipeline would bring the projection anchor date forward by three years.
- A scenario comparison mode where two knob configurations are shown side by side on the same time series would let a viewer directly compare, for example, a high-demand future against a conservation scenario
- A richer multimodal version could let the user scrub through time and hear the score transition continuously, making the sonification more than a background layer and turning it into a deliberate temporal storytelling device

---

## References

Include all six sources from the citations, plus any additional sources used for ONNX, XGBoost, the U.S. Drought Monitor, MERRA-2, MODIS, GRACE, USGS NWIS, and NRCS SNOTEL as appropriate.

- https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022WR033847
- https://www.mdpi.com/2072-4292/15/4/873
- https://www.sciencedirect.com/science/article/abs/pii/S0048969723041323
- https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2025.1564670/full
- https://www.mdpi.com/2306-5338/11/5/66
- https://github.com/onnx/onnx
