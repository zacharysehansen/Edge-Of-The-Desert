# Southwest Water Sustainability Visualizer
## Final Project Design

---

### Overview

An interactive visualization that lets users manipulate environmental and human demand conditions across the American Southwest and see the predicted water sustainability score update in real time. The system combines satellite remote sensing data, hydrological measurements, and human consumption data to train a regression model, which is exported and run entirely in the browser via ONNX. The D3.js frontend visualizes both the user-controlled inputs and the model output simultaneously. Phase 1 data assembly is currently Arizona-focused within that broader Southwest framing, with the current monthly endpoint source files stored in `data/Final`.

---

### Data Sources

| Feature | Source | Raw Resolution | Model Resolution |
|---|---|---|---|
| NDVI | MODIS MOD13A3 | Monthly | Monthly |
| 2m Temperature | MERRA-2 | Daily | Monthly avg |
| Specific Humidity | MERRA-2 | Daily | Monthly avg |
| Precipitation | MERRA-2 | Daily | Monthly avg |
| Snow Water Equivalent | Arizona SNOTEL bundle (NRCS / WRCC) -> `snotel_swe.csv` | Daily | Monthly endpoint |
| Streamflow | USGS NWIS -> `usgs_streamflow.csv` | Daily | Monthly endpoint |
| GRACE Groundwater Anomaly | NASA GRACE | Monthly | Monthly |
| GRACE/GRACE-FO Derived Recharge Estimate | NASA GRACE / GRACE-FO | Monthly | Monthly |
| Lake Powell Operations | Manual Powell bundle -> `powell_combined.csv` | Daily / irregular | Monthly endpoint |
| Irrigation Total Withdrawal | USGS Arizona HUC12 source -> `irrigation_huc12_monthly_az_2000_2020.csv` | Monthly | Monthly statewide endpoint |
| Public-Supply Total | NWAA Arizona HUC12 source -> `nwaa_public_supply_az_monthly.csv` | Monthly | Monthly statewide endpoint |
| Population | Arizona `AZPOP` observations -> `azpop_monthly.csv` | Annual | Monthly endpoint via interpolation |
| Inverted USDM Score | USDM -> `usdm_sustainability.csv` | Weekly | Monthly endpoint (target) |

Project target range: 2000-2023. Join key: `year_month`. The endpoint/source CSVs we were looking for now live in `data/Final`: `modis_ndvi.csv`, `usgs_streamflow.csv`, `usdm_sustainability.csv`, `snotel_swe.csv`, `irrigation_huc12_monthly_az_2000_2020.csv`, `nwaa_public_supply_az_monthly.csv`, `powell_combined.csv`, and `azpop_monthly.csv`. Coverage is still shorter for some sources: `snotel_swe.csv` spans `2000-10` through `2018-07`, and `irrigation_huc12_monthly_az_2000_2020.csv`, `nwaa_public_supply_az_monthly.csv`, `powell_combined.csv`, and `azpop_monthly.csv` span `2000-01` through `2020-12`, while `modis_ndvi.csv`, `usgs_streamflow.csv`, and `usdm_sustainability.csv` extend through `2023-12`. Earlier filenames such as `snotel_swe_daily.csv`, raw Powell inputs, `AZPOP.csv`, and the HUC12 staging matrices were upstream inputs used to produce these endpoint files.

---

### Target Variable

The U.S. Drought Monitor weekly categorical score (D0-D4) is averaged to monthly and converted to a 0-100 numeric scale, then inverted so that 100 represents maximum sustainability and 0 represents crisis conditions. This gives a continuous regression target that is grounded in an established expert-curated index rather than an arbitrary composite.

---

### Model

**Algorithm:** XGBoost Regressor (via scikit-learn API)

XGBoost is chosen over linear regression because the relationships between atmospheric, hydrological, and human demand variables are nonlinear and interact with each other in ways a linear model cannot capture. It is also chosen over a neural network because the dataset is tabular and small enough, on the order of a few hundred monthly rows once the final overlap window is chosen, that a tree-based ensemble will outperform a neural net and train in seconds. Feature importances come for free and are useful for the writeup.

**Train/test split:** Evaluated using expanding window cross-validation via scikit-learn's `TimeSeriesSplit`, where the training window grows across 5-6 folds and predictions always run forward in time. This avoids the thin test set problem of a single year-based split and gives more reliable error estimates from the same few-hundred-row monthly dataset. Final model is trained on the full dataset after cross-validation confirms generalization.

**Export:** Trained pipeline exported to ONNX via `skl2onnx`. Loaded in the browser using `onnxruntime-web` running on WebAssembly. No backend server required.

---

### Visualization

The frontend is built in D3.js and has two zones:

**Control Panel**

A set of sliders, one per input feature, with labels and realistic ranges drawn from the historical data distribution. Non-intuitive features like specific humidity and GRACE anomaly are given plain-language labels ("Atmospheric Moisture", "Aquifer Health"). Each slider is initialized to the historical median for that feature. As the user adjusts sliders, the model runs inference in real time and all visual elements update.

Knobs available for user interaction:
- Precipitation
- Temperature
- Snowpack
- Lake Mead Level
- Irrigation Withdrawal
- Public-Supply Total
- Population

**Visualization Panel**

Three visual elements:

1. **Water sustainability gauge** - a large 0-100 arc gauge that is the primary output, color coded from red (crisis) to blue (healthy), with labeled thresholds

2. **Animated water table cross-section** - an SVG illustration of a cross-section of the ground showing the water table level rising or falling based on the GRACE anomaly and Lake Mead knob values, with the sustainability score driving the overall fill color and saturation

3. **Historical time series** - a D3 line chart of the real historical sustainability score from 2000-2023, with a highlighted dot showing where the user's current knob configuration falls relative to history. This grounds the hypothetical in real context and lets users see how their scenario compares to actual conditions like the 2002 or 2012 drought years

---

### Phased Plan

**Phase 1 (Weeks 1-3): Data Collection**
- Set up `earthaccess`, pull MERRA-2 and MODIS for the Southwest bounding box
- Download or assemble the Arizona endpoint inputs for USDM, SNOTEL SWE, USGS streamflow, and Lake Powell operations
- Pull GRACE groundwater anomaly from NASA and derive the monthly recharge estimate from the GRACE / GRACE-FO storage series
- Use the Arizona monthly endpoint/source files already prepared in `data/Final` (`snotel_swe.csv`, `irrigation_huc12_monthly_az_2000_2020.csv`, `nwaa_public_supply_az_monthly.csv`, `powell_combined.csv`, `azpop_monthly.csv`, `modis_ndvi.csv`, `usgs_streamflow.csv`, `usdm_sustainability.csv`)
- Join all sources into one flat monthly CSV

**Phase 2 (Weeks 4-5): Modeling**
- Exploratory analysis, check correlations, handle missing values
- Interpolate only the lower-frequency sources that still need it, such as population, and resolve missing values caused by differing source coverage windows
- Train XGBoost regressor, evaluate with R² and MAE using expanding window cross-validation (`TimeSeriesSplit`)
- Inspect feature importances
- Export trained pipeline to ONNX via `skl2onnx`

**Phase 3 (Weeks 6-9): Visualization**
- Build D3.js interface: sliders, gauge, water table SVG, time series
- Integrate `onnxruntime-web` so slider changes drive live ONNX inference
- Load and render real historical data on the time series chart
- Connect knob values to animated water table SVG elements

**Phase 4 (Weeks 10-12): Polish and Writeup**
- Refine slider ranges and default values based on testing
- Write up the project as a visualization system contribution
- Document data sources, modeling decisions, and design rationale

---

### Project Type Classification

This fits cleanly into the "building a visual solution to a particular data analysis problem" category from the project brief. The novel contribution is the integration of a pre-trained ONNX model with an interactive D3.js environment to let non-expert users explore the relationship between environmental conditions and water sustainability in an intuitive, hypothesis-driven way.
