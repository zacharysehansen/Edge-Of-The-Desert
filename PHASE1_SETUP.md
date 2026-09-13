# Southern Arizona Water and Land Model: Dataset Inventory

## End Goal

Build a set of regression models for an eight-county central and southern Arizona region (Pima, Pinal, Santa Cruz, Cochise, Maricopa, Greenlee, Yuma, Gila — the documents said Graham and La Paz until 2026-09-12; the code's FIPS list never did, see PROBLEMS.md P8) that take in human and environmental pressure variables and predict three separate environmental outcomes. This is a planning-stage tool, meant to let someone adjust inputs like population growth, irrigation demand, or urbanization and see how vegetation health, water levels, and wildfire risk would likely respond.

Inputs: population, irrigation withdrawal, water management (reservoir behavior), urbanization, and water stress.

Outputs: vegetation index (NDVI), ground and surface water levels, wildfire risk, and wildlife abundance.

Each output gets its own separate XGBoost regression model trained on the same shared input set, rather than one combined model or a single bottleneck variable that all inputs route through first.


## Research Question

How do population growth, agricultural water demand, urban development, reservoir management decisions, and drought conditions influence vegetation health, water availability, wildfire risk, and wildlife abundance in Southern Arizona?

Version 2 is designed around a simple conceptual framework:

**Human Pressures and Environmental Conditions → Environmental Responses**

Rather than combining all variables into a single sustainability metric, the project models several environmental systems independently so that users can observe how each responds to the same set of pressures.

---

## Input Variables

The model inputs represent human pressures, land-use change, water management decisions, and regional drought conditions.

| Input Variable           | Description                                                        | Source                             |
| ------------------------ | ------------------------------------------------------------------ | ---------------------------------- |
| Population               | Regional population pressure                                       | U.S. Census Bureau                 |
| Irrigation Withdrawal    | Agricultural water demand                                          | HUC12 irrigation datasets          |
| Public Supply Withdrawal | Municipal groundwater demand                                       | NWAA groundwater datasets          |
| Reservoir Operations     | Water management behavior represented through Lake Mead operations | Bureau of Reclamation              |
| Urbanization             | Impervious surface coverage and land development                   | NLCD Fractional Impervious Surface |
| USDM DSCI                | Regional drought severity and coverage index                       | U.S. Drought Monitor               |

These variables form the shared input set used by every model in the system.

---

## Output Variables

The model predicts ecological and hydrological responses to the selected input conditions.

| Output Variable             | Source                              | Temporal Resolution |
| --------------------------- | ----------------------------------- | ------------------- |
| Vegetation Health (NDVI)    | MODIS                               | Monthly             |
| Groundwater Storage Anomaly | GRACE / GRACE-FO                    | Monthly             |
| Surface Water Conditions    | Regional water-level indicators     | Monthly             |
| Wildfire Risk               | Regional fire event database        | Annual              |
| Wildlife Abundance          | North American Breeding Bird Survey | Annual              |

Together, these outputs provide a multi-dimensional picture of environmental health across Southern Arizona.

---

## Modeling Framework

Version 2 uses multiple independent machine learning models rather than a single composite environmental index.

Each model receives the same set of input variables and predicts one environmental outcome.

| Model   | Prediction Target           |
| ------- | --------------------------- |
| Model 1 | NDVI                        |
| Model 2 | Groundwater Storage Anomaly |
| Model 4 | Surface Water Conditions    |
| Model 5 | Wildfire Risk               |
| Model 6 | Wildlife Abundance          |

This approach improves interpretability and allows users to examine how different environmental systems respond to the same set of human and environmental pressures.

---

## Visualization

### Control Panel

Users manipulate the major drivers of environmental change:

* Population
* Irrigation Withdrawal
* Public Supply Withdrawal
* Reservoir Operations
* Urbanization
* USDM DSCI

### Environmental Response Panel

The system displays predicted outcomes across multiple environmental domains.

**Vegetation Health**

NDVI indicators show expected changes in vegetation productivity.

**Groundwater Conditions**

Groundwater anomaly and well-level visualizations display subsurface water conditions.

**Surface Water Conditions**

Water-level indicators summarize predicted surface-water response.

**Wildfire Risk**

A wildfire-risk indicator displays projected fire pressure under the selected scenario.

**Wildlife Abundance**

Ecological indicators display predicted changes in regional wildlife populations.

**Historical Context**

Time-series visualizations compare predicted outcomes with historical observations.

## Datasets Already Available

These were collected for an earlier, differently-scoped project and exist as files already, though several need to be re-aggregated to fit the new eight-county boundary.

**Irrigation total withdrawal.** A pre-aggregated statewide monthly file exists (`irrigation_huc12_monthly_az_2000_2020.csv`), built from an underlying HUC12-level source matrix that is also still available. Since the statewide file can't be un-aggregated, the HUC12-level source needs to be re-filtered to only the eight counties and re-summed.

**Public supply groundwater withdrawal.** Same situation as irrigation: a statewide pre-aggregated file exists (`nwaa_public_supply_az_monthly.csv`), and the underlying HUC12-level shards are also available for re-filtering and re-aggregation to the eight counties.

**GRACE groundwater anomaly.** A statewide monthly file exists (`grace_groundwater_anomaly.csv`), derived from raw satellite `.nc4` files that are also still available. Because GRACE is a coarse-resolution satellite product (roughly 100km grid cells), it was never truly statewide-precise to begin with, so re-extracting it with a bounding box around the eight counties from the raw files is straightforward.

**MODIS NDVI.** A statewide monthly file exists (`modis_ndvi.csv`). This is the vegetation output target. It can likely be re-clipped to the eight counties if the original raw MODIS pull is still available, or re-pulled fresh from Earthdata scoped to the new region otherwise.

**Population.** A statewide monthly file exists (`azpop_monthly.csv`), interpolated from annual Census figures. This needs to be replaced rather than re-aggregated, since the original interpolation was done at the state level with no county breakdown retained.

## Datasets That Need To Be Created

These don't exist yet in any form and require pulling new source data.

**Population, county-level.** U.S. Census Bureau county population estimates for the eight counties, summed to a regional annual total, then interpolated to monthly using the same method as the original statewide interpolation.

**Lake Mead water management data.** Pulled from the Bureau of Reclamation's HydroData Navigator, reservoir ID 921. Four confirmed CSV endpoints, all daily and starting in 1935:

- Pool elevation: `https://www.usbr.gov/uc/water/hydrodata/reservoir_data/921/csv/49.csv`
- Storage: `https://www.usbr.gov/uc/water/hydrodata/reservoir_data/921/csv/17.csv`
- Total release: `https://www.usbr.gov/uc/water/hydrodata/reservoir_data/921/csv/42.csv`
- Release volume: `https://www.usbr.gov/uc/water/hydrodata/reservoir_data/921/csv/43.csv`

Each needs filtering to the model's date range and aggregation from daily to monthly.

**CAP water deliveries (added 2026-09-12).** Monthly Central Arizona Project deliveries, 1999-01
onward, from CAP's own delivery reports at `library.cap-az.com` — "Monthly Deliveries" by
classification (1999–2018) and "Year to Date Deliveries by Contract Type" (2018–). Reclamation's
decree accounting reports (`usbr.gov/lc/region/g4000/4200Rpts/DecreeRpt/`) carry the diversion at
Lake Havasu and are used only as an annual cross-check: within a year the diversion runs at
r = −0.2 to deliveries, because CAP fills Lake Pleasant in winter and delivers from it in summer.
2008 and 2009 are scanned images; their totals rows are transcribed in the script and checked
against the printed annual totals. `scripts/phase1/cap_deliveries.py` fetches every PDF it lacks.

**GLDAS land-surface state (added 2026-09-12).** GLDAS-2.1 Noah monthly (`GLDAS_NOAH025_M` v2.1,
GES DISC), 2000-01 to 2023-12: soil moisture in four layers, snow water equivalent, canopy storage,
evapotranspiration and rainfall forcing, averaged over the study bounding box (293 cells at 0.25°).
Each month is fetched as a DAP4 subset from `opendap.earthdata.nasa.gov` (~90 KB) rather than as the
24 MB global granule. Needs an Earthdata Login bearer token in `~/.config/earthdata/token` (mode 600)
or `$EARTHDATA_TOKEN`, and the "NASA GESDISC DATA ARCHIVE" application approved on the account —
until it is, every request returns `403 EULA Acceptance Failure`. `scripts/phase1/gldas.py`. The
GLDAS storage change is the GRACE and groundwater models' main input (§26–§29); runoff and
root-zone soil moisture were tested for streamflow, wildfire and NDVI and are not used (§33).

**Humidity and VPD (added 2026-09-13).** MERRA-2 M2TMNXSLV v5.12.4 (GES DISC), 2000-01 to
2023-12: 2-m specific humidity, dew point, air temperature and surface pressure as DAP4 bounding-box
subsets; VPD computed exactly from air and dew-point temperature. The directory
`data/raw/merra_specific_humidity_2m/` does NOT hold humidity — it is a second copy of the
temperature statistics files. `scripts/phase1/humidity.py`, same token and approval as GLDAS.
Tested for wildfire and NDVI and not used (§33).

**Urbanization.** USGS Annual NLCD (1985-2023), Fractional Impervious Surface product, pulled from the public cloud bucket (`s3://usgs-landcover/annual-nlcd/c1/v0/cu/mosaic/`). This is raster data covering the whole continental US, not pre-aggregated to any region, so it needs to be downloaded year by year, clipped to the eight-county boundary using a county shapefile, and averaged into a single impervious-surface percentage per year. The result is annual and needs interpolation to monthly.

New private housing permit data from FRED was considered as an alternative urbanization signal and rejected. Permits measure new construction activity rather than the existing built footprint, and only the Tucson metro area has county-level coverage in FRED, leaving seven of the eight counties unrepresented.

**Water stress (USDM-derived score).** The original statewide USDM-based sustainability score (`100 - usdm_dsci / 5`) needs to be re-derived or re-extracted for the eight-county region specifically, since the existing file is statewide.

**Wildfire event data, regionally filtered.** The user already holds a wildfire CSV (`OBJECTID, FIRE_NAME, FIRE_Number, FireID, Acres, FIRE_YEAR, Z, KM2, Source1, Source2, Shape__Area, Shape__Length`). This needs to be filtered to fires located within the eight counties, then converted into an annual wildfire risk index combining fire count and log-transformed total acreage, since the raw file is annual-grain with no monthly date field and the acreage distribution is heavily skewed by a small number of large fires.

**Wildlife abundance.** North American Breeding Bird Survey (BBS), USGS, hosted on ScienceBase, the 2025 release covering 1966 through 2024. Routes are fixed physical roadside survey lines, each with a stable latitude and longitude, sampled annually during June. Route metadata (route ID, name, latitude, longitude, state, stratum, active status) and yearly species-level counts are separate tables joined by route ID and year. Data must be filtered to routes whose coordinates fall inside the eight counties, then aggregated to a single annual abundance index per year (pooled count across species, or a species richness count, still to be decided). No data exists for 2020, since BBS field activity was cancelled that year. Access is through the `sciencebasepy` Python package, which calls the ScienceBase REST API directly. The host `sciencebase.gov` is not reachable from this sandbox's network, so this dataset must be pulled from an environment with open internet access rather than from inside the current pipeline tooling.

