# Data Inventory — Southern Arizona Environmental Model

## Project Overview

This document catalogs all data inputs for the Southern Arizona Water and Land Sustainability Model, covering an eight-county region (Pima, Pinal, Santa Cruz, Cochise, Maricopa, Greenlee, Yuma, Gila) [2]. **Corrected 2026-09-12:** this document, like every other, named Graham and La Paz; the FIPS codes the pipeline filters on (04013, 04007) are Maricopa and Gila, and always were. The set was kept on purpose — PROBLEMS.md P8. The model uses human pressures and environmental conditions as inputs to predict environmental responses across six independent XGBoost regression models [2].

**Project temporal range:** 2000-01 through 2023-12 [3]
**Spatial extent:** Bounding box (-114.81, 31.33, -109.05, 34.5) [3]

---

## Input Variables (Human Pressures & Environmental Conditions)

### 1. Population

| Field | Detail |
|-------|--------|
| **Description** | Regional population pressure — eight-county sum, interpolated to monthly [2] |
| **Source** | U.S. Census Bureau Population Estimates Program [2] |
| **Specific Endpoints** | 2000-2010: Census API intercensal estimates (`https://api.census.gov/data/2000/pep/int_population`); 2010-2020: Direct CSV (`https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/totals/co-est2020-alldata.csv`); 2020-2023: Direct CSV (`https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv`) |
| **Temporal Resolution** | Annual estimates (July 1), interpolated to monthly via cubic spline [1] |
| **Spatial Resolution** | County-level, summed to eight-county regional total [1] |
| **Date Range** | 2000–2023 |
| **Output File** | `data/Final/azpop_monthly.csv` |
| **Output Columns** | `year_month, population` [1] |
| **Processing** | Three vintage sources merged, county populations summed to regional total, cubic spline interpolation using July 1 anchor points [1] |
| **Access Date** | June 2025 |

---

### 2. Irrigation Withdrawal

| Field | Detail |
|-------|--------|
| **Description** | Agricultural water demand — total irrigation withdrawal for HUC12 watersheds within the eight counties [2] |
| **Source** | USGS National Water-Use Assessment (NWAA), HUC12-level irrigation source matrix [2] |
| **Raw File** | `data/raw/ir_huc12_tot_wd_az_2000_2020.csv` [3] |
| **Format** | Wide-format CSV with Year, Month columns and HUC12 codes as column headers; values are withdrawal rates in MGD [3] |
| **Temporal Resolution** | Monthly [1] |
| **Spatial Resolution** | HUC12 watershed level, filtered to eight-county region and summed [1] |
| **Date Range** | 2000-01 through 2020-12 (gap: 2021-2023 not available) [3] |
| **Output File** | `data/Final/irrigation_monthly.csv` |
| **Output Columns** | `year_month, irrigation_total_withdrawal_mgd` [1] |
| **Processing** | HUC12 columns identified, spatially filtered to eight-county boundary (via WBD shapefile centroid matching or fallback to all Arizona HUC12s), summed per month [1] |

---

### 3. Public Supply Groundwater Withdrawal

| Field | Detail |
|-------|--------|
| **Description** | Municipal groundwater demand — public supply withdrawal for HUC12 watersheds within the eight counties [2] |
| **Source** | USGS National Water-Use Assessment (NWAA), HUC12-level public supply shards [2] |
| **Raw File** | `data/raw/ps_huc12_tot_az_2000_2020.csv` [3] |
| **Format** | Wide-format CSV, same structure as irrigation [3] |
| **Temporal Resolution** | Monthly [1] |
| **Spatial Resolution** | HUC12 watershed level, filtered to eight-county region and summed [1] |
| **Date Range** | 2000-01 through 2020-12 (gap: 2021-2023 not available) [3] |
| **Output File** | `data/Final/public_supply_monthly.csv` |
| **Output Columns** | `year_month, public_supply_groundwater_mgd` [1] |
| **Processing** | Same pattern as irrigation.py — HUC12 spatial filter then row-wise sum [1] |

---

### 4. Lake Mead Reservoir Operations

| Field | Detail |
|-------|--------|
| **Description** | Water management behavior represented through Lake Mead pool elevation, storage, total release, and release volume [2] |
| **Source** | Bureau of Reclamation HydroData Navigator, Reservoir ID 921 [2] |
| **Specific Endpoints** | Pool elevation: `https://www.usbr.gov/uc/water/hydrodata/reservoir_data/921/csv/49.csv`; Storage: `.../17.csv`; Total release: `.../42.csv`; Release volume: `.../43.csv` [2] |
| **Temporal Resolution** | Daily (aggregated to monthly) [1] |
| **Date Range** | 2000-01 through 2023-12 (source data starts 1935) [3] |
| **Output File** | `data/Final/lake_mead_monthly.csv` |
| **Output Columns** | `year_month, mead_pool_elevation, mead_storage, mead_total_release, mead_release_volume` [1] |
| **Processing** | Daily CSVs fetched, parsed, filtered to project date range, aggregated to monthly (mean for elevation/storage/release rate, sum for release volume) [1] |
| **Access Date** | June 2025 |

---

### 5. Urbanization (Impervious Surface)

| Field | Detail |
|-------|--------|
| **Description** | Impervious surface coverage representing land development and urban growth [2] |
| **Source** | USGS Multi-Resolution Land Characteristics Consortium (MRLC), Annual NLCD Fractional Impervious Surface (1985-2024), Collection 1, Version 1 [2] |
| **Specific Endpoints** | Three MRLC data bundles: `https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_FctImp_1995-2004_CU_C1V1.zip`; `...2005-2014_CU_C1V1.zip`; `...2015-2024_CU_C1V1.zip` |
| **Format** | GeoTIFF rasters, CONUS-wide, ~30m resolution, Albers Equal Area projection |
| **Temporal Resolution** | Annual (interpolated to monthly via cubic spline) [1] |
| **Spatial Resolution** | 30m pixels, clipped to eight-county bounding box via windowed read [1] |
| **Date Range** | 2000–2023 [3] |
| **Output File** | `data/Final/urbanization_monthly.csv` |
| **Output Columns** | `year_month, impervious_pct` [1] |
| **Processing** | Pre-extracted TIF rasters read from `data/raw/NLCD/`, bounding box transformed to Albers Equal Area, windowed read (~14,600 × 19,500 pixels per year), valid pixels (0-100) averaged per year, cubic spline interpolation to monthly using July 1 anchor points [1] |
| **Access Date** | June 2025 |

---

### 6. Water Stress (USDM DSCI)

| Field | Detail |
|-------|--------|
| **Description** | Regional drought severity and coverage index — area-weighted across the eight counties [2] |
| **Source** | U.S. Drought Monitor, USDM Statistics API [2] |
| **Specific Endpoint** | `https://usdmdataservices.unl.edu/api/CountyStatistics/GetDSCI?aoi={FIPS}&startdate=1/1/2000&enddate=12/31/2023&statisticsType=1` (queried per county) |
| **Format** | CSV response: `State, County, FIPS, MapDate, DSCI` |
| **Temporal Resolution** | Weekly (aggregated to monthly mean) [1] |
| **Spatial Resolution** | County-level DSCI, area-weighted to eight-county regional score |
| **Date Range** | 2000-01 through 2023-12 [3] |
| **Output File** | `data/Final/water_stress_monthly.csv` |
| **Output Columns** | `year_month, usdm_dsci, water_stress_score` [1] |
| **Processing** | County DSCI pulled individually, area-weighted by county land area (sq mi), averaged weekly→monthly, stress score derived as `100 - (dsci / 5)` [1][2] |
| **Area Weights** | As shipped: Pima 9,189; Pinal 5,374; Santa Cruz 1,238; Cochise 6,219; **04013 (Maricopa) 4,641 — Graham's area**; Greenlee 1,848; Yuma 5,519; **04007 (Gila) 4,513 — La Paz's area**. Corrected in `water_stress.py` 2026-09-12 (Maricopa 9,226; Gila 4,795, TIGER EPSG:5070) but the CSV is not regenerated — correct weights move the index by r = 0.9992 (PROBLEMS.md P8) |
| **Access Date** | June 2025 |

---

### 7. Temperature (2-meter)

| Field | Detail |
|-------|--------|
| **Description** | Regional mean 2-meter air temperature — key driver of evapotranspiration, fire weather, and vegetation phenology |
| **Source** | NASA MERRA-2 M2SMNXSLV (Stationary Monthly means, Single-Level diagnostics) [3] |
| **Variable** | `T2MMEAN` (monthly mean 2-meter air temperature) [3] |
| **Raw Files** | `data/raw/merra_temperature_2m/*.nc4` |
| **Format** | NetCDF4, global grid, ~0.5° × 0.625° resolution |
| **Temporal Resolution** | Monthly [3] |
| **Spatial Resolution** | ~50km grid cells, subset to eight-county bounding box [3] |
| **Date Range** | 2000-01 through 2023-12 [3] |
| **Output File** | `data/Final/temperature_monthly.csv` |
| **Output Columns** | `year_month, temperature_2m_c` [3] |
| **Processing** | NetCDF opened with xarray, bounding box subset, spatial mean computed, Kelvin converted to Celsius (−273.15) [3] |
| **Unit Transform** | `kelvin_to_celsius` [3] |
| **Access Method** | NASA Earthdata (`earthaccess` Python package) |
| **Access Date** | June 2025 |

---

### 8. Precipitation

| Field | Detail |
|-------|--------|
| **Description** | Regional mean total precipitation rate — controls vegetation productivity, groundwater recharge, and fire fuel moisture |
| **Source** | NASA MERRA-2 M2TMNXFLX (Monthly mean, Time-averaged, Single-Level, Full Horizontal Resolution) [3] |
| **Variable** | `PRECTOT` (total precipitation) [3] |
| **Raw Files** | `data/raw/merra_precipitation/*.nc4` |
| **Format** | NetCDF4, global grid, ~0.5° × 0.625° resolution |
| **Temporal Resolution** | Monthly [3] |
| **Spatial Resolution** | ~50km grid cells, subset to eight-county bounding box [3] |
| **Date Range** | 2000-01 through 2023-12 [3] |
| **Output File** | `data/Final/precipitation_monthly.csv` |
| **Output Columns** | `year_month, precipitation_mm_day` [3] |
| **Processing** | NetCDF opened with xarray, bounding box subset, spatial mean computed, kg/m²/s converted to mm/day (×86400) [3] |
| **Unit Transform** | `kg_m2_s_to_mm_day` [3] |
| **Access Method** | NASA Earthdata (`earthaccess` Python package) |
| **Access Date** | June 2025 |

---

## Output Variables (Environmental Responses)

### 9. Vegetation Health (NDVI)

| Field | Detail |
|-------|--------|
| **Description** | Normalized Difference Vegetation Index — primary indicator of vegetation productivity and health [2] |
| **Source** | NASA MODIS MOD13A3, Version 061, 1km monthly NDVI [3] |
| **Variable** | `1 km monthly NDVI` subdataset [3] |
| **Raw Files** | `data/raw/modis_ndvi/*.hdf` |
| **Format** | HDF4 (EOS), MODIS sinusoidal tile projection, 1km resolution |
| **Temporal Resolution** | Monthly [2] |
| **Spatial Resolution** | 1km pixels, windowed to eight-county bounding box [3] |
| **Date Range** | 2000-01 through 2023-12 [3] |
| **Output File** | `data/Final/ndvi_monthly.csv` |
| **Output Columns** | `year_month, ndvi` [1] |
| **Processing** | HDF subdataset extracted, bounding box transformed to sinusoidal CRS, windowed read, invalid values (<−2000 raw) masked, scale factor applied (÷10000), spatial mean computed, small gaps (≤3 months) interpolated [3] |
| **Scale Factor** | 10000 (raw integers ÷ 10000 = NDVI in [-1, 1]) [3] |
| **Invalid Threshold** | Raw values below −2000 [3] |
| **Access Method** | NASA Earthdata (`earthaccess` Python package) |
| **Access Date** | June 2025 |

---

### 10. Groundwater Storage Anomaly (GRACE)

| Field | Detail |
|-------|--------|
| **Description** | Satellite-derived liquid water equivalent thickness anomaly — represents total water storage changes including groundwater [2] |
| **Source** | NASA GRACE / GRACE-FO, JPL RL06 Mascon Solution [3] |
| **Short Names** | `TELLUS_GRAC_L3_JPL_RL06_LND_v04` (GRACE, 2002-04 to 2017-06); `TELLUS_GRFO_L3_JPL_RL06.3_LND_v04` (GRACE-FO, 2018-06 to 2023-12) [3] |
| **Variable** | `lwe_thickness` (liquid water equivalent thickness, cm) [3] |
| **Raw Files** | `data/raw/grace_groundwater_anomaly/*.nc4` |
| **Format** | NetCDF4, global grid, 0.5° or 1° resolution (~300km effective) [2] |
| **Temporal Resolution** | Monthly [2] |
| **Spatial Resolution** | ~300km effective (gridded to 0.5°/1°), subset to bounding box [2][3] |
| **Date Range** | 2002-04 through 2023-12 (with inter-mission gap 2017-07 to 2018-05) [3] |
| **Output File** | `data/Final/grace_monthly.csv` |
| **Output Columns** | `year_month, grace_groundwater_anomaly, grace_available` [1] |
| **Processing** | NetCDF opened with xarray, bounding box subset (with 0-360° longitude handling), spatial mean computed per timestep, pre-GRACE months (2000-01 to 2002-03) filled with 0.0, inter-mission gap (2017-07 to 2018-05) linearly interpolated, `grace_available` flag marks real vs. filled data [1] |
| **Known Gaps** | Pre-mission: 2000-01 to 2002-03 (filled with 0.0); Inter-mission: 2017-07 to 2018-05 (interpolated); **plus routine single-month instrument dropouts scattered across the record** (2003-06, 2011-01, 2011-06, 2011-12, 2012-05, 2012-10, 2013-03, 2013-08/09, 2014-02, 2014-07, 2014-12, 2015-05/06, 2015-10/11, 2016-04, 2016-09, 2016-10, 2017-02, 2018-08/09) — GRACE powers down its accelerometers in low-solar-cycle months, so months are simply missing [1] |
| **Fill Extent** | **62 of 288 months are filled, not measured** — 27 pre-mission zeros plus **33 interpolated months in 23 separate blocks** inside the 2002-10+ modeling window. `grace_available == 0` marks every one. Any model using `grace_groundwater_anomaly` as a **target** must exclude these rows (and the row after each block, whose lag1 anchor is fill). See `_MONTHLY_MODEL_SPECS["grace"]["require_real_target"]` in `scripts/phase2/features.py` |
| **Access Method** | NASA Earthdata (`earthaccess` Python package) |
| **Access Date** | June 2025 |

---

### 11. Groundwater Well Levels

| Field | Detail |
|-------|--------|
| **Description** | Regional mean depth to water level from USGS monitoring wells — in-situ complement to GRACE satellite anomaly [2] |
| **Source** | USGS National Water Information System (NWIS) Daily Values Service [2] |
| **Specific Endpoint** | `https://waterservices.usgs.gov/nwis/dv/` with `siteType=GW` |
| **Parameters** | 72019 (depth to water level, ft below land surface), 72008 (depth to water in well, periodic), 62610 (GW level above NGVD 1929, ft), 62611 (GW level above NAVD 1988, ft) |
| **Format** | JSON responses parsed per site/timestep |
| **Temporal Resolution** | Daily (aggregated to monthly mean) [1] |
| **Spatial Resolution** | Individual monitoring wells, queried by county FIPS for eight counties [1] |
| **Date Range** | 2000-01 through 2020-12 [1] |
| **Output File** | `data/Final/groundwater_levels_monthly.csv` |
| **Output Columns** | `year_month, depth_to_water_ft_mean` [1] |
| **Processing** | Daily values fetched per county per parameter per year (JSON format), combined, aggregated to monthly regional mean using parameter 72019 as primary [1] |
| **Row Count** | 252 monthly rows (full coverage) [1] |
| **Access Date** | June 2025 |

---

### 11b. Surface Water Conditions

| Field | Detail |
|-------|--------|
| **Description** | Regional mean stream discharge and gage height from USGS stream gages — surface water availability indicator [2] |
| **Source** | USGS National Water Information System (NWIS) Daily Values Service [2] |
| **Specific Endpoint** | `https://waterservices.usgs.gov/nwis/dv/` with `siteType=ST` |
| **Parameters** | 00060 (discharge, cubic feet per second), 00065 (gage height, feet) |
| **Format** | JSON responses parsed per site/timestep |
| **Temporal Resolution** | Daily (aggregated to monthly mean) [1] |
| **Spatial Resolution** | Individual stream gages, queried by county FIPS for eight counties [1] |
| **Date Range** | 2000-01 through 2020-12 [1] |
| **Output File** | `data/Final/water_surface_monthly.csv` |
| **Output Columns** | `year_month, discharge_cfs_mean, gage_height_ft_mean` [1] |
| **Processing** | Daily values fetched per county per parameter per year (JSON format), combined, pivoted so each parameter is its own column, aggregated to monthly regional mean [1] |
| **Row Count** | 252 monthly rows (full coverage) [1] |
| **Note** | Some months may have missing gage height values where fewer gages report that parameter |
| **Access Date** | June 2025 |

---

### 12. Wildfire Risk

| Field | Detail |
|-------|--------|
| **Description** | Annual wildfire risk index combining fire frequency and magnitude within the eight counties [2] |
| **Source** | User-provided Arizona wildfire event database [2] |
| **Raw File** | `data/raw/az_wildfires.csv` |
| **Format** | CSV with columns: `OBJECTID, FIRE_NAME, FIRE_Number, FireID, Acres, FIRE_YEAR, Z, KM2, Source1, Source2, Shape__Area, Shape__Length` [2] |
| **Temporal Resolution** | Annual (no monthly date field available) [2] |
| **Spatial Resolution** | Point locations filtered to eight-county boundary [1] |
| **Date Range** | Variable (limited by source data availability) |
| **Output File** | `data/Final/wildfire_annual.csv` |
| **Output Columns** | `year, fire_count, log_acres_total, wildfire_risk_index` [1] |
| **Processing** | Spatial filter to eight counties (point-in-polygon if lat/lon columns present), aggregated to annual fire count and total acres, log1p transform on acres (heavy right skew), risk index = 0.5 × norm(fire_count) + 0.5 × norm(log_acres_total) [1][2] |
| **Row Count** | ~50 annual rows after filtering (at the minimum threshold for XGBoost cross-validation) [1] |

---

### 12b. Wildfire Risk (Monthly)

| Field | Detail |
|-------|--------|
| **Description** | Monthly large-wildfire risk index combining fire frequency and log-transformed acreage, on true ignition dates [2] |
| **Source** | **MTBS** (Monitoring Trends in Burn Severity) fire-occurrence points — USGS/USFS, national, 1984–present |
| **Raw File** | `data/raw/wildfire/mtbs/mtbs_FODpoints_DD.shp` |
| **Re-pull** | `curl -sL -o mtbs.zip https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/fod_pt_shapefile/mtbs_fod_pts_data.zip` (3.5 MB, no auth) |
| **Format** | Point shapefile; key fields `ig_date` (**true ignition date**), `burnbndac` (acres), `incid_type` |
| **Temporal Resolution** | Monthly, aggregated on `ig_date` |
| **Spatial Resolution** | Point-in-polygon clip to the eight-county boundary via `region.filter_points()` |
| **Date Range** | 2000-01 through 2023-12 (**MTBS supports 1984**; 105 additional in-region fires available pre-2000) |
| **Output File** | `data/Final/wildfire_monthly.csv` |
| **Output Columns** | `year_month, fire_count, total_acres, log_acres, wildfire_risk_index` [1] |
| **Processing** | Drop `Prescribed Fire`/`Other` (keep `Wildfire`, `Wildland Fire Use`), clip to region, aggregate by ignition month → fire count + total acres, log1p transform, risk index = 0.5 × norm(fire_count) + 0.5 × norm(log_acres) |
| **Row Count** | 288 monthly rows; 326 fires. **177 months (61%) are zero** — a month with no large fire is the most common outcome, and those zeros are real data, not missing values |
| **Coverage caveat** | MTBS only maps fires ≥ ~1000 acres in the West, so `fire_count` means **large fires**, not all ignitions |
| **⚠ History** | This series was previously built from `InterAgencyFirePerimeterHistory` with the month parsed from **`DATE_CUR` — a database-maintenance timestamp, not an ignition date**. 57% of AZ fires landed on five calendar days (Feb 1 alone held 1,221), the implied fire season peaked in *February*, and the target was anti-correlated with temperature (−0.27). **It was not measuring wildfire.** The perimeter file has no ignition-date field at any grain finer than `FIRE_YEAR`. Do not go back to it. `wildfire_monthly.py` now refuses to write the CSV if the peak ignition month falls outside April–August. |

---

### 13. Wildlife Abundance

| Field | Detail |
|-------|--------|
| **Description** | Annual bird abundance and species richness index — ecological health indicator [2] |
| **Source** | USGS North American Breeding Bird Survey (BBS), 2025 Release (covering 1966–2024) [2] |
| **ScienceBase Item** | `691cfb53d4be021d1d89b482` (2025 Release - North American Breeding Bird Survey Dataset) |
| **Raw Files** | `data/raw/bbs/Routes.csv` (latin-1 encoding); `data/raw/bbs/Arizona.csv` (utf-8 encoding) |
| **Routes File Columns** | `CountryNum, StateNum, Route, RouteName, Active, Latitude, Longitude, Stratum, BCR, RouteTypeID, RouteTypeDetailID` |
| **Counts File Columns** | `RouteDataID, CountryNum, StateNum, Route, RPID, Year, AOU, Count10, Count20, Count30, Count40, Count50, StopTotal, SpeciesTotal` |
| **Temporal Resolution** | Annual (one survey per route per year, June) [2] |
| **Spatial Resolution** | Fixed roadside survey routes, filtered to eight-county boundary via `region.filter_points()` [1] |
| **Date Range** | 2000–2024 (excluding 2020 — field work cancelled) [2] |
| **Output File** | `data/Final/wildlife_annual.csv` |
| **Output Columns** | `year, route_count, total_abundance, species_richness, abundance_index` [1] |
| **Processing** | Routes filtered spatially (40 routes in study area), counts joined by (StateNum, Route), SpeciesTotal summed across species/routes per year, species richness = unique AOU codes, abundance_index = min-max normalized log1p(total_abundance) [1] |
| **Count Column Used** | `SpeciesTotal` (sum across all 50 stops per species per route per year) |
| **Known Gaps** | 2020 excluded (BBS cancelled due to COVID-19) [2] |
| **Row Count** | 24 annual rows (below the 50-row warning threshold) [1] |
| **Access Date** | June 2025 |

---

## Spatial Boundary Definition

| Field | Detail |
|-------|--------|
| **Source** | U.S. Census Bureau TIGER/Line County Shapefile, 2023 vintage |
| **Download** | `https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2023&layergroup=Counties+(and+equivalent)` |
| **Local Path** | `data/raw/tiger/tl_2023_us_county.shp` [1] |
| **Filter Column** | `GEOID` (five-digit FIPS code) [1] |
| **Counties** | Pima (04019), Pinal (04021), Santa Cruz (04023), Cochise (04003), Maricopa (04013), Greenlee (04011), Yuma (04027), Gila (04007) [3] — 04013 and 04007 were labelled Graham and La Paz until 2026-09-12; the codes were always these |
| **Processing** | Filtered to eight counties, dissolved to single polygon, reprojected to EPSG:4326 [1] |
| **Used By** | Every script in the pipeline via `region.py` [1] |

---

## Data Coverage Summary

| Dataset | Temporal Grain | Start | End | Full Coverage? |
|---------|---------------|-------|-----|----------------|
| Population | Monthly | 2000-01 | 2023-12 | ✓ |
| Irrigation | Monthly | 2000-01 | 2020-12 | Gap: 2021-2023 |
| Public Supply | Monthly | 2000-01 | 2020-12 | Gap: 2021-2023 |
| Lake Mead | Monthly | 2000-01 | 2023-12 | ✓ |
| Urbanization | Monthly | 2000-01 | 2023-12 | ✓ |
| Water Stress | Monthly | 2000-01 | 2023-12 | ✓ |
| Temperature | Monthly | 2000-01 | 2023-12 | ✓ |
| Precipitation | Monthly | 2000-01 | 2023-12 | ✓ |
| NDVI | Monthly | 2000-01 | 2023-12 | ✓ |
| GRACE | Monthly | 2000-01 | 2023-12 | Filled pre-2002 & gap |
| Groundwater Levels | Monthly | 2000-01 | 2020-12 | ✓ (252 rows) |
| Surface Water | Monthly | 2000-01 | 2020-12 | ✓ (252 rows) |
| Wildfire (annual) | Annual | 2000 | 2023 | ✓ |
| Wildfire (monthly) | Monthly | 2000-01 | 2023-12 | ✓ (288 rows) |
| Wildlife | Annual | 2000 | 2024 | Missing 2020 (24 rows) |

---

## Known Limitations

1. **Irrigation and Public Supply end in 2020** — handled by *exclusion*, per model. Models whose target runs past 2020 (NDVI, GRACE, wildfire) drop these two series and every feature derived from them, and run to 2023-12. Models whose target ends in 2020 anyway (groundwater, surface water) keep them. Do **not** extrapolate them; they are near-pure month-of-year templates and dropping them costs almost nothing (measured: −0.0075 NDVI skill).
2. **Groundwater levels and Surface water end in 2020** — these are *targets*, so those two models are structurally capped at 2020-12 and cannot score current conditions.
3. **GRACE fill is more extensive than the mission gaps suggest** — 62 of 288 months are filled, not measured: pre-mission zeros (2000-01 to 2002-03) *and* 33 interpolated months scattered in 23 blocks. All flagged `grace_available=0`. Never train or score a GRACE-targeted model on them [1]
4. **Wildlife row count (24 rows)** — below the 50-row warning threshold; cross-validation will be unstable [1]
5. **Wildfire row count (50 rows)** — at the minimum acceptable threshold [1]
6. **Water stress uses statewide DSCI as proxy** — updated to county-level API when available, but the API returns statewide-equivalent DSCI values per county [2]
7. **GRACE spatial resolution (~300km)** — too coarse to distinguish within-region gradients [2]

---

*Document generated June 2025. Data access confirmed during pipeline development.*