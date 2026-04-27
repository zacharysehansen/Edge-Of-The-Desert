# Phase 1 Setup

This repo now includes an API-first phase 1 pipeline at `scripts/phase1.py`, plus Arizona-focused endpoint/source CSVs in [`data/Final`](/home/zacharyhansen/Documents/GitHub/Edge-Of-The-Desert/data/Final).

## Quick start

1. Install dependencies from `requirements.txt`.
2. Make sure your NASA Earthdata credentials are available to `earthaccess`.
3. Copy `config/phase1.example.json` to your own config file if you want to change the study area, site IDs, or source behavior.
4. Run:

```bash
python scripts/phase1.py collect
python scripts/phase1.py build --allow-partial
```

Or run both steps together:

```bash
python scripts/phase1.py run --allow-partial
```

Usable monthly endpoint/source files will be written to `data/Final/`, raw API snapshots will be written to `data/raw/`, and the joined dataset will be written to `data/processed/phase1_monthly_dataset.csv`.

## What is automated already

- MODIS monthly NDVI via NASA Earthdata / `earthaccess`
- MERRA-2 monthly temperature, specific humidity, and precipitation via NASA Earthdata / `earthaccess`
- GRACE land water storage anomaly via NASA Earthdata / `earthaccess`
- GRACE/GRACE-FO-derived recharge proxy, computed from month-to-month positive changes in the GRACE storage series
- USGS NWIS daily streamflow via Water Services
- U.S. Drought Monitor DSCI / sustainability target via the USDM REST service

## Current endpoint files in repo

These now live in [`data/Final`](/home/zacharyhansen/Documents/GitHub/Edge-Of-The-Desert/data/Final).

This folder is now the final endpoint for the usable monthly source CSVs we were looking for before model training begins.

1. `snotel_swe.csv`
   Current columns: `year_month, snow_water_equivalent_in`
   Coverage: `2000-10` through `2018-07`
   Notes: monthly Arizona SNOTEL snow-water-equivalent feature aggregated from the Baker Butte, Happy Jack, Mormon Mountain, and Hannagan Meadows station bundle.

2. `irrigation_huc12_monthly_az_2000_2020.csv`
   Current columns: `year_month, irrigation_total_withdrawal_mgd, irrigation_total_withdrawal_gallons_per_day, irrigation_total_withdrawal_acre_feet_month, huc12_count`
   Coverage: `2000-01` through `2020-12`
   Notes: Arizona statewide monthly irrigation endpoint derived from the USGS irrigation HUC12 source matrix.

3. `nwaa_public_supply_az_monthly.csv`
   Current columns: `year_month, public_supply_groundwater_mgd, public_supply_groundwater_gallons_per_day, public_supply_groundwater_acre_feet_month, huc12_count`
   Coverage: `2000-01` through `2020-12`
   Notes: Arizona statewide monthly public-supply groundwater endpoint aggregated from NWAA HUC12 shards.

4. `powell_combined.csv`
   Current columns: `year_month, powell_evaporation, powell_total_release, powell_inflow, powell_storage, powell_pool_elevation`
   Coverage: `2000-01` through `2020-12`
   Notes: Lake Powell monthly operational feature bundle merged from manually gathered source files in `data/raw`.

5. `azpop_monthly.csv`
   Current columns: `year_month, AZPOP`
   Coverage: `2000-01` through `2020-12`
   Notes: Arizona monthly population endpoint expanded from annual `AZPOP` observations.

6. `modis_ndvi.csv`
   Current columns: `year_month, ndvi`
   Coverage: `2000-02` through `2023-12`
   Notes: Arizona-wide monthly NDVI endpoint collected from MODIS Earthdata.

7. `merra_precipitation.csv`
   Current columns: `year_month, precipitation_mm_day`
   Coverage: `2000-01` through `2023-12`
   Notes: Arizona-wide monthly MERRA-2 precipitation endpoint derived from the locally downloaded `data/raw/merra_precipitation/*.nc4` files.

8. `merra_temperature_2m.csv`
   Current columns: `year_month, temperature_2m_c`
   Coverage: `2000-01` through `2023-12`
   Notes: Arizona-wide monthly MERRA-2 2m temperature endpoint derived from the locally downloaded `data/raw/merra_temperature_2m/*.nc4` files.

9. `grace_groundwater_anomaly.csv`
   Current columns: `year_month, grace_groundwater_anomaly`
   Coverage: `2000-01` through `2020-12`
   Notes: Arizona-wide monthly GRACE groundwater anomaly endpoint derived from the GRACE / GRACE-FO raw files in `data/raw/grace_groundwater_anomaly/`.

10. `usgs_streamflow.csv`
   Current columns: `year_month, streamflow_cfs`
   Coverage: `2000-01` through `2023-12`
   Notes: monthly mean streamflow endpoint derived from the configured USGS NWIS daily gauge set.

11. `usdm_sustainability.csv`
   Current columns: `year_month, usdm_dsci, usdm_sustainability`
   Coverage: `2000-01` through `2023-12`
   Notes: monthly Arizona drought endpoint built from U.S. Drought Monitor DSCI output, including the derived sustainability score used by the pipeline.

Intermediate HUC12-level files, summaries, and build artifacts remain in
[`data/processed`](/home/zacharyhansen/Documents/GitHub/Edge-Of-The-Desert/data/processed).

## Earlier staging inputs you may still see

Some earlier notes, scripts, and config values still reference upstream staging inputs rather than the final endpoint files above. The most common examples are:

- `snotel_swe_daily.csv`
- `AZPOP.csv`
- raw Powell source files such as `powell_evaporation.csv` and `powell_storage.csv`
- HUC12 staging matrices such as `ir_huc12_tot_wd_az_2000_2020.csv` and `ps_huc12_tot_az_2000_2020.csv`

Treat those older names as upstream inputs used to produce the `data/Final/` endpoint CSVs, not as the final target files we were trying to end up with.

## Config values you should review before trusting the dataset

- `project.start_year_month` / `project.end_year_month`
  The default build window is still `2000-01` through `2023-12`, but the endpoint files do not all fully span that range: `snotel_swe.csv` stops in `2018-07`, and `irrigation_huc12_monthly_az_2000_2020.csv`, `nwaa_public_supply_az_monthly.csv`, `powell_combined.csv`, `azpop_monthly.csv`, and `grace_groundwater_anomaly.csv` stop in `2020-12`. `merra_precipitation.csv`, `merra_temperature_2m.csv`, `usgs_streamflow.csv`, and `usdm_sustainability.csv` extend through `2023-12`. Expect missing values outside the shorter source windows unless you trim the build period or backfill additional data.

- `study_area.bbox`
  The default config now uses an Arizona-wide bounding box.

- `usgs_streamflow.site_numbers`
  The default config now uses:
  `09380000` Colorado River at Lees Ferry,
  `09498500` Salt River near Roosevelt,
  `09506000` Verde River near Camp Verde.

- `grace_groundwater_anomaly.short_names`
  The config currently stitches the older GRACE JPL land product with the GRACE-FO successor; keep this if it matches your intended groundwater proxy, or swap to a different official GRACE product.

- `grace_recharge_estimate`
  This is now derived from the GRACE/GRACE-FO anomaly series inside the pipeline rather than coming from a separate manual recharge-gap CSV.

## Outputs worth checking

- `data/processed/phase1_status.json`
  Collection status, failures, and missing sources.

- `data/processed/phase1_build_report.json`
  Join summary and which source CSVs were missing when the flat file was built.
