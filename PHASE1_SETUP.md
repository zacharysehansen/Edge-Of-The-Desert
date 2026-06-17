# Phase 1 Setup

This repo now includes an API-first phase 1 pipeline at `scripts/phase1`, plus Arizona-focused endpoint/source CSVs in [`data/Final`](/home/zacharyhansen/Documents/GitHub/Edge-Of-The-Desert/data/Final).

## Quick start

1. Install dependencies from `requirements.txt`.
2. Make sure your NASA Earthdata credentials are available to `earthaccess`.
3. Copy `config/phase1.example.json` to your own config file if you want to change the study area, site IDs, or source behavior.
4. Run:

`# Southern Arizona Pipeline: File Plan

## Structure

One Python file per dataset. Each file is responsible for one source only: pulling or loading raw data, filtering to the eight-county region, aggregating to the model's time grain, and writing a single clean endpoint CSV. No file depends on another file's output except the join script at the end.

Shared region boundary logic (the county list, the shapefile load, the point-in-polygon helper) lives in one common module so it is not duplicated eight times.

```
phase1/
    region.py
    population.py
    irrigation.py
    public_supply.py
    lake_mead.py
    urbanization.py
    water_stress.py
    ndvi.py
    grace_groundwater.py
    groundwater_wells.py
    wildfire.py
    wildlife_bbs.py
    build_dataset.py
```

## region.py

Holds the eight-county definition (Pima, Pinal, Santa Cruz, Cochise, Graham, Greenlee, Yuma, La Paz) and loads the county boundary shapefile (Census TIGER/Line). Exposes a function that takes a list of latitude/longitude points or a raster and returns only what falls inside the boundary. Every other file that needs spatial filtering imports from here instead of repeating the boundary logic.

## population.py

Pulls Census Bureau county population estimates for the eight counties, sums to a regional annual total, interpolates to monthly. Replaces the old statewide `azpop_monthly.csv` rather than re-aggregating it, since the original interpolation kept no county breakdown.

Output: `population_monthly.csv` with `year_month, population`.

## irrigation.py

Loads the existing HUC12-level irrigation source matrix, filters to HUC12s within the eight counties using `region.py`, re-aggregates to a monthly regional total the same way the original statewide file was built.

Output: `irrigation_monthly.csv` with `year_month, irrigation_total_withdrawal_mgd`.

## public_supply.py

Same pattern as `irrigation.py`, loading the NWAA HUC12 shards instead.

Output: `public_supply_monthly.csv` with `year_month, public_supply_groundwater_mgd`.

## lake_mead.py

Pulls the four Lake Mead CSV endpoints from the Bureau of Reclamation HydroData Navigator (reservoir ID 921: pool elevation, storage, total release, release volume), filters each to the model's date range, aggregates daily to monthly, merges into one table.

Output: `lake_mead_monthly.csv` with `year_month, mead_pool_elevation, mead_storage, mead_total_release, mead_release_volume`.

## urbanization.py

Downloads the relevant year's NLCD Fractional Impervious Surface raster from the S3 bucket, clips to the eight-county boundary using `region.py`, computes mean impervious percentage per year, interpolates to monthly.

Output: `urbanization_monthly.csv` with `year_month, impervious_pct`.

## water_stress.py

Derives or re-extracts the USDM score for the eight-county region, 

Output: `water_stress_monthly.csv` with `year_month, usdm_dsci,.

## ndvi.py

Loads or re-pulls MODIS NDVI, clips to the eight-county boundary.

Output: `ndvi_monthly.csv` with `year_month, ndvi`.

## grace_groundwater.py

Re-extracts GRACE groundwater anomaly from the raw `.nc4` files using a bounding box around the eight counties, applies the same `grace_available` flag and pre-2002-04 neutral fill as the original pipeline.

Output: `grace_monthly.csv` with `year_month, grace_groundwater_anomaly, grace_available`.

## groundwater_wells.py

Pulls USGS groundwater well levels via the USGS Water Data API, filtered to wells in the Tucson and Santa Cruz Active Management Areas and any other AMA inside the eight counties, aggregates to a monthly regional mean or median water level.

Output: `groundwater_wells_monthly.csv` with `year_month, well_level_ft`.

## wildfire.py

Loads the user-provided fire-event CSV, filters to fires located within the eight counties, builds the annual risk index from fire count and log-transformed total acres.

Output: `wildfire_annual.csv` with `year, fire_count, log_acres_total, wildfire_risk_index`.

## wildlife_bbs.py

Uses `sciencebasepy` to pull the BBS route metadata table and the yearly count table from the 2025 release, filters routes to those falling inside the eight counties using `region.py`, aggregates to an annual regional abundance index (pooled count or species richness, decided before writing this file). Must be run from an environment with access to `sciencebase.gov`, since that host is not reachable from this sandbox.

Output: `wildlife_annual.csv` with `year, route_count, total_abundance` or `year, route_count, species_richness`, depending on which index is chosen.

## build_dataset.py

Joins all of the above outputs on `year_month` (monthly files) or `year` (annual files: wildfire, wildlife), producing the final modeling tables. Since wildfire and wildlife are annual while the rest are monthly, this file is also responsible for deciding how the join handles grain mismatch, either by aggregating the monthly inputs to annual for those two specific targets, or by broadcasting the annual values across the twelve months of each year if a monthly wildfire or wildlife row is ever needed. Does not fetch any new data itself.

## Notes

`region.py` is the only file every other script depends on. Get the county shapefile and boundary logic right first, since a mistake there silently changes every other file's regional filter.

`wildfire.py` and `wildlife_bbs.py` are the only two files producing annual rather than monthly output, and both need their own row-count check before any cross-validation split is chosen, since the eight-county filter will shrink both datasets and the annual grain already means far fewer rows than the monthly files.

`lake_mead.py` and `urbanization.py` are the only two files that depend on an external resource not yet test-pulled in full (the exact Mead CSV parameter codes are confirmed, but the NLCD S3 bucket's file naming convention is not). Both should be the first two scripts run end to end before the others, to surface any remaining access problems early.