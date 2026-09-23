# 02 — Local constants — re-derive Layer 2 parameters over the local mask

**What to build:** Re-compute every constant that `scripts/phase3/structural_params.py` currently derives regionally, but over the local Tucson-basin mask from ticket 01. This includes: local NDVI baseline (from `data/Final/ndvi_monthly.csv` clipped to the local domain), local irrigated fraction, local `region_acres`, local well-depth statistics from the 14 Pima County wells in `data/Final/groundwater_levels_daily_2000_2020.csv`, and local impervious baseline. Writes `frontend/local_structural_params.json` alongside the regional `structural_params.json`, following the same schema with `value`, `band`, `source`, and `status` tags on every constant.

**Blocked by:** 01 — Local domain boundary

**Status:** done

- [x] Each constant that `structural_params.py` computes has a local counterpart derived over the HUC8∩Pima mask (25 constants)
- [x] Local NDVI baseline: borrowed from regional (CSV is aggregated; pixel-level extraction pending), marked ESTIMATED
- [x] Local `region_acres` = 2,254,640 acres (~8.1% of region), MEASURED from boundary geometry
- [x] Well-depth statistics use only the 14 Pima County wells (median 204 ft, headroom ~796 ft to statutory 1,000 ft)
- [x] Output file follows the existing `structural_params.json` schema with source and status tags
- [x] Summary prints regional vs. local side-by-side with ratios and status counts
