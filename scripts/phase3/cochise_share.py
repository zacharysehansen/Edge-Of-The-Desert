"""
cochise_share.py
----------------
What share of the region's groundwater pumping happens in Cochise County?
(PHASE3_PLAN.md §29.)

The groundwater output is now an index over the Cochise County wells (§28, option
A). Layer 2's pumping levers are built from REGIONAL series — the eight-county
irrigation and public-supply withdrawals, and population × GPCD — divided by an
aquifer storage coefficient. For the irrigation lever that coefficient is
calibrated empirically against the target (aquifer_calibration.py regresses the
Cochise index on the regional irrigation anomaly), so the region-to-Cochise share
is already inside it. For public supply and population there is no calibration:
they reuse the irrigation coefficient, and a regional municipal withdrawal applied
to Cochise wells would overstate the response by the inverse of Cochise's share of
it. This script measures that share from the HUC12 withdrawal matrices, assigning
each regional HUC12 to the county its representative point falls in.

Writes model/cochise_share.json; structural_params.py reads it.

Run:
    python scripts/phase3/cochise_share.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase1"))

from phase1 import irrigation as irr  # noqa: E402
from phase1 import public_supply as ps  # noqa: E402
from phase1.region import COUNTY_FIPS, STATE_FIPS  # noqa: E402

TIGER = ROOT / "data" / "raw" / "tiger" / "tl_2023_us_county.shp"
OUTPUT = ROOT / "model" / "cochise_share.json"
COCHISE = "04003"


def huc12_county() -> pd.Series:
    """HUC12 code -> county GEOID, by representative point, for the study counties."""
    counties = gpd.read_file(TIGER)
    counties = counties[(counties.STATEFP == STATE_FIPS) & counties.GEOID.isin(COUNTY_FIPS)]
    shapes = gpd.read_file(irr.HUC12_SHAPEFILE).to_crs(counties.crs)
    pts = shapes[["HUC12", "geometry"]].copy()
    pts["geometry"] = pts.geometry.representative_point()
    joined = gpd.sjoin(pts, counties[["GEOID", "geometry"]], how="inner", predicate="within")
    return joined.groupby("HUC12")["GEOID"].first()


def share_of(module, label: str, county_of: pd.Series) -> dict:
    frame = module._load_raw(module.INPUT_FILE)
    cols = module._identify_huc12_columns(frame)
    regional = [c for c in module._get_regional_huc12_columns(cols)]
    values = frame[regional].replace(list(getattr(module, "NODATA_SENTINELS", [])), np.nan)
    by_col = values.mean()  # mean withdrawal per HUC12 over the record
    county = pd.Series({c: county_of.get(str(c).strip()) for c in regional})
    total = float(by_col.sum())
    cochise = float(by_col[county == COCHISE].sum())
    by_county = by_col.groupby(county).sum().sort_values(ascending=False)
    # Year-by-year, so the band is the min-max over the record rather than a point.
    years = frame["Year"] if "Year" in frame.columns else None
    yearly = []
    if years is not None:
        for y, g in values.groupby(years):
            m = g.mean()
            yearly.append(float(m[county == COCHISE].sum() / m.sum()))
    return {
        "series": label,
        "n_huc12_regional": int(len(regional)),
        "n_huc12_cochise": int((county == COCHISE).sum()),
        "regional_mean_mgd": total,
        "cochise_mean_mgd": cochise,
        "share": cochise / total if total else float("nan"),
        "share_band_by_year": [min(yearly), max(yearly)] if yearly else None,
        "by_county_share": {str(k): float(v / total) for k, v in by_county.items()},
    }


def main() -> None:
    county_of = huc12_county()
    out = {
        "irrigation": share_of(irr, "irrigation_total_withdrawal_mgd", county_of),
        "public_supply": share_of(ps, "public_supply_groundwater_mgd", county_of),
    }
    for k, v in out.items():
        print(f"{k:14s} Cochise {v['cochise_mean_mgd']:8.1f} of {v['regional_mean_mgd']:8.1f} MGD = share {v['share']:.4f}"
              f"  (by year {v['share_band_by_year'][0]:.3f}..{v['share_band_by_year'][1]:.3f});  HUC12s {v['n_huc12_cochise']}/{v['n_huc12_regional']}")
        print("   by county:", {k2: round(v2, 3) for k2, v2 in v["by_county_share"].items()})
    OUTPUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
