"""
groundwater_diagnosis.py
------------------------
What is `depth_to_water_anomaly_ft` an index OF? (PHASE3_PLAN.md §28, roadmap step 1
for the groundwater model.)

The target is a mean of per-well anomalies over whichever USGS daily-value wells
report each month — 10 to 42 of 66. By county the roster is 44 Cochise, 14 Pima,
5 Yuma, 2 Maricopa, 1 Pinal: two thirds Willcox/Douglas-basin agriculture, three
wells in the Phoenix and Pinal AMAs where the region's pumping and all of its CAP
water are. This script measures what that composition does to the series:

  1. Rebuilds the shipped index from the daily pull with the same rules as
     groundwater_levels.py (param 72019, well-month means, wells with >= 24 months,
     per-well centring) and checks it reproduces data/Final.
  2. Builds a sub-index per county the same way, and asks whether Cochise and Pima
     move together — in level, in month-to-month change, and in trend.
  3. Measures the roster: Cochise's share of reporting wells by month, and how much
     of the shipped index's monthly change is explained by an index over the 27
     wells with 10+ years (roster churn survives per-well centring if wells with
     different trends enter and leave).
  4. Correlates each sub-index's monthly change with the panel's pumping proxies
     (deseasonalized irrigation withdrawal, CAP deliveries) and with the climate the
     GRACE model uses (GLDAS storage change, rain).

Changes nothing that ships. Writes model/groundwater_diagnosis.json.

Run:
    python scripts/phase3/groundwater_diagnosis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "data" / "Final" / "groundwater_levels_daily_2000_2020.csv"
MONTHLY = ROOT / "data" / "Final" / "groundwater_levels_monthly.csv"
IRR = ROOT / "data" / "Final" / "irrigation_monthly.csv"
CAP = ROOT / "data" / "Final" / "cap_deliveries_monthly.csv"
GLDAS = ROOT / "data" / "Final" / "gldas_monthly.csv"
PRECIP = ROOT / "data" / "Final" / "precipitation_monthly.csv"
OUTPUT = ROOT / "model" / "groundwater_diagnosis.json"

DEPTH_CD = "72019"
MIN_MONTHS_PER_WELL = 24
LONG_WELL_MONTHS = 120
COUNTY = {4013: "Maricopa", 4021: "Pinal", 4019: "Pima", 4003: "Cochise", 4023: "Santa Cruz",
          4027: "Yuma", 4007: "Gila", 4011: "Greenlee"}


def well_months() -> pd.DataFrame:
    d = pd.read_csv(DAILY, low_memory=False)
    d = d[d["param_cd"].astype(str) == DEPTH_CD].copy()
    d["site_no"] = d["site_no"].astype(str)
    d["year_month"] = pd.to_datetime(d["datetime"]).dt.to_period("M").astype(str)
    wm = d.groupby(["site_no", "year_month"]).agg(value=("value", "mean"), county=("county_fips", "first")).reset_index()
    wm["county"] = wm["county"].map(COUNTY)
    counts = wm.groupby("site_no")["year_month"].transform("size")
    wm = wm[counts >= MIN_MONTHS_PER_WELL].copy()
    wm["months"] = counts[wm.index]
    wm["anomaly"] = wm["value"] - wm.groupby("site_no")["value"].transform("mean")
    return wm


def index_of(wm: pd.DataFrame) -> pd.Series:
    return wm.groupby("year_month")["anomaly"].mean()


def deseasonalize(s: pd.Series) -> pd.Series:
    m = pd.PeriodIndex(s.index, freq="M").month
    return s - s.groupby(m).transform("mean")


def trend_ft_per_year(s: pd.Series) -> float:
    t = np.arange(len(s)) / 12.0
    return float(np.polyfit(t, s.to_numpy(), 1)[0])


def main() -> None:
    wm = well_months()
    shipped = index_of(wm).rename("rebuilt")
    final = pd.read_csv(MONTHLY).set_index("year_month")["depth_to_water_anomaly_ft"]
    j = pd.concat([shipped, final.rename("final")], axis=1).dropna()
    repro = float(j.rebuilt.corr(j.final))
    max_dev = float((j.rebuilt - j.final).abs().max())

    out: dict = {"reproduces_shipped_index": {"r": repro, "max_abs_dev_ft": max_dev, "n": int(len(j))}}
    print("=== groundwater target diagnosis ===\n")
    print(f"rebuilt index vs data/Final: r = {repro:.6f}, max |dev| = {max_dev:.4f} ft over {len(j)} months\n")

    # --- roster composition ---
    per_month = wm.groupby(["year_month", "county"])["site_no"].nunique().unstack(fill_value=0)
    share = (per_month.div(per_month.sum(axis=1), axis=0)).fillna(0)
    print("wells reporting per month by county (median):", per_month.median().astype(int).to_dict())
    print("Cochise share of the roster: mean %.2f, min %.2f, max %.2f" % (share.get("Cochise", 0).mean(), share.get("Cochise", 0).min(), share.get("Cochise", 0).max()))
    out["roster"] = {
        "wells_per_month_median": {k: int(v) for k, v in per_month.median().items()},
        "cochise_share": {"mean": float(share["Cochise"].mean()), "min": float(share["Cochise"].min()), "max": float(share["Cochise"].max())},
        "cochise_share_by_year": {str(y): float(v) for y, v in share["Cochise"].groupby(share.index.str[:4]).mean().items()},
    }

    # --- county sub-indices ---
    sub = {c: index_of(wm[wm.county == c]) for c in ("Cochise", "Pima")}
    both = pd.concat([sub["Cochise"].rename("cochise"), sub["Pima"].rename("pima"), shipped], axis=1).dropna()
    d = both.diff().dropna()
    print("\nCochise vs Pima sub-index over %d shared months:" % len(both))
    print("  level r = %+.3f   monthly-change r = %+.3f" % (both.cochise.corr(both.pima), d.cochise.corr(d.pima)))
    print("  trend  Cochise %+.2f ft/yr   Pima %+.2f ft/yr   shipped %+.2f ft/yr" % (trend_ft_per_year(both.cochise), trend_ft_per_year(both.pima), trend_ft_per_year(both.rebuilt)))
    print("  sd of monthly change: Cochise %.2f  Pima %.2f  shipped %.2f ft" % (d.cochise.std(), d.pima.std(), d.rebuilt.std()))
    print("  shipped change vs Cochise change r = %+.3f ; vs Pima change r = %+.3f" % (d.rebuilt.corr(d.cochise), d.rebuilt.corr(d.pima)))
    out["cochise_vs_pima"] = {
        "n": int(len(both)),
        "level_r": float(both.cochise.corr(both.pima)),
        "change_r": float(d.cochise.corr(d.pima)),
        "trend_ft_per_year": {"cochise": trend_ft_per_year(both.cochise), "pima": trend_ft_per_year(both.pima), "shipped": trend_ft_per_year(both.rebuilt)},
        "change_sd_ft": {"cochise": float(d.cochise.std()), "pima": float(d.pima.std()), "shipped": float(d.rebuilt.std())},
        "shipped_change_r_with": {"cochise": float(d.rebuilt.corr(d.cochise)), "pima": float(d.rebuilt.corr(d.pima))},
    }

    # --- roster churn: long-well index vs shipped ---
    long_idx = index_of(wm[wm.months >= LONG_WELL_MONTHS]).rename("long")
    n_long = wm[wm.months >= LONG_WELL_MONTHS].site_no.nunique()
    jj = pd.concat([shipped, long_idx], axis=1).dropna()
    dj = jj.diff().dropna()
    print(f"\nindex over the {n_long} wells with >= {LONG_WELL_MONTHS} months vs shipped:")
    print("  level r = %+.3f   monthly-change r = %+.3f   (1 - r² of the change = the part of the shipped month-to-month move that a fixed roster does not share)" % (jj.rebuilt.corr(jj.long), dj.rebuilt.corr(dj.long)))
    out["fixed_roster"] = {"n_wells": int(n_long), "level_r": float(jj.rebuilt.corr(jj.long)), "change_r": float(dj.rebuilt.corr(dj.long))}

    # --- what moves each sub-index? ---
    irr = pd.read_csv(IRR).set_index("year_month")["irrigation_total_withdrawal_mgd"]
    cap = pd.read_csv(CAP).set_index("year_month")["cap_deliveries_af"]
    gl = pd.read_csv(GLDAS).set_index("year_month")["gldas_tws_proxy_mm"].diff()
    pr = pd.read_csv(PRECIP).set_index("year_month")
    pr = pr[[c for c in pr.columns if "precip" in c][0]]
    drivers = pd.concat({"irr_des": deseasonalize(irr), "cap_des": deseasonalize(cap), "gldas_delta": gl, "precip": pr}, axis=1)
    rows = {}
    print("\nmonthly change of each index vs drivers (Pearson r, shared months):")
    for name, s in (("shipped", shipped), ("cochise", sub["Cochise"]), ("pima", sub["Pima"])):
        x = pd.concat([s.diff().rename("dy"), drivers], axis=1).dropna()
        rows[name] = {c: float(x.dy.corr(x[c])) for c in drivers.columns} | {"n": int(len(x))}
        print("  %-8s n=%3d  " % (name, len(x)) + "  ".join(f"{c} {rows[name][c]:+.3f}" for c in drivers.columns))
    out["change_vs_drivers"] = rows

    OUTPUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
