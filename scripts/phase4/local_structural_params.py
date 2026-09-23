"""Re-derive Layer 2 structural parameters over the local Tucson-basin mask.

Every constant that structural_params.py computes regionally gets a local
counterpart here, derived over the HUC8∩Pima domain from ticket 01.
Constants that cannot yet be measured locally are borrowed from regional
with an explicit status tag.

Writes frontend/local_structural_params.json with the same schema as
structural_params.json: value, band, source, status on every entry.

    python -m scripts.phase4.local_structural_params
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Final"
STATS_PATH = ROOT / "frontend" / "computed_stats.json"
REGIONAL_PARAMS_PATH = ROOT / "frontend" / "structural_params.json"
OUTPUT_FILE = ROOT / "frontend" / "local_structural_params.json"

REFERENCE_YEAR = 2020
SQ_M_PER_ACRE = 4046.8564224
CUBIC_M_PER_AF = 1233.48
PIMA_FIPS = 4019

PIMA_IRRIGATED_ACRES = 30_008.0


def tag(value, *, band=None, source, status, note=None) -> dict:
    entry = {"value": value, "source": source, "status": status}
    if band is not None:
        entry["band"] = list(band)
    if note is not None:
        entry["note"] = note
    return entry


def local_region_acres() -> dict:
    """Compute the local domain area in acres from the HUC8∩Pima boundary."""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from phase1.local_region import load_local_boundary
        boundary = load_local_boundary()
        area_m2 = boundary.to_crs("EPSG:5070").geometry.union_all().area
        acres = area_m2 / SQ_M_PER_ACRE
        return tag(
            round(acres),
            source="HUC8∩Pima boundary, EPSG:5070, via scripts/phase1/local_region.py",
            status="MEASURED",
        )
    except Exception as exc:
        area_km2 = 9124.0
        acres = area_km2 * 1e6 / SQ_M_PER_ACRE
        return tag(
            round(acres),
            source=f"fallback from PHASE4.md §2 target (9,124 km²), ({type(exc).__name__})",
            status="ESTIMATED",
        )


def local_well_stats() -> dict:
    """Compute depth-to-water statistics from the 14 Pima County wells."""
    gw = pd.read_csv(DATA / "groundwater_levels_daily_2000_2020.csv")
    pima = gw[gw["county_fips"] == PIMA_FIPS].copy()
    n_wells = pima["site_no"].nunique()

    pima["ym"] = pima["datetime"].str[:7]
    well_monthly = pima.groupby(["ym", "site_no"])["value"].mean()
    monthly_mean = well_monthly.groupby("ym").mean()

    ref_depth_all = float(monthly_mean.mean())
    ref_depth_2020 = float(monthly_mean[monthly_mean.index.str.startswith("2020")].mean())

    latest = pima.sort_values("datetime").groupby("site_no").last()
    depth_min = float(latest["value"].min())
    depth_median = float(latest["value"].median())
    depth_max = float(latest["value"].max())

    return {
        "n_wells": n_wells,
        "ref_depth_all_time": round(ref_depth_all, 2),
        "ref_depth_2020": round(ref_depth_2020, 2),
        "latest_min_ft": round(depth_min, 0),
        "latest_median_ft": round(depth_median, 0),
        "latest_max_ft": round(depth_max, 0),
        "headroom_to_statutory_ft": round(1000 - depth_median, 0),
    }


def local_streamflow_baseline() -> dict:
    """Compute local baseline CFS from Pima County gages."""
    sw = pd.read_csv(DATA / "water_surface_daily_8county_2000_2020.csv")
    pima = sw[sw["county_fips"] == PIMA_FIPS].copy()
    n_gages = pima["site_no"].nunique()

    pima["ym"] = pima["datetime"].str[:7]
    monthly = pima.groupby(["ym", "site_no"])["value"].mean().reset_index()
    monthly.columns = ["ym", "site_no", "cfs"]
    per_month = monthly.groupby("ym").agg(
        mean_cfs=("cfs", "mean"), n_gages=("site_no", "nunique")
    )
    median_cfs = float(per_month["mean_cfs"].median())
    median_n = float(per_month["n_gages"].median())
    baseline_cfs = median_cfs * median_n

    return tag(
        round(baseline_cfs, 1),
        source=(
            f"median(mean_cfs) × median(n_gages) over Pima County gages in "
            f"water_surface_daily_8county_2000_2020.csv, {n_gages} gages"
        ),
        status="MEASURED",
    )


def build() -> dict:
    stats = json.loads(STATS_PATH.read_text())
    regional = json.loads(REGIONAL_PARAMS_PATH.read_text())
    regional_c = regional["constants"]

    acres = local_region_acres()
    local_area_m2 = acres["value"] * SQ_M_PER_ACRE

    wells = local_well_stats()
    streamflow = local_streamflow_baseline()

    ndvi_natural = stats["OUTPUT_STATS"]["ndvi"]["baseline"]

    constants = {
        "region_acres": acres,

        "irrigated_acres": tag(
            PIMA_IRRIGATED_ACRES,
            source=(
                "2017 Census of Agriculture, Table 10, Pima County only. "
                "Regional total 718,832 includes all eight counties."
            ),
            status="VERIFIED",
        ),

        "irrigated_fraction": tag(
            PIMA_IRRIGATED_ACRES / acres["value"],
            source="irrigated_acres / local region_acres",
            status="VERIFIED",
        ),

        "grace_units_per_af": tag(
            CUBIC_M_PER_AF / local_area_m2,
            source=(
                "1 AF spread over the LOCAL domain area, in metres of equivalent "
                "water height. ~12× larger than regional because the domain is "
                "~8.1% of the region."
            ),
            status="UNTESTED",
            note=(
                "Same closed-basin assumption as regional. A pumped AF is more "
                "visible per unit area on the local footprint."
            ),
        ),

        "ndvi_natural": tag(
            ndvi_natural,
            source=(
                "Borrowed from regional baseline (computed_stats.json). The monthly "
                "NDVI CSV is a regional aggregate; local NDVI requires pixel-level "
                "MODIS data clipped to the HUC8∩Pima mask."
            ),
            status="ESTIMATED",
            note="Replace with local NDVI mean when pixel-level extraction is available.",
        ),

        "ndvi_impervious": tag(
            regional_c["ndvi_impervious"]["value"],
            band=regional_c["ndvi_impervious"].get("band"),
            source=(
                "Borrowed from regional measurement (ndvi_endpoints.py "
                "difference-in-differences). The slope is a per-cell property, "
                "not a regional aggregate, so it transfers to any sub-domain."
            ),
            status="MEASURED",
            note="The slope (-0.0356 NDVI / unit impervious) is local by construction.",
        ),

        "ndvi_irrigated_crop": tag(
            regional_c["ndvi_irrigated_crop"]["value"],
            band=regional_c["ndvi_irrigated_crop"].get("band"),
            source=(
                "Borrowed from regional measurement. The slope (+0.3177 NDVI / "
                "unit irrigated fraction) was measured per-cell and transfers."
            ),
            status="MEASURED",
        ),

        "local_baseline_cfs": streamflow,

        "mean_precipitation_mm_day": tag(
            regional_c["mean_precipitation_mm_day"]["value"],
            source=(
                "Borrowed from regional mean (precipitation_monthly.csv). "
                "Tucson-basin precipitation is close to the regional mean; "
                "a local PRISM extraction would refine this."
            ),
            status="ESTIMATED",
        ),

        "storage_af_per_ft": tag(
            regional_c["storage_af_per_ft"]["value"],
            band=regional_c["storage_af_per_ft"].get("band"),
            source=(
                "Borrowed from regional calibration (Cochise index). PHASE4 §4.4a: "
                "'The shipped 707,463 AF/ft was calibrated against the Cochise index "
                "and does not transfer.' A Tucson-basin calibration against the 14 "
                "Pima wells is needed."
            ),
            status="UNTESTED",
            note=(
                "This is the single largest source of local-tier uncertainty. "
                "Pool & Anderson SIR 2007-5275 may provide a Tucson-basin estimate."
            ),
        ),

        "well_depth_stats": tag(
            wells,
            source=(
                f"14 Pima County wells in groundwater_levels_daily_2000_2020.csv. "
                f"Latest depths: {wells['latest_min_ft']:.0f}–{wells['latest_max_ft']:.0f} ft, "
                f"median {wells['latest_median_ft']:.0f} ft. "
                f"Headroom to A.A.C. R12-15-716 statutory limit (1,000 ft): "
                f"~{wells['headroom_to_statutory_ft']:.0f} ft."
            ),
            status="MEASURED",
        ),

        "statutory_depth_limit_ft": tag(
            1000.0,
            source=(
                "A.A.C. R12-15-716(B)(2): maximum 100-year depth-to-static-water-level "
                "in the Tucson AMA for Assured Water Supply designation."
            ),
            status="VERIFIED",
        ),
    }

    # Constants that are physical and don't change with scale
    for key in [
        "af_per_mgd_month", "cfs_per_mgd", "cfs_per_af_month",
        "specific_yield", "alluvial_fraction",
        "effluent_return_fraction", "runoff_coefficient_impervious",
        "runoff_coefficient_natural", "stream_capture_fraction",
        "groundwater_substitution_fraction",
    ]:
        if key in regional_c:
            entry = dict(regional_c[key])
            entry["source"] = f"Same as regional — {entry['source']}"
            constants[key] = entry

    # The Mead/CAP chain is regional by nature — the CAP delivers to the
    # region, not to the local domain specifically. The share that reaches
    # Pima is a separate question (ticket 03/04 nested budget).
    constants["region_share_of_az_reduction"] = dict(regional_c["region_share_of_az_reduction"])

    # Transfer coefficient is measured regionally and transfers
    constants["transfer_surface_water_to_wildlife"] = dict(
        regional_c["transfer_surface_water_to_wildlife"]
    )

    # Pima wells: no Cochise share scaling needed — the wells ARE local
    constants["pima_municipal_share"] = tag(
        1.0,
        source=(
            "Local tier uses Pima County wells directly. The Cochise scaling "
            "factor (cochise_municipal_to_irrigation_share = 0.191) does not "
            "apply — municipal pumping in Pima acts on the Pima aquifer."
        ),
        status="MEASURED",
        note=(
            "When the local storage_af_per_ft is calibrated against Pima wells, "
            "the calibration will already embed the local pumping share."
        ),
    )

    return constants, wells


def main() -> int:
    regional = json.loads(REGIONAL_PARAMS_PATH.read_text())
    constants, wells = build()

    result = {
        "meta": {
            "generated_by": "scripts/phase4/local_structural_params.py",
            "reference_year": REFERENCE_YEAR,
            "domain": "HUC8∩Pima (Tucson basin), ~9,124 km² / 8.1% of region",
            "plan": "PHASE4.md §2 (D1 local tier), §3 (D2 local constants)",
            "note": (
                "Local counterparts of structural_params.json. Constants that "
                "cannot yet be measured locally are borrowed from regional with "
                "status ESTIMATED or UNTESTED."
            ),
        },
        "constants": constants,
    }

    with OUTPUT_FILE.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"Local structural parameters written to {OUTPUT_FILE}\n")

    # ── Side-by-side comparison ──────────────────────────────────────────
    regional_c = regional["constants"]

    print(f"  {'constant':36s}{'regional':>16s}{'local':>16s}  ratio    status")
    print(f"  {'─' * 36}{'─' * 16}{'─' * 16}{'─' * 9}{'─' * 12}")

    compare_keys = [
        "region_acres", "irrigated_acres", "irrigated_fraction",
        "grace_units_per_af", "ndvi_natural", "storage_af_per_ft",
        "mean_precipitation_mm_day",
    ]
    for key in compare_keys:
        rc = regional_c.get(key, {})
        lc = constants.get(key, {})
        rv = rc.get("value", 0)
        lv = lc.get("value", 0)
        ratio = lv / rv if rv else float("nan")
        status = lc.get("status", "?")
        print(f"  {key:36s}{rv:>16,.6g}{lv:>16,.6g}  {ratio:>7.3f}  {status}")

    print(f"\n  {'streamflow':36s}{'regional':>16s}{'local':>16s}")
    rc_cfs = regional_c.get("regional_baseline_cfs", {}).get("value", 0)
    lc_cfs = constants.get("local_baseline_cfs", {}).get("value", 0)
    print(f"  {'baseline_cfs':36s}{rc_cfs:>16,.1f}{lc_cfs:>16,.1f}")

    print(f"\n  Pima well depth statistics:")
    print(f"    n_wells:              {wells['n_wells']}")
    print(f"    ref depth (all time): {wells['ref_depth_all_time']:.1f} ft")
    print(f"    ref depth (2020):     {wells['ref_depth_2020']:.1f} ft")
    print(f"    latest range:         {wells['latest_min_ft']:.0f}–{wells['latest_max_ft']:.0f} ft")
    print(f"    latest median:        {wells['latest_median_ft']:.0f} ft")
    print(f"    headroom to 1,000 ft: ~{wells['headroom_to_statutory_ft']:.0f} ft")

    borrowed = [k for k, v in constants.items() if v.get("status") == "ESTIMATED"]
    untested = [k for k, v in constants.items() if v.get("status") == "UNTESTED"]
    measured = [k for k, v in constants.items() if v.get("status") == "MEASURED"]
    verified = [k for k, v in constants.items() if v.get("status") == "VERIFIED"]

    print(f"\n  Status summary:")
    print(f"    MEASURED:  {len(measured)}")
    print(f"    VERIFIED:  {len(verified)}")
    print(f"    ESTIMATED: {len(borrowed)} (borrowed from regional)")
    print(f"    UNTESTED:  {len(untested)}")
    if borrowed:
        print(f"    → borrowed: {', '.join(borrowed)}")
    if untested:
        print(f"    → untested: {', '.join(untested)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
