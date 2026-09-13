"""Generate the Layer 2 structural coefficients (PHASE3_PLAN.md §4).

Layer 1 — the learned models — is a climate forecaster and is left alone. Layer 2
supplies the human-lever response it cannot carry, with signs and magnitudes fixed by
water balance and published policy rather than fitted from a collinear 219-row panel.
PHASE3_PLAN.md §10 measured why that is necessary: of 21 lever/target pairs with a
declared physical sign, the panel identified 3 and found nothing for 12.

Every coefficient written here carries `value`, `band`, `source` and `status`, using
the claim tags from PROBLEMS.md. Nothing is tuned to make an acceptance test pass —
PHASE3_PARAMS.md §2 is explicit that doing so would defeat the premise of the plan,
and §5 there says to let the test fail where the physics is small.

WHICH STORAGE COEFFICIENT SHIPS, AND WHY
----------------------------------------
PHASE3_PLAN.md §12 measured `S_y * A` two ways and got 110,891 AF/ft (regression on
monthly pumping anomalies, t = +3.16) against 1,875,000 AF/ft (S_y times the alluvial
basin area). They are the short-run and long-run limits of a spreading drawdown cone,
and their 16.9x ratio is the ratio of the two areas.

**The long-run value ships.** The slider is a SUSTAINED policy control (that is what
D5's reparameterization made it), and the coefficient for a sustained change is the
long-run one. Using the short-run value instead would require a reversion rate of
about 0.47/month -- a 2.1-month e-folding -- to keep the 36-month total physical, and
that contradicts the aquifer's own measured lambda of 0.0157 (63.8 months, §4b). The
consistent reading of a large short-run coefficient that recovers in weeks is local
well interference around the pumped fields, not basin storage depletion.

So the measured regression does two jobs here, and neither is setting the magnitude:
it confirms the sign and the mechanism at t = +3.16, and it sets the upper end of the
reported band.

    python scripts/phase3/structural_params.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Final"
STATS_PATH = ROOT / "frontend" / "computed_stats.json"
CALIBRATION_PATH = ROOT / "model" / "aquifer_calibration.json"
NDVI_ENDPOINTS_PATH = ROOT / "model" / "ndvi_endpoints.json"
CAP_CALIBRATION_PATH = ROOT / "model" / "cap_calibration.json"
TRANSFER_PATH = ROOT / "model" / "transfer_calibration.json"
OUTPUT_FILE = ROOT / "frontend" / "structural_params.json"

REFERENCE_YEAR = 2020

# 1 Mgal = 3.06889 acre-ft; sustained for 365 days that is 1,120 AF/yr.
AF_PER_MGD_YEAR = 1120.0
AF_PER_MGD_MONTH = AF_PER_MGD_YEAR / 12.0
CUBIC_M_PER_AF = 1233.48
SECONDS_PER_MONTH = 365.2425 / 12 * 86400
CFS_PER_CMS = 35.3147
CFS_PER_MGD = 1.547229
# 1 AF delivered evenly over a month, as cubic feet per second.
CFS_PER_AF_MONTH = CUBIC_M_PER_AF / SECONDS_PER_MONTH * CFS_PER_CMS
SQ_M_PER_ACRE = 4046.8564224

# Recorded in PHASE3_PARAMS.md §2 as MEASURED from the dissolved eight-county TIGER
# boundary in EPSG:5070. Recomputed in-repo when geopandas/rasterio are installed;
# this is the fallback so the script runs without the geospatial stack.
REGION_ACRES_RECORDED = 27_779_840.0


def tag(value, *, band=None, source, status, note=None) -> dict:
    entry = {"value": value, "source": source, "status": status}
    if band is not None:
        entry["band"] = list(band)
    if note is not None:
        entry["note"] = note
    return entry


def _region_share_constant() -> dict:
    """Lost CAP delivery per declared Arizona cut — measured from CAP's own delivery
    record by scripts/phase3/cap_calibration.py (PHASE3_PLAN.md §23). Every acre-foot
    CAP does not deliver is undelivered inside the region (all three CAP counties are
    in-region, PROBLEMS.md P8), so this is the share of the declared cut that reaches
    the region as lost water. Falls back to the sourced assumption if the calibration
    has not been run."""
    if not CAP_CALIBRATION_PATH.exists():
        return tag(
            0.95,
            band=(0.85, 1.00),
            source=(
                "Arizona's shortage reduction is 'borne almost entirely by the CAP system' "
                "(ADWR-CAP joint shortage statement, 2021), and CAP delivers only to "
                "Maricopa, Pinal and Pima - all three in-region (PROBLEMS.md P8)."
            ),
            status="UNTESTED",
            note="run scripts/phase3/cap_calibration.py to replace this with the measurement",
        )
    cal = json.loads(CAP_CALIBRATION_PATH.read_text())
    meas = cal["region_share_of_az_reduction"]
    years = cal["specification"]["judged_years"]
    lo, hi = meas["band"]
    # The lever multiplies a DECLARED cut, so the shipped band is capped at 1.0: the
    # ratios above 1 in 2023-2025 are compensated system conservation running on top
    # of the tier, which is real water not delivered but not the tier's doing.
    return tag(
        min(meas["value"], 1.0),
        band=(round(lo, 3), min(round(hi, 3), 1.0)),
        source=(
            f"measured: (baseline - actual) annual CAP deliveries / declared AZ cut, mean "
            f"{meas['value']:.3f} over {years[0]}-{years[-1]} against a 2015-2019 baseline "
            f"({cal['baseline_kaf']['2015-2019']:,.0f} kAF/yr); min-max over years x two "
            f"baselines {lo:.3f}-{hi:.3f}, shipped capped at 1.0. scripts/phase3/cap_calibration.py, "
            "from CAP's monthly delivery reports (scripts/phase1/cap_deliveries.py)."
        ),
        status="MEASURED",
        note=(
            "Was 0.60 (0.40-0.80) while the documents said Maricopa was out-of-region, then "
            "0.95 (0.85-1.00) by reasoning after PROBLEMS.md P8. The delivery record says a "
            "Tier 1+ cut arrives in the region essentially in full: 2022 alone gives 0.82-0.99, "
            "and 2023-2025 exceed 1.0 because compensated conservation ran alongside the tier. "
            "The Tier 0 contributions of 2020-2021 (192 kAF) did NOT show up as lost deliveries "
            "(-0.10 and 0.43), which is why those years are reported but not judged: DCP let "
            "them be met from ICS and conservation credits."
        ),
    )


def region_acres() -> dict:
    try:
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        from phase1.region import load_county_boundary

        boundary = load_county_boundary()
        area_m2 = boundary.to_crs("EPSG:5070").geometry.union_all().area
        return tag(
            round(area_m2 / SQ_M_PER_ACRE),
            source="dissolved eight-county TIGER boundary, EPSG:5070, via scripts/phase1/region.py",
            status="MEASURED",
        )
    except Exception as exc:  # noqa: BLE001 - geospatial stack is optional here
        return tag(
            REGION_ACRES_RECORDED,
            source="PHASE3_PARAMS.md §2, measured from the same boundary",
            status="MEASURED",
            note=f"recomputation unavailable in this environment ({type(exc).__name__})",
        )


def reference_mean(filename: str, column: str) -> float:
    frame = pd.read_csv(DATA / filename)
    year = pd.PeriodIndex(frame["year_month"], freq="M").year
    rows = frame.loc[year == REFERENCE_YEAR, column].dropna()
    if rows.empty:
        raise ValueError(f"No {REFERENCE_YEAR} rows for {column} in {filename}")
    return float(rows.mean())


# Lower Basin Operating Agreement, Table 1, "Combined DCP Contributions and 2007
# Interim Guidelines Shortages". Drought Contingency Plan Agreements, Final Review
# Draft 2018-10-05, Bureau of Reclamation. Transcribed verbatim from the primary PDF
# (PHASE3_PARAMS.md §1). Thousand acre-feet per year of Arizona reduction, keyed on
# the projected January 1 Lake Mead elevation in feet msl.
#
# Descending thresholds: the first row whose `above` the elevation exceeds applies.
# The 0-above-1,090 row is not in Table 1 — it is the complement of the table's
# coverage, and asserting it is safe because no reduction is required above 1,090.
MEAD_TIERS = [
    {"above": 1090.0, "az_combined_kaf": 0},
    {"above": 1075.0, "az_combined_kaf": 192},
    {"above": 1050.0, "az_combined_kaf": 512},
    {"above": 1045.0, "az_combined_kaf": 592},
    {"above": 1025.0, "az_combined_kaf": 640},
    # Catch-all for "below 1,025". Written as 0.0 rather than -inf because JSON has no
    # infinity literal and JSON.parse rejects Python's `-Infinity`; Mead's dead pool is
    # 895 ft, so no reachable elevation falls through this row.
    {"above": 0.0, "az_combined_kaf": 720},
]


def build() -> dict:
    stats = json.loads(STATS_PATH.read_text())
    slider_stats = stats["SLIDER_STATS"]
    calibration = json.loads(CALIBRATION_PATH.read_text())
    ndvi_endpoints = json.loads(NDVI_ENDPOINTS_PATH.read_text())
    transfers = json.loads(TRANSFER_PATH.read_text())

    acres = region_acres()
    region_m2 = acres["value"] * SQ_M_PER_ACRE

    population_2020 = reference_mean("azpop_monthly.csv", "population")
    public_supply_2020 = reference_mean(
        "public_supply_monthly.csv", "public_supply_groundwater_mgd"
    )
    gpcd = public_supply_2020 * 1e6 / population_2020

    specific_yield = 0.15
    alluvial_fraction = 0.45
    storage_long_run = specific_yield * alluvial_fraction * acres["value"]
    storage_short_run = calibration["irrigation"]["+ climate + trend"][
        "storage_af_per_ft"
    ]
    # PHASE3_PLAN.md §12's spreading-cone test, run in aquifer_calibration.py. The
    # coefficient is measured AT the scenario horizon rather than picked from either
    # limit, because the test showed both limits are wrong for a 12-month slider: the
    # cone does spread (809k -> 1.63M acres from 1 to 12 months) but it plateaus at
    # two to three times the irrigated area, never approaching the alluvial basin.
    horizon = calibration["horizon_sweep"]
    storage_at_horizon = horizon["storage_af_per_ft_at_default"]
    storage_band = tuple(horizon["plateau_band_af_per_ft"])

    # Total flow the gage network sees, from the project's own record. The surface
    # water target is the mean over gages of ln(Q/Q_normal), so a flow change
    # expressed as a PROPORTION of this converts straight to a log anomaly.
    surface = pd.read_csv(DATA / "water_surface_monthly.csv")
    regional_cfs = float(
        surface["discharge_cfs_mean"].median() * surface["n_gages"].median()
    )
    mean_precip_mm_day = float(
        pd.read_csv(DATA / "precipitation_monthly.csv")["precipitation_mm_day"].mean()
    )

    # PHASE3_PLAN.md §16 item 1 / PHASE3_PARAMS.md §4b. The lever multiplies the
    # DIFFERENCE (ndvi_impervious - ndvi_natural), which is exactly the regression
    # slope of NDVI on impervious fraction, so the slope is what gets adopted and the
    # endpoint is derived from it. Band is the interquartile range across the 287
    # months fitted: a per-pixel standard error would be fiction, since 131,000 MODIS
    # cells in one month are nowhere near independent.
    # The DIFFERENCE-IN-DIFFERENCES slope, not the cross-sectional one. The
    # cross-section compares cities to the desert around them and cannot separate
    # "this land is paved" from "this land was always different" — cities sit on
    # valley floors and alluvial fans, which were never a random sample of the region.
    # Differencing each cell against its own past removes every time-invariant
    # characteristic at once. It was chosen on identification, before the magnitude
    # was known; it happens to come out 42% larger.
    ndvi_slope = ndvi_endpoints["difference_in_differences"]["slope"]
    # Same design for the irrigation endpoint, baseline-controlled because mean
    # reversion would otherwise inflate it. See PHASE3_PLAN.md §20.
    irr_did = ndvi_endpoints["irrigation_difference_in_differences"]
    irr_slope = irr_did["slope_baseline_controlled"]
    irr_band = sorted((irr_slope, irr_did["slope"]))
    # Band spans the cross-sectional estimate at one end and the matched urbanised-vs-
    # control contrast at the other: the honest width is the spread across designs,
    # which is wider than any one design's standard error.
    ndvi_slope_band = [
        min(ndvi_slope, ndvi_endpoints["slope"]["median"]),
        max(ndvi_slope, ndvi_endpoints["slope"]["median"]),
    ]

    # 2017 Census of Agriculture Table 10, summed over the eight counties the
    # pipeline actually selects (PROBLEMS.md P8): Cochise 86,008 + Gila 1,296 +
    # Greenlee 5,136 + Maricopa 180,214 + Pima 30,008 + Pinal 232,224 +
    # Santa Cruz 2,551 + Yuma 181,395. Was 681,143, which summed Graham and La Paz
    # instead of Maricopa and Gila — a numerator over the wrong set, divided by
    # region_acres from the right one.
    irrigated_acres = 718_832.0
    ndvi_natural = stats["OUTPUT_STATS"]["ndvi"]["baseline"]

    constants = {
        "af_per_mgd_month": tag(
            AF_PER_MGD_MONTH,
            source="1 Mgal = 3.06889 acre-ft, x 365 d / 12 mo",
            status="VERIFIED",
        ),
        "region_acres": acres,
        "irrigated_acres": tag(
            irrigated_acres,
            source=(
                "2017 Census of Agriculture, Table 10 'Irrigation: 2017 and 2012', "
                "USDA NASS Vol 1 Ch 2 Arizona county-level, summed over the eight counties "
                "the pipeline selects (Maricopa and Gila in, Graham and La Paz out - PROBLEMS.md P8)"
            ),
            status="VERIFIED",
        ),
        "irrigated_fraction": tag(
            irrigated_acres / acres["value"],
            source="irrigated_acres / region_acres",
            status="VERIFIED",
        ),
        "specific_yield": tag(
            specific_yield,
            band=(0.10, 0.20),
            source=(
                "Midpoint of the ADWR basin-fill modelling range. USGS SIR 2007-5275 "
                "(Pool & Anderson) gives 0.16-0.21 for comparable alluvial aquifers. "
                "The exact ADWR value was not retrieved: azwater.gov returns HTTP 403."
            ),
            status="UNTESTED",
        ),
        "alluvial_fraction": tag(
            alluvial_fraction,
            band=(0.30, 0.60),
            source="assumed share of the region that is alluvial basin rather than mountain block",
            status="UNTESTED",
        ),
        "storage_af_per_ft": tag(
            storage_at_horizon,
            band=storage_band,
            source=(
                "measured at the {horizon['default_scenario_months']}-month scenario horizon by "
                "aquifer_calibration.py's spreading-cone sweep: total drawdown over the "
                "window regressed on total volume pumped across it, climate and trend "
                f"controlled, Newey-West. {storage_at_horizon:,.0f} AF/ft "
                f"(t = {horizon['t_at_default']:+.2f}), i.e. A_eff = "
                f"{storage_at_horizon / specific_yield:,.0f} acres. The band is the plateau "
                "across 3-24 month horizons."
            ),
            status="MEASURED",
            note=(
                "Still the dominant uncertainty in Layer 2, but no longer a choice between "
                f"two assumptions. PHASE3_PLAN.md §12 prescribed the short-run limit "
                f"({storage_short_run:,.0f} AF/ft) and the first implementation shipped the "
                f"long-run one ({storage_long_run:,.0f}); the sweep measured that the cone "
                "spreads but plateaus at two to three times the irrigated area, so the "
                f"long-run figure is {storage_long_run / storage_at_horizon:.1f}x past the end "
                "of the evidence and the short-run figure understates a 12-month scenario by "
                f"{storage_at_horizon / storage_short_run:.1f}x. See §12."
            ),
        ),
        "grace_units_per_af": tag(
            CUBIC_M_PER_AF / region_m2,
            source=(
                "1 AF spread over the region area, in metres of equivalent water height. "
                "The GRACE series is lwe_thickness in METRES despite the 'cm' label in "
                "the source docstring: its sd is 0.0479, which is 4.8 cm (plausible for "
                "a regional anomaly) and not 0.05 mm (not)."
            ),
            status="UNTESTED",
            note=(
                "A closed-basin upper bound: it assumes every acre-foot pumped shows up "
                "in the GRACE footprint. The measured irrigation->GRACE slope "
                f"({calibration['irrigation_grace']['+ climate + trend']['beta_ft_per_af_month']:.3e} "
                "per AF/month, t = -1.93) is LARGER than this bound, which means it is "
                "not a clean structural coefficient — it is used for sign only."
            ),
        ),
        "gpcd_groundwater": tag(
            gpcd,
            source=(
                f"public_supply_groundwater_mgd x 1e6 / population, {REFERENCE_YEAR} means "
                f"({public_supply_2020:,.1f} MGD / {population_2020:,.0f} people)"
            ),
            status="MEASURED",
        ),
        "ndvi_natural": tag(
            ndvi_natural,
            source="regional NDVI baseline from frontend/computed_stats.json",
            status="MEASURED",
        ),
        "ndvi_impervious": tag(
            ndvi_natural + ndvi_slope,
            band=(ndvi_natural + ndvi_slope_band[0], ndvi_natural + ndvi_slope_band[1]),
            source=(
                f"measured: difference-in-differences on MOD13A3 NDVI against NLCD "
                f"impervious change, {ndvi_endpoints['difference_in_differences']['n_cells']:,} "
                f"MODIS cells differenced against themselves "
                f"({ndvi_endpoints['difference_in_differences']['early_years'][0]}-"
                f"{ndvi_endpoints['difference_in_differences']['early_years'][1]} vs "
                f"{ndvi_endpoints['difference_in_differences']['late_years'][0]}-"
                f"{ndvi_endpoints['difference_in_differences']['late_years'][1]}). "
                f"Slope {ndvi_slope:+.4f} NDVI per unit impervious fraction "
                f"(HC1 t = {ndvi_endpoints['difference_in_differences']['t']:+.1f}). "
                f"Cross-sectional fit over {ndvi_endpoints['n_months']} months agrees "
                f"at {ndvi_endpoints['slope']['median']:+.4f}. "
                "See scripts/phase3/ndvi_endpoints.py."
            ),
            status="MEASURED",
            note=(
                "Stated so that (ndvi_impervious - ndvi_natural) IS the measured slope, "
                "because that difference is the whole of what the lever multiplies. The "
                f"directly measured NDVI of a fully impervious cell is "
                f"{ndvi_endpoints['ndvi_impervious']['median']:.4f}; the "
                f"{ndvi_endpoints['intercept_median'] - ndvi_natural:+.4f} gap to the value "
                "used here is the difference between the regression's zero-impervious "
                f"intercept ({ndvi_endpoints['intercept_median']:.4f}, the NDVI of the land "
                f"actually being paved) and ndvi_natural ({ndvi_natural:.4f}, a p50 over "
                "months of the whole-region mean). Anchoring on the slope removes that "
                "mismatch instead of inheriting it. "
                "THE EFFECT IS ALMOST ENTIRELY ABOUT WHAT THE PAVEMENT REPLACED: split by "
                "each cell's pre-2005 NDVI, paving dry desert costs "
                f"{ndvi_endpoints['difference_in_differences']['by_prior_land_cover']['dry_desert']['slope']:+.4f} "
                "(not distinguishable from zero — desert NDVI is already near what "
                "pavement reads), while paving cropland or riparian land costs "
                f"{ndvi_endpoints['difference_in_differences']['by_prior_land_cover']['very_green_cropland_riparian']['slope']:+.4f}, "
                "a 45x difference. PHASE3_PARAMS.md §4b's assumed 0.08 endpoint implies "
                "-0.1367, which is close to the cropland-conversion case: the assumption "
                "was not absurd, it was describing the wrong half of the region. The value "
                "adopted here is the historical mix, which is the right one for a slider "
                "that adds impervious cover the way this region actually adds it."
            ),
        ),
        "ndvi_irrigated_crop": tag(
            ndvi_natural + irr_slope,
            band=(ndvi_natural + irr_band[0], ndvi_natural + irr_band[1]),
            source=(
                f"measured: difference-in-differences of MOD13A3 NDVI against HUC12 "
                f"irrigation withdrawal, {irr_did['n_cells']:,} MODIS cells differenced "
                f"against themselves ({irr_did['early_years'][0]}-{irr_did['early_years'][1]} "
                f"vs {irr_did['late_years'][0]}-{irr_did['late_years'][1]}). Slope "
                f"{irr_slope:+.4f} NDVI per unit irrigated fraction, baseline-controlled, "
                f"HC1 t = {irr_did['t_baseline_controlled']:+.1f} "
                f"({irr_did['slope']:+.4f} uncontrolled). "
                "See scripts/phase3/ndvi_endpoints.py."
            ),
            status="MEASURED",
            note=(
                "PHASE3_PARAMS.md §4b assumed 0.55 (band 0.45-0.65) and was very nearly "
                f"right: the measurement is {ndvi_natural + irr_slope:.4f}, inside the "
                f"assumed band and {abs(ndvi_natural + irr_slope - 0.55) / 0.55:.1%} from its "
                "midpoint (0.4% before the irrigated-acres correction of PROBLEMS.md P8; the "
                "calibration acres rose 5.5% and the slope fell by the same factor, so the "
                "lever irrigated_fraction x slope is unchanged). Stated relative to ndvi_natural "
                "so the difference IS the measured slope, as for ndvi_impervious. "
                "THE BAND IS NARROW BECAUSE TWO DESIGNS AGREE, NOT BECAUSE THE QUANTITY IS "
                "PRECISELY KNOWN: it spans the baseline-controlled and uncontrolled fits, "
                "and is deliberately not a confidence interval, since 126,655 MODIS cells "
                "are nowhere near independent. There is no cropland mask in the repo, so "
                "the predictor is HUC12 withdrawal turned into irrigated area by one "
                "calibration on the whole record - coarser than the impervious predictor, "
                "and identified off ~339 irrigated HUC12s rather than off fields. The "
                "cross-sectional fit is unusable here and visibly so (+0.0486): binned by "
                "irrigated fraction, mean NDVI DIPS before it rises, because HUC12s with no "
                "irrigation include the mountains while HUC12s with a little are low desert "
                "valleys - that fit is measuring elevation."
            ),
        ),
        "region_share_of_az_reduction": _region_share_constant(),
        # ── surface water (PHASE3_PLAN.md §14) ──────────────────────────────
        "regional_baseline_cfs": tag(
            regional_cfs,
            source="median(discharge_cfs_mean) x median(n_gages) over data/Final/water_surface_monthly.csv",
            status="MEASURED",
        ),
        "cfs_per_mgd": tag(
            CFS_PER_MGD, source="1 MGD = 1.547229 cfs", status="VERIFIED"
        ),
        "cfs_per_af_month": tag(
            CFS_PER_AF_MONTH,
            source="1 acre-foot delivered evenly over a mean month, in cubic feet per second",
            status="VERIFIED",
        ),
        "mean_precipitation_mm_day": tag(
            mean_precip_mm_day,
            source="mean of data/Final/precipitation_monthly.csv over the full record",
            status="MEASURED",
        ),
        "effluent_return_fraction": tag(
            0.55,
            band=(0.45, 0.70),
            source=(
                "share of municipal supply returned as treated effluent — indoor use "
                "reaches the sewer, outdoor use is consumed. The mechanism is primary: "
                "'The Santa Cruz River is largely dependent on discharges from water "
                "reclamation facilities' (A Living River: Santa Cruz River 2024, "
                "Downtown Tucson to Marana, Supplementary Report, Sonoran Institute)."
            ),
            status="UNTESTED",
            note="The fraction itself is a standard water-balance figure, not a cited value.",
        ),
        "runoff_coefficient_impervious": tag(
            0.85,
            band=(0.75, 0.95),
            source="standard rational-method runoff coefficient for paved surface",
            status="UNTESTED",
        ),
        "runoff_coefficient_natural": tag(
            0.15,
            band=(0.05, 0.25),
            source="standard rational-method runoff coefficient for desert soils",
            status="UNTESTED",
            note=(
                "The same mechanism the Living River report names — 'runoff after storms "
                "depends on the amount of impervious surface' — and the reason "
                "features.py calls precip_x_impervious the dominant urban-desert discharge term."
            ),
        ),
        "stream_capture_fraction": tag(
            0.10,
            band=(0.05, 0.25),
            source=(
                "share of regional groundwater pumping captured from streamflow rather "
                "than from storage. Stream-aquifer depletion is the documented mechanism "
                "behind the loss of perennial reach on the Santa Cruz and San Pedro."
            ),
            status="UNTESTED",
            note="Not sourced to a number. Most regional pumping is far from a stream, hence the low central value.",
        ),
        "transfer_surface_water_to_wildlife": tag(
            transfers["surface_water_to_wildlife"]["+ climate + trend"]["beta"],
            band=(
                transfers["surface_water_to_wildlife"]["+ climate + trend"]["beta"],
                transfers["surface_water_to_wildlife"]["+ climate"]["beta"],
            ),
            source=(
                "regression of bbs_abundance_anomaly on annual mean discharge_log_anomaly, "
                "n = 44 (1980-2024), PDSI-controlled. Riparian corridors carry "
                "disproportionate bird abundance in the Southwest."
            ),
            status="MEASURED",
            note=(
                "The trend-controlled (conservative) estimate ships; the band's upper end "
                "is the climate-only fit, which is significant (t = +2.19, p = 0.029). The "
                "sign is positive in both, unlike Mead, which flipped."
            ),
        ),
        "groundwater_substitution_fraction": tag(
            0.50,
            band=(0.30, 0.70),
            source=(
                "share of lost CAP water replaced by pumping rather than fallowing; "
                "Pinal's DCP mitigation explicitly funded new wells"
            ),
            status="UNTESTED",
            note="Not sourced.",
        ),
    }

    # MGD of groundwater pumping per one unit of each slider's policy delta.
    irrigation_baseline = slider_stats["irrigation_total_withdrawal_mgd"]["policy"]["baseline"]
    public_supply_baseline = slider_stats["public_supply_groundwater_mgd"]["policy"]["baseline"]
    pumping = {
        "irrigation_total_withdrawal_mgd": tag(
            irrigation_baseline / 100.0,
            source=f"1% of the {REFERENCE_YEAR} deseasonalized baseline, {irrigation_baseline:,.1f} MGD",
            status="MEASURED",
        ),
        "public_supply_groundwater_mgd": tag(
            public_supply_baseline / 100.0,
            source=f"1% of the {REFERENCE_YEAR} deseasonalized baseline, {public_supply_baseline:,.1f} MGD",
            status="MEASURED",
        ),
        "population": tag(
            gpcd / 1e6,
            source="one person x GPCD, converted to MGD — closes the population -> public supply -> aquifer loop",
            status="MEASURED",
        ),
    }
    return constants, pumping, slider_stats, calibration


# Every lever Layer 2 supplies, with the tier PHASE3_PLAN.md §11.1 assigns it.
#
#   kind "rate"  — a forcing per month that accumulates and mean-reverts. Pumping
#                  changes a storage balance, so the response builds over time.
#   kind "level" — a persistent offset applied immediately. Converting desert to
#                  pavement does not accumulate; the ground is either paved or not.
#
# "corroborated" means an independent regression on the panel agrees on the sign
# after climate and trend controls (PHASE3_PLAN.md §12). It does NOT mean the panel
# set the magnitude — no magnitude here is fitted.
LEVERS = [
    {
        "id": "irrigation_to_groundwater",
        "slider": "irrigation_total_withdrawal_mgd",
        "output": "groundwater",
        "kind": "rate",
        "path": "pumping_to_depth",
        "sign": +1,
        "tier": "corroborated",
        "mechanism": "ΔDepth/month = ΔPumping / (S_y · A). More withdrawal, deeper water.",
        "evidence": "PHASE3_PLAN.md §12: +9.018e-06 ft per AF/month, t = +3.16, n = 246.",
    },
    {
        "id": "public_supply_to_groundwater",
        "slider": "public_supply_groundwater_mgd",
        "output": "groundwater",
        "kind": "rate",
        "path": "pumping_to_depth",
        "sign": +1,
        "tier": "structural-only",
        "mechanism": "Same storage balance; municipal groundwater is pumped from the same aquifer.",
        "evidence": "None. PHASE3_PLAN.md §12 finds t = +0.17 / -0.11 / +0.91 across specifications.",
    },
    {
        "id": "population_to_groundwater",
        "via": "municipal pumping",
        "slider": "population",
        "output": "groundwater",
        "kind": "rate",
        "path": "pumping_to_depth",
        "sign": +1,
        "tier": "structural-only",
        "mechanism": "Δpopulation × per-capita municipal groundwater draw, into the storage balance.",
        "evidence": "None. PHASE3_PLAN.md §10 finds 0/5 folds and an effect of -0.113 sd, wrong-signed.",
    },
    {
        "id": "mead_to_groundwater",
        "via": "DCP shortage tier",
        "slider": "mead_pool_elevation",
        "output": "groundwater",
        "kind": "rate",
        "path": "mead_tier_to_depth",
        "sign": -1,
        "tier": "structural-only",
        "mechanism": (
            "Elevation → DCP shortage tier → Arizona reduction (kAF/yr) → in-region share "
            "→ share replaced by pumping → storage balance. A lower reservoir means more pumping."
        ),
        "evidence": (
            "Published law (LBOps Table 1), but no panel support: PHASE3_PLAN.md §12 finds the "
            "wrong sign in every specification and t falling to +0.12 under a time control. "
            "§10's 5/5-fold agreement was five folds sharing one secular trend."
        ),
    },
    {
        "id": "irrigation_to_grace",
        "slider": "irrigation_total_withdrawal_mgd",
        "output": "grace",
        "kind": "rate",
        "path": "pumping_to_storage",
        "sign": -1,
        "tier": "corroborated",
        "mechanism": "Pumped water leaves the storage GRACE measures, as equivalent water height.",
        "evidence": "PHASE3_PLAN.md §12: correct sign in all three specifications, t = -3.61 to -1.93.",
    },
    {
        "id": "public_supply_to_grace",
        "slider": "public_supply_groundwater_mgd",
        "output": "grace",
        "kind": "rate",
        "path": "pumping_to_storage",
        "sign": -1,
        "tier": "structural-only",
        "mechanism": "Same storage accounting as irrigation.",
        "evidence": "None.",
    },
    {
        "id": "population_to_grace",
        "via": "municipal pumping",
        "slider": "population",
        "output": "grace",
        "kind": "rate",
        "path": "pumping_to_storage",
        "sign": -1,
        "tier": "structural-only",
        "mechanism": "Population → municipal pumping → storage.",
        "evidence": "None.",
    },
    {
        "id": "mead_to_grace",
        "via": "DCP shortage tier",
        "slider": "mead_pool_elevation",
        "output": "grace",
        "kind": "rate",
        "path": "mead_tier_to_storage",
        "sign": +1,
        "tier": "structural-only",
        "mechanism": "A lower reservoir substitutes pumping for CAP water, drawing down storage.",
        "evidence": "None; see mead_to_groundwater.",
    },
    # ── surface water (PHASE3_PLAN.md §14) ───────────────────────────────────
    # Three paths with OPPOSING signs, which is the point: more people means more
    # effluent and more flow, while more pumping means less baseflow. The Santa Cruz
    # is perennial today because of the first and intermittent historically because
    # of the second.
    {
        "id": "population_to_surface_water",
        "via": "effluent",
        "slider": "population",
        "output": "surface_water",
        "kind": "level",
        "path": "effluent_to_flow",
        "sign": +1,
        "tier": "structural-only",
        "mechanism": (
            "Δpopulation × per-capita municipal draw × effluent return fraction → added "
            "discharge, as a proportion of total gaged flow. Counterintuitive and correct: "
            "the perennial reaches of the Santa Cruz are treated wastewater."
        ),
        "evidence": "Mechanism is primary (Living River 2024); the return fraction is assumed.",
    },
    {
        "id": "urbanization_to_surface_water",
        "via": "storm runoff",
        "slider": "impervious_pct",
        "output": "surface_water",
        "kind": "level",
        "path": "runoff_to_flow",
        "sign": +1,
        "tier": "structural-only",
        "mechanism": (
            "Δimpervious area × (runoff coefficient paved − runoff coefficient desert) × "
            "mean precipitation → added storm runoff."
        ),
        "evidence": "None from the panel; features.py already treats precip x impervious as the dominant urban-desert discharge term.",
    },
    {
        "id": "irrigation_to_surface_water",
        "via": "stream capture",
        "slider": "irrigation_total_withdrawal_mgd",
        "output": "surface_water",
        "kind": "level",
        "path": "capture_to_flow",
        "sign": -1,
        "tier": "structural-only",
        "mechanism": "A share of pumping is captured from streamflow rather than from storage.",
        "evidence": "None from the panel. Stream depletion is the documented cause of lost perennial reach on the Santa Cruz and San Pedro.",
    },
    {
        "id": "public_supply_to_surface_water",
        "via": "stream capture",
        "slider": "public_supply_groundwater_mgd",
        "output": "surface_water",
        "kind": "level",
        "path": "capture_to_flow",
        "sign": -1,
        "tier": "structural-only",
        "mechanism": "Same stream capture as irrigation.",
        "evidence": "None.",
    },
    {
        "id": "population_capture_to_surface_water",
        "via": "stream capture",
        "slider": "population",
        "output": "surface_water",
        "kind": "level",
        "path": "capture_to_flow",
        "sign": -1,
        "tier": "structural-only",
        "mechanism": (
            "The municipal pumping that population drives is also captured from streams. "
            "This runs AGAINST the effluent lever above; the net sign is an output of the "
            "model rather than an assumption, and it is positive at the shipped parameters."
        ),
        "evidence": "None.",
    },
    {
        "id": "mead_to_surface_water",
        "via": "stream capture",
        "slider": "mead_pool_elevation",
        "output": "surface_water",
        "kind": "level",
        "path": "mead_capture_to_flow",
        "sign": +1,
        "tier": "structural-only",
        "mechanism": "A lower reservoir substitutes pumping for CAP water, and part of that pumping is stream capture.",
        "evidence": "None.",
    },
    {
        "id": "urbanization_to_ndvi",
        "slider": "impervious_pct",
        "output": "ndvi",
        "kind": "level",
        "path": "landcover_impervious",
        "sign": -1,
        "tier": "structural-only",
        "mechanism": "ΔNDVI = (Δimpervious/100) × (NDVI_impervious − NDVI_natural). Pavement is not green.",
        # evidence, tier and local_effect are filled in from model/ndvi_endpoints.json
        # by _apply_ndvi_measurement() so the numbers on the card cannot drift from the
        # measurement that produced them.
        "evidence": None,
    },
    {
        "id": "irrigation_to_ndvi",
        "slider": "irrigation_total_withdrawal_mgd",
        "output": "ndvi",
        "kind": "level",
        "path": "landcover_irrigated",
        "sign": +1,
        "tier": "structural-only",
        "mechanism": (
            "ΔNDVI = Δfraction_of_baseline_withdrawal × irrigated_fraction × "
            "(NDVI_crop − NDVI_natural). Positive on irrigated pixels while negative on "
            "groundwater — the tension PHASE3_PLAN.md §4 wants shown, not hidden."
        ),
        # filled in from model/ndvi_endpoints.json by _apply_ndvi_measurement()
        "evidence": None,
    },
]

# Transfer edges: an output's structural displacement propagating to another output.
# This is how surface water and wildlife get a human response at all — the outputs
# form a chain, so ONE coefficient per edge carries every lever that feeds the source.
#
# There is no ndvi -> wildlife edge on purpose. It was estimated
# (scripts/phase3/transfer_calibration.py) and came back a null: t = -0.19 without a
# trend control and +0.17 with one, p ~ 0.85 either way, and the sign flips between
# specifications. With no panel support AND no citable published elasticity, building
# it would be inventing a number, which is the thing Layer 2 exists to avoid. The
# frontend's stale `ndvi feedsInto wildlife` claim is corrected rather than honoured.
TRANSFERS = [
    {
        "id": "surface_water_to_wildlife",
        "from": "surface_water",
        "to": "wildlife",
        "constant": "transfer_surface_water_to_wildlife",
        "sign": +1,
        "tier": "structural-only",
        "mechanism": (
            "Riparian corridors carry disproportionate bird abundance in the Southwest, "
            "so streamflow drives habitat. This edge is what lets every human lever reach "
            "wildlife at all."
        ),
        "evidence": (
            "Regression on the project's own annual panel, n = 44 (1980-2024): "
            "+0.1953 (t = +2.19, p = 0.029) PDSI-controlled, attenuating to +0.0536 "
            "(t = +0.42) with a linear trend added. Positive in both; the conservative "
            "estimate ships."
        ),
    },
]

# Wildfire gets nothing, and that is a conclusion rather than an omission. Human
# ignitions dominate US fire counts, but ignition is not the limiting factor for
# large-fire extent in the Southwest -- fuel and weather are -- and the MTBS target
# measures exactly that extent. No coefficient from population or impervious cover to
# large-fire risk could be sourced, and PHASE3_PLAN.md §10 measured both at under
# 0.05 sd. It ships as climate-only and the interface says so.
CLIMATE_ONLY_OUTPUTS = ["wildfire"]


def _apply_ndvi_measurement(levers: list[dict], endpoints: dict) -> None:
    """Fill the urbanization -> NDVI lever in from the raster measurement.

    Two things change beyond the coefficient. The lever's `evidence` used to read "None.
    PHASE3_PLAN.md §10 finds an effect of -0.010 sd" — that was the panel's verdict, and
    the panel could not see this: an eight-county monthly mean has no way to separate a
    2%-of-area land-cover change from weather. The rasters can, because they resolve the
    change where it happens. So the tier goes to `corroborated`: §11.1 defines that as a
    structural mechanism plus an independent empirical check that agrees on the sign
    after climate and trend controls, and a difference-in-differences across 130,833
    cells with regional drift in the intercept is exactly that check.

    `local_effect` exists because the regional number is honest and useless on its own.
    -0.97 points region-wide is what a 2%-of-area change does to a regional mean; it says
    nothing about what happens to the land that was actually paved, which is the thing a
    person moving an urbanization slider is picturing.
    """
    did = endpoints["difference_in_differences"]
    strata = did["by_prior_land_cover"]
    natural = endpoints["ndvi_natural_in_repo"]
    irr = endpoints["irrigation_difference_in_differences"]

    for lever in levers:
        if lever["id"] == "irrigation_to_ndvi":
            # Same promotion, same reason: the panel could not see this lever (§10
            # found 1 of 5 folds, wrong-signed) because an eight-county monthly mean
            # cannot separate a 2.5%-of-area land-cover effect from weather. The
            # rasters resolve it where it happens.
            lever["tier"] = "corroborated"
            lever["evidence"] = (
                f"Difference-in-differences on MOD13A3 against HUC12 irrigation "
                f"withdrawal: {irr['n_cells']:,} MODIS cells differenced against "
                f"themselves ({irr['early_years'][0]}-{irr['early_years'][1]} vs "
                f"{irr['late_years'][0]}-{irr['late_years'][1]}), slope "
                f"{irr['slope_baseline_controlled']:+.4f} NDVI per unit irrigated "
                f"fraction, HC1 t = {irr['t_baseline_controlled']:+.1f}. The dose-response "
                "is monotone: cells that LOST irrigation fell below the regional drift, "
                "cells that gained rose above it."
            )
            continue
        if lever["id"] != "urbanization_to_ndvi":
            continue
        lever["tier"] = "corroborated"
        lever["evidence"] = (
            f"Difference-in-differences on the project's own MOD13A3 and NLCD rasters: "
            f"{did['n_cells']:,} MODIS cells differenced against themselves "
            f"({did['early_years'][0]}-{did['early_years'][1]} vs "
            f"{did['late_years'][0]}-{did['late_years'][1]}), slope {did['slope']:+.4f} "
            f"NDVI per unit impervious fraction, HC1 t = {did['t']:+.1f}. A cross-sectional "
            f"fit over {endpoints['n_months']} months agrees at "
            f"{endpoints['slope']['median']:+.4f}."
        )
        lever["local_effect"] = {
            "value": did["slope"],
            "as_percent_of_baseline": did["slope"] / natural * 100,
            "headline": (
                f"On the land actually paved, NDVI falls "
                f"{abs(did['slope'] / natural * 100):.0f}%."
            ),
            "detail": (
                "The region-wide number is small because the paving is small: the slider's "
                "full range is about 2% of the eight counties, and a regional mean is built "
                "to average that away. The same coefficient applied to the whole region "
                "would be about 40 score points. What the pavement REPLACED is most of the "
                f"story — paving dry desert costs "
                f"{strata['dry_desert']['slope']:+.4f} NDVI, which is not distinguishable "
                f"from zero because desert is already near what pavement reads, while paving "
                f"cropland or riparian land costs "
                f"{strata['very_green_cropland_riparian']['slope']:+.4f}, about "
                f"{abs(strata['very_green_cropland_riparian']['slope'] / natural * 100):.0f}% "
                "of the region's whole vegetation signal."
            ),
        }


def main() -> None:
    constants, pumping, slider_stats, calibration = build()
    _apply_ndvi_measurement(LEVERS, json.loads(NDVI_ENDPOINTS_PATH.read_text()))

    # Mean-reversion rates from PHASE3_PLAN.md §4b, fitted on the observed panel by
    # `slider_sensitivity.py --mode lambda`. These bound the rate levers: a sustained
    # forcing f converges to f/lambda instead of ramping without limit, which §4a
    # measured to be the difference between a scenario and an unbounded number.
    reversion = {
        "grace": tag(
            0.0386,
            source="regression of dy on (y_lag1 - ybar), t = -2.28, e-folding 25.9 months",
            status="MEASURED",
        ),
        "groundwater": tag(
            0.0157,
            source="same regression, t = -1.24, e-folding 63.8 months",
            status="MEASURED",
            note="Not significant. An aquifer barely reverting is physically right, but the rate is weakly identified.",
        ),
        # Needed for Layer 3's integration of the LEARNED residual, not for any
        # structural lever — neither of these outputs has one.
        "ndvi": tag(
            0.2221,
            source="same regression, t = -5.99, e-folding 4.5 months",
            status="MEASURED",
            note="Vegetation recovering within a season.",
        ),
        "surface_water": tag(
            0.3511,
            source="same regression, t = -10.81, e-folding 2.8 months",
            status="MEASURED",
            note="Flow recedes fast.",
        ),
    }

    integrate_learned = tag(
        False,
        source="PHASE3_PLAN.md §13 — measured both ways, see the table there",
        status="MEASURED",
        note=(
            "Whether Layer 3 also integrates the LEARNED residual, not just the "
            "structural forcing. Off: §4a established the exported models are not "
            "dynamical systems, and integrating them amplifies D2's wrong signs "
            "(Mead -> NDVI goes from -3.27 to -14.01 score points) without adding "
            "any dynamics the models actually contain. The structural layer is "
            "integrated either way, because a storage balance really does accumulate."
        ),
    )

    result = {
        "meta": {
            "generated_by": "scripts/phase3/structural_params.py",
            "reference_year": REFERENCE_YEAR,
            "plan": "PHASE3_PLAN.md §4 (Layer 2), §4b (Layer 3), §12 (calibration)",
            "note": (
                "Magnitudes are structural: water balance, land-cover arithmetic and "
                "published policy. None is fitted. `tier: corroborated` means an "
                "independent regression agrees on the SIGN, not that it set the value."
            ),
        },
        "constants": constants,
        "pumping_mgd_per_slider_unit": pumping,
        "mead_tiers": MEAD_TIERS,
        "mead_baseline_elevation": slider_stats["mead_pool_elevation"]["policy"]["baseline"],
        "irrigation_baseline_mgd": slider_stats["irrigation_total_withdrawal_mgd"]["policy"]["baseline"],
        "reversion_per_month": reversion,
        "integrate_learned_residual": integrate_learned,
        "levers": LEVERS,
        "transfers": TRANSFERS,
        "climate_only_outputs": CLIMATE_ONLY_OUTPUTS,
    }

    with OUTPUT_FILE.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"Structural parameters written to {OUTPUT_FILE}\n")
    print(f"  {'constant':36s}{'value':>16s}  status")
    for name, entry in constants.items():
        print(f"  {name:36s}{entry['value']:>16,.6g}  {entry['status']}")
    print(f"\n  {'lever':30s}{'output':14s}{'kind':7s}{'sign':>5s}  tier")
    for lever in LEVERS:
        print(
            f"  {lever['id']:30s}{lever['output']:14s}{lever['kind']:7s}"
            f"{lever['sign']:+5d}  {lever['tier']}"
        )
    print(f"\n  {'transfer':30s}{'to':14s}{'coefficient':>13s}  tier")
    for transfer in TRANSFERS:
        print(
            f"  {transfer['id']:30s}{transfer['to']:14s}"
            f"{constants[transfer['constant']]['value']:>13.4f}  {transfer['tier']}"
        )
    counts = {"corroborated": 0, "structural-only": 0}
    for lever in LEVERS:
        counts[lever["tier"]] += 1
    print(f"\n  {counts['corroborated']} corroborated, {counts['structural-only']} structural-only")
    print(f"  climate-only outputs (no structural path): {', '.join(CLIMATE_ONLY_OUTPUTS)}")


if __name__ == "__main__":
    main()
