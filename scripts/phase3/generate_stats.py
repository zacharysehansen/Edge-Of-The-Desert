import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "Final"
OUTPUT_FILE = REPO_ROOT / "frontend" / "computed_stats.json"

FILE_TO_KEY = {
    # Sliders
    "azpop_monthly.csv": {
        "json_key": "population",
        "column": "population",
    },
    "irrigation_monthly.csv": {
        "json_key": "irrigation_total_withdrawal_mgd",
        "column": "irrigation_total_withdrawal_mgd",
    },
    "public_supply_monthly.csv": {
        "json_key": "public_supply_groundwater_mgd",
        "column": "public_supply_groundwater_mgd",
    },
    "urbanization_monthly.csv": {
        "json_key": "impervious_pct",
        "column": "impervious_pct",
    },
    "lake_mead_monthly.csv": {
        "json_key": "mead_pool_elevation",
        "column": "mead_pool_elevation",
    },
    "precipitation_monthly.csv": {
        "json_key": "precipitation_mm_day",
        "column": "precipitation_mm_day",
    },
    "water_stress_monthly.csv": {
        "json_key": "usdm_dsci",
        "column": "usdm_dsci",
    },
    "nclimdiv_monthly.csv": {
        # THE drought slider. PDSI is signed (- dry, + wet), runs back to 1895, and is
        # the
        # exact input for the two strongest models (surface water, wildlife).
        # usdm_dsci is
        # still a slider stat because NDVI/GRACE/wildfire use it, but the frontend no
        # longer
        # exposes it directly — it derives it from PDSI. See DROUGHT_PDSI_TO_DSCI in
        # frontend/models.js. Zero is "normal", a real reading, so it must not be
        # dropped.
        "json_key": "nclimdiv_pdsi",
        "column": "nclimdiv_pdsi",
        "zero_is_data": True,
    },
    "temperature_monthly.csv": {
        "json_key": "temperature_2m_c",
        "column": "temperature_2m_c",
    },
    # Outputs
    "grace_monthly.csv": {
        "json_key": "grace",
        "column": "grace_groundwater_anomaly",
    },
    "ndvi_monthly.csv": {
        "json_key": "ndvi",
        "column": "ndvi",
    },
    "groundwater_levels_monthly.csv": {
        "json_key": "groundwater",
        # The per-well anomaly, not the raw roster mean — this is what the model
        # predicts. Positive = deeper than normal = less groundwater. A zero here is
        # "exactly normal", a real and meaningful reading, so it must not be dropped.
        "column": "depth_to_water_anomaly_ft",
        "zero_is_data": True,
    },
    "water_surface_monthly.csv": {
        "json_key": "surface_water",
        # Per-gage log anomaly. Zero = normal flow, a real reading.
        "column": "discharge_log_anomaly",
        "zero_is_data": True,
    },
    "wildfire_monthly.csv": {
        "json_key": "wildfire",
        "column": "wildfire_risk_index",
        # A zero here is a real observation — 61% of months have no large fire.
        # Dropping zeros would put the frontend's resting wildfire risk at 0.41
        # (the median of fire months only) instead of 0.0.
        "zero_is_data": True,
    },
    "wildlife_annual.csv": {
        "json_key": "wildlife",
        # The per-route anomaly, not the old min-max index. The old one was a sum over
        # whichever routes were surveyed, so it tracked survey effort (r = +0.94 with
        # route_count) rather than birds. Zero here means "an average year", a real and
        # meaningful reading, so it must not be dropped.
        "column": "abundance_anomaly",
        "zero_is_data": True,
    },
}

SLIDER_KEYS = {
    "population",
    "irrigation_total_withdrawal_mgd",
    "public_supply_groundwater_mgd",
    "impervious_pct",
    "mead_pool_elevation",
    "precipitation_mm_day",
    "temperature_2m_c",
    "usdm_dsci",
    "nclimdiv_pdsi",
}

OUTPUT_KEYS = {
    "grace",
    "ndvi",
    "groundwater",
    "surface_water",
    "wildfire",
    "wildlife",
}


# ── D5: policy reparameterization ─────────────────────────────────────────────
#
# The raw min/max above are the p5/p95 of the pooled monthly series, and for most
# of these sliders that is not a policy axis at all (PHASE3_PLAN.md D5).
# Irrigation is 95.2% a repeating month-of-year template, so its raw "max" means
# *June*, not "we pumped hard"; population is 0.4% within-year, so its raw "min"
# means *2002*. Dragging either asks a question month_sin/month_cos already
# answers, which is one of the reasons the learned models find nothing there.
#
# So every exposed slider also gets a `policy` block: a delta around a
# DESEASONALIZED baseline, with the seasonal shape supplied by the month selector
# where it belongs. The frontend reconstructs the raw value the models were
# trained on:
#
#     mode "scale"   raw(month, d) = baseline * (1 + d/100) * seasonal[month-1]
#     mode "offset"  raw(month, d) = baseline + d + seasonal[month-1]
#
# so d = 0 is the climatological normal for that month, and d is a sustained
# policy change in units a person can defend.
#
# The raw min/max/default stay in the file: models.js still needs the raw scale
# for the ONNX inputs, and slider_sensitivity.py reads them for the historical
# sweep reported in PHASE3_PLAN.md §1.

# The last calendar year for which *every* human series exists. Irrigation and
# public supply both stop at 2020; population, impervious and Mead run to 2023.
# Using one common reference year keeps the default scenario from mixing epochs.
REFERENCE_YEAR = 2020

# `range_from="sustained"` derives the delta range from the observed spread of
# deseasonalized ANNUAL means, not monthly ones. Used for the climate sliders,
# which have no policy scenario to range them on — "how much has it varied" is
# the only available answer.
#
# The annual grain matters and is a deliberate departure from PHASE3_PARAMS.md §6,
# which specified the monthly residual p5/p95. A slider is held for the whole
# scenario duration, so its range has to be what a SUSTAINED excursion can reach,
# and a sustained monthly extreme is not a year:
#
#     precipitation   monthly p5/p95  -93% .. +160%     annual  -32% .. +45%
#     temperature     monthly p5/p95  -2.29 .. +2.52 C  annual  -0.71 .. +0.79 C
#     PDSI            monthly p5/p95  -2.64 .. +3.64    annual  -1.96 .. +2.42
#
# No year in the record came close to +160% precipitation; the wettest was +73.5%.
# Ranging on the monthly figure and then holding it for twelve months measured
# precipitation -> surface water at +121 points on a 0-100 scale, which is not a
# scenario, it is a saturated bar. §6's stated reason for the monthly choice was
# to keep climate magnitudes comparable to PHASE3_PLAN.md §1 (precip -> surface
# water +40.1); the annual range is what actually delivers that.
#
# The five human levers get explicit policy ranges instead, because "how much
# could we plausibly change this" is a different question from "how much has it
# varied".
POLICY_SPECS = {
    "population": {
        "mode": "offset",
        "unit": "people added",
        "min": -500_000,
        "max": 3_000_000,
        "step": 50_000,
    },
    "irrigation_total_withdrawal_mgd": {
        "mode": "scale",
        "unit": "% of annual withdrawal",
        "min": -60,
        "max": 40,
        "step": 1,
    },
    "public_supply_groundwater_mgd": {
        "mode": "scale",
        "unit": "% of annual withdrawal",
        "min": -40,
        "max": 60,
        "step": 1,
    },
    "impervious_pct": {
        "mode": "offset",
        "unit": "points of impervious cover",
        "min": -0.4,
        "max": 2.0,
        "step": 0.05,
    },
    # The range matters here beyond ergonomics: the DCP shortage tiers this lever
    # drives (PHASE3_PARAMS.md §1) key on elevations down to 1,025 ft, and the raw
    # p5/p95 range (1,058-1,195) cannot reach them. Around the 2020 deseasonalized
    # baseline of ~1,088.8 ft this spans ~994-1,219 ft and reaches every tier.
    "mead_pool_elevation": {
        "mode": "offset",
        "unit": "ft",
        "min": -95,
        "max": 130,
        "step": 1,
    },
    "temperature_2m_c": {
        "mode": "offset",
        "unit": "\u00b0C",
        "step": 0.1,
        "range_from": "sustained",
    },
    "precipitation_mm_day": {
        "mode": "scale",
        "unit": "% of normal",
        "step": 1,
        "range_from": "sustained",
    },
    "nclimdiv_pdsi": {
        "mode": "offset",
        "unit": "PDSI",
        "step": 0.1,
        "range_from": "sustained",
        # PDSI runs back to 1895. The seasonal shape and the baseline have to come
        # from the same era as everything else, or "normal" means the 20th century.
        "since_year": 2000,
    },
}


def compute_policy(
    year_month: pd.Series, values: pd.Series, spec: dict
) -> dict[str, object]:
    """Build the deseasonalized `policy` block for one slider.

    `seasonal[m]` is the month-of-year climatology expressed as a factor (scale)
    or an offset (offset), so the deseasonalized series is the raw series with
    the calendar divided/subtracted out. `baseline` is the mean of that
    deseasonalized series over REFERENCE_YEAR.
    """
    mode = spec["mode"]
    period = pd.PeriodIndex(year_month, freq="M")
    frame = pd.DataFrame(
        {"value": values.to_numpy(), "month": period.month, "year": period.year}
    ).dropna()

    since = spec.get("since_year")
    if since is not None:
        frame = frame[frame["year"] >= since]

    record_mean = float(frame["value"].mean())
    monthly_mean = frame.groupby("month")["value"].mean()
    if sorted(monthly_mean.index) != list(range(1, 13)):
        raise ValueError(f"Need all 12 months to deseasonalize; got {list(monthly_mean.index)}")

    if mode == "scale":
        seasonal = monthly_mean / record_mean
        deseasonalized = frame["value"].to_numpy() / seasonal.reindex(frame["month"]).to_numpy()
    elif mode == "offset":
        seasonal = monthly_mean - record_mean
        deseasonalized = frame["value"].to_numpy() - seasonal.reindex(frame["month"]).to_numpy()
    else:
        raise ValueError(f"Unknown policy mode '{mode}'")

    reference = deseasonalized[(frame["year"] == REFERENCE_YEAR).to_numpy()]
    if reference.size == 0:
        raise ValueError(f"No {REFERENCE_YEAR} rows to anchor the policy baseline on")
    baseline = float(np.mean(reference))

    policy = {
        "mode": mode,
        "unit": spec["unit"],
        "baseline": round(baseline, 4),
        "reference_year": REFERENCE_YEAR,
        "seasonal": [round(float(seasonal[m]), 6) for m in range(1, 13)],
        "default": 0,
        "step": spec["step"],
    }

    if spec.get("range_from") == "sustained":
        # The deseasonalized series as a departure from its own record mean, then
        # averaged over each complete calendar year: the size of a departure that
        # actually persisted for as long as the slider holds it.
        if mode == "scale":
            residual = (deseasonalized / record_mean - 1.0) * 100.0
        else:
            residual = deseasonalized - record_mean
        by_year = pd.Series(residual, index=frame["year"].to_numpy()).groupby(level=0)
        annual = by_year.mean()[by_year.count() == 12]
        if annual.size < 10:
            raise ValueError(
                f"Only {annual.size} complete years available to range this slider"
            )
        policy["min"] = round(float(np.percentile(annual, 5)), 2)
        policy["max"] = round(float(np.percentile(annual, 95)), 2)
        policy["range_basis"] = "sustained (annual-mean) deseasonalized p5/p95"
    else:
        policy["min"] = spec["min"]
        policy["max"] = spec["max"]
        policy["range_basis"] = "policy scenario"

    return policy


def compute_stats(values: pd.DataFrame, zero_is_data: bool = False) -> dict[str, float]:
    """
    Return 5th, 50th, and 95th percentiles after removing NaNs.

    Zeros are dropped by default: for most variables here (population, discharge,
    withdrawal) a zero is a missing reading rather than a measurement. Set
    `zero_is_data` for series where zero is a real observation — wildfire, where
    a month with no large fire is the single most common outcome.
    """
    values = values.dropna()
    if not zero_is_data:
        values = values[values != 0]

    if values.empty:
        raise ValueError("No values left after filtering.")

    return {
        "min": round(float(np.percentile(values, 5)), 4),
        "median": round(float(np.percentile(values, 50)), 4),
        "max": round(float(np.percentile(values, 95)), 4),
    }


def main() -> None:
    slider_stats = {}
    output_stats = {}

    for filename, config in FILE_TO_KEY.items():
        filepath = DATA_PATH / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Missing file: {filepath}")

        df = pd.read_csv(filepath)

        key = config["json_key"]
        column = config["column"]

        if column not in df.columns:
            raise ValueError(
                f"{filename} does not contain expected column '{column}'. "
                f"Available columns: {list(df.columns)}"
            )

        stats = compute_stats(df[column], config.get("zero_is_data", False))

        if key in SLIDER_KEYS:
            entry = {
                "min": stats["min"],
                "max": stats["max"],
                "default": stats["median"],
            }
            # usdm_dsci has no policy block on purpose: the frontend does not expose
            # it as a control, it derives it from the PDSI slider (see
            # DROUGHT_PDSI_TO_DSCI in frontend/models.js).
            if key in POLICY_SPECS:
                entry["policy"] = compute_policy(
                    df["year_month"], df[column], POLICY_SPECS[key]
                )
            slider_stats[key] = entry
        elif key in OUTPUT_KEYS:
            output_stats[key] = {
                "min": stats["min"],
                "max": stats["max"],
                "baseline": stats["median"],
            }
        else:
            raise KeyError(f"Unknown key '{key}' in FILE_TO_KEY.")

    missing_policy = POLICY_SPECS.keys() - {
        k for k, v in slider_stats.items() if "policy" in v
    }
    if missing_policy:
        raise RuntimeError(f"Missing policy blocks for: {sorted(missing_policy)}")

    missing_sliders = SLIDER_KEYS - slider_stats.keys()
    missing_outputs = OUTPUT_KEYS - output_stats.keys()

    if missing_sliders:
        raise RuntimeError(f"Missing slider stats for: {sorted(missing_sliders)}")

    if missing_outputs:
        raise RuntimeError(f"Missing output stats for: {sorted(missing_outputs)}")

    result = {
        "SLIDER_STATS": slider_stats,
        "OUTPUT_STATS": output_stats,
    }

    with OUTPUT_FILE.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"Stats written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
