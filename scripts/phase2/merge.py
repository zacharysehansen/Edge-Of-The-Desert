"""
merge.py
--------
Load all data/Final/ CSVs and join them into two clean panels:

  data/processed/monthly_panel.csv  — 2000-01 through 2023-12, one row per month
  data/processed/annual_panel.csv   — one row per year, monthly inputs aggregated

Run directly to see a coverage summary:
    python scripts/phase2/merge.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL = REPO_ROOT / "data" / "Final"
PROCESSED = REPO_ROOT / "data" / "processed"

MONTHLY_PANEL_PATH = PROCESSED / "monthly_panel.csv"
ANNUAL_PANEL_PATH = PROCESSED / "annual_panel.csv"

# Modeling window bookends — used by features.py and model scripts.
#
# The start is fixed by GRACE: everything before 2002-04 is zero-fill, not
# measurement, so no model may see it. Six months of roll warmup puts the first
# usable row at 2002-10.
MONTHLY_WINDOW_START = "2002-10"

# The end is *per model*, not global. Two source series (irrigation, public supply)
# and one target (well depth) stop at 2020-12; most of the rest run to 2023-12, and
# discharge now runs to 2025-12. A model whose target outlives the short series is
# capped at 2020 only if it carries features derived from them — so those models drop
# them instead. See MONTHLY_CAPPED_2020_BASES and features._monthly_dataset.
MONTHLY_WINDOW_END = "2020-12"
MONTHLY_WINDOW_END_EXTENDED = "2023-12"

# Base columns whose source series end 2020-12. Any feature derived from one of
# these is NaN after that date, so a single one of them in a model's X drags the
# whole model back to the short window via the not-null mask.
MONTHLY_CAPPED_2020_BASES = [
    "irrigation_total_withdrawal_mgd",
    "public_supply_groundwater_mgd",
    "depth_to_water_ft_mean",
    "depth_to_water_anomaly_ft",
    "n_wells",
    # NOTE: the discharge columns used to live here. They no longer do —
    # water_surface.py now
    # pulls 1980-2025, so discharge is the *longest* monthly series in the panel, not
    # one of
    # the shortest. Groundwater is still 2000-2020 and still belongs here.
]

# The monthly panel's index floor. 1968 is the BBS floor — the earliest target
# anywhere in the project. nClimDiv reaches 1895, so this is bounded by data we
# have, not by data we lack.
PANEL_START = "1968-01"

# The panel ceiling. NWIS discharge now runs to 2025; everything else stops earlier
# and is simply NaN up here, which each model's not-null mask handles.
PANEL_END = "2025-12"

# The annual window is now bounded by the *target*, not by a 2000-floored feature.
# Wildlife (BBS) runs 1968-2024. Pre-2000 rows carry nClimDiv drought/temperature/
# precipitation and nothing else — every satellite input (NDVI, GRACE) and the USDM
# are NaN out there, so any annual model reaching back must drop them.
# See _WILDLIFE_FEATURES in features.py.
ANNUAL_WINDOW_START = 1968
ANNUAL_WINDOW_END = 2024


# ---------------------------------------------------------------------------
# Monthly aggregation rules for producing the annual panel
# ---------------------------------------------------------------------------
# "mean"   — continuous rate (temperature, NDVI, drought index, impervious %)
# "sum"    — cumulative volume over the year (precipitation, withdrawals)
# "june"   — June value only (pre-monsoon snapshot; BBS surveys happen in June)
# "jja"    — June–July–August mean (peak heat/stress season)
# "last"   — December value (end-of-year stock, e.g. lake elevation)

MONTHLY_TO_ANNUAL_RULES = {
    # nClimDiv — the only inputs that exist before 2000. Every annual model
    # reaching past the satellite era is built from these three and nothing else.
    "nclimdiv_pdsi": ["mean", "jja"],
    "nclimdiv_temperature_c": ["mean", "jja"],
    "nclimdiv_precipitation_mm_day": ["mean", "sum", "jja"],
    "population": ["mean", "june"],
    "irrigation_total_withdrawal_mgd": ["sum", "mean"],
    "public_supply_groundwater_mgd": ["sum", "mean"],
    "mead_pool_elevation": ["mean", "june", "last"],
    "mead_total_release": ["sum", "mean"],
    "usdm_dsci": ["mean", "jja"],
    "water_stress_score": ["mean", "jja"],
    "temperature_2m_c": ["mean", "jja"],
    "precipitation_mm_day": ["mean", "sum", "jja"],
    "grace_groundwater_anomaly": ["mean"],
    "grace_available": ["mean"],
    "ndvi": ["mean", "jja"],
    "impervious_pct": ["mean"],
}


def _read_monthly(filename: str, rename: dict | None = None) -> pd.DataFrame:
    """Read a year_month-indexed monthly CSV from data/Final/."""
    path = FINAL / filename
    df = pd.read_csv(path, parse_dates=["year_month"])
    df["year_month"] = df["year_month"].dt.to_period("M")
    df = df.set_index("year_month").sort_index()
    if rename:
        df = df.rename(columns=rename)
    return df


def _read_annual(filename: str) -> pd.DataFrame:
    """Read a year-indexed annual CSV from data/Final/."""
    path = FINAL / filename
    df = pd.read_csv(path)
    df = df.rename(columns={"year": "year"})
    df["year"] = df["year"].astype(int)
    df = df.set_index("year").sort_index()
    return df


def build_monthly_panel() -> pd.DataFrame:
    """
    Join all monthly CSVs on a complete 2000-01 to 2023-12 PeriodIndex.
    Returns a DataFrame with every column from every source file.
    Columns missing for a sub-range (irrigation ends 2020-12) are NaN
    outside their coverage window — this is intentional and reported.
    """
    # The panel starts at the BBS floor (1968), not 2000. Every satellite/USDM
    # source is simply NaN out there, which is correct and harmless — each model
    # declares its own window, and the monthly models all start at
    # MONTHLY_WINDOW_START (2002-10) regardless.
    #
    # What the extra rows buy is the *annual* panel: nClimDiv carries drought,
    # temperature and precipitation back to 1895, so the annual models can now run
    # as far back as their target reaches rather than as far back as their shortest
    # feature reaches.
    full_index = pd.period_range(PANEL_START, PANEL_END, freq="M")

    frames = [
        _read_monthly("nclimdiv_monthly.csv"),
        _read_monthly("azpop_monthly.csv"),
        _read_monthly("irrigation_monthly.csv"),
        _read_monthly("public_supply_monthly.csv"),
        _read_monthly("lake_mead_monthly.csv"),
        _read_monthly("water_stress_monthly.csv"),
        _read_monthly("temperature_monthly.csv"),
        _read_monthly("precipitation_monthly.csv"),
        _read_monthly("grace_monthly.csv"),
        _read_monthly("ndvi_monthly.csv"),
        _read_monthly("wildfire_monthly.csv"),
        _read_monthly("urbanization_monthly.csv"),
        _read_monthly("groundwater_levels_monthly.csv"),
        _read_monthly("water_surface_monthly.csv"),
        # GLDAS-2.1 land-surface state, 2000-01..2023-12 (PHASE3_PLAN.md §25-§26).
        # Consumed only by the GRACE model, as its storage-change input.
        _read_monthly("gldas_monthly.csv"),
    ]

    panel = pd.DataFrame(index=full_index)
    panel.index.name = "year_month"

    for df in frames:
        for col in df.columns:
            panel[col] = df[col]

    panel["month"] = panel.index.month.astype(int)

    return panel


# ---------------------------------------------------------------------------
# Aggregate monthly → annual
# ---------------------------------------------------------------------------


def _agg_column(monthly: pd.DataFrame, col: str, rules: list[str]) -> pd.DataFrame:
    """Apply a list of aggregation rules for one column, return annual DataFrame."""
    # Work on a plain datetime-indexed copy for resample
    s = monthly[col].copy()
    s.index = s.index.to_timestamp()

    frames = []
    for rule in rules:
        if rule == "mean":
            agg = s.resample("YE").mean().rename(f"{col}_annual_mean")
        elif rule == "sum":
            agg = s.resample("YE").sum(min_count=1).rename(f"{col}_annual_sum")
        elif rule == "june":
            june = s[s.index.month == 6]  # noqa: PLR2004
            agg = june.resample("YE").first().rename(f"{col}_june")
        elif rule == "jja":
            jja = s[s.index.month.isin([6, 7, 8])]
            agg = jja.resample("YE").mean().rename(f"{col}_jja_mean")
        elif rule == "last":
            agg = s.resample("YE").last().rename(f"{col}_year_end")
        else:
            raise ValueError(f"Unknown aggregation rule '{rule}' for column '{col}'")
        frames.append(agg)

    result = pd.concat(frames, axis=1)
    result.index = result.index.year.astype(int)
    result.index.name = "year"
    return result


def build_annual_panel(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the monthly panel to annual grain and join annual target files.
    Returns a DataFrame indexed by year (integer).
    """
    frames = []
    for col, rules in MONTHLY_TO_ANNUAL_RULES.items():
        if col not in monthly.columns:
            continue
        frames.append(_agg_column(monthly, col, rules))

    annual = pd.concat(frames, axis=1)

    # Log-transformed annual precipitation total
    if "precipitation_mm_day_annual_sum" in annual.columns:
        annual["log_precip_annual"] = np.log1p(
            annual["precipitation_mm_day_annual_sum"]
        )

    # The same, on the long-record nClimDiv series, so pre-2000 rows have one too.
    if "nclimdiv_precipitation_mm_day_annual_sum" in annual.columns:
        annual["nclimdiv_log_precip_annual"] = np.log1p(
            annual["nclimdiv_precipitation_mm_day_annual_sum"]
        )

    # Linear year index
    annual["year_linear"] = annual.index - ANNUAL_WINDOW_START

    # Join wildfire targets
    wildfire = _read_annual("wildfire_annual.csv")
    annual = annual.join(wildfire, how="left")

    wildlife = _read_annual("wildlife_annual.csv")
    wildlife = wildlife.rename(
        columns={
            "route_count": "bbs_route_count",
            "total_abundance": "bbs_total_abundance",
            "species_richness": "bbs_species_richness",
            # The target. Per-route log-abundance anomaly — effort-corrected.
            "abundance_anomaly": "bbs_abundance_anomaly",
            # LEGACY. A sum over whichever routes were surveyed, so it correlates +0.94
            # with route_count over 1968-2024: a survey-effort index, not a bird index.
            # Kept for comparison only. Never model on it. See
            # scripts/phase1/wildlife.py.
            "abundance_index": "bbs_abundance_index",
        }
    )
    annual = annual.join(wildlife, how="left")

    return annual


def _coverage_report(panel: pd.DataFrame, name: str) -> None:
    """Print a per-column NaN summary for the panel."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {len(panel)} rows  |  {len(panel.columns)} columns")
    print(f"{'='*60}")

    total = len(panel)
    for col in panel.columns:
        n_null = panel[col].isna().sum()
        if n_null == 0:
            status = "full"
        else:
            pct = 100 * n_null / total
            status = f"{n_null} NaN  ({pct:.1f}%)"
        print(f"  {col:<45} {status}")


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build both panels, write to processed/, and return (monthly, annual).
    Called by features.py and run_all.py.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)

    monthly = build_monthly_panel()
    annual = build_annual_panel(monthly)

    monthly.to_csv(MONTHLY_PANEL_PATH)
    annual.to_csv(ANNUAL_PANEL_PATH)

    return monthly, annual


def main() -> None:
    monthly, annual = run()
    _coverage_report(monthly, "Monthly Panel  →  data/processed/monthly_panel.csv")
    _coverage_report(annual, "Annual Panel   →  data/processed/annual_panel.csv")
    print(f"\nMonthly panel  : {MONTHLY_PANEL_PATH}")
    print(f"Annual panel   : {ANNUAL_PANEL_PATH}")


if __name__ == "__main__":
    main()
