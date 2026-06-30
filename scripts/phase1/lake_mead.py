"""
lake_mead.py
------------
Pulls four Lake Mead CSV endpoints from the Bureau of Reclamation
HydroData Navigator (reservoir ID 921), filters each to the project
date range, aggregates daily to monthly, and merges into one table.

Input  : Bureau of Reclamation HydroData Navigator (direct CSV URLs)
         - Pool elevation (parameter 49)
         - Storage (parameter 17)
         - Total release (parameter 42)
         - Release volume (parameter 43)

Output : data/processed/lake_mead_monthly.csv

Columns in output:
    year_month              - str, format YYYY-MM
    mead_pool_elevation     - float, monthly mean pool elevation (ft)
    mead_storage            - float, monthly mean storage (acre-ft)
    mead_total_release      - float, monthly mean total release (cfs)
    mead_release_volume     - float, monthly total release volume (acre-ft)

All four endpoints are daily CSVs starting from 1935. Each is filtered
to the project date range (2000-01 to 2023-12) [3] and aggregated to
monthly before merging.

Aggregation logic:
    - Pool elevation : monthly mean (smooth level indicator)
    - Storage        : monthly mean (smooth volume indicator)
    - Total release  : monthly mean (flow rate, cfs)
    - Release volume : monthly sum  (total volume released in the month)

This script is flagged in the setup notes [1] as one of the two files
that should be run first to surface access problems early, since the
exact CSV parameter codes need to be confirmed against live data.

Date range: 2000-01 to 2023-12 [3]
"""

import logging
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = ROOT / "data" / "Final"
OUTPUT_FILE = PROCESSED_DIR / "lake_mead_monthly.csv"

START_DATE = "2000-01-01"
END_DATE = "2023-12-31"

# Bureau of Reclamation HydroData endpoints [2]
# Reservoir ID 921 = Lake Mead
# Each URL returns a daily CSV with columns varying by parameter.
ENDPOINTS = [
    {
        "label": "pool_elevation",
        "url": "https://www.usbr.gov/uc/water/hydrodata/reservoir_data/921/csv/49.csv",
        "output_col": "mead_pool_elevation",
        "agg": "mean",
        "description": "Pool elevation (ft above sea level)",
    },
    {
        "label": "storage",
        "url": "https://www.usbr.gov/uc/water/hydrodata/reservoir_data/921/csv/17.csv",
        "output_col": "mead_storage",
        "agg": "mean",
        "description": "Storage (acre-ft)",
    },
    {
        "label": "total_release",
        "url": "https://www.usbr.gov/uc/water/hydrodata/reservoir_data/921/csv/42.csv",
        "output_col": "mead_total_release",
        "agg": "mean",
        "description": "Total release (cfs)",
    },
    {
        "label": "release_volume",
        "url": "https://www.usbr.gov/uc/water/hydrodata/reservoir_data/921/csv/43.csv",
        "output_col": "mead_release_volume",
        "agg": "sum",
        "description": "Release volume (acre-ft)",
    },
]


def _fetch_csv(url: str, label: str, retries: int = 3) -> str:
    """
    Fetch a CSV from the Bureau of Reclamation with retry logic.
    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    for attempt in range(1, retries + 1):
        try:
            log.info("Fetching %s (attempt %d/%d)...", label, attempt, retries)
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            log.info(
                "  Status: %d, size: %d bytes",
                response.status_code,
                len(response.content),
            )
            return response.text
        except requests.exceptions.RequestException as e:
            log.warning("  Attempt %d failed: %s", attempt, e)
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to fetch {label} after {retries} attempts.\n"
                    f"URL: {url}\nError: {e}"
                ) from e
            time.sleep(2**attempt)


def _parse_reclamation_csv(raw_text: str, label: str) -> pd.DataFrame:  # noqa: C901
    """
    Parse a Bureau of Reclamation daily CSV into a clean DataFrame.
    This function handles common quirks:
        - Skips comment/metadata lines starting with non-numeric chars
        - Attempts multiple date column name candidates
        - Coerces the value column to numeric

    """
    try:
        df = pd.read_csv(StringIO(raw_text))
    except Exception:
        # Try skipping initial rows if the first attempt fails
        lines = raw_text.strip().split("\n")
        header_idx = 0
        for i, line in enumerate(lines):
            if "date" in line.lower() or "time" in line.lower():
                header_idx = i
                break
        df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))

    log.info("  Raw columns: %s", df.columns.tolist())
    log.info("  Raw shape  : %s", df.shape)

    date_candidates = ["datetime", "date", "Date", "DateTime", "DATETIME"]
    date_col = None
    for candidate in date_candidates:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        date_col = df.columns[0]
        log.info("  No named date column found — using first column: '%s'", date_col)

    value_col = None
    for col in df.columns:
        if col == date_col:
            continue
        test = pd.to_numeric(df[col], errors="coerce")
        if test.notna().sum() > len(df) * 0.5:
            value_col = col
            break

    if value_col is None:
        raise ValueError(
            f"Could not identify a numeric value column in {label}.\n"
            f"Columns: {df.columns.tolist()}\n"
            f"First 3 rows:\n{df.head(3).to_string()}"
        )

    log.info("  Using date column: '%s', value column: '%s'", date_col, value_col)

    result = pd.DataFrame()
    result["date"] = pd.to_datetime(df[date_col], errors="coerce")
    result["value"] = pd.to_numeric(df[value_col], errors="coerce")

    result = result.dropna(subset=["date", "value"])

    return result.reset_index(drop=True)


def _filter_date_range(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Filter a parsed DataFrame to the project date range [3].

    Parameters
    ----------
    df : DataFrame
        Must have a 'date' column (datetime64).
    label : str
        For logging.

    Returns
    -------
    Filtered DataFrame.
    """
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    pre = len(df)
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    log.info(
        "  Date filter (%s to %s): %d → %d rows.",
        START_DATE,
        END_DATE,
        pre,
        len(df),
    )

    if df.empty:
        raise ValueError(
            f"No data remains for {label} after filtering to "
            f"{START_DATE} – {END_DATE}. Check the source CSV date range."
        )

    return df.reset_index(drop=True)


def _aggregate_to_monthly(
    df: pd.DataFrame,
    agg: str,
    label: str,
) -> pd.DataFrame:
    """
    Aggregate daily values to monthly.

    Parameters
    ----------
    df : DataFrame
        Must have 'date' (datetime64) and 'value' (float) columns.
    agg : str
        Aggregation method: 'mean' or 'sum'.
    label : str
        For logging.

    Returns
    -------
    DataFrame with columns: year_month (str YYYY-MM), value (float).
    """
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    if agg == "mean":
        monthly = df.groupby("year_month")["value"].mean()
    elif agg == "sum":
        monthly = df.groupby("year_month")["value"].sum()
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")

    monthly = monthly.reset_index()
    monthly["value"] = monthly["value"].round(4)

    log.info(
        "  Monthly %s aggregation: %d months (%s to %s).",
        agg,
        len(monthly),
        monthly["year_month"].iloc[0],
        monthly["year_month"].iloc[-1],
    )

    return monthly


def _process_endpoint(endpoint: dict) -> pd.DataFrame:
    """
    Fetch, parse, filter, and aggregate one Reclamation endpoint.
    """
    label = endpoint["label"]
    url = endpoint["url"]
    output_col = endpoint["output_col"]
    agg = endpoint["agg"]

    log.info("--- Processing: %s ---", label)

    raw_text = _fetch_csv(url, label)
    df = _parse_reclamation_csv(raw_text, label)
    df = _filter_date_range(df, label)
    monthly = _aggregate_to_monthly(df, agg, label)
    monthly = monthly.rename(columns={"value": output_col})

    return monthly


def _merge_endpoints(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all monthly endpoint DataFrames on year_month.
    """
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="year_month", how="outer")

    merged = merged.sort_values("year_month").reset_index(drop=True)

    null_counts = merged.isna().sum()
    if null_counts.sum() > 0:
        log.warning(
            "Null values after merge:\n%s",
            null_counts[null_counts > 0].to_string(),
        )
    else:
        log.info("No null values after merge — all endpoints aligned.")

    return merged


def _sanity_checks(df: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the merged output.
    """
    expected = (2023 - 2000 + 1) * 12
    if len(df) != expected:
        log.warning(
            "Expected %d monthly rows but got %d. "
            "Check for gaps in the source data.",
            expected,
            len(df),
        )
    else:
        log.info("Row count correct: %d monthly rows.", len(df))


def main() -> None:
    log.info("=== lake_mead.py start ===")

    frames = []
    for endpoint in ENDPOINTS:
        try:
            monthly = _process_endpoint(endpoint)
            frames.append(monthly)
        except Exception as e:
            log.error("Failed to process %s: %s", endpoint["label"], e)
            log.error(
                "Continuing without this endpoint. "
                "Output will have a missing column."
            )

    if not frames:
        raise RuntimeError(
            "All four Lake Mead endpoints failed. "
            "Check network access and Bureau of Reclamation availability."
        )

    merged = _merge_endpoints(frames)

    _sanity_checks(merged)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    log.info("Wrote %d rows to %s", len(merged), OUTPUT_FILE)


if __name__ == "__main__":
    main()
