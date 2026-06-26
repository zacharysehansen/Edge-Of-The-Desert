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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = ROOT / "data" / "Final"
OUTPUT_FILE = PROCESSED_DIR / "lake_mead_monthly.csv"

# ---------------------------------------------------------------------------
# Config [3]
# ---------------------------------------------------------------------------
START_DATE = "2000-01-01"
END_DATE = "2023-12-31"

# ---------------------------------------------------------------------------
# Bureau of Reclamation HydroData endpoints [2]
# Reservoir ID 921 = Lake Mead
# Each URL returns a daily CSV with columns varying by parameter.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_csv(url: str, label: str, retries: int = 3) -> str:
    """
    Fetch a CSV from the Bureau of Reclamation with retry logic.

    Parameters
    ----------
    url : str
        Direct CSV download URL.
    label : str
        Human-readable label for logging.
    retries : int
        Number of retry attempts on transient HTTP errors.

    Returns
    -------
    str
        Raw CSV text content.

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

    The Reclamation CSVs have varying formats but generally include:
        - A date column (often 'datetime' or the first column)
        - A value column (often the second numeric column)
        - Possible header rows or metadata lines at the top

    This function handles common quirks:
        - Skips comment/metadata lines starting with non-numeric chars
        - Attempts multiple date column name candidates
        - Coerces the value column to numeric

    Parameters
    ----------
    raw_text : str
        Raw CSV text from the HTTP response.
    label : str
        Human-readable label for error messages.

    Returns
    -------
    DataFrame with columns: date (datetime64), value (float).
    """
    # Try reading the CSV — Reclamation files sometimes have leading
    # whitespace or irregular headers
    try:
        df = pd.read_csv(StringIO(raw_text))
    except Exception:
        # Try skipping initial rows if the first attempt fails
        lines = raw_text.strip().split("\n")
        # Find the header row (first row that looks like column names)
        header_idx = 0
        for i, line in enumerate(lines):
            if "date" in line.lower() or "time" in line.lower():
                header_idx = i
                break
        df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))

    log.info("  Raw columns: %s", df.columns.tolist())
    log.info("  Raw shape  : %s", df.shape)

    # Identify the date column
    date_candidates = ["datetime", "date", "Date", "DateTime", "DATETIME"]
    date_col = None
    for candidate in date_candidates:
        if candidate in df.columns:
            date_col = candidate
            break

    # If no named date column, assume the first column is the date
    if date_col is None:
        date_col = df.columns[0]
        log.info("  No named date column found — using first column: '%s'", date_col)

    # Identify the value column (first numeric column that isn't the date)
    value_col = None
    for col in df.columns:
        if col == date_col:
            continue
        # Check if the column has numeric-looking data
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

    # Build clean output
    result = pd.DataFrame()
    result["date"] = pd.to_datetime(df[date_col], errors="coerce")
    result["value"] = pd.to_numeric(df[value_col], errors="coerce")

    # Drop rows where date or value is null
    pre = len(result)
    result = result.dropna(subset=["date", "value"])
    if len(result) < pre:
        log.info("  Dropped %d rows with null date or value.", pre - len(result))

    log.info(
        "  Parsed: %d rows, date range %s to %s.",
        len(result),
        result["date"].min().strftime("%Y-%m-%d"),
        result["date"].max().strftime("%Y-%m-%d"),
    )

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


# ---------------------------------------------------------------------------
# Process one endpoint end-to-end
# ---------------------------------------------------------------------------


def _process_endpoint(endpoint: dict) -> pd.DataFrame:
    """
    Fetch, parse, filter, and aggregate one Reclamation endpoint.

    Parameters
    ----------
    endpoint : dict
        One entry from the ENDPOINTS list.

    Returns
    -------
    DataFrame with columns: year_month, <output_col>.
    """
    label = endpoint["label"]
    url = endpoint["url"]
    output_col = endpoint["output_col"]
    agg = endpoint["agg"]

    log.info("--- Processing: %s ---", label)

    # Fetch
    raw_text = _fetch_csv(url, label)

    # Parse
    df = _parse_reclamation_csv(raw_text, label)

    # Filter to date range
    df = _filter_date_range(df, label)

    # Aggregate to monthly
    monthly = _aggregate_to_monthly(df, agg, label)

    # Rename value column to the output name
    monthly = monthly.rename(columns={"value": output_col})

    return monthly


# ---------------------------------------------------------------------------
# Merge all endpoints
# ---------------------------------------------------------------------------


def _merge_endpoints(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all monthly endpoint DataFrames on year_month.

    Uses an outer join so that months missing from one endpoint do not
    drop data from the others. Missing values are left as NaN and logged
    as a warning.

    Parameters
    ----------
    frames : list of DataFrames
        Each must have a 'year_month' column plus one data column.

    Returns
    -------
    Merged DataFrame with year_month + all four data columns.
    """
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="year_month", how="outer")

    merged = merged.sort_values("year_month").reset_index(drop=True)

    # Check for nulls
    null_counts = merged.isna().sum()
    if null_counts.sum() > 0:
        log.warning(
            "Null values after merge:\n%s",
            null_counts[null_counts > 0].to_string(),
        )
    else:
        log.info("No null values after merge — all endpoints aligned.")

    return merged


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _sanity_checks(df: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the merged output.
    """
    # Row count — expect one row per month from 2000-01 to 2023-12
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

    # Pool elevation range check — Mead has declined from ~1200ft to ~1050ft
    # over this period. Values outside 800-1250 would be suspicious.
    lake_mead_ranges = [800, 1250]
    if "mead_pool_elevation" in df.columns:
        elev = df["mead_pool_elevation"]
        if elev.min() < lake_mead_ranges[0] or elev.max() > lake_mead_ranges[1]:
            log.warning(
                "Pool elevation range [%.1f, %.1f] exceeds expected "
                "bounds [800, 1250]. Check source data.",
                elev.min(),
                elev.max(),
            )
        else:
            log.info(
                "Pool elevation range check passed: [%.1f, %.1f] ft.",
                elev.min(),
                elev.max(),
            )

    # Declining trend check — Lake Mead has generally declined 2000-2023
    if "mead_pool_elevation" in df.columns:
        first_year = df[df["year_month"] < "2001-01"]["mead_pool_elevation"].mean()
        last_year = df[df["year_month"] >= "2023-01"]["mead_pool_elevation"].mean()
        if last_year > first_year:
            log.warning(
                "Pool elevation increased from 2000 (%.1f) to 2023 (%.1f). "
                "This is unexpected — Lake Mead has generally declined.",
                first_year,
                last_year,
            )
        else:
            log.info(
                "Declining trend check passed: 2000 mean=%.1f, " "2023 mean=%.1f.",
                first_year,
                last_year,
            )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(output_file: Path = OUTPUT_FILE) -> pd.DataFrame:
    """
    Full Lake Mead pipeline:
        fetch four endpoints → parse → filter → aggregate → merge →
        sanity checks → write CSV.

    Parameters
    ----------
    output_file : Path
        Path for output CSV.
        Defaults to data/processed/lake_mead_monthly.csv.

    Returns
    -------
    DataFrame
        Final monthly Lake Mead table, also written to output_file.
    """
    log.info("=== lake_mead.py start ===")

    # Process each endpoint
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

    # Merge all endpoints
    merged = _merge_endpoints(frames)

    # Sanity checks
    _sanity_checks(merged)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)
    log.info("Wrote %d rows to %s", len(merged), output_file)

    log.info("=== lake_mead.py complete ===")
    return merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build monthly Lake Mead water management table from "
        "Bureau of Reclamation HydroData Navigator."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Path for output CSV (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    result = run(output_file=args.output)
    print(result.head(12).to_string(index=False))
    print("...")
    print(result.tail(12).to_string(index=False))
