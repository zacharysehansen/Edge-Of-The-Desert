"""Parse Arizona OEO population projections for Pima County.

Reads the three series (Low / Medium / High) from the downloaded zip,
extracts annual total-population rows from each county workbook's
"1. Total Pop & Components" sheet, and writes a tidy CSV:

    data/processed/projections/pima_population_oeo.csv
        year,low,medium,high

Runnable as:
    python -m scripts.phase4.parse_oeo_population
"""
from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = REPO_ROOT / "data" / "raw" / "projections" / "AZ_OEO_All_Series_2025-2060.zip"
OUT_DIR = REPO_ROOT / "data" / "processed" / "projections"
OUT_PATH = OUT_DIR / "pima_population_oeo.csv"

# Map series names to the filenames inside the zip.
# Note: naming is inconsistent (spaces vs no spaces) across series folders.
SERIES_FILES = {
    "low": "All_Series_2025-2060/Low/Pima_Low Series.xlsx",
    "medium": "All_Series_2025-2060/Medium/Pima_Medium Series.xlsx",
    "high": "All_Series_2025-2060/High/Pima_High Series.xlsx",
}

SHEET = "1. Total Pop & Components"

# Verification checkpoints from PHASE4.md section 4.6.
EXPECTED = {
    "low":    {2025: 1_093_761, 2030: 1_106_178, 2040: 1_093_872, 2050: 1_052_351, 2060: 1_002_678},
    "medium": {2025: 1_093_761, 2030: 1_121_464, 2040: 1_147_353, 2050: 1_148_212, 2060: 1_143_575},
    "high":   {2025: 1_093_761, 2030: 1_166_019, 2040: 1_288_856, 2050: 1_398_818, 2060: 1_519_907},
}


def _read_series(zf: zipfile.ZipFile, inner_path: str) -> pd.Series:
    """Return a year-indexed Series of total population from one workbook."""
    raw = zf.read(inner_path)
    df = pd.read_excel(BytesIO(raw), sheet_name=SHEET, header=None)

    # The header row contains "Year" in column 0 and "Population" in column 1.
    header_idx = None
    for i, val in enumerate(df.iloc[:, 0]):
        if str(val).strip() == "Year":
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not locate header row in {inner_path}")

    data = pd.read_excel(BytesIO(raw), sheet_name=SHEET, header=header_idx)
    data.columns = [str(c).strip() for c in data.columns]

    # Keep only rows where Year is a valid integer (drop footnotes).
    data = data[pd.to_numeric(data["Year"], errors="coerce").notna()].copy()
    data["Year"] = data["Year"].astype(int)
    data["Population"] = data["Population"].astype(float)

    return data.set_index("Year")["Population"]


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"ERROR: zip not found at {ZIP_PATH}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.Series] = {}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for series_name, inner_path in SERIES_FILES.items():
            frames[series_name] = _read_series(zf, inner_path)

    result = pd.DataFrame(frames)
    result.index.name = "year"
    # Round to nearest integer — the raw values have sub-integer precision
    # from the cohort-component model; OEO publishes rounded integers.
    result = result.round(0).astype(int)

    result.to_csv(OUT_PATH)
    print(f"Wrote {OUT_PATH}  ({len(result)} rows, {result.columns.tolist()})")
    print()

    # ---------- Summary & verification ----------
    print("Summary (spot-check years):")
    for yr in [2025, 2030, 2040, 2050, 2060]:
        row = result.loc[yr]
        print(f"  {yr}:  Low {row['low']:>12,}   Med {row['medium']:>12,}   High {row['high']:>12,}")
    print()

    mismatches = 0
    for series_name, checks in EXPECTED.items():
        for yr, expected_val in checks.items():
            actual = result.loc[yr, series_name]
            if actual != expected_val:
                print(f"  MISMATCH  {series_name} {yr}: got {actual:,}, expected {expected_val:,}")
                mismatches += 1

    if mismatches:
        print(f"\n{mismatches} verification failure(s)!")
        return 1

    print("All verification checkpoints PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
