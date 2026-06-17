"""
test_population_sources.py
--------------------------
Tests the hybrid approach for population data:
    - 2000-2010 : Census Bureau API (intercensal estimates)
    - 2010-2020 : Manual CSV download
    - 2020-2023 : Manual CSV download

Manual files should be downloaded from:
    2010-2020: https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/totals/co-est2020-alldata.csv
    2020-2023: https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv

Place them in:
    data/raw/census/co-est2020-alldata.csv
    data/raw/census/co-est2023-alldata.csv
"""

import os
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KEY = os.environ.get("CENSUS_API_KEY", "2782b36ff5bb3ba01fe126016c9baf6a3c82ee1e")

COUNTY_FIPS = [
    "04019",  # Pima
    "04021",  # Pinal
    "04023",  # Santa Cruz
    "04003",  # Cochise
    "04013",  # Graham
    "04011",  # Greenlee
    "04027",  # Yuma
    "04007",  # La Paz
]
STATE_FIPS = "04"
COUNTY_CODES = [f[2:] for f in COUNTY_FIPS]

ROOT = Path(__file__).resolve().parents[2]
CENSUS_DIR = ROOT / "data" / "raw" / "census"
FILE_2010_2020 = CENSUS_DIR / "co-est2020-alldata.csv"
FILE_2020_2023 = CENSUS_DIR / "co-est2023-alldata.csv"

# ---------------------------------------------------------------------------
# Test 1: Census API 2000-2010
# ---------------------------------------------------------------------------


def test_api_2000_2010():
    print("\n" + "=" * 60)
    print("TEST 1: Census API 2000-2010 intercensal")
    print("=" * 60)

    url = "https://api.census.gov/data/2000/pep/int_population"
    params = {
        "get": "POP,DATE_DESC",
        "for": f"county:{','.join(COUNTY_CODES)}",
        "in": f"state:{STATE_FIPS}",
        "key": KEY,
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        print(f"  Status     : {r.status_code}")
        print(f"  URL called : {r.url}")

        if r.status_code == 200:
            data = r.json()
            headers = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=headers)
            df["county_fips"] = df["state"] + df["county"]
            df = df[df["county_fips"].isin(COUNTY_FIPS)]
            df["POP"] = pd.to_numeric(df["POP"], errors="coerce")

            print(f"  Columns    : {df.columns.tolist()}")
            print(f"  Rows       : {len(df)}")
            print(f"  Counties   : {df['county_fips'].nunique()} of 8")
            print("  DATE_DESC sample:")
            for v in df["DATE_DESC"].unique()[:5]:
                print(f"    {v}")
            print("\n  Sample rows:")
            print(
                df[["DATE_DESC", "county_fips", "POP"]].head(10).to_string(index=False)
            )
            return df
        else:
            print(f"  FAILED: {r.text[:300]}")
            return None

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ---------------------------------------------------------------------------
# Test 2: Manual CSV 2010-2020
# ---------------------------------------------------------------------------


def test_manual_2010_2020():
    print("\n" + "=" * 60)
    print("TEST 2: Manual CSV 2010-2020")
    print("=" * 60)

    if not FILE_2010_2020.exists():
        print(f"  FILE NOT FOUND: {FILE_2010_2020}")
        print("  Download from:")
        print("  https://www2.census.gov/programs-surveys/popest/datasets/")
        print("  2010-2020/counties/totals/co-est2020-alldata.csv")
        print("  and place in data/raw/census/")
        return None

    try:
        df = pd.read_csv(FILE_2010_2020, encoding="latin-1")
        print(f"  Loaded     : {len(df)} rows")
        print(f"  Columns    : {df.columns.tolist()[:15]}")

        # Build FIPS and filter
        df["county_fips"] = df["STATE"].astype(str).str.zfill(2) + df["COUNTY"].astype(
            str
        ).str.zfill(3)
        df = df[df["county_fips"].isin(COUNTY_FIPS)].copy()
        print(f"  After filter: {len(df)} rows, {df['county_fips'].nunique()} counties")

        # Show population columns available
        pop_cols = [c for c in df.columns if "POPESTIMATE" in c]
        print(f"  Pop columns : {pop_cols}")

        print("\n  Sample (POPESTIMATE columns):")
        print(df[["county_fips", "CTYNAME"] + pop_cols[:5]].to_string(index=False))
        return df

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ---------------------------------------------------------------------------
# Test 3: Manual CSV 2020-2023
# ---------------------------------------------------------------------------


def test_manual_2020_2023():
    print("\n" + "=" * 60)
    print("TEST 3: Manual CSV 2020-2023")
    print("=" * 60)

    if not FILE_2020_2023.exists():
        print(f"  FILE NOT FOUND: {FILE_2020_2023}")
        print("  Download from:")
        print("  https://www2.census.gov/programs-surveys/popest/datasets/")
        print("  2020-2023/counties/totals/co-est2023-alldata.csv")
        print("  and place in data/raw/census/")
        return None

    try:
        df = pd.read_csv(FILE_2020_2023, encoding="latin-1")
        print(f"  Loaded     : {len(df)} rows")
        print(f"  Columns    : {df.columns.tolist()[:15]}")

        # Build FIPS and filter
        df["county_fips"] = df["STATE"].astype(str).str.zfill(2) + df["COUNTY"].astype(
            str
        ).str.zfill(3)
        df = df[df["county_fips"].isin(COUNTY_FIPS)].copy()
        print(f"  After filter: {len(df)} rows, {df['county_fips'].nunique()} counties")

        # Show population columns available
        pop_cols = [c for c in df.columns if "POPESTIMATE" in c]
        print(f"  Pop columns : {pop_cols}")

        print("\n  Sample (POPESTIMATE columns):")
        print(df[["county_fips", "CTYNAME"] + pop_cols].to_string(index=False))
        return df

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ---------------------------------------------------------------------------
# Test 4: Overlap check between sources
# ---------------------------------------------------------------------------


def test_overlap(df_api, df_2010, df_2020):
    print("\n" + "=" * 60)
    print("TEST 4: Overlap check at vintage boundaries")
    print("=" * 60)

    # Check 2010 value from API vs manual 2010-2020 file
    if df_api is not None and df_2010 is not None:
        print("\n  2010 regional total comparison:")

        # API: extract 2010 from DATE_DESC
        api_2010 = df_api[df_api["DATE_DESC"].str.contains("2010", na=False)].copy()
        api_2010["POP"] = pd.to_numeric(api_2010["POP"], errors="coerce")
        api_regional_2010 = api_2010.groupby("county_fips")["POP"].max()
        api_total = api_regional_2010.sum()
        print(f"    API 2010 regional total    : {api_total:,.0f}")

        # Manual: POPESTIMATE2010
        if "POPESTIMATE2010" in df_2010.columns:
            manual_total = df_2010["POPESTIMATE2010"].sum()
            print(f"    Manual 2010 regional total : {manual_total:,.0f}")
            diff = abs(api_total - manual_total)
            pct = diff / manual_total * 100 if manual_total > 0 else 0
            print(f"    Difference                 : {diff:,.0f} ({pct:.2f}%)")
            if pct < 1:
                print("    ✓ Sources agree within 1% at 2010 boundary")
            else:
                print("    ✗ Sources diverge at 2010 — review before merging")
        else:
            print("    POPESTIMATE2010 not found in manual file")

    # Check 2020 value from manual 2010-2020 vs manual 2020-2023 file
    if df_2010 is not None and df_2020 is not None:
        print("\n  2020 regional total comparison:")

        if (
            "POPESTIMATE2020" in df_2010.columns
            and "POPESTIMATE2020" in df_2020.columns
        ):
            total_2010_file = df_2010["POPESTIMATE2020"].sum()
            total_2020_file = df_2020["POPESTIMATE2020"].sum()
            diff = abs(total_2010_file - total_2020_file)
            pct = diff / total_2020_file * 100 if total_2020_file > 0 else 0
            print(f"    2010-2020 file POPESTIMATE2020 : {total_2010_file:,.0f}")
            print(f"    2020-2023 file POPESTIMATE2020 : {total_2020_file:,.0f}")
            print(f"    Difference                     : {diff:,.0f} ({pct:.2f}%)")
            if pct < 1:
                print("    ✓ Sources agree within 1% at 2020 boundary")
            else:
                print("    ✗ Sources diverge at 2020 — review before merging")
        else:
            print("    POPESTIMATE2020 not found in one or both files")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Population source test — hybrid API + manual CSV approach")
    print(f"Census dir : {CENSUS_DIR}")
    print(f"API key    : {KEY[:8]}...")

    df_api = test_api_2000_2010()
    df_2010 = test_manual_2010_2020()
    df_2020 = test_manual_2020_2023()

    test_overlap(df_api, df_2010, df_2020)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    api_status = "✓ OK" if df_api is not None else "✗ FAILED"
    csv2010_status = "✓ OK" if df_2010 is not None else "✗ FAILED — download file first"
    csv2020_status = "✓ OK" if df_2020 is not None else "✗ FAILED — download file first"

    print(f"API 2000-2010  : {api_status}")
    print(f"CSV 2010-2020 : {csv2010_status}")
    print(f"CSV 2020-2023 : {csv2020_status}")
