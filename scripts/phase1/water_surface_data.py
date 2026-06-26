"""
Download USGS water quality sample data for Arizona (2000-2020).

Uses the USGS Samples Data API:
https://api.waterdata.usgs.gov/samples-data/docs

Data is fetched year-by-year to avoid timeouts on large requests,
then combined into a single CSV.
"""

import io
import time

import pandas as pd
import requests

# -------------------------------------------------------------------
# Configuration — edit these as needed
# -------------------------------------------------------------------

# Profile controls which columns come back.
# Options: fullphyschem, basicphyschem, fullbio, basicbio,
#          narrow, resultdetectionquantitationlimit, labsampleprep, count
# "basicphyschem" is a good default: key physical + chemical columns.
PROFILE = "basicphyschem"

# Arizona state FIPS code
STATE_FIPS = "US:04"

# Year range (inclusive)
YEAR_START = 2000
YEAR_END = 2020

# Optional: get a free API key at https://api.waterdata.usgs.gov/docs/api-keys/
# Leave as empty string if you don't have one (rate limits will be lower).
API_KEY = ""

# Output file
OUTPUT_FILE = "arizona_water_samples_2000_2020.csv"

# Seconds to wait between requests (be polite to the API)
REQUEST_DELAY = 1.5

# -------------------------------------------------------------------
# API endpoint
# -------------------------------------------------------------------

BASE_URL = f"https://api.waterdata.usgs.gov/samples-data/results/{PROFILE}"


def fetch_year(year: int) -> pd.DataFrame:
    """
    Fetch all water quality sample results for Arizona in a given year.
    Handles pagination if the API returns a 'next' link in the headers.
    """
    params = {
        "mimeType": "text/csv",
        "stateFips": STATE_FIPS,
        "activityStartDateLower": f"{year}-01-01",
        "activityStartDateUpper": f"{year}-12-31",
    }
    if API_KEY:
        params["api_key"] = API_KEY

    frames = []
    url = BASE_URL

    while url:
        resp = requests.get(
            url, params=params if url == BASE_URL else None, timeout=300
        )

        # Some years may have no data — the API may return an empty CSV or 404
        if resp.status_code == 404:  # noqa: PLR2004
            print(f"    No data found for {year}")
            return pd.DataFrame()

        resp.raise_for_status()

        text = resp.text.strip()
        if not text or text.startswith("No results"):
            break

        df = pd.read_csv(io.StringIO(text))
        frames.append(df)

        # Check for a pagination 'next' link in the response headers.
        # If the API returns one, keep fetching pages.
        link_header = resp.headers.get("Link", "")
        next_url = None
        for part in link_header.split(","):
            cleaned_part = part.strip()
            if 'rel="next"' in cleaned_part:
                next_url = cleaned_part.split(";")[0].strip().strip("<>")
                break
        url = next_url

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    all_frames = []

    for year in range(YEAR_START, YEAR_END + 1):
        print(f"Fetching {year}...", end=" ", flush=True)

        try:
            df = fetch_year(year)
            row_count = len(df)
            print(f"{row_count} rows")

            if row_count > 0:
                all_frames.append(df)

        except requests.HTTPError as e:
            print(f"HTTP error: {e}")
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(REQUEST_DELAY)

    if not all_frames:
        print("No data was retrieved. Check your parameters and try again.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # Add year and month columns for easy monthly grouping.
    # The date column name in the basicphyschem profile is Activity_StartDate.
    date_col = "Activity_StartDate"
    if date_col in combined.columns:
        combined[date_col] = pd.to_datetime(combined[date_col], errors="coerce")
        combined.insert(0, "month", combined[date_col].dt.month)
        combined.insert(0, "year", combined[date_col].dt.year)

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. Saved {len(combined):,} rows to '{OUTPUT_FILE}'")
    print(f"Columns: {list(combined.columns[:10])} ...")


if __name__ == "__main__":
    main()
