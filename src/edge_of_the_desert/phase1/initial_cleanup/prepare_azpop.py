from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import progress

START = "2000-01-01"
END = "2020-12-31"


def ensure_azpop_monthly(
    repo_root: Path,
    config_path: Path,
    raw_dir: Path,
    processed_dir: Path,
    final_dir: Path,
) -> None:
    output_path = final_dir / "azpop_monthly.csv"
    if output_path.exists():
        progress("azpop_monthly.csv already exists, skipping")
        return

    raw_path = raw_dir / "AZPOP.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Expected AZPOP source file at {raw_path}")

    frame = pd.read_csv(raw_path, parse_dates=["observation_date"])
    frame = frame[(frame["observation_date"] >= START) & (frame["observation_date"] <= END)]
    frame["AZPOP"] = pd.to_numeric(frame["AZPOP"], errors="coerce") * 1000
    frame["year"] = frame["observation_date"].dt.year

    annual = (
        frame.groupby("year", as_index=False)["AZPOP"]
        .mean()
        .sort_values("year")
    )

    monthly_index = pd.date_range(start=START, end=END, freq="MS")
    expanded = pd.DataFrame({"date": monthly_index})
    expanded["year"] = expanded["date"].dt.year
    expanded = expanded.merge(annual, on="year", how="left")
    expanded["AZPOP"] = expanded["AZPOP"].interpolate(method="linear", limit_direction="both")
    expanded["year_month"] = expanded["date"].dt.strftime("%Y-%m")

    result = expanded[["year_month", "AZPOP"]].reset_index(drop=True)
    result.to_csv(output_path, index=False)
    progress(f"wrote {output_path.relative_to(repo_root)} with {len(result)} rows")
