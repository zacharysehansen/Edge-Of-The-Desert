from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import progress

POWELL_FILES = {
    "powell_evaporation": "powell_evaporation.csv",
    "powell_total_release": "powell_total_release.csv",
    "powell_inflow": "powell_inflow.csv",
    "powell_storage": "powell_storage.csv",
    "powell_pool_elevation": "powell_pool_elevation.csv",
}

START = "2000-01-01"
END = "2020-12-31"


def ensure_powell_monthly(
    repo_root: Path,
    config_path: Path,
    raw_dir: Path,
    processed_dir: Path,
    final_dir: Path,
) -> None:
    output_path = final_dir / "powell_combined.csv"
    if output_path.exists():
        progress("powell_combined.csv already exists, skipping")
        return

    frames = []
    for feature, filename in POWELL_FILES.items():
        raw_path = raw_dir / filename
        if not raw_path.exists():
            raise FileNotFoundError(f"Expected Powell source file at {raw_path}")

        frame = pd.read_csv(raw_path, parse_dates=["datetime"])
        frame = frame.rename(columns={frame.columns[1]: feature})
        frame = frame[(frame["datetime"] >= START) & (frame["datetime"] <= END)]
        frame["year_month"] = frame["datetime"].dt.strftime("%Y-%m")
        monthly = (
            frame.groupby("year_month", as_index=False)[feature]
            .mean()
            .sort_values("year_month")
        )
        frames.append(monthly)
        progress(f"processed {filename}: {len(monthly)} monthly rows")

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="year_month", how="outer")

    combined = combined.sort_values("year_month").reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    progress(f"wrote {output_path.relative_to(repo_root)} with {len(combined)} rows")
