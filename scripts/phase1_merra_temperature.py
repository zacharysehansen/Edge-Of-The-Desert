import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_of_the_desert.phase1.pipeline import (
    Phase1Error,
    import_xarray,
    reduce_spatial_monthly,
    subset_data_array_to_bbox,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "phase1.example.json"
DEFAULT_SOURCE_NAME = "merra_temperature_2m"


def log(message: str) -> None:
    print(f"[phase1_merra_temperature] {message}", flush=True)


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise Phase1Error(f"Config file was not found: {config_path}")
    return json.loads(config_path.read_text())


def pick_source(config: dict, source_name: str) -> dict:
    for source in config.get("sources", []):
        if source.get("name") == source_name:
            return source
    raise Phase1Error(f"Source {source_name!r} was not found in the config.")


def build_temperature_endpoint(config_path: Path, source_name: str) -> Path:
    config = load_config(config_path)
    paths = config.get("paths", {})
    source = pick_source(config, source_name)

    raw_dir = REPO_ROOT / paths.get("raw_dir", "data/raw") / source_name
    final_dir = REPO_ROOT / paths.get("final_dir", "data/Final")
    output_path = final_dir / f"{source_name}.csv"

    if not raw_dir.exists():
        raise Phase1Error(f"Raw directory was not found: {raw_dir}")

    raw_files = sorted(raw_dir.glob("*.nc4"))
    if not raw_files:
        raise Phase1Error(f"No .nc4 files were found in {raw_dir}")

    xr = import_xarray()
    bbox = source["bbox"]
    variable_name = source["variable"]
    transform_name = source.get("transform", "identity")
    feature_name = source["feature_name"]

    log(f"reading {len(raw_files)} raw files from {raw_dir.relative_to(REPO_ROOT)}")

    records: list[dict[str, float | str]] = []
    for index, file_path in enumerate(raw_files, start=1):
        if index == 1 or index % 50 == 0 or index == len(raw_files):
            log(f"processing file {index}/{len(raw_files)}: {file_path.name}")

        with xr.open_dataset(file_path) as dataset:
            if variable_name not in dataset:
                raise Phase1Error(
                    f"Variable {variable_name!r} was not found in {file_path.name}"
                )
            data_array = dataset[variable_name]
            subset = subset_data_array_to_bbox(data_array, bbox)
            value_by_time = reduce_spatial_monthly(subset, transform_name)
            for timestamp, value in value_by_time.items():
                records.append(
                    {
                        "year_month": timestamp.strftime("%Y-%m"),
                        feature_name: float(value),
                    }
                )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise Phase1Error("No monthly temperature values were extracted from the raw files.")

    monthly = (
        frame.groupby("year_month", as_index=False)[feature_name]
        .mean()
        .sort_values("year_month")
        .reset_index(drop=True)
    )

    final_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_path, index=False)

    log(
        "wrote "
        f"{len(monthly)} rows to {output_path.relative_to(REPO_ROOT)} "
        f"covering {monthly['year_month'].iloc[0]} -> {monthly['year_month'].iloc[-1]}"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Arizona-wide monthly MERRA-2 2m temperature endpoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the phase 1 config JSON.",
    )
    parser.add_argument(
        "--source-name",
        default=DEFAULT_SOURCE_NAME,
        help="Source name to extract from the config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_temperature_endpoint(args.config.resolve(), args.source_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
