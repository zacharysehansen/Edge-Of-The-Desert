from __future__ import annotations
import csv
from pathlib import Path
import pandas as pd
from .common import load_arizona_huc12_ids, progress, write_json


def ensure_ir_huc12_total_to_az(
    *,
    repo_root: Path,
    raw_dir: Path,
    processed_dir: Path,
    final_dir: Path,
    **_: Path,
) -> Path:
    input_path = raw_dir / "IR_HUC12_Tot_WD_monthly_2000_2020.csv"
    output_path = processed_dir / "ir_huc12_tot_wd_az_2000_2020.csv"
    summary_path = processed_dir / "ir_huc12_tot_wd_az_summary.json"

    if output_path.exists():
        progress(f"irrigation AZ subset already present at {output_path.relative_to(repo_root)}")
        return output_path

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    az_huc12_ids = load_arizona_huc12_ids(raw_dir)
    if not az_huc12_ids:
        raise RuntimeError("Could not derive Arizona HUC12 IDs from the NWAA shard files.")

    progress("filtering irrigation HUC12 matrix to Arizona")

    with input_path.open(newline="") as src, output_path.open("w", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader)
        selected_pairs = [
            (index, column_name)
            for index, column_name in enumerate(header)
            if index < 2 or column_name in az_huc12_ids
        ]
        selected_indexes = [index for index, _ in selected_pairs]
        selected_columns = [column_name for _, column_name in selected_pairs]
        writer.writerow(selected_columns)
        row_count = 0
        for row in reader:
            writer.writerow([row[index] for index in selected_indexes])
            row_count += 1

    summary = {
        "input_file": input_path.name,
        "output_file": output_path.relative_to(repo_root).as_posix(),
        "year_month_row_count": row_count,
        "selected_huc12_count": len(selected_columns) - 2,
        "selected_column_count": len(selected_columns),
        "year_month_min": "2000-01",
        "year_month_max": "2020-12",
        "selection_basis": "Arizona HUC12 IDs present in data/raw/nwaa_data_*_of_6.csv",
    }
    write_json(summary_path, summary)
    progress(f"wrote {output_path.relative_to(repo_root)}")
    return output_path


def ensure_ir_huc12_az_monthly(
    *,
    repo_root: Path,
    raw_dir: Path,
    processed_dir: Path,
    final_dir: Path,
    **_: Path,
) -> Path:
    input_path = ensure_ir_huc12_total_to_az(
        repo_root=repo_root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        final_dir=final_dir,
    )
    output_path = final_dir / "irrigation_huc12_monthly_az_2000_2020.csv"
    summary_path = processed_dir / "ir_huc12_tot_wd_az_monthly_summary.json"

    if output_path.exists():
        progress(f"irrigation AZ monthly already present at {output_path.relative_to(repo_root)}")
        return output_path

    df = pd.read_csv(input_path, dtype=str)
    huc12_cols = df.columns[2:]

    monthly = pd.DataFrame()
    monthly["year_month"] = (
        df["Year"].str.zfill(4) + "-" + df["Month"].str.zfill(2)
    )
    monthly["irrigation_total_withdrawal_mgd"] = df[huc12_cols].apply(
        pd.to_numeric, errors="raise"
    ).sum(axis=1)
    monthly["huc12_count"] = len(huc12_cols)

    monthly_dates = pd.to_datetime(monthly["year_month"] + "-01")
    days_in_month = monthly_dates.dt.days_in_month

    monthly["irrigation_total_withdrawal_gallons_per_day"] = (
        monthly["irrigation_total_withdrawal_mgd"] * 1_000_000.0
    )
    monthly["irrigation_total_withdrawal_acre_feet_month"] = (
        monthly["irrigation_total_withdrawal_gallons_per_day"] * days_in_month / 325851.429
    )

    monthly = monthly[
        [
            "year_month",
            "irrigation_total_withdrawal_mgd",
            "irrigation_total_withdrawal_gallons_per_day",
            "irrigation_total_withdrawal_acre_feet_month",
            "huc12_count",
        ]
    ]
    monthly = monthly.sort_values("year_month").reset_index(drop=True)
    monthly.to_csv(output_path, index=False)

    summary = {
        "input_file": input_path.relative_to(repo_root).as_posix(),
        "output_file": output_path.relative_to(repo_root).as_posix(),
        "row_count": int(len(monthly)),
        "year_month_min": str(monthly["year_month"].min()),
        "year_month_max": str(monthly["year_month"].max()),
        "huc12_count": int(len(huc12_cols)),
        "min_irrigation_total_withdrawal_mgd": float(monthly["irrigation_total_withdrawal_mgd"].min()),
        "max_irrigation_total_withdrawal_mgd": float(monthly["irrigation_total_withdrawal_mgd"].max()),
    }
    write_json(summary_path, summary)
    progress(f"wrote {output_path.relative_to(repo_root)}")
    return output_path

def main() -> int:
    from .common import load_paths
    paths = load_paths(Path("config/phase1.example.json"))
    ensure_ir_huc12_az_monthly(**paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
