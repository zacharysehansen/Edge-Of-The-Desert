from __future__ import annotations

import argparse
import io
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


USGS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
USDM_API_ROOT = "https://usdmdataservices.unl.edu/api"
RISE_RESULTS_URL = "https://data.usbr.gov/rise/api/result"


class Phase1Error(RuntimeError):
    """Base error for the phase 1 pipeline."""


class MissingSourceError(Phase1Error):
    """Raised when a local source file is missing."""


@dataclass
class PipelineContext:
    repo_root: Path
    config_path: Path
    config: dict[str, Any]
    raw_dir: Path
    processed_dir: Path
    source_dir: Path
    final_dir: Path
    status_path: Path
    build_report_path: Path
    dataset_path: Path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ctx = load_context(Path(args.config))

    if args.command == "collect":
        progress(f"[collect] starting with config {ctx.config_path.relative_to(ctx.repo_root)}")
        selected = set(args.sources or [])
        status = collect_sources(
            ctx,
            selected_sources=selected if selected else None,
            fail_fast=args.fail_fast,
            refresh=args.refresh,
        )
        progress("[collect] finished")
        print(json.dumps(status, indent=2))
        return 0

    if args.command == "build":
        progress(f"[build] starting with config {ctx.config_path.relative_to(ctx.repo_root)}")
        merged, report = build_dataset(ctx, allow_partial=args.allow_partial)
        progress("[build] finished")
        print(f"Wrote {len(merged)} monthly rows to {ctx.dataset_path}")
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "run":
        progress(f"[run] starting with config {ctx.config_path.relative_to(ctx.repo_root)}")
        selected = set(args.sources or [])
        status = collect_sources(
            ctx,
            selected_sources=selected if selected else None,
            fail_fast=args.fail_fast,
            refresh=args.refresh,
        )
        merged, report = build_dataset(ctx, allow_partial=args.allow_partial)
        progress("[run] finished")
        print(json.dumps({"collect": status, "build": report}, indent=2))
        print(f"Wrote {len(merged)} monthly rows to {ctx.dataset_path}")
        return 0

    raise Phase1Error(f"Unsupported command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1 data pipeline for the Southwest Water Sustainability Visualizer."
    )
    parser.add_argument(
        "--config",
        default="config/phase1.example.json",
        help="Path to the phase 1 JSON config file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect and normalize source datasets.")
    collect_parser.add_argument(
        "--sources",
        nargs="*",
        help="Optional list of source names to collect.",
    )
    collect_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first source error instead of continuing and reporting failures.",
    )
    collect_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild normalized source files even if they already exist in the usable source directory.",
    )

    build_parser_cmd = subparsers.add_parser("build", help="Join normalized monthly sources.")
    build_parser_cmd.add_argument(
        "--allow-partial",
        action="store_true",
        help="Build the monthly dataset even if some sources are still missing.",
    )

    run_parser = subparsers.add_parser("run", help="Collect sources and then build the joined dataset.")
    run_parser.add_argument(
        "--sources",
        nargs="*",
        help="Optional list of source names to collect before building.",
    )
    run_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first source error instead of continuing and reporting failures.",
    )
    run_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild normalized source files even if they already exist in the usable source directory.",
    )
    run_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Build the monthly dataset even if some sources are still missing.",
    )
    return parser


def load_context(config_path: Path) -> PipelineContext:
    repo_root = Path(__file__).resolve().parents[3]
    resolved_config = config_path if config_path.is_absolute() else repo_root / config_path
    config = json.loads(resolved_config.read_text())
    paths = config.get("paths", {})

    raw_dir = resolve_repo_path(repo_root, paths.get("raw_dir", "data/raw"))
    processed_dir = resolve_repo_path(repo_root, paths.get("processed_dir", "data/processed"))
    final_dir_value = paths.get("final_dir", "data/Final")
    final_dir = resolve_repo_path(repo_root, final_dir_value)
    source_dir_value = paths.get("source_dir")
    source_dir = (
        resolve_repo_path(repo_root, source_dir_value)
        if source_dir_value
        else final_dir
    )

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    return PipelineContext(
        repo_root=repo_root,
        config_path=resolved_config,
        config=config,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        source_dir=source_dir,
        final_dir=final_dir,
        status_path=processed_dir / "phase1_status.json",
        build_report_path=processed_dir / "phase1_build_report.json",
        dataset_path=processed_dir / "phase1_monthly_dataset.csv",
    )


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def collect_sources(
    ctx: PipelineContext,
    selected_sources: set[str] | None = None,
    fail_fast: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "config": str(ctx.config_path.relative_to(ctx.repo_root)),
        "collected": [],
        "missing": [],
        "failed": [],
        "skipped": [],
    }

    for source in ctx.config.get("sources", []):
        name = source["name"]
        if not source.get("enabled", True):
            status["skipped"].append({"name": name, "reason": "disabled"})
            continue
        if selected_sources and name not in selected_sources:
            status["skipped"].append({"name": name, "reason": "not_selected"})
            continue

        output_path = normalized_source_path(ctx, name)
        if output_path.exists() and not refresh:
            progress(f"[collect] {name}: skipping existing {output_path.relative_to(ctx.repo_root)}")
            status["skipped"].append(
                {
                    "name": name,
                    "reason": f"already_present:{output_path.relative_to(ctx.repo_root)}",
                }
            )
            continue

        try:
            progress(f"[collect] {name}: starting")
            monthly = collect_source(ctx, source)
            monthly = normalize_monthly_frame(monthly)
            monthly.to_csv(output_path, index=False)
            progress(
                f"[collect] {name}: wrote {output_path.relative_to(ctx.repo_root)} "
                f"with {len(monthly)} rows"
            )
            status["collected"].append(
                {
                    "name": name,
                    "output": str(output_path.relative_to(ctx.repo_root)),
                    "rows": int(len(monthly)),
                    "columns": [column for column in monthly.columns if column != "year_month"],
                }
            )
        except MissingSourceError as exc:
            progress(f"[collect] {name}: missing source ({exc})")
            status["missing"].append({"name": name, "error": str(exc)})
            if fail_fast:
                break
        except Exception as exc:  # pragma: no cover - defensive logging path
            progress(f"[collect] {name}: failed ({exc})")
            status["failed"].append({"name": name, "error": str(exc)})
            if fail_fast:
                break

    write_json(ctx.status_path, status)
    return status


def collect_source(ctx: PipelineContext, source: dict[str, Any]) -> pd.DataFrame:
    kind = source["kind"]
    if kind == "usgs_daily_value":
        return collect_usgs_daily_value(ctx, source)
    if kind == "usdm_dsci":
        return collect_usdm_dsci(ctx, source)
    if kind == "rise_results":
        return collect_rise_results(ctx, source)
    if kind == "census_population_download":
        return collect_census_population_download(ctx, source)
    if kind == "manual_daily_csv":
        return collect_manual_daily_csv(ctx, source)
    if kind == "manual_monthly_csv":
        return collect_manual_monthly_csv(ctx, source)
    if kind == "manual_annual_csv":
        return collect_manual_annual_csv(ctx, source)
    if kind == "derived_monthly_delta":
        return collect_derived_monthly_delta(ctx, source)
    if kind == "earthdata_modis_hdf":
        return collect_earthdata_modis_hdf(ctx, source)
    if kind == "earthdata_grid_netcdf":
        return collect_earthdata_grid_netcdf(ctx, source)
    if kind == "manual_huc12_wide_csv":
        return collect_manual_huc12_wide_csv(ctx, source)
    raise Phase1Error(f"Unknown source kind: {kind}")

def collect_manual_huc12_wide_csv(ctx: PipelineContext, source: dict[str, Any]) -> pd.DataFrame:
    path = resolve_final_source_path(ctx, source)
    frame = pd.read_csv(path)

    year_col = source.get("year_column", "Year")
    month_col = source.get("month_column", "Month")

    if year_col not in frame.columns or month_col not in frame.columns:
        raise Phase1Error(
            f"{path} must include '{year_col}' and '{month_col}' columns."
        )

    frame["year_month"] = (
        frame[year_col].astype(int).astype(str)
        + "-"
        + frame[month_col].astype(int).astype(str).str.zfill(2)
    )

    huc_columns = [
        col for col in frame.columns
        if col not in {year_col, month_col, "year_month"}
    ]

    values = frame[huc_columns].apply(pd.to_numeric, errors="coerce")

    # handle 999 sentinel values (VERY IMPORTANT)
    if source.get("treat_999_as_nan", True):
        values = values.replace(999, np.nan)

    if source.get("aggregation", "sum") == "mean":
        aggregated = values.mean(axis=1)
    else:
        aggregated = values.sum(axis=1)

    result = pd.DataFrame({   

        "year_month": frame["year_month"],
        source["feature_name"]: aggregated,
    })

    write_raw_csv(ctx, source["name"], frame)

    return result.sort_values("year_month")

def collect_usgs_daily_value(ctx: PipelineContext, source: dict[str, Any]) -> pd.DataFrame:
    requests = import_requests()
    params = {
        "format": "json",
        "sites": ",".join(source["site_numbers"]),
        "parameterCd": source.get("parameter_code", "00060"),
        "startDT": source["start_date"],
        "endDT": source["end_date"],
        "siteStatus": "all",
    }
    response = requests.get(USGS_DV_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    records: list[dict[str, Any]] = []
    for series in payload.get("value", {}).get("timeSeries", []):
        site_codes = series.get("sourceInfo", {}).get("siteCode", [])
        site_number = site_codes[0].get("value") if site_codes else None
        values = series.get("values", [])
        for value_block in values:
            for point in value_block.get("value", []):
                value = coerce_float(point.get("value"))
                if value is None:
                    continue
                records.append(
                    {
                        "site_number": site_number,
                        "date": point.get("dateTime", "")[:10],
                        "value": value,
                    }
                )

    raw = pd.DataFrame.from_records(records)
    write_raw_csv(ctx, source["name"], raw)
    if raw.empty:
        raise Phase1Error(f"No USGS streamflow rows returned for {source['name']}")

    monthly = aggregate_daily_to_monthly(
        raw,
        date_column="date",
        value_column="value",
        feature_name=source["feature_name"],
        group_column="site_number",
        group_aggregation=source.get("group_aggregation", "mean"),
    )
    return monthly


def collect_usdm_dsci(ctx: PipelineContext, source: dict[str, Any]) -> pd.DataFrame:
    requests = import_requests()
    url = f"{USDM_API_ROOT}/{source['area_type']}/GetDSCI"
    params = {
        "aoi": source["aoi"],
        "startdate": format_usdm_date(source["start_date"]),
        "enddate": format_usdm_date(source["end_date"]),
        "statisticsType": source.get("statistics_type", 1),
    }
    response = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=120)
    response.raise_for_status()

    if "application/json" in response.headers.get("Content-Type", ""):
        raw = pd.DataFrame(response.json())
    else:
        raw = pd.read_csv(io.StringIO(response.text))

    if raw.empty:
        raise Phase1Error(f"No USDM rows returned for {source['name']}")

    date_column = find_column(raw, ["MapDate", "mapDate", "date", "Date"])
    dsci_column = find_column(raw, ["DSCI", "dsci"])
    raw = raw.rename(columns={date_column: "date", dsci_column: "usdm_dsci"})
    raw["date"] = pd.to_datetime(raw["date"])
    raw["usdm_dsci"] = pd.to_numeric(raw["usdm_dsci"], errors="coerce")
    raw = raw.dropna(subset=["date", "usdm_dsci"])
    write_raw_csv(ctx, source["name"], raw)

    monthly = (
        raw.assign(year_month=raw["date"].dt.strftime("%Y-%m"))
        .groupby("year_month", as_index=False)["usdm_dsci"]
        .mean()
        .sort_values("year_month")
    )
    monthly[source["feature_name"]] = 100.0 - (monthly["usdm_dsci"] / 5.0)
    return monthly


def collect_rise_results(ctx: PipelineContext, source: dict[str, Any]) -> pd.DataFrame:
    requests = import_requests()
    params = {
        "locationId": source["location_id"],
        "dateTime[after]": source["start_date"],
        "dateTime[before]": source["end_date"],
        "catalogItem.isModeled": str(source.get("modeled", False)).lower(),
        "page[size]": source.get("page_size", 200),
    }
    if "parameter_id" in source and source["parameter_id"] is not None:
        params["parameterId"] = source["parameter_id"]

    rows: list[dict[str, Any]] = []
    next_url: str | None = RISE_RESULTS_URL
    next_params: dict[str, Any] | None = params
    while next_url:
        response = requests.get(
            next_url,
            params=next_params,
            headers={"Accept": "application/vnd.api+json, application/json"},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        for item in data:
            attributes = item.get("attributes", {})
            rows.append(
                {
                    "date": pick_first(
                        attributes,
                        source.get(
                            "date_fields",
                            ["dateTime", "date", "observedDateTime", "resultDateTime"],
                        ),
                    ),
                    "value": coerce_float(
                        pick_first(
                            attributes,
                            source.get(
                                "value_fields",
                                ["result", "value", "resultValue", "result_va"],
                            ),
                        )
                    ),
                    "parameter_id": pick_first(attributes, ["parameterId", "parameter_id"]),
                    "catalog_item_id": pick_first(attributes, ["catalogItemId", "catalog_item_id"]),
                }
            )

        next_url = extract_next_link(payload)
        next_params = None

    raw = pd.DataFrame.from_records(rows)
    raw = raw.dropna(subset=["date", "value"])
    if raw.empty:
        raise Phase1Error(f"No RISE rows returned for {source['name']}")

    if "catalog_item_id" in source:
        raw = raw[raw["catalog_item_id"].astype(str) == str(source["catalog_item_id"])]

    write_raw_csv(ctx, source["name"], raw)
    monthly = aggregate_daily_to_monthly(
        raw,
        date_column="date",
        value_column="value",
        feature_name=source["feature_name"],
    )
    return monthly


def collect_census_population_download(ctx: PipelineContext, source: dict[str, Any]) -> pd.DataFrame:
    requests = import_requests()
    state_fips = str(source["state_fips"]).zfill(2)
    rows: list[dict[str, Any]] = []
    annual_values: dict[int, float] = {}

    for url in source["urls"]:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        state_column = find_column(frame, ["STATE", "state"])
        frame[state_column] = frame[state_column].astype(str).str.zfill(2)
        state_rows = frame[frame[state_column] == state_fips].copy()
        if state_rows.empty:
            continue
        row = state_rows.iloc[0]
        extracted = extract_popestimate_columns(row)
        for year, value in extracted.items():
            annual_values[year] = value
            rows.append({"source_url": url, "year": year, source["feature_name"]: value})

    raw = pd.DataFrame(rows)
    write_raw_csv(ctx, source["name"], raw)
    if not annual_values:
        raise Phase1Error(f"No Census population rows returned for state {state_fips}")

    annual = (
        pd.DataFrame(
            {
                "year": sorted(annual_values),
                source["feature_name"]: [annual_values[year] for year in sorted(annual_values)],
            }
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    monthly = annual_points_to_monthly(
        annual,
        year_column="year",
        feature_columns=[source["feature_name"]],
        start_year_month=source["start_year_month"],
        end_year_month=source["end_year_month"],
        anchor_month=source.get("anchor_month", 7),
    )
    return monthly


def collect_manual_daily_csv(ctx: PipelineContext, source: dict[str, Any]) -> pd.DataFrame:
    path = resolve_final_source_path(ctx, source)
    frame = pd.read_csv(path)
    date_column = source.get("date_column", "date")
    value_column = source["value_column"]
    group_column = source.get("group_column")
    write_raw_csv(ctx, source["name"], frame)
    monthly = aggregate_daily_to_monthly(
        frame,
        date_column=date_column,
        value_column=value_column,
        feature_name=source["feature_name"],
        group_column=group_column,
        group_aggregation=source.get("group_aggregation", "mean"),
    )
    return monthly


def collect_manual_monthly_csv(ctx: PipelineContext, source: dict[str, Any]) -> pd.DataFrame:
    path = resolve_final_source_path(ctx, source)
    frame = pd.read_csv(path)
    if "year_month" not in frame.columns:
        if "date" in frame.columns:
            frame["year_month"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m")
        else:
            raise MissingSourceError(
                f"{path.relative_to(ctx.repo_root)} must include a year_month column."
            )
    keep_columns = ["year_month", *source["feature_columns"]]
    return frame[keep_columns].copy()


def collect_manual_annual_csv(ctx: PipelineContext, source: dict[str, Any]) -> pd.DataFrame:
    path = resolve_final_source_path(ctx, source)
    frame = pd.read_csv(path)
    feature_columns = source.get("feature_columns", [source["feature_name"]])
    monthly = annual_points_to_monthly(
        frame,
        year_column=source.get("year_column", "year"),
        feature_columns=feature_columns,
        start_year_month=source["start_year_month"],
        end_year_month=source["end_year_month"],
        anchor_month=source.get("anchor_month", 1),
    )
    write_raw_csv(ctx, source["name"], frame)
    return monthly


def collect_derived_monthly_delta(ctx: PipelineContext, source: dict[str, Any]) -> pd.DataFrame:
    input_path = normalized_source_path(ctx, source["input_source"])
    if not input_path.exists():
        raise MissingSourceError(
            f"Derived source {source['name']} needs {input_path.relative_to(ctx.repo_root)} first."
        )

    frame = pd.read_csv(input_path, dtype={"year_month": "string"})
    input_column = source["input_column"]
    if input_column not in frame.columns:
        raise Phase1Error(
            f"Derived source {source['name']} could not find input column {input_column!r} "
            f"in {input_path.relative_to(ctx.repo_root)}"
        )

    derived = frame[["year_month", input_column]].copy()
    derived["date"] = pd.to_datetime(derived["year_month"] + "-01")
    derived[input_column] = pd.to_numeric(derived[input_column], errors="coerce")
    derived = derived.sort_values("date")
    values = derived[input_column].diff()

    if source.get("positive_only", False):
        values = values.clip(lower=0)
    if source.get("negative_only", False):
        values = values.clip(upper=0)
    if "fill_value" in source:
        values = values.fillna(source["fill_value"])

    derived[source["feature_name"]] = values
    return derived[["year_month", source["feature_name"]]]


def collect_earthdata_modis_hdf(ctx: PipelineContext, source: dict[str, Any]) -> pd.DataFrame:
    earthaccess = import_earthaccess()
    gdal = import_gdal()
    progress(f"[collect] {source['name']}: logging into Earthdata")
    earthaccess.login()

    bbox = source["bbox"]
    results = []
    if "short_names" in source:
        short_names = source["short_names"]
    else:
        short_names = [source["short_name"]]

    for short_name in short_names:
        progress(f"[collect] {source['name']}: searching Earthdata short_name={short_name}")
        results.extend(
            earthaccess.search_data(
                short_name=short_name,
                temporal=(source["start_date"], source["end_date"]),
                bounding_box=tuple(bbox),
                count=source.get("count", 5000),
            )
        )

    if not results:
        raise Phase1Error(f"No Earthdata granules found for {source['name']}")
    progress(f"[collect] {source['name']}: found {len(results)} Earthdata granules")

    download_dir = ctx.raw_dir / source["name"]
    download_dir.mkdir(parents=True, exist_ok=True)
    progress(f"[collect] {source['name']}: downloading granules to {download_dir.relative_to(ctx.repo_root)}")
    files = earthaccess.download(results, download_dir)
    if not files:
        raise Phase1Error(f"Earthaccess did not download any files for {source['name']}")
    progress(f"[collect] {source['name']}: downloaded {len(files)} files")

    records: list[dict[str, Any]] = []
    for file_path in map(Path, files):
        year_month = parse_modis_year_month(file_path.name)
        dataset = gdal.Open(str(file_path))
        subdatasets = dataset.GetSubDatasets()
        target = pick_modis_subdataset(subdatasets, source["subdataset_contains"])
        subdataset = gdal.Open(target)
        region = extract_modis_region(
            subdataset,
            bbox,
            scale_factor=source.get("scale_factor", 10000.0),
            invalid_below=source.get("invalid_below", -2000),
        )
        valid = region[~np.isnan(region)]
        if valid.size == 0:
            continue
        records.append(
            {
                "year_month": year_month,
                "_sum": float(np.nansum(valid)),
                "_count": int(valid.size),
            }
        )

    raw = pd.DataFrame.from_records(records)
    if raw.empty:
        raise Phase1Error(f"No valid MODIS pixels were extracted for {source['name']}")

    monthly = (
        raw.groupby("year_month", as_index=False)[["_sum", "_count"]]
        .sum()
        .sort_values("year_month")
    )
    monthly[source["feature_name"]] = monthly["_sum"] / monthly["_count"]
    return monthly[["year_month", source["feature_name"]]]


def collect_earthdata_grid_netcdf(ctx: PipelineContext, source: dict[str, Any]) -> pd.DataFrame:
    earthaccess = import_earthaccess()
    xr = import_xarray()
    progress(f"[collect] {source['name']}: logging into Earthdata")
    earthaccess.login()

    bbox = source["bbox"]
    results = []
    if "short_names" in source:
        short_names = source["short_names"]
    else:
        short_names = [source["short_name"]]

    for short_name in short_names:
        progress(f"[collect] {source['name']}: searching Earthdata short_name={short_name}")
        results.extend(
            earthaccess.search_data(
                short_name=short_name,
                temporal=(source["start_date"], source["end_date"]),
                bounding_box=tuple(bbox),
                count=source.get("count", 5000),
            )
        )

    if not results:
        raise Phase1Error(f"No Earthdata granules found for {source['name']}")
    progress(f"[collect] {source['name']}: found {len(results)} Earthdata granules")

    download_dir = ctx.raw_dir / source["name"]
    download_dir.mkdir(parents=True, exist_ok=True)
    progress(f"[collect] {source['name']}: downloading granules to {download_dir.relative_to(ctx.repo_root)}")
    files = earthaccess.download(results, download_dir)
    if not files:
        raise Phase1Error(f"Earthaccess did not download any files for {source['name']}")
    progress(f"[collect] {source['name']}: downloaded {len(files)} files")

    records: list[dict[str, Any]] = []
    for file_path in map(Path, files):
        with xr.open_dataset(file_path) as dataset:
            variable_name = source["variable"]
            if variable_name not in dataset:
                raise Phase1Error(
                    f"Variable {variable_name!r} was not found in {file_path.name} for {source['name']}"
                )
            data_array = dataset[variable_name]
            subset = subset_data_array_to_bbox(data_array, bbox)
            value_by_time = reduce_spatial_monthly(subset, source.get("transform", "identity"))
            for timestamp, value in value_by_time.items():
                records.append(
                    {
                        "year_month": timestamp.strftime("%Y-%m"),
                        source["feature_name"]: value,
                    }
                )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise Phase1Error(f"No valid Earthdata values were extracted for {source['name']}")

    monthly = (
        frame.groupby("year_month", as_index=False)[source["feature_name"]]
        .mean()
        .sort_values("year_month")
    )
    return monthly


def build_dataset(ctx: PipelineContext, allow_partial: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    timeline = monthly_timeline(
        ctx.config["project"]["start_year_month"],
        ctx.config["project"]["end_year_month"],
    )
    progress(f"[build] joining sources onto {len(timeline)} monthly timestamps")
    merged = pd.DataFrame({"year_month": timeline})
    missing_sources: list[str] = []
    source_columns: dict[str, list[str]] = {}

    for source in ctx.config.get("sources", []):
        if not source.get("enabled", True):
            continue
        path = normalized_source_path(ctx, source["name"])
        if not path.exists() and is_manual_source(source):
            try:
                progress(f"[build] {source['name']}: materializing from local source input")
                monthly = collect_source(ctx, source)
                monthly = normalize_monthly_frame(monthly)
                monthly.to_csv(path, index=False)
                progress(f"[build] {source['name']}: wrote {path.relative_to(ctx.repo_root)}")
            except MissingSourceError:
                pass
        if not path.exists():
            progress(f"[build] {source['name']}: missing normalized source")
            missing_sources.append(source["name"])
            continue
        frame = pd.read_csv(path, dtype={"year_month": "string"})
        frame = normalize_monthly_frame(frame)
        value_columns = [column for column in frame.columns if column != "year_month"]
        source_columns[source["name"]] = value_columns
        merged = merged.merge(frame, on="year_month", how="left")
        progress(
            f"[build] {source['name']}: merged {path.relative_to(ctx.repo_root)} "
            f"with {len(frame)} rows"
        )

    if missing_sources and not allow_partial:
        raise MissingSourceError(
            "Missing normalized source files for: "
            + ", ".join(missing_sources)
            + ". Run collect first or use --allow-partial."
        )

    merged.to_csv(ctx.dataset_path, index=False)
    report = {
        "dataset": str(ctx.dataset_path.relative_to(ctx.repo_root)),
        "rows": int(len(merged)),
        "columns": list(merged.columns),
        "missing_sources": missing_sources,
        "source_columns": source_columns,
    }
    write_json(ctx.build_report_path, report)
    return merged, report


def aggregate_daily_to_monthly(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    feature_name: str,
    group_column: str | None = None,
    group_aggregation: str = "mean",
) -> pd.DataFrame:
    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.dropna(subset=[date_column, value_column])
    if data.empty:
        return pd.DataFrame(columns=["year_month", feature_name])

    data["year_month"] = data[date_column].dt.strftime("%Y-%m")
    if group_column and group_column in data.columns:
        grouped = (
            data.groupby(["year_month", group_column], as_index=False)[value_column]
            .mean()
            .rename(columns={value_column: feature_name})
        )
        if group_aggregation == "sum":
            monthly = grouped.groupby("year_month", as_index=False)[feature_name].sum()
        else:
            monthly = grouped.groupby("year_month", as_index=False)[feature_name].mean()
        return monthly.sort_values("year_month")

    monthly = data.groupby("year_month", as_index=False)[value_column].mean()
    return monthly.rename(columns={value_column: feature_name}).sort_values("year_month")


def annual_points_to_monthly(
    frame: pd.DataFrame,
    year_column: str,
    feature_columns: list[str],
    start_year_month: str,
    end_year_month: str,
    anchor_month: int = 1,
) -> pd.DataFrame:
    annual = frame.copy()
    annual[year_column] = pd.to_numeric(annual[year_column], errors="coerce")
    annual = annual.dropna(subset=[year_column])
    annual[year_column] = annual[year_column].astype(int)
    annual["anchor_date"] = pd.to_datetime(
        annual[year_column].astype(str) + f"-{anchor_month:02d}-01"
    )
    monthly_index = pd.date_range(
        start=f"{start_year_month}-01",
        end=f"{end_year_month}-01",
        freq="MS",
    )
    expanded = pd.DataFrame(index=monthly_index)
    expanded = expanded.join(annual.set_index("anchor_date")[feature_columns], how="left")
    expanded[feature_columns] = expanded[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    expanded[feature_columns] = expanded[feature_columns].interpolate(
        method="time",
        limit_direction="both",
    )
    expanded = expanded.reset_index().rename(columns={"index": "date"})
    expanded["year_month"] = expanded["date"].dt.strftime("%Y-%m")
    return expanded[["year_month", *feature_columns]]


def normalize_monthly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "year_month" not in frame.columns:
        raise Phase1Error("Monthly frames must include a year_month column.")
    monthly = frame.copy()
    monthly["year_month"] = (
        pd.to_datetime(monthly["year_month"].astype(str) + "-01")
        .dt.strftime("%Y-%m")
    )
    monthly = monthly.groupby("year_month", as_index=False).first()
    monthly = monthly.sort_values("year_month").reset_index(drop=True)
    return monthly


def monthly_timeline(start_year_month: str, end_year_month: str) -> list[str]:
    dates = pd.date_range(
        start=f"{start_year_month}-01",
        end=f"{end_year_month}-01",
        freq="MS",
    )
    return dates.strftime("%Y-%m").tolist()


def resolve_final_source_path(ctx: PipelineContext, source: dict[str, Any]) -> Path:
    path = Path(source["path"])
    if path.is_absolute():
        if not path.exists():
            raise MissingSourceError(
                f"Expected source file {path} for {source['name']}"
            )
        return path

    candidates = [
        ctx.final_dir / path,
        ctx.processed_dir / path,
        ctx.processed_dir / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    expected = candidates[0].relative_to(ctx.repo_root)
    fallback = candidates[1].relative_to(ctx.repo_root)
    raise MissingSourceError(
        f"Expected source file {expected} for {source['name']} "
        f"(fallback checked {fallback})"
    )


def normalized_source_path(ctx: PipelineContext, source_name: str) -> Path:
    return ctx.source_dir / f"{source_name}.csv"


def is_manual_source(source: dict[str, Any]) -> bool:
    return source.get("kind") in {
        "manual_daily_csv",
        "manual_monthly_csv",
        "manual_annual_csv",
        "manual_huc12_wide_csv",
    }

def write_raw_csv(ctx: PipelineContext, source_name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    raw_csv_path = ctx.raw_dir / f"{source_name}.csv"
    frame.to_csv(raw_csv_path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def format_usdm_date(value: str) -> str:
    timestamp = pd.to_datetime(value)
    return f"{timestamp.month}/{timestamp.day}/{timestamp.year}"


def extract_popestimate_columns(row: pd.Series) -> dict[int, float]:
    extracted: dict[int, float] = {}
    for column, value in row.items():
        match = re.fullmatch(r"POPESTIMATE(\d{4})", str(column))
        if match:
            extracted[int(match.group(1))] = float(value)
    return extracted


def extract_next_link(payload: dict[str, Any]) -> str | None:
    links = payload.get("links", {})
    next_link = links.get("next")
    if isinstance(next_link, str):
        return next_link
    if isinstance(next_link, dict):
        return next_link.get("href")
    return None


def pick_first(mapping: dict[str, Any], candidates: list[str]) -> Any:
    for candidate in candidates:
        if candidate in mapping and mapping[candidate] not in (None, ""):
            return mapping[candidate]
    return None


def coerce_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    lowered = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        lowered_candidate = candidate.lower()
        if lowered_candidate in lowered:
            return lowered[lowered_candidate]
    raise Phase1Error(
        "None of the expected columns were found. "
        f"Expected one of {candidates}, got {list(frame.columns)}"
    )


def parse_modis_year_month(filename: str) -> str:
    match = re.search(r"\.A(\d{4})(\d{3})\.", filename)
    if not match:
        raise Phase1Error(f"Could not parse MODIS date from {filename}")
    year = int(match.group(1))
    day_of_year = int(match.group(2))
    date = datetime.strptime(f"{year}-{day_of_year:03d}", "%Y-%j")
    return date.strftime("%Y-%m")


def pick_modis_subdataset(subdatasets: list[tuple[str, str]], contains: str) -> str:
    contains_lower = contains.lower()
    for path, description in subdatasets:
        if contains_lower in description.lower():
            return path
    raise Phase1Error(f"Could not find a MODIS subdataset matching {contains!r}")


def extract_modis_region(
    dataset: Any,
    bbox: list[float],
    scale_factor: float,
    invalid_below: float,
) -> np.ndarray:
    min_lon, min_lat, max_lon, max_lat = bbox
    radius = 6371007.181

    def latlon_to_sinusoidal(lon: float, lat: float) -> tuple[float, float]:
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        x_value = radius * lon_rad * np.cos(lat_rad)
        y_value = radius * lat_rad
        return x_value, y_value

    geotransform = dataset.GetGeoTransform()
    full = dataset.ReadAsArray()
    rows, cols = full.shape

    ul_x, ul_y = latlon_to_sinusoidal(min_lon, max_lat)
    lr_x, lr_y = latlon_to_sinusoidal(max_lon, min_lat)

    col_min = int((ul_x - geotransform[0]) / geotransform[1])
    row_min = int((ul_y - geotransform[3]) / geotransform[5])
    col_max = int((lr_x - geotransform[0]) / geotransform[1])
    row_max = int((lr_y - geotransform[3]) / geotransform[5])

    row_min = max(0, min(row_min, rows - 1))
    row_max = max(0, min(row_max, rows))
    col_min = max(0, min(col_min, cols - 1))
    col_max = max(0, min(col_max, cols))

    region = full[min(row_min, row_max) : max(row_min, row_max), min(col_min, col_max) : max(col_min, col_max)]
    scaled = region.astype(float) / scale_factor
    scaled[region < invalid_below] = np.nan
    return scaled


def subset_data_array_to_bbox(data_array: Any, bbox: list[float]) -> Any:
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_name = find_coord_name(data_array, ["lat", "latitude", "y"])
    lon_name = find_coord_name(data_array, ["lon", "longitude", "x"])

    lat_values = data_array[lat_name]
    lon_values = data_array[lon_name]

    if float(lon_values.max()) > 180 and min_lon < 0:
        min_lon = min_lon % 360
        max_lon = max_lon % 360

    lat_slice = slice(min_lat, max_lat) if lat_values[0] <= lat_values[-1] else slice(max_lat, min_lat)
    lon_slice = slice(min_lon, max_lon) if lon_values[0] <= lon_values[-1] else slice(max_lon, min_lon)
    return data_array.sel({lat_name: lat_slice, lon_name: lon_slice})


def find_coord_name(data_array: Any, candidates: list[str]) -> str:
    available = {name.lower(): name for name in data_array.coords}
    for candidate in candidates:
        if candidate.lower() in available:
            return available[candidate.lower()]
    raise Phase1Error(
        f"Could not find any of the coordinate names {candidates} in {list(data_array.coords)}"
    )


def reduce_spatial_monthly(data_array: Any, transform_name: str) -> dict[pd.Timestamp, float]:
    transform = build_transform(transform_name)
    transformed = transform(data_array)
    time_name = next((dim for dim in transformed.dims if dim.lower() == "time"), None)
    if time_name is None:
        timestamp = pd.Timestamp("1970-01-01")
        return {timestamp: float(transformed.mean().item())}

    spatial_dims = [dim for dim in transformed.dims if dim != time_name]
    reduced = transformed.mean(dim=spatial_dims, skipna=True)
    series = reduced.to_series().dropna()
    return {pd.Timestamp(index): float(value) for index, value in series.items()}


def build_transform(name: str):
    if name == "identity":
        return lambda array: array
    if name == "kelvin_to_celsius":
        return lambda array: array - 273.15
    if name == "kg_m2_s_to_mm_day":
        return lambda array: array * 86400.0
    raise Phase1Error(f"Unsupported transform: {name}")


def import_requests():
    import requests

    return requests


def import_earthaccess():
    import earthaccess

    return earthaccess


def import_xarray():
    import xarray as xr

    return xr


def import_gdal():
    from osgeo import gdal

    gdal.UseExceptions()
    return gdal


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
