from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .common import progress


FIELD_COUNT = 7
PLAIN_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
VALUE_WITH_FLAGS_RE = re.compile(r"^(?P<value>-?\d+(?:\.\d+)?)(?P<flags>[A-Za-z]*)$")
SITE_LINE_RE = re.compile(r"^:SNOTEL Site:\s*(?P<name>.+?)\s+WRCC Number:\s*(?P<wrcc>\d{6})$")
ID_LINE_RE = re.compile(
    r"^: Snotel ID:\s*(?P<snotel>\S+)\s+NWS Handbook 5 ID:\s*(?P<nws>\S+)$"
)


@dataclass(frozen=True)
class StationInput:
    station_id: str
    candidate_paths: tuple[str, ...]


STATION_INPUTS = (
    StationInput("baker_butte", ("test.txt", "data/raw/snotel_baker_butte_2000-2020.txt")),
    StationInput("happy_jack", ("test2.txt", "data/raw/snotel_happy_jack_2000-2020.txt")),
    StationInput("mormon_mountain", ("test3.txt", "data/raw/snotel_mormon_mountain_2000-2020.txt")),
    StationInput(
        "hannagan_meadows",
        (
            "test4.txt",
            "data/raw/snotel_hannagan_meadows_2000-2020.txt",
            "data/raw/snotel_hannigan_meadows_2000-2020.txt",
        ),
    ),
)


def ensure_snotel_monthly(
    *,
    repo_root: Path,
    processed_dir: Path,
    final_dir: Path,
    **_: Path,
) -> Path:
    monthly_output = final_dir / "snotel_swe.csv"
    if monthly_output.exists():
        progress(f"snotel monthly source already present at {monthly_output.relative_to(repo_root)}")
        return monthly_output

    daily_input = find_existing_daily_bundle(repo_root, processed_dir, final_dir)
    if daily_input is None:
        daily_input = build_daily_bundle(repo_root, final_dir)

    progress(f"aggregating {daily_input.relative_to(repo_root)} to monthly SNOTEL source")
    frame = pd.read_csv(daily_input)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["swe_in"] = pd.to_numeric(frame["swe_in"], errors="coerce")
    frame = frame.dropna(subset=["date", "swe_in"])
    frame["year_month"] = frame["date"].dt.strftime("%Y-%m")
    grouped = (
        frame.groupby(["year_month", "station_id"], as_index=False)["swe_in"]
        .mean()
        .rename(columns={"swe_in": "snow_water_equivalent_in"})
    )
    monthly = (
        grouped.groupby("year_month", as_index=False)["snow_water_equivalent_in"]
        .mean()
        .sort_values("year_month")
    )
    monthly.to_csv(monthly_output, index=False)
    progress(f"wrote {monthly_output.relative_to(repo_root)}")
    return monthly_output


def find_existing_daily_bundle(repo_root: Path, processed_dir: Path, final_dir: Path) -> Path | None:
    candidates = [
        final_dir / "snotel_swe_daily.csv",
        processed_dir / "snotel_swe_daily.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            progress(f"using existing SNOTEL daily bundle at {candidate.relative_to(repo_root)}")
            return candidate
    return None


def build_daily_bundle(repo_root: Path, final_dir: Path) -> Path:
    all_rows: list[dict[str, object]] = []
    for station in STATION_INPUTS:
        source_path = resolve_station_input(repo_root, station)
        rows = parse_station_file(source_path, station.station_id)
        all_rows.extend(rows)

    all_rows.sort(key=lambda row: (row["date"], row["station_id"]))
    if not all_rows:
        raise RuntimeError("Could not build SNOTEL bundle because no station rows were parsed.")

    combined_path = final_dir / "snotel_swe_daily.csv"
    write_csv(
        combined_path,
        all_rows,
        ["date", "station_id", "station_name", "wrcc_no", "snotel_id", "nws_id", "swe_in"],
    )
    progress(f"wrote {combined_path.relative_to(repo_root)}")
    return combined_path


def resolve_station_input(repo_root: Path, station: StationInput) -> Path:
    for candidate in station.candidate_paths:
        path = repo_root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing source file for {station.station_id}. Checked: {', '.join(station.candidate_paths)}"
    )


def parse_station_file(path: Path, station_id: str) -> list[dict[str, object]]:
    site_name = None
    wrcc_no = None
    snotel_id = None
    nws_id = None
    lines = path.read_text(errors="replace").splitlines()

    for line in lines:
        site_match = SITE_LINE_RE.match(line)
        if site_match:
            site_name = site_match.group("name").strip()
            wrcc_no = site_match.group("wrcc")
            continue

        id_match = ID_LINE_RE.match(line)
        if id_match:
            snotel_id = id_match.group("snotel")
            nws_id = id_match.group("nws")

    if not all([site_name, wrcc_no, snotel_id, nws_id]):
        raise RuntimeError(f"Could not parse station metadata from {path.name}")

    rows: list[dict[str, object]] = []
    for line in lines:
        if not line.startswith(wrcc_no):
            continue

        tokens = line.split()
        record = tokens[0]
        fields = fold_fields(tokens[1:])
        if len(fields) != FIELD_COUNT:
            continue
        if any(field == "-999M" for field in fields):
            continue

        parsed_fields = [parse_field(field) for field in fields]
        if any(field is None for field in parsed_fields):
            continue

        swe_in = parsed_fields[2]["value"] / 10.0
        rows.append(
            {
                "date": datetime.strptime(record[6:14], "%Y%m%d").date().isoformat(),
                "station_id": station_id,
                "station_name": site_name,
                "wrcc_no": wrcc_no,
                "snotel_id": snotel_id,
                "nws_id": nws_id,
                "swe_in": round(swe_in, 4),
            }
        )

    rows.sort(key=lambda row: row["date"])
    return rows


def fold_fields(tokens: list[str]) -> list[str]:
    folded: list[str] = []
    index = 0
    while index < len(tokens) and len(folded) < FIELD_COUNT:
        token = tokens[index]
        if (
            PLAIN_NUMBER_RE.fullmatch(token)
            and index + 1 < len(tokens)
            and re.fullmatch(r"[A-Za-z]", tokens[index + 1])
        ):
            folded.append(token + tokens[index + 1])
            index += 2
        else:
            folded.append(token)
            index += 1
    return folded


def parse_field(token: str) -> dict[str, object] | None:
    match = VALUE_WITH_FLAGS_RE.fullmatch(token)
    if not match:
        return None
    return {"value": float(match.group("value")), "flags": match.group("flags")}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    from .common import load_paths

    paths = load_paths(Path("config/phase1.example.json"))
    ensure_snotel_monthly(**paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
