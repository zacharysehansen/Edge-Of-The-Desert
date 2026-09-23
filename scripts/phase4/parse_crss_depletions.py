"""Parse CRSS Final EIS Appendix M — Arizona depletion schedules.

Extracts every Arizona table (Priority 1–4 and their sub-tables) from the
Bureau of Reclamation FEIS PDF using ``pdftotext -layout``, then writes a
tidy long-form CSV:

    data/processed/projections/az_depletions_crss.csv
        year,priority,water_user,depletion_af

Each row is one water-user / year pair. The ``priority`` column carries the
CRSS priority class (1–4). ``depletion_af`` is in acre-feet.

The script also emits per-priority totals and the CAP schedule for quick
verification against the PDF.

Runnable as:
    python -m scripts.phase4.parse_crss_depletions
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

PDF_PATH = REPO_ROOT / "data" / "raw" / "crss" / "FEIS_AppM_LowerBasin_DemandSchedule.pdf"
OUT_DIR = REPO_ROOT / "data" / "processed" / "projections"
OUT_PATH = OUT_DIR / "az_depletions_crss.csv"

# Tables to parse.  key = table id, value = CRSS priority class.
AZ_TABLES: dict[str, int] = {
    "M-3a": 1, "M-3b": 1, "M-3c": 1,
    "M-4": 2,
    "M-5a": 3, "M-5b": 3, "M-5c": 3, "M-5d": 3,
    "M-6a": 4, "M-6b": 4, "M-6c": 4, "M-6d": 4,
    "M-6e": 4, "M-6f": 4, "M-6g": 4, "M-6h": 4, "M-6i": 4,
}

# Known totals from the PDF for spot-checking (acre-feet).
EXPECTED_PRIORITY_TOTALS_2027 = {1: 570_388, 2: 18_588, 3: 639_636, 4: 1_571_388}
EXPECTED_PRIORITY_TOTALS_2040 = {1: 597_676, 2: 39_471, 3: 720_268, 4: 1_442_586}

# Known CAP values for verification.
EXPECTED_CAP = {2027: 1_511_829, 2040: 1_334_531, 2060: 1_334_531}


def _extract_text() -> str:
    """Run pdftotext -layout and return the full text."""
    result = subprocess.run(
        ["pdftotext", "-layout", str(PDF_PATH), "-"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def _parse_number(s: str) -> int | None:
    """Parse an integer that may contain commas. Return None if not numeric."""
    s = s.strip().replace(",", "")
    if re.match(r"^-?\d+$", s):
        return int(s)
    return None


def _find_column_slices(data_rows: list[str]) -> list[tuple[int, int]]:
    """Return (start, end) character-position pairs for each column.

    A column occupies a range of character positions where at least one
    data row has a non-space character.  Columns are separated by runs of
    positions that are spaces in ALL sample rows.
    """
    if not data_rows:
        return []

    sample = data_rows[:min(8, len(data_rows))]
    max_len = max(len(r) for r in sample)

    # Classify each position as "has content in at least one row" or not.
    has_content = [False] * max_len
    for row in sample:
        for j, ch in enumerate(row):
            if ch != " ":
                has_content[j] = True

    # Walk has_content to find contiguous content runs.
    slices: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for j in range(max_len):
        if has_content[j] and not in_run:
            start = j
            in_run = True
        elif not has_content[j] and in_run:
            slices.append((start, j))
            in_run = False
    if in_run:
        slices.append((start, max_len))

    return slices


def _extract_header_name(
    header_lines: list[str], col_start: int, col_end: int
) -> str:
    """Extract and clean a column name from the header lines at the given
    character-position span.

    Because headers span multiple lines and are wider than data columns,
    we expand the extraction window a few characters on each side to
    capture text that is centered over the data column.
    """
    # Expand the window to capture centered header text.
    margin = 4
    lo = max(0, col_start - margin)
    hi = col_end + margin

    parts: list[str] = []
    for hline in header_lines:
        segment = hline[lo:hi].strip()
        if segment and segment not in ("Calendar",):
            parts.append(segment)

    name = " ".join(parts)
    name = re.sub(r"\s+", " ", name).strip()
    # Strip trailing footnote markers like "1", "2" at end of names.
    name = re.sub(r"\d+$", "", name).strip()
    return name


def _parse_table(lines: list[str], table_id: str, priority: int) -> list[dict]:
    """Parse one table block into records."""
    records: list[dict] = []

    # --- Locate the header band (Calendar ... Year) --------------------------
    cal_idx: int | None = None
    year_idx: int | None = None
    for i, line in enumerate(lines):
        if "Calendar" in line and cal_idx is None:
            cal_idx = i
        if cal_idx is not None and re.search(r"\bYear\b", line) and i >= cal_idx:
            year_idx = i
            break
    if year_idx is None:
        return records

    # Collect all header lines (Calendar through Year, plus continuations).
    header_lines = list(lines[cal_idx: year_idx + 1])
    extra_header_end = year_idx + 1
    for i in range(year_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if tokens and re.match(r"^20\d{2}$", tokens[0]):
            break
        header_lines.append(lines[i])
        extra_header_end = i + 1

    # --- Collect data rows ---------------------------------------------------
    data_rows: list[str] = []
    for i in range(extra_header_end, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if tokens and re.match(r"^20\d{2}$", tokens[0]):
            data_rows.append(lines[i])
        elif data_rows:
            break

    if not data_rows:
        return records

    # --- Column detection ----------------------------------------------------
    col_slices = _find_column_slices(data_rows)
    # First slice is the year column; remaining are value columns.
    value_slices = col_slices[1:]

    # Build column names from headers.
    col_names: list[str] = []
    for start, end in value_slices:
        name = _extract_header_name(header_lines, start, end)
        col_names.append(name if name else f"unknown_{start}")

    # Verify the token count matches.
    n_value_tokens = len(data_rows[0].split()) - 1
    if len(col_names) != n_value_tokens:
        # Fallback — number generically.
        col_names = [f"col_{j}" for j in range(n_value_tokens)]

    for row_line in data_rows:
        tokens = row_line.split()
        year = int(tokens[0])
        values = tokens[1:]

        if len(values) != len(col_names):
            continue

        for name, val_str in zip(col_names, values):
            val = _parse_number(val_str)
            if val is not None:
                records.append({
                    "year": year,
                    "priority": priority,
                    "water_user": name,
                    "table": table_id,
                    "depletion_af": val,
                })

    return records


def _find_table_blocks(text: str) -> dict[str, list[str]]:
    """Split the full PDF text into per-table line blocks for Arizona tables."""
    lines_all = text.split("\n")
    blocks: dict[str, list[str]] = {}

    table_pattern = re.compile(r"Table\s+(M-\d+[a-z]?)\b")
    table_starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines_all):
        m = table_pattern.search(line)
        if m:
            tid = m.group(1)
            if tid in AZ_TABLES:
                table_starts.append((i, tid))

    for k, (start_i, tid) in enumerate(table_starts):
        end_i = (
            table_starts[k + 1][0] if k + 1 < len(table_starts)
            else len(lines_all)
        )
        blocks[tid] = lines_all[start_i:end_i]

    return blocks


def main() -> int:
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    text = _extract_text()
    blocks = _find_table_blocks(text)

    all_records: list[dict] = []
    for tid, priority in AZ_TABLES.items():
        if tid not in blocks:
            print(f"WARNING: table {tid} not found in PDF text", file=sys.stderr)
            continue
        recs = _parse_table(blocks[tid], tid, priority)
        all_records.extend(recs)
        users = sorted({r["water_user"] for r in recs})
        year_count = len({r["year"] for r in recs})
        print(f"  {tid:6s}  priority {priority}  {len(users):3d} users  "
              f"{year_count:2d} years  {len(recs):5d} records")

    if not all_records:
        print("ERROR: no records parsed", file=sys.stderr)
        return 1

    df = pd.DataFrame(all_records)

    # --- Exclude "Total" summary columns -------------------------------------
    # The last sub-table of each priority group has a "Total State of Arizona
    # Priority N" column.  We keep it for verification but exclude from the
    # output CSV.
    total_mask = df["water_user"].str.contains(
        r"^Total|^of Arizona|Arizona Priority", case=False, regex=True, na=False
    )
    totals_df = df[total_mask].copy()
    users_df = df[~total_mask].copy()

    out = users_df[["year", "priority", "water_user", "depletion_af"]].copy()
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}  ({len(out)} rows)")

    # ---------- Summary & verification ----------
    print("\nPer-priority totals (acre-feet, summed from individual users):")
    for yr in [2027, 2040, 2060]:
        print(f"\n  Year {yr}:")
        yr_df = users_df[users_df["year"] == yr]
        for p in sorted(users_df["priority"].unique()):
            total = yr_df[yr_df["priority"] == p]["depletion_af"].sum()
            print(f"    Priority {p}: {total:>12,}")

    # CAP schedule (the single largest user).
    cap = users_df[users_df["water_user"].str.contains("CAP", case=False, na=False)]
    if not cap.empty:
        print("\nCAP depletion schedule (sample years):")
        for yr in [2027, 2030, 2040, 2050, 2060]:
            row = cap[cap["year"] == yr]
            if not row.empty:
                print(f"  {yr}: {row['depletion_af'].values[0]:>12,} af")

    # Spot-check priority totals against the PDF's printed total columns.
    print("\nVerification against PDF total columns:")
    mismatches = 0
    for yr, expected_map in [
        (2027, EXPECTED_PRIORITY_TOTALS_2027),
        (2040, EXPECTED_PRIORITY_TOTALS_2040),
    ]:
        yr_df = users_df[users_df["year"] == yr]
        for p, expected_total in expected_map.items():
            actual = yr_df[yr_df["priority"] == p]["depletion_af"].sum()
            status = "OK" if actual == expected_total else f"DELTA {actual - expected_total:+,}"
            if actual != expected_total:
                mismatches += 1
            print(f"  {yr} P{p}: {actual:>12,}  (expected {expected_total:>12,})  {status}")

    for yr, expected_af in EXPECTED_CAP.items():
        cap_yr = cap[cap["year"] == yr]
        if not cap_yr.empty:
            actual = cap_yr["depletion_af"].values[0]
            status = "OK" if actual == expected_af else f"DELTA {actual - expected_af:+,}"
            if actual != expected_af:
                mismatches += 1
            print(f"  CAP {yr}: {actual:>12,}  (expected {expected_af:>12,})  {status}")

    total_all = users_df[users_df["year"] == 2027]["depletion_af"].sum()
    print(f"\nAZ total depletion 2027 (all priorities): {total_all:,} af")
    print(f"AZ apportionment: 2,800,000 af")

    if mismatches:
        print(f"\n{mismatches} verification delta(s).")
    else:
        print("\nAll verification checkpoints PASS.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
