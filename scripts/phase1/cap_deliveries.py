"""
cap_deliveries.py
-----------------
Monthly Central Arizona Project (CAP) water deliveries, 1999-01 onward, from CAP's
own published delivery reports, with Reclamation's Lake Havasu diversion as a
cross-check where the decree accounting reports carry it in extractable text.

WHY THIS EXISTS
---------------
PROBLEMS.md P5 / Part 3 item 5 and PHASE3_PLAN.md §16 item 2: CAP water is the
substitute for groundwater pumping in the three counties that take it (Maricopa,
Pinal, Pima — all in-region, PROBLEMS.md P8). When CAP delivery is high, pumping is
low; when the shortage tiers cut CAP, pumping rises. It is the one remaining
*monthly* pumping proxy the project can acquire, and it responds to policy
(shortage tiers, allocation cuts), unlike ADWR's annual pumpage or the HUC12
withdrawal matrix, which are calendar templates.

Input  : data/raw/cap/deliveries/monthly_delivery_report_{1999..2018}.pdf
           CAP "Monthly Deliveries" reports — one table per classification
           (M&I, Ag, Federal/Indian), each with a TOTAL row, plus TOTAL DELIVERIES.
         data/raw/cap/deliveries/ytd_by_contract_type_{2018..}.pdf
           CAP "Year to Date Deliveries by Contract Type" — a summary table with
           one row per contract type and a Total row.
         data/raw/cap/decree/decree_{2000..2025}.pdf
           Reclamation "Colorado River Accounting and Water Use Report" (the
           decree accounting report). Carries "Central Arizona Project / Pumped
           from Lake Havasu / Diversion" as 12 monthly acre-feet values. Used as a
           cross-check only; the row is machine-readable for 2010-2025 and a few
           earlier years, and the script records which.
         All are fetched from the URLs in SOURCES if missing.

Output : data/Final/cap_deliveries_monthly.csv
    year_month              str   YYYY-MM
    cap_deliveries_af       float total CAP deliveries in the month, acre-feet
    cap_mi_af               float M&I subcontract deliveries (long-term municipal)
    cap_ag_af               float agricultural / excess-pool deliveries
    cap_federal_af          float federal (tribal, on- and off-reservation)
    cap_havasu_diversion_af float Reclamation's diversion at Lake Havasu (NaN where
                                  the decree row could not be read)
    cap_source              str   "monthly_report" | "ytd_report" | "transcribed"

Classification note. CAP's tables changed shape in 2018: before, deliveries were
listed by classification (M&I / AG / FEDERAL or INDIAN); after, by contract type
(M&I Subcontract / Federal On-Res / Federal Off-Res / Excess - Ag Pool / Excess -
Other Excess), and from 2023 by rate (Federal / M&I Subcontract / Reclamation
wheeling), with no Ag Pool row at all because Tier 2a cut that pool to zero. The
three columns here map the later shapes onto the first (Ag Pool + Other Excess ->
ag; On-Res + Off-Res -> federal; from 2023, ag = total - M&I - Federal). The TOTAL
is the same quantity in every era and is the series the models use; the split is
context.

Two years (2008, 2009) are scanned images with no text layer. Tesseract OCR
recovered the 2008 grand-total row with two mis-read cells (67,501 for 67,591;
166,834 for 156,834) and could not find 2009's at all, so the totals rows for
those two years were transcribed by eye from the page images (TRANSCRIBED below,
with the page each row sits on). Every year, transcribed or not, must pass the
same guard: the twelve months must sum to the printed annual total within 0.2 %,
or the year is rejected and written as NaN rather than silently shipped. Both
transcribed years pass it exactly. That is the lesson of PROBLEMS.md P7.

Cross-check note. Reclamation's Havasu diversion does NOT track deliveries month by
month (within-year r is about -0.2): CAP pumps at Havasu in winter to fill Lake
Pleasant and delivers out of it in summer. The two agree on annual totals
(deliveries / diversion ~ 0.97). The delivery series is the pumping proxy; the
diversion is kept only as an annual-scale check that the parse is right.

Run:
    python -m scripts.phase1.cap_deliveries
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw" / "cap"
DELIVERIES_DIR = RAW_DIR / "deliveries"
DECREE_DIR = RAW_DIR / "decree"
OUTPUT_FILE = ROOT / "data" / "Final" / "cap_deliveries_monthly.csv"

CAP_LIBRARY = "https://library.cap-az.com/documents/departments/water-operations/"
DECREE_BASE = "https://www.usbr.gov/lc/region/g4000/4200Rpts/DecreeRpt/"

MONTHLY_REPORT_YEARS = range(1999, 2019)
YTD_REPORT_YEARS = range(2018, 2027)
DECREE_YEARS = range(2000, 2026)
# Transcribed from the scanned reports (see docstring). Each row: 12 months then the
# printed annual total, acre-feet, exactly as printed. Page numbers are of the PDF.
TRANSCRIBED: dict[int, dict[str, list[float]]] = {
    2008: {
        # TOTAL DELIVERIES, page 6
        "total": [64_972, 67_591, 141_106, 173_379, 156_834, 183_449, 170_451, 195_254,
                  146_110, 106_848, 73_122, 68_908, 1_548_024],
        # TOTAL (AG), page 4
        "ag": [7_212, 13_617, 48_228, 70_196, 56_052, 70_618, 58_310, 37_915,
               21_683, 9_623, 5_811, 6_385, 405_650],
        # TOTAL (FEDERAL), page 6
        "federal": [14_975, 11_719, 26_380, 30_433, 33_280, 18_672, 15_071, 14_805,
                    10_992, 9_030, 4_420, 2_559, 192_336],
    },
    2009: {
        # TOTAL DELIVERIES, page 5
        "total": [76_539, 76_551, 133_808, 155_827, 171_233, 184_114, 196_222, 204_572,
                  136_875, 119_470, 59_039, 95_987, 1_610_237],
        # TOTAL (AG), page 4
        "ag": [11_983, 17_952, 40_823, 62_325, 55_184, 54_605, 57_367, 44_800,
               24_845, 15_478, 3_081, 9_251, 397_694],
        # TOTAL (FEDERAL), page 5
        "federal": [12_811, 22_658, 27_452, 17_298, 34_808, 34_915, 17_278, 21_492,
                    20_214, 11_648, 5_272, 7_693, 233_539],
    },
}

# Row sums must match the printed annual total to this tolerance.
TOTAL_TOLERANCE = 0.002
# Annual deliveries outside this range mean the parse grabbed the wrong row.
PLAUSIBLE_ANNUAL_AF = (600_000, 1_900_000)

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------
def _url_for(kind: str, year: int) -> str:
    if kind == "monthly":
        return f"{CAP_LIBRARY}monthly-delivery-report-{year}.pdf"
    if kind == "ytd":
        # CAP's 2023 file drops the hyphen in "contracttype".
        stem = "year-to-date-by-contracttype" if year == 2023 else "year-to-date-by-contract-type"  # noqa: PLR2004
        return f"{CAP_LIBRARY}{stem}-{year}.pdf"
    if kind == "decree":
        return f"{DECREE_BASE}{year}/{year}.pdf" if year >= 2003 else f"{DECREE_BASE}{year}DecreeRpt.pdf"  # noqa: PLR2004
    raise ValueError(kind)


def _path_for(kind: str, year: int) -> Path:
    if kind == "monthly":
        return DELIVERIES_DIR / f"monthly_delivery_report_{year}.pdf"
    if kind == "ytd":
        return DELIVERIES_DIR / f"ytd_by_contract_type_{year}.pdf"
    return DECREE_DIR / f"decree_{year}.pdf"


def _ensure(kind: str, year: int) -> Path | None:
    """Return the local PDF, fetching it if absent. None if it cannot be had."""
    path = _path_for(kind, year)
    if path.exists() and path.stat().st_size > 10_000:  # noqa: PLR2004
        return path
    import requests  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    url = _url_for(kind, year)
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (eotd)"}, timeout=240)
            if r.status_code == 200 and r.content[:4] == b"%PDF":  # noqa: PLR2004
                path.write_bytes(r.content)
                log.info("fetched %s (%d bytes)", path.name, len(r.content))
                return path
            log.warning("%s -> HTTP %d", url, r.status_code)
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("attempt %d for %s failed: %s", attempt, url, e)
            time.sleep(2**attempt)
    return None


# ---------------------------------------------------------------------------
# text extraction
# ---------------------------------------------------------------------------
def _pdftotext(path: Path) -> str:
    try:
        return subprocess.run(
            ["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True
        ).stdout
    except FileNotFoundError as e:
        raise RuntimeError("pdftotext (poppler-utils) is required to read the CAP reports") from e


_NUM = re.compile(r"-?\d[\d,]*")


def _numbers(line: str) -> list[float]:
    return [float(t.replace(",", "")) for t in _NUM.findall(line)]


def _row_after_label(text: str, label: re.Pattern[str], n_min: int = 13) -> list[float] | None:
    """First line matching `label` that carries at least 12 months + a total."""
    for line in text.splitlines():
        if label.search(line):
            nums = _numbers(line.split("|")[-1] if "|" in line else line)
            # Drop anything that came from the label itself (e.g. "$54/AF").
            if len(nums) >= n_min:
                return nums
    return None


def _check_year(
    year: int, months: list[float], printed_total: float, label: str, partial: bool = False
) -> bool:
    total = float(np.nansum(months))
    if printed_total <= 0 or abs(total - printed_total) / printed_total > TOTAL_TOLERANCE:
        log.error(
            "%d %s: months sum to %s but the printed total is %s — REJECTED",
            year, label, f"{total:,.0f}", f"{printed_total:,.0f}",
        )
        return False
    if not partial and not PLAUSIBLE_ANNUAL_AF[0] <= total <= PLAUSIBLE_ANNUAL_AF[1]:
        log.error(
            "%d %s: annual total %s is outside the plausible range — REJECTED",
            year, label, f"{total:,.0f}",
        )
        return False
    return True


def transcribed_year(year: int) -> dict | None:
    rows = TRANSCRIBED[year]
    if not _check_year(year, rows["total"][:12], rows["total"][12], "transcribed total"):
        return None
    for key in ("ag", "federal"):
        if not _check_year(year, rows[key][:12], rows[key][12], f"transcribed {key}", partial=True):
            return None
    total, ag, fed = (np.asarray(rows[k][:12], dtype=float) for k in ("total", "ag", "federal"))
    return {"total": list(total), "mi": list(total - ag - fed), "ag": list(ag), "federal": list(fed)}


# ---------------------------------------------------------------------------
# CAP monthly delivery reports, 1999-2018
# ---------------------------------------------------------------------------
_TOTAL_DELIVERIES = re.compile(r"^\s*_*\s*TOTAL DELIVERIES\b", re.I)
_TOTAL_MI = re.compile(r"^\s*TOTAL\s*\(M\s*&\s*I\)", re.I)
_TOTAL_AG = re.compile(r"^\s*TOTAL\s*\(AG\)", re.I)
_TOTAL_FED = re.compile(r"^\s*TOTAL\s*\((FEDERAL|INDIAN)\)", re.I)
# 2015-2018 use title case and drop the parentheses.
_TOTAL_DELIVERIES_TC = re.compile(r"^\s*Total Deliveries\b")
_TOTAL_MI_TC = re.compile(r"^\s*Total M\s*&\s*I\b", re.I)
_TOTAL_AG_TC = re.compile(r"^\s*Total (Ag|Agricultur)", re.I)
_TOTAL_FED_TC = re.compile(r"^\s*Total (Federal|Indian)", re.I)


def parse_monthly_report(year: int, text: str) -> dict | None:
    grand = _row_after_label(text, _TOTAL_DELIVERIES) or _row_after_label(text, _TOTAL_DELIVERIES_TC)
    if grand is None:
        log.error("%d: no TOTAL DELIVERIES row found", year)
        return None
    months, printed = grand[:12], grand[12]
    if not _check_year(year, months, printed, "total deliveries"):
        return None

    def part(pattern_a: re.Pattern[str], pattern_b: re.Pattern[str]) -> list[float] | None:
        row = _row_after_label(text, pattern_a) or _row_after_label(text, pattern_b)
        return row[:12] if row else None

    return {
        "total": months,
        "mi": part(_TOTAL_MI, _TOTAL_MI_TC),
        "ag": part(_TOTAL_AG, _TOTAL_AG_TC),
        "federal": part(_TOTAL_FED, _TOTAL_FED_TC),
    }


# ---------------------------------------------------------------------------
# CAP year-to-date by contract type, 2018-
# ---------------------------------------------------------------------------
_YTD_ROWS = {
    "ag_pool": re.compile(r"^\s*Excess\s*[-\u2013\u2014]\s*Ag Pool\b"),
    "other_excess": re.compile(r"^\s*Excess\s*[-\u2013\u2014]\s*Other Excess\b"),
    "fed_off": re.compile(r"^\s*Federal Off[-\u2013\u2014]Res\b"),
    "fed_on": re.compile(r"^\s*Federal On[-\u2013\u2014]Res\b"),
    # 2023 onward the report is "Deliveries by Rate": one Federal row, no Ag Pool row
    # (the pool was cut to zero under Tier 2a), plus small Reclamation wheeling rows.
    "fed_all": re.compile(r"^\s*Federal\s+[\d,]"),
    "mi": re.compile(r"^\s*M&I Subcontract\b"),
    "total": re.compile(r"^\s*Total\b"),
}


def parse_ytd_report(year: int, text: str, through_month: int | None) -> dict | None:
    """The summary table is the first block; its rows carry Jan..Dec, Delivered,
    Scheduled, Remaining. Months not yet delivered print as 0 and are NaN here."""
    head = text.split("Deliveries by Contract Type", 1)[-1][:6000]
    rows: dict[str, list[float]] = {}
    for key, pat in _YTD_ROWS.items():
        for line in head.splitlines():
            if pat.search(line):
                nums = _numbers(line)
                if len(nums) >= 13:  # noqa: PLR2004
                    rows[key] = nums
                    break
    if "total" not in rows:
        log.error("%d: no Total row in the YTD summary", year)
        return None
    months, printed = rows["total"][:12], rows["total"][12]
    if through_month is not None:
        months = months[:through_month] + [np.nan] * (12 - through_month)
        printed_months = months[:through_month]
        if not _check_year(year, printed_months, printed, "ytd total (partial year)", partial=True):
            return None
    elif not _check_year(year, months, printed, "ytd total"):
        return None

    def get(*keys: str) -> list[float] | None:
        parts = [rows[k][:12] for k in keys if k in rows]
        if not parts:
            return None
        out = np.sum(parts, axis=0).astype(float)
        if through_month is not None:
            out[through_month:] = np.nan
        return list(out)

    federal = get("fed_on", "fed_off") if ("fed_on" in rows or "fed_off" in rows) else get("fed_all")
    ag = get("ag_pool", "other_excess")
    mi = get("mi")
    if ag is None and federal is not None and mi is not None:
        # "Deliveries by Rate" era: no Ag Pool row exists, so ag is what is left after
        # M&I and Federal — wheeling included, which is a few hundred AF a year.
        ag = list(np.asarray(months, dtype=float) - np.asarray(mi) - np.asarray(federal))
    return {"total": months, "mi": mi, "ag": ag, "federal": federal}


def _partial_year_extent(text: str) -> int | None:
    """For the current year: how many months are delivered. The summary Total row
    prints 0 for undelivered months; the last nonzero month is the extent."""
    head = text.split("Deliveries by Contract Type", 1)[-1][:6000]
    for line in head.splitlines():
        if _YTD_ROWS["total"].search(line):
            nums = _numbers(line)
            if len(nums) >= 13:  # noqa: PLR2004
                months = nums[:12]
                nonzero = [i for i, v in enumerate(months) if v > 0]
                return (max(nonzero) + 1) if nonzero else 0
    return None


# ---------------------------------------------------------------------------
# Reclamation decree report cross-check
# ---------------------------------------------------------------------------
_HAVASU = re.compile(r"Pumped from Lake Havasu.*?Diversion\s+([\d,\s]+)$", re.I)


def parse_decree(year: int, text: str) -> list[float] | None:
    for line in text.splitlines():
        m = _HAVASU.search(line)
        if m:
            nums = _numbers(m.group(1))
            if len(nums) >= 13 and PLAUSIBLE_ANNUAL_AF[0] <= nums[12] <= PLAUSIBLE_ANNUAL_AF[1]:  # noqa: PLR2004
                if abs(sum(nums[:12]) - nums[12]) / nums[12] <= TOTAL_TOLERANCE:
                    return nums[:12]
    return None


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def build() -> pd.DataFrame:
    records: dict[int, dict] = {}
    source: dict[int, str] = {}

    for year in MONTHLY_REPORT_YEARS:
        path = _ensure("monthly", year)
        if path is None:
            continue
        if year in TRANSCRIBED:
            parsed = transcribed_year(year)
            source[year] = "transcribed"
        else:
            parsed = parse_monthly_report(year, _pdftotext(path))
            source[year] = "monthly_report"
        if parsed:
            records[year] = parsed
        else:
            source.pop(year, None)

    current_year = pd.Timestamp.today().year
    for year in YTD_REPORT_YEARS:
        if year in records:  # 2018 exists in both; the monthly report wins
            continue
        path = _ensure("ytd", year)
        if path is None:
            continue
        text = _pdftotext(path)
        through = _partial_year_extent(text) if year >= current_year else None
        parsed = parse_ytd_report(year, text, through)
        if parsed:
            records[year] = parsed
            source[year] = "ytd_report"

    decree: dict[int, list[float]] = {}
    for year in DECREE_YEARS:
        path = _ensure("decree", year)
        if path is None:
            continue
        row = parse_decree(year, _pdftotext(path))
        if row:
            decree[year] = row
    log.info(
        "decree cross-check readable for %d of %d years: %s",
        len(decree), len(DECREE_YEARS), ", ".join(str(y) for y in sorted(decree)),
    )

    rows = []
    for year in sorted(records):
        rec = records[year]
        for m in range(12):
            rows.append(
                {
                    "year_month": f"{year}-{m + 1:02d}",
                    "cap_deliveries_af": rec["total"][m],
                    "cap_mi_af": rec["mi"][m] if rec["mi"] else np.nan,
                    "cap_ag_af": rec["ag"][m] if rec["ag"] else np.nan,
                    "cap_federal_af": rec["federal"][m] if rec["federal"] else np.nan,
                    "cap_havasu_diversion_af": decree[year][m] if year in decree else np.nan,
                    "cap_source": source[year],
                }
            )
    df = pd.DataFrame(rows)
    df = df[df["cap_deliveries_af"].notna()].reset_index(drop=True)
    return df


def _report(df: pd.DataFrame) -> None:
    annual = df.assign(year=df.year_month.str[:4]).groupby("year")["cap_deliveries_af"].sum()
    log.info("years covered: %s .. %s (%d months)", df.year_month.min(), df.year_month.max(), len(df))
    log.info("annual totals (kAF):\n%s", (annual / 1000).round(0).to_string())
    both = df.dropna(subset=["cap_havasu_diversion_af"])
    if len(both):
        r = np.corrcoef(both.cap_deliveries_af, both.cap_havasu_diversion_af)[0, 1]
        ratio = both.cap_deliveries_af.sum() / both.cap_havasu_diversion_af.sum()
        log.info(
            "cross-check vs Reclamation Havasu diversion over %d months: r = %.4f, "
            "deliveries / diversion = %.3f (the remainder is canal loss and recharge timing)",
            len(both), r, ratio,
        )
    by_month = df.assign(m=df.year_month.str[5:].astype(int)).groupby("m")["cap_deliveries_af"].mean()
    log.info("seasonal shape (mean AF by month): peak %s, trough %s", by_month.idxmax(), by_month.idxmin())


def main() -> None:
    log.info("=== cap_deliveries.py start ===")
    df = build()
    if df.empty:
        raise RuntimeError("no CAP delivery data could be parsed")
    _report(df)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    log.info("wrote %s (%d rows)", OUTPUT_FILE, len(df))


if __name__ == "__main__":
    main()
