"""
cap_calibration.py
------------------
Measure the first link of the Lake Mead lever from CAP's own delivery record:
how much of a declared Arizona shortage reduction actually shows up as water not
delivered in the region.

Layer 2 (frontend/structural.js, `mead_tier_to_depth`) runs
    elevation -> DCP tier -> AZ combined reduction (kAF/yr)
              -> x region_share_of_az_reduction -> x groundwater_substitution_fraction
              -> pumping -> storage -> depth.
`region_share_of_az_reduction` was assumed (0.60, then 0.95 after PROBLEMS.md P8
established that all three CAP counties are in-region). With
`data/Final/cap_deliveries_monthly.csv` on disk it can be measured: every acre-foot
CAP does not deliver is an acre-foot not delivered inside the region, so

    share = (baseline annual deliveries - actual annual deliveries) / declared AZ cut

for each year a Tier 1 or deeper shortage was in force. A ratio below 1 means part
of the declared cut was absorbed without a delivery loss (ICS releases, AWBA
firming, conservation credits); above 1 means deliveries fell by more than the
tier alone required (compensated system conservation, which ran alongside the
tiers from 2023).

THE SPECIFICATION, FIXED BEFORE THE NUMBERS WERE LOOKED AT
----------------------------------------------------------
- Shortage years and declared cuts, from Reclamation's August 24-Month Study
  determinations and the DCP table in PHASE3_PARAMS.md §1:
    2020 Tier 0  192 kAF    2021 Tier 0  192 kAF    (DCP contributions only)
    2022 Tier 1  512 kAF    2023 Tier 2a 592 kAF
    2024 Tier 1  512 kAF    2025 Tier 1  512 kAF
- The VALUE is the mean ratio over the Tier >= 1 years (2022-2025) against the
  2015-2019 baseline (the last five full years before any DCP contribution).
- The BAND is the min-max over those years x two baselines (2015-2019 and
  2010-2019), clipped to [0, 1.2] — above 1 is physically possible for the reason
  above, but the constant multiplies a declared cut, so it is reported and the
  shipped band top is capped at 1.0 where the lever uses it.
- Tier 0 years are reported but do not enter the value: a 192 kAF contribution
  that DCP allowed to be met from ICS and conservation credits is not the same
  mechanism as a Tier 1 delivery cut, and the ratios show it.

Writes model/cap_calibration.json; structural_params.py reads it.

Run:
    python scripts/phase3/cap_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAP_FILE = ROOT / "data" / "Final" / "cap_deliveries_monthly.csv"
OUTPUT_FILE = ROOT / "model" / "cap_calibration.json"

DECLARED_CUT_KAF = {2020: 192, 2021: 192, 2022: 512, 2023: 592, 2024: 512, 2025: 512}
TIER = {2020: "0", 2021: "0", 2022: "1", 2023: "2a", 2024: "1", 2025: "1"}
JUDGED_YEARS = [2022, 2023, 2024, 2025]
BASELINES = {"2015-2019": range(2015, 2020), "2010-2019": range(2010, 2020)}
PRIMARY_BASELINE = "2015-2019"
BAND_CAP = 1.2


def main() -> None:
    df = pd.read_csv(CAP_FILE)
    df["year"] = df["year_month"].str[:4].astype(int)
    annual = df.groupby("year")["cap_deliveries_af"].agg(["sum", "count"])
    annual = annual[annual["count"] == 12]["sum"] / 1000.0  # kAF, full years only

    baselines = {name: float(annual.loc[list(years)].mean()) for name, years in BASELINES.items()}
    rows = []
    for year, cut in DECLARED_CUT_KAF.items():
        if year not in annual.index:
            continue
        for bname, base in baselines.items():
            drop = base - float(annual.loc[year])
            rows.append(
                {
                    "year": year,
                    "tier": TIER[year],
                    "declared_cut_kaf": cut,
                    "baseline": bname,
                    "baseline_kaf": round(base, 1),
                    "delivered_kaf": round(float(annual.loc[year]), 1),
                    "drop_kaf": round(drop, 1),
                    "ratio": round(drop / cut, 4),
                }
            )
    table = pd.DataFrame(rows)

    judged = table[(table.year.isin(JUDGED_YEARS))]
    primary = judged[judged.baseline == PRIMARY_BASELINE]
    value = float(primary.ratio.mean())
    band = [
        float(np.clip(judged.ratio.min(), 0, BAND_CAP)),
        float(np.clip(judged.ratio.max(), 0, BAND_CAP)),
    ]

    print("=== CAP deliveries vs declared Arizona shortage cuts ===\n")
    print("annual deliveries (kAF):")
    print((annual.round(0)).to_string())
    print("\nbaselines:", {k: round(v, 1) for k, v in baselines.items()})
    print("\n", table.to_string(index=False))
    print(f"\nregion_share_of_az_reduction, measured:")
    print(f"  value  {value:.3f}   (mean over {JUDGED_YEARS}, baseline {PRIMARY_BASELINE})")
    print(f"  band   {band[0]:.3f} .. {band[1]:.3f}   (min-max over judged years x baselines)")
    tier0 = table[table.year.isin([2020, 2021]) & (table.baseline == PRIMARY_BASELINE)]
    print(f"  Tier 0 years, for the record: {dict(zip(tier0.year, tier0.ratio))}")

    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "specification": {
                    "judged_years": JUDGED_YEARS,
                    "primary_baseline": PRIMARY_BASELINE,
                    "baselines": {k: [min(v), max(v)] for k, v in BASELINES.items()},
                    "declared_cut_kaf": DECLARED_CUT_KAF,
                    "tier": TIER,
                    "band_cap": BAND_CAP,
                },
                "annual_deliveries_kaf": {int(k): round(float(v), 1) for k, v in annual.items()},
                "baseline_kaf": baselines,
                "table": rows,
                "region_share_of_az_reduction": {
                    "value": value,
                    "band": band,
                    "n_years": int(len(primary)),
                    "status": "MEASURED",
                },
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
