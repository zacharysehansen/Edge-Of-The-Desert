"""
experiment_grace_gldas.py
-------------------------
PHASE3_PLAN.md §16 item 5 / PROBLEMS.md P5 Option C: does GLDAS land-surface
state — the fast, weather-driven part of monthly storage change — give GRACE any
skill? The roadmap's step 2, after CAP (steps §23/§24) was closed as a null.

GRACE's target is the month-to-month change in total water storage. GLDAS-2.1 Noah
gives an observationally-forced estimate of the part of that change that lives in
soil, snow and canopy. `scripts/phase1/gldas.py` acquired it over the same bounding
box the GRACE target is averaged on, 2000-01..2023-12.

THIS SCRIPT CHANGES NOTHING THAT SHIPS. Same harness as `experiment_grace_cap.py`:
both arms under `model_grace.py`'s real nested tuner on identical rows, five shared
test blocks, feature set the only moving part. It runs on GRACE's own window
(2002-10..2023-12, the 204 rows the deployed model reports), because §24
established that the shorter human-block window inflates first-fold effects.

Arms:
  shipped           the deployed GRACE feature set, as exported
  shipped+gldas     shipped plus the GLDAS block, RAW — this is the judged arm.
                    GLDAS is a physical state, not a calendar template like
                    irrigation or CAP; its seasonality is the seasonality of the
                    quantity GRACE measures, so removing it would remove the signal
  shipped+gldas_ds  the same block deseasonalized in-fold — reported, NOT judged

The GLDAS block (8 columns): the TWS proxy (soil 0-200 cm + SWE + canopy) and its
month-to-month change; soil moisture and its change; SWE and its change; ET; and
the TWS proxy's 3-month trailing mean. The changes are contemporaneous with the
target's month, exactly as precipitation and temperature already are.

THE DECISION RULE, FIXED BEFORE THE FIRST RUN
---------------------------------------------
The rule of §11.5, §23 and §24, unchanged, applied to the raw arm. GLDAS is REAL
only if ALL THREE hold on target R² meaned over the five shared blocks:
  1. Δ target R² > 0,
  2. it wins in at least 4 of 5 folds,
  3. the paired t across folds is |t| ≥ 2.0.
Anything else is NULL and the shipped feature set stands. One run, one window, no
second block. REAL deploys the block to GRACE; NULL closes the last feature route
for GRACE and the roadmap's step 5 (formal demotion) becomes the honest shipping
fix for that model.

Run:
    python -m scripts.phase2.experiment_grace_gldas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from scripts.phase2.experiment_grace_cap import full_window_rows, run_arm
from scripts.phase2.experiment_grace_nested import paired_t
from scripts.phase2.experiment_human_block import COMMON_START, N_SPLITS, build_variants
from scripts.phase2.features import _engineer_monthly
from scripts.phase2.merge import MONTHLY_WINDOW_END_EXTENDED, build_monthly_panel

ROOT = Path(__file__).resolve().parents[2]
GLDAS_FILE = ROOT / "data" / "Final" / "gldas_monthly.csv"
OUTPUT_FILE = ROOT / "model" / "experiment_grace_gldas.json"

MODEL_KEY = "grace"
WIN_FOLDS_REQUIRED = 4
T_REQUIRED = 2.0


def gldas_block(index: pd.Index) -> pd.DataFrame:
    g = pd.read_csv(GLDAS_FILE)
    g.index = pd.PeriodIndex(g["year_month"], freq="M")
    tws, soil, swe = g["gldas_tws_proxy_mm"], g["gldas_soil_moisture_mm"], g["gldas_swe_mm"]
    block = pd.DataFrame(index=g.index)
    block["gldas_tws_proxy_mm"] = tws
    block["gldas_tws_proxy_delta"] = tws.diff()
    block["gldas_soil_moisture_mm"] = soil
    block["gldas_soil_moisture_delta"] = soil.diff()
    block["gldas_swe_mm"] = swe
    block["gldas_swe_delta"] = swe.diff()
    block["gldas_evap_mm_day"] = g["gldas_evap_mm_day"]
    block["gldas_tws_proxy_roll3"] = tws.shift(1).rolling(3, min_periods=2).mean()
    if isinstance(index, pd.DatetimeIndex):
        block.index = block.index.to_timestamp()
    return block.reindex(index)


def physics_check(monthly: pd.DataFrame, gldas: pd.DataFrame, common: pd.Index) -> dict:
    """Before any model: does GLDAS's storage change track GRACE's? Same-month
    correlation of the two changes on the scored rows, raw and deseasonalized. If
    this is near zero the model test is moot."""
    y = (monthly["grace_groundwater_anomaly"] - monthly["grace_groundwater_anomaly_lag1"]).reindex(common)
    x = gldas["gldas_tws_proxy_delta"].reindex(common) / 10.0  # mm -> cm, GRACE's unit
    frame = pd.DataFrame({"grace_d_cm": y, "gldas_d_cm": x}).dropna()
    months = frame.index.month
    des = frame - frame.groupby(months).transform("mean")
    return {
        "n": int(len(frame)),
        "r_raw": float(frame.corr().iloc[0, 1]),
        "r_deseasonalized": float(des.corr().iloc[0, 1]),
        "sd_grace_change_cm": float(frame.grace_d_cm.std()),
        "sd_gldas_change_cm": float(frame.gldas_d_cm.std()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="wiring check only — NOT a result")
    args = ap.parse_args()
    if args.quick:
        import scripts.phase2.experiment_grace_cap as cap  # noqa: PLC0415
        import scripts.phase2.experiment_grace_nested as nested  # noqa: PLC0415
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: PLC0415

        quick = lambda x_tr, y_tr, x_te: (  # noqa: E731
            GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=0).fit(x_tr, y_tr).predict(x_te)
        )
        nested.nested_fit_predict = quick  # type: ignore[assignment]
        cap.nested_fit_predict = quick  # type: ignore[assignment]

    monthly = _engineer_monthly(build_monthly_panel())
    variants, y, lag1, _ = build_variants(monthly, MODEL_KEY)
    shipped = variants["shipped"]
    common = full_window_rows(monthly, shipped, y, lag1)
    y_target = y - lag1

    gldas = gldas_block(monthly.index)
    cols = list(gldas.columns)
    with_gldas = pd.concat([shipped, gldas], axis=1)
    complete = gldas.reindex(common).notna().all(axis=1)
    if not complete.all():
        raise RuntimeError(f"GLDAS block incomplete on {int((~complete).sum())} scored rows")

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    blocks = [common[idx] for _, idx in splitter.split(np.arange(len(common)))]

    print("=== GLDAS land-surface state as GRACE features, real nested pipeline, full window ===\n")
    print(f"  rows scored        {len(common)}  ({common[0]}..{common[-1]})")
    print(f"  window             {COMMON_START}..{MONTHLY_WINDOW_END_EXTENDED}")
    print(f"  folds              {len(blocks)} shared blocks, identical for every arm")
    print(f"  GLDAS block        {len(cols)} columns: {', '.join(cols)}")
    print(f"  target sd          {float(y_target.loc[common].std()):.6f}\n")

    physics = physics_check(monthly, gldas, common)
    print("  physics check (GLDAS storage change vs GRACE change, same month, scored rows):")
    for k, v in physics.items():
        print(f"    {k:26s} {v:+.4f}" if isinstance(v, float) else f"    {k:26s} {v}")
    print()

    arms = {}
    for name, frame, des in [
        ("shipped", shipped, []),
        ("shipped+gldas", with_gldas, []),
        ("shipped+gldas_ds", with_gldas, cols),
    ]:
        print(f"  running {name} ...", flush=True)
        arms[name] = run_arm(frame, y, lag1, y_target, common, blocks, des)

    a, b = arms["shipped"], arms["shipped+gldas"]
    diffs = [fb["target_r2"] - fa["target_r2"] for fa, fb in zip(a["folds"], b["folds"], strict=True)]
    wins = sum(d > 0 for d in diffs)
    t_stat = paired_t(diffs)
    d_target = b["target_r2"] - a["target_r2"]

    print(f"\n{'arm':18s} {'features':>9s} {'target R2':>11s} {'level R2':>10s} {'skill':>9s}")
    for name, r in arms.items():
        print(f"{name:18s} {r['n_features']:9d} {r['target_r2']:+11.4f} {r['level_r2']:+10.4f} {r['skill_vs_persistence']:+9.4f}")
    print("\n  per-fold target R2 difference (shipped+gldas - shipped):")
    for fa, fb, d in zip(a["folds"], b["folds"], diffs, strict=True):
        print(f"    fold {fa['fold']}  {fa['test_start']}..{fa['test_end']}  {fa['target_r2']:+8.4f} -> {fb['target_r2']:+8.4f}   {d:+8.4f}  {'win ' if d > 0 else 'loss'}")
    print(f"\n  fold-to-fold spread of the difference: sd {np.std(diffs, ddof=1):.4f}")

    print("\n=== verdict, against the rule fixed in the docstring ===")
    checks = [
        ("Δ target R² > 0", d_target > 0, f"{d_target:+.4f}"),
        (f"wins ≥ {WIN_FOLDS_REQUIRED} of {len(diffs)} folds", wins >= WIN_FOLDS_REQUIRED, f"{wins}/{len(diffs)}"),
        (f"|paired t| ≥ {T_REQUIRED}", abs(t_stat) >= T_REQUIRED, f"t = {t_stat:+.2f}"),
    ]
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label:28s} {detail}")
    real = all(ok for _, ok, _ in checks)
    verdict = "REAL" if real else "NULL"
    print(f"\n  verdict: {verdict}" + ("" if real else " — the shipped feature set stands"))
    if args.quick:
        print("\n  (--quick: NOT a result)")
        return

    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "question": "does a GLDAS land-surface block lift GRACE target R2 under the real nested tuner?",
                "rule": {"delta_target_r2_gt_0": True, "wins_required": WIN_FOLDS_REQUIRED, "t_required": T_REQUIRED, "judged_arm": "shipped+gldas"},
                "rows_scored": int(len(common)),
                "rows": [str(common[0]), str(common[-1])],
                "gldas_block": cols,
                "physics_check": physics,
                "arms": arms,
                "delta_target_r2": d_target,
                "wins": int(wins),
                "paired_t": t_stat,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"\n  wrote {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
