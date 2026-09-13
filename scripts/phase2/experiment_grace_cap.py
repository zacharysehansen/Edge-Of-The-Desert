"""
experiment_grace_cap.py
-----------------------
PHASE3_PLAN.md §16 item 2 / PROBLEMS.md Part 3 item 5: does a monthly CAP delivery
series — the one remaining monthly pumping proxy — give GRACE any skill?

GRACE is the only model with negative skill, and PROBLEMS.md P5 says why: the
month-to-month change in a storage integral is driven by pumping the panel does not
observe. Every proxy tried so far was a calendar template (irrigation withdrawal is
95 % seasonal; ADWR pumpage is annual). CAP deliveries are different in kind: when
CAP water is delivered, groundwater is not pumped for that demand, and deliveries
respond to shortage tiers and allocation cuts rather than to the calendar alone.
`scripts/phase1/cap_deliveries.py` acquired them, 1999-2025, from CAP's own reports.

THIS SCRIPT CHANGES NOTHING THAT SHIPS. It follows `experiment_grace_nested.py`
exactly: both arms run under `model_grace.py`'s real nested tuner
(`RandomizedSearchCV(n_iter=60, cv=TimeSeriesSplit(3))` inside every training
fold), on the rows `build_variants` guarantees are identical, with the same five
shared test blocks. The only moving part is the feature set.

Arms:
  shipped         the deployed GRACE feature set, as exported
  shipped+cap     shipped plus a CAP block: deliveries, lag1, lag3, roll3, roll6,
                  roll12 and a 12-month trailing SUM (a storage integral responds to
                  cumulative delivery, not to one month's) — each column
                  deseasonalized inside the fold, the way §10 deseasonalized the
                  human block, because CAP delivery is itself ~90 % a summer
                  template and the residual is the policy signal
  shipped+cap_raw the same block without deseasonalizing — reported, NOT judged

THE DECISION RULE, FIXED BEFORE THE FIRST RUN
---------------------------------------------
The same rule §11.5 declared for the human block, applied to the deseasonalized
arm only. Primary metric is target R² (the residual the model is trained on),
meaned over the five shared blocks. CAP is called REAL only if ALL THREE hold:
  1. Δ target R² > 0,
  2. it wins in at least 4 of 5 folds,
  3. the paired t across folds is |t| ≥ 2.0.
Anything else is NULL, the shipped feature set stands, and the series is recorded
as acquired-but-unused. The raw arm is printed for context and cannot rescue a
NULL: a template that "helps" without deseasonalizing is month_sin by another name.

A NULL here is still a result: it would mean that even the one monthly pumping
proxy that responds to policy cannot lift GRACE off its floor, which settles P5's
"weakly observed" at "unobserved at monthly grain".

THE SECOND RUN — `--window full` — DECLARED BEFORE IT WAS RUN (PHASE3_PLAN.md §24)
---------------------------------------------------------------------------------
The first run (2026-09-12, `--window common`) was NULL: +0.0665, 5 of 5 folds,
t = +1.52, with 72 % of the gain in the 28-training-row first fold. Its scored rows
stopped at 2020-12 only because the harness inherits the human-block window, where
irrigation and public supply end. GRACE's shipped features run to 2023-12 and CAP
deliveries run to 2026, so the model's own window — 2002-10..2023-12, the 204 rows
`grace_cv_results.json` reports — is available and has no such cap. This run scores
on that window: every rule, block count, tuner, feature and deseasonalizing step is
unchanged; only the window is longer, so the first fold trains on more rows.

This is the second and LAST CAP run. Its rule is the same three-part rule above,
declared here before the numbers were seen. REAL deploys the block to GRACE
(feature list, panel merge, frontend constant driver, retrain); NULL closes CAP for
GRACE permanently, and no third window, rule or block will be tried.

Run:
    python -m scripts.phase2.experiment_grace_cap                # 2002-10..2020-12, the §23 run
    python -m scripts.phase2.experiment_grace_cap --window full  # 2002-10..2023-12, the §24 run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from scripts.phase2.experiment_grace_nested import nested_fit_predict, paired_t
from scripts.phase2.experiment_human_block import (
    COMMON_END,
    COMMON_START,
    N_SPLITS,
    build_variants,
    deseasonalize,
)
from scripts.phase2.features import _MONTHLY_MODEL_SPECS, _engineer_monthly
from scripts.phase2.merge import MONTHLY_WINDOW_END_EXTENDED, build_monthly_panel

ROOT = Path(__file__).resolve().parents[2]
CAP_FILE = ROOT / "data" / "Final" / "cap_deliveries_monthly.csv"
OUTPUT_FILES = {
    "common": ROOT / "model" / "experiment_grace_cap.json",
    "full": ROOT / "model" / "experiment_grace_cap_full.json",
}

MODEL_KEY = "grace"
WIN_FOLDS_REQUIRED = 4
T_REQUIRED = 2.0
CAP_BASE = "cap_deliveries_af"


def cap_block(index: pd.Index) -> pd.DataFrame:
    """The CAP feature block, built the way features.py builds every base input."""
    cap = pd.read_csv(CAP_FILE)
    cap.index = pd.PeriodIndex(cap["year_month"], freq="M")
    s = cap[CAP_BASE].astype(float)
    block = pd.DataFrame(index=s.index)
    block[CAP_BASE] = s
    block[f"{CAP_BASE}_lag1"] = s.shift(1)
    block[f"{CAP_BASE}_lag3"] = s.shift(3)
    block[f"{CAP_BASE}_roll3"] = s.shift(1).rolling(3, min_periods=2).mean()
    block[f"{CAP_BASE}_roll6"] = s.shift(1).rolling(6, min_periods=4).mean()
    block[f"{CAP_BASE}_roll12"] = s.shift(1).rolling(12, min_periods=8).mean()
    block[f"{CAP_BASE}_sum12"] = s.rolling(12, min_periods=12).sum()
    if isinstance(index, pd.DatetimeIndex):
        block.index = block.index.to_timestamp()
    return block.reindex(index)


def full_window_rows(
    monthly: pd.DataFrame, shipped: pd.DataFrame, y: pd.Series, lag1: pd.Series
) -> pd.Index:
    """GRACE's own window: 2002-10 .. MONTHLY_WINDOW_END_EXTENDED, rows where the
    shipped features are complete and both the target and its lag-1 anchor are real
    measurements (`require_real_target`), exactly as build_variants does inside the
    shorter window. The human-inclusive completeness test is dropped because the
    human block is not an arm here."""
    window = pd.period_range(COMMON_START, MONTHLY_WINDOW_END_EXTENDED, freq="M")
    usable = monthly.index.intersection(window)
    complete = (
        shipped.loc[usable].notna().all(axis=1) & y.loc[usable].notna() & lag1.loc[usable].notna()
    )
    flag = _MONTHLY_MODEL_SPECS["grace"]["require_real_target"]
    real = monthly[flag] == 1
    complete &= (real & real.shift(1, fill_value=False)).loc[usable]
    return usable[complete.to_numpy()]


def run_arm(
    frame: pd.DataFrame,
    y: pd.Series,
    lag1: pd.Series,
    y_target: pd.Series,
    common: pd.Index,
    blocks: list[pd.Index],
    deseason_columns: list[str],
) -> dict:
    folds = []
    for i, block in enumerate(blocks):
        train_index = frame.index[
            frame.notna().all(axis=1)
            & (frame.index < block[0])
            & y_target.reindex(frame.index).notna()
        ].intersection(common)
        if len(train_index) < 24:  # noqa: PLR2004
            continue
        x_tr, x_te = frame.loc[train_index], frame.loc[block]
        if deseason_columns:
            x_tr, x_te = deseasonalize(x_tr, x_te, deseason_columns)
        prediction = nested_fit_predict(x_tr, y_target.loc[train_index], x_te)
        level = prediction + lag1.loc[block].to_numpy()
        folds.append(
            {
                "fold": i,
                "test_start": str(block[0]),
                "test_end": str(block[-1]),
                "n_train": int(len(train_index)),
                "n_test": int(len(block)),
                "target_r2": float(r2_score(y_target.loc[block], prediction)),
                "level_r2": float(r2_score(y.loc[block], level)),
            }
        )
    persistence = [float(r2_score(y.loc[b], lag1.loc[b])) for b in blocks]
    return {
        "n_features": int(frame.shape[1]),
        "deseasonalized_columns": deseason_columns,
        "target_r2": float(np.mean([f["target_r2"] for f in folds])),
        "level_r2": float(np.mean([f["level_r2"] for f in folds])),
        "skill_vs_persistence": float(
            np.mean([f["level_r2"] for f in folds]) - np.mean(persistence)
        ),
        "folds": folds,
    }


def proxy_sanity(monthly: pd.DataFrame, cap: pd.DataFrame, common: pd.Index) -> dict:
    """Before any model: is the series a pumping proxy at all? Deseasonalize both
    CAP deliveries and irrigation withdrawal on the scored rows and correlate.
    Water balance says the sign should be NEGATIVE (more CAP, less pumping) if CAP
    substitutes for irrigation pumping, and the GRACE target residual should move
    POSITIVELY with deseasonalized cumulative delivery."""
    frame = pd.DataFrame(
        {
            "cap": cap[CAP_BASE].reindex(common),
            "cap_sum12": cap[f"{CAP_BASE}_sum12"].reindex(common),
            "irr": monthly["irrigation_total_withdrawal_mgd"].reindex(common),
            "ps": monthly["public_supply_groundwater_mgd"].reindex(common),
            "grace_d": (
                monthly["grace_groundwater_anomaly"] - monthly["grace_groundwater_anomaly_lag1"]
            ).reindex(common),
        }
    ).dropna()
    months = frame.index.month
    des = frame.copy()
    for c in frame.columns:
        des[c] = frame[c] - frame[c].groupby(months).transform("mean")
    corr = des.corr()
    return {
        "n": int(len(des)),
        "seasonal_share_of_cap_variance": float(
            1 - des["cap"].var() / frame["cap"].var()
        ),
        "r_cap_vs_irrigation_deseason": float(corr.loc["cap", "irr"]),
        "r_cap_vs_public_supply_deseason": float(corr.loc["cap", "ps"]),
        "r_cap_vs_grace_change_deseason": float(corr.loc["cap", "grace_d"]),
        "r_cap_sum12_vs_grace_change_deseason": float(corr.loc["cap_sum12", "grace_d"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="wiring check only — NOT a result")
    ap.add_argument(
        "--window", choices=("common", "full"), default="common",
        help="common: the human-block window to 2020-12 (§23); full: GRACE's own window to 2023-12 (§24)",
    )
    args = ap.parse_args()
    output_file = OUTPUT_FILES[args.window]
    if args.quick:
        import scripts.phase2.experiment_grace_nested as nested  # noqa: PLC0415
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: PLC0415

        nested.nested_fit_predict = lambda x_tr, y_tr, x_te: (  # type: ignore[assignment]
            GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=0)
            .fit(x_tr, y_tr)
            .predict(x_te)
        )
        globals()["nested_fit_predict"] = nested.nested_fit_predict

    monthly = _engineer_monthly(build_monthly_panel())
    variants, y, lag1, common = build_variants(monthly, MODEL_KEY)
    shipped = variants["shipped"]
    if args.window == "full":
        common = full_window_rows(monthly, shipped, y, lag1)
    y_target = y - lag1

    cap = cap_block(monthly.index)
    cap_cols = list(cap.columns)
    with_cap = pd.concat([shipped, cap], axis=1)

    # Rows all arms can be scored on: CAP must exist there too. It starts 1999-01, so
    # this should not remove a row from §11.5's common set; assert rather than assume.
    cap_complete = cap.reindex(common).notna().all(axis=1)
    if not cap_complete.all():
        missing = common[~cap_complete]
        raise RuntimeError(f"CAP block incomplete on {len(missing)} scored rows: {missing[:5]}")

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    blocks = [common[idx] for _, idx in splitter.split(np.arange(len(common)))]

    print(f"=== CAP deliveries as a GRACE feature, under model_grace.py's real nested pipeline [{args.window} window] ===\n")
    print(f"  rows scored        {len(common)}  ({common[0]}..{common[-1]})")
    print(f"  window             {COMMON_START}..{COMMON_END if args.window == 'common' else MONTHLY_WINDOW_END_EXTENDED}")
    print(f"  folds              {len(blocks)} shared blocks, identical for every arm")
    print(f"  CAP block          {len(cap_cols)} columns: {', '.join(cap_cols)}")
    print(f"  target sd          {float(y_target.loc[common].std()):.6f}\n")

    sanity = proxy_sanity(monthly, cap, common)
    print("  proxy sanity (deseasonalized, scored rows):")
    for k, v in sanity.items():
        print(f"    {k:42s} {v:+.4f}" if isinstance(v, float) else f"    {k:42s} {v}")
    print()

    arms = {}
    for name, frame, des in [
        ("shipped", shipped, []),
        ("shipped+cap", with_cap, cap_cols),
        ("shipped+cap_raw", with_cap, []),
    ]:
        print(f"  running {name} ...", flush=True)
        arms[name] = run_arm(frame, y, lag1, y_target, common, blocks, des)

    a, b = arms["shipped"], arms["shipped+cap"]
    diffs = [
        fb["target_r2"] - fa["target_r2"] for fa, fb in zip(a["folds"], b["folds"], strict=True)
    ]
    wins = sum(d > 0 for d in diffs)
    t_stat = paired_t(diffs)
    d_target = b["target_r2"] - a["target_r2"]

    print(f"\n{'arm':18s} {'features':>9s} {'target R2':>11s} {'level R2':>10s} {'skill':>9s}")
    for name, r in arms.items():
        print(
            f"{name:18s} {r['n_features']:9d} {r['target_r2']:+11.4f} "
            f"{r['level_r2']:+10.4f} {r['skill_vs_persistence']:+9.4f}"
        )
    print("\n  per-fold target R2 difference (shipped+cap - shipped):")
    for fa, fb, d in zip(a["folds"], b["folds"], diffs, strict=True):
        print(
            f"    fold {fa['fold']}  {fa['test_start']}..{fa['test_end']}  "
            f"{fa['target_r2']:+8.4f} -> {fb['target_r2']:+8.4f}   {d:+8.4f}  "
            f"{'win ' if d > 0 else 'loss'}"
        )
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

    output_file.write_text(
        json.dumps(
            {
                "question": "does a monthly CAP delivery block lift GRACE target R2 under the real nested tuner?",
                "window": args.window,
                "rule": {
                    "delta_target_r2_gt_0": True,
                    "wins_required": WIN_FOLDS_REQUIRED,
                    "t_required": T_REQUIRED,
                    "judged_arm": "shipped+cap",
                },
                "rows_scored": int(len(common)),
                "rows": [str(common[0]), str(common[-1])],
                "cap_block": cap_cols,
                "proxy_sanity": sanity,
                "arms": arms,
                "delta_target_r2": d_target,
                "wins": int(wins),
                "paired_t": t_stat,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"\n  wrote {output_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
