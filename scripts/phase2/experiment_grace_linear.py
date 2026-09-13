"""
experiment_grace_linear.py
--------------------------
The roadmap's step 2b (PHASE3_PLAN.md §25 → §26): is GRACE's floor the estimator?

§25 measured that a one-coefficient OLS on GLDAS's same-month storage change scores
+0.244 target R² out of fold on the five shared blocks, where the shipped 45-feature
XGBoost scores +0.021 and the same XGBoost with the GLDAS block added scores
+0.089. Three feature blocks (human, CAP, GLDAS) have been NULL under the rule
"add it to the shipped model"; the OLS says a signal is there that the shipped
model class cannot use. This script tests the model class, not a feature.

THIS SCRIPT CHANGES NOTHING THAT SHIPS. Same harness as §23-§25: GRACE's own
window (2002-10..2023-12), the same five TimeSeriesSplit blocks, training strictly
before each block, target = residual over the lag-1 anchor, level = prediction +
lag1. The shipped arm is re-run under model_grace.py's real nested tuner so the
comparison is self-contained.

Arms:
  shipped         XGBoost, 45 features, RandomizedSearchCV(60) inside each fold
  linear_physical JUDGED. Ridge on FOUR inputs, standardised, alpha chosen by an
                  inner TimeSeriesSplit(3) inside each fold:
                    gldas_tws_proxy_delta      the physics (§25 slope +0.8 m/m)
                    precipitation_mm_day        this month's rain
                    precipitation_mm_day_lag1   last month's rain (recharge lag)
                    temperature_2m_c_anomaly    ET / drought departure
                  The set is the smallest one that names a driver for each part of
                  the monthly storage change; it was chosen before this script was
                  run, and it is deliberately NOT tuned by adding or removing
                  columns afterwards.
  linear_gldas1   OLS on gldas_tws_proxy_delta alone — the §25 floor, reported
  linear_all      Ridge on the shipped 45 + the 8-column GLDAS block, same inner
                  tuning — reported: if this scores like linear_physical, the win
                  is linearity; if it scores like the XGBoost arms, the win is the
                  small feature set

THE DECISION RULE, FIXED BEFORE THE FIRST RUN
---------------------------------------------
The same three-part rule as §11.5, §23, §24 and §25, applied to linear_physical
against shipped, on target R² meaned over the five blocks:
  1. Δ target R² > 0,
  2. it wins in at least 4 of 5 folds,
  3. the paired t across folds is |t| ≥ 2.0.
REAL means GRACE's shipping model becomes a linear residual model on these four
inputs — an architecture change (model_grace.py estimator, ONNX export, a GLDAS
driver in the frontend, and the drought-index regeneration that has been waiting
for a retrain). NULL means the estimator was not the floor after all, and the
roadmap's step 5 — demote GRACE to climatology plus Layer 2 — is the honest fix.
One run. No feature-set edits after the numbers are seen.

Run:
    python -m scripts.phase2.experiment_grace_linear
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.phase2.experiment_grace_cap import full_window_rows
from scripts.phase2.experiment_grace_gldas import gldas_block
from scripts.phase2.experiment_grace_nested import nested_fit_predict, paired_t
from scripts.phase2.experiment_human_block import COMMON_START, N_SPLITS, build_variants
from scripts.phase2.features import _engineer_monthly
from scripts.phase2.merge import MONTHLY_WINDOW_END_EXTENDED, build_monthly_panel

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = ROOT / "model" / "experiment_grace_linear.json"

MODEL_KEY = "grace"
WIN_FOLDS_REQUIRED = 4
T_REQUIRED = 2.0

PHYSICAL = [
    "gldas_tws_proxy_delta",
    "precipitation_mm_day",
    "precipitation_mm_day_lag1",
    "temperature_2m_c_anomaly",
]
ALPHAS = np.logspace(-3, 3, 13)


def ridge_fit_predict(x_tr: pd.DataFrame, y_tr: pd.Series, x_te: pd.DataFrame) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(), RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=3))
    )
    return model.fit(x_tr, y_tr).predict(x_te)


def ols_fit_predict(x_tr: pd.DataFrame, y_tr: pd.Series, x_te: pd.DataFrame) -> np.ndarray:
    return LinearRegression().fit(x_tr, y_tr).predict(x_te)


def run_arm(frame, y, lag1, y_target, common, blocks, fit_predict) -> dict:
    folds = []
    for i, block in enumerate(blocks):
        train_index = frame.index[
            frame.notna().all(axis=1)
            & (frame.index < block[0])
            & y_target.reindex(frame.index).notna()
        ].intersection(common)
        if len(train_index) < 24:  # noqa: PLR2004
            continue
        prediction = fit_predict(frame.loc[train_index], y_target.loc[train_index], frame.loc[block])
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
        "features": list(frame.columns) if frame.shape[1] <= 12 else None,  # noqa: PLR2004
        "target_r2": float(np.mean([f["target_r2"] for f in folds])),
        "level_r2": float(np.mean([f["level_r2"] for f in folds])),
        "skill_vs_persistence": float(np.mean([f["level_r2"] for f in folds]) - np.mean(persistence)),
        "persistence_level_r2": float(np.mean(persistence)),
        "folds": folds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reuse-shipped", action="store_true",
        help="wiring check: take the shipped arm from model/experiment_grace_gldas.json instead of re-running it — NOT a result",
    )
    args = ap.parse_args()

    monthly = _engineer_monthly(build_monthly_panel())
    variants, y, lag1, _ = build_variants(monthly, MODEL_KEY)
    shipped = variants["shipped"]
    common = full_window_rows(monthly, shipped, y, lag1)
    y_target = y - lag1

    gldas = gldas_block(monthly.index)
    panel = pd.concat([monthly, gldas], axis=1)
    physical = panel[PHYSICAL]
    gldas1 = panel[["gldas_tws_proxy_delta"]]
    all_linear = pd.concat([shipped, gldas], axis=1)
    for name, frame in [("physical", physical), ("gldas1", gldas1), ("all", all_linear)]:
        missing = frame.reindex(common).isna().any(axis=1)
        if missing.any():
            raise RuntimeError(f"{name} frame incomplete on {int(missing.sum())} scored rows")

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    blocks = [common[idx] for _, idx in splitter.split(np.arange(len(common)))]

    print("=== GRACE model class: linear residual model vs the shipped XGBoost, full window ===\n")
    print(f"  rows scored        {len(common)}  ({common[0]}..{common[-1]})")
    print(f"  window             {COMMON_START}..{MONTHLY_WINDOW_END_EXTENDED}")
    print(f"  folds              {len(blocks)} shared blocks, identical for every arm")
    print(f"  judged arm         linear_physical = ridge on {PHYSICAL}")
    print(f"  target sd          {float(y_target.loc[common].std()):.6f}\n")

    arms = {}
    if args.reuse_shipped:
        prev = json.loads((ROOT / "model" / "experiment_grace_gldas.json").read_text())
        arms["shipped"] = prev["arms"]["shipped"]
        print("  shipped arm reused from experiment_grace_gldas.json (wiring check)")
    else:
        print("  running shipped (nested XGBoost) ...", flush=True)
        arms["shipped"] = run_arm(shipped, y, lag1, y_target, common, blocks, nested_fit_predict)
    for name, frame, fp in [
        ("linear_physical", physical, ridge_fit_predict),
        ("linear_gldas1", gldas1, ols_fit_predict),
        ("linear_all", all_linear, ridge_fit_predict),
    ]:
        print(f"  running {name} ...", flush=True)
        arms[name] = run_arm(frame, y, lag1, y_target, common, blocks, fp)

    a, b = arms["shipped"], arms["linear_physical"]
    diffs = [fb["target_r2"] - fa["target_r2"] for fa, fb in zip(a["folds"], b["folds"], strict=True)]
    wins = sum(d > 0 for d in diffs)
    t_stat = paired_t(diffs)
    d_target = b["target_r2"] - a["target_r2"]

    print(f"\n{'arm':18s} {'features':>9s} {'target R2':>11s} {'level R2':>10s} {'skill':>9s}")
    for name, r in arms.items():
        print(f"{name:18s} {r['n_features']:9d} {r['target_r2']:+11.4f} {r['level_r2']:+10.4f} {r['skill_vs_persistence']:+9.4f}")
    print(f"  (persistence level R2 on these blocks: {arms['linear_physical']['persistence_level_r2']:+.4f})")
    print("\n  per-fold target R2 (shipped -> linear_physical):")
    for fa, fb, d in zip(a["folds"], b["folds"], diffs, strict=True):
        print(f"    fold {fa['fold']}  {fa['test_start']}..{fa['test_end']}  {fa['target_r2']:+8.4f} -> {fb['target_r2']:+8.4f}   {d:+8.4f}  {'win ' if d > 0 else 'loss'}")
    print(f"\n  fold-to-fold spread of the difference: sd {np.std(diffs, ddof=1):.4f}")

    # The judged model's coefficients on the full scored window, for the record.
    fit = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=3)))
    fit.fit(physical.loc[common], y_target.loc[common])
    ridge = fit.named_steps["ridgecv"]
    coefs = dict(zip(PHYSICAL, [float(c) for c in ridge.coef_], strict=True))
    print(f"\n  linear_physical on all {len(common)} rows: alpha {ridge.alpha_:g}, standardised coefficients {json.dumps({k: round(v, 5) for k, v in coefs.items()})}")

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
    print(f"\n  verdict: {verdict}" + ("" if real else " — the shipped model class stands"))
    if args.reuse_shipped:
        print("\n  (--reuse-shipped: NOT a result)")
        return

    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "question": "does a linear residual model on four physical inputs beat the shipped XGBoost for GRACE?",
                "rule": {"delta_target_r2_gt_0": True, "wins_required": WIN_FOLDS_REQUIRED, "t_required": T_REQUIRED, "judged_arm": "linear_physical"},
                "rows_scored": int(len(common)),
                "rows": [str(common[0]), str(common[-1])],
                "physical_features": PHYSICAL,
                "alphas": [float(a) for a in ALPHAS],
                "arms": arms,
                "full_window_fit": {"alpha": float(ridge.alpha_), "standardised_coefficients": coefs},
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
