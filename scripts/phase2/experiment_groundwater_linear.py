"""
experiment_groundwater_linear.py
--------------------------------
Roadmap step 2 for the groundwater model (PHASE3_PLAN.md §28): is its floor the
estimator, the way GRACE's was (§26)?

The shipped groundwater model is an in-fold competition of five candidates
(XGBoost x2, a blend, ridge, elastic net) over a 63-column catalogue with in-fold
feature selection, and it has sat at zero skill through five re-runs. GRACE's
lesson was that a large catalogue on ~200 rows lets a flexible model fit the wrong
thing, and that four physically chosen inputs beat it ten-fold. This runs the same
test on groundwater, and this time the physical set includes the one human input
the panel is known to see: irrigation withdrawal, correctly signed and significant
at ~2 % of the variance of monthly change (PROBLEMS.md P5).

THIS SCRIPT CHANGES NOTHING THAT SHIPS. Groundwater's own window (2002-10..2020-12),
the five shared TimeSeriesSplit blocks build_variants provides, training strictly
before each, residual over the lag-1 anchor. The shipped arm is re-run with
model_groundwater.py's real _fit_predict (the in-fold competition), so the
comparison is self-contained.

Arms:
  shipped          the deployed competition on its 16 exported features
  linear_physical  JUDGED. Ridge on FIVE inputs, standardised, alpha by an inner
                   TimeSeriesSplit(3) inside each fold:
                     irrigation_total_withdrawal_mgd  DESEASONALIZED IN-FOLD — the
                                                      pumping proxy; 95 % of it is a
                                                      calendar template and the
                                                      anomaly is the signal (§10)
                     gldas_tws_proxy_delta            the land-surface storage change
                     precipitation_mm_day             this month's rain
                     precipitation_mm_day_lag1        last month's rain
                     temperature_2m_c_anomaly         ET / drought departure
  linear_climate   the same without irrigation — reported: says whether the human
                   input is doing anything a climate-only model cannot
  linear_gldas1    OLS on gldas_tws_proxy_delta alone — reported

A deployment note, declared now so it cannot be argued later: if REAL, the
irrigation column ships PINNED at climatology in the frontend (climateOnly() in
state.js zeroes every human delta before Layer 1 runs), exactly as mead_total_release
is pinned today. Layer 2 keeps the whole human response; the learned irrigation
coefficient only improves the forecast, never moves a slider. The no-double-count
gate enforces this and would fail otherwise.

THE DECISION RULE, FIXED BEFORE THE FIRST RUN
---------------------------------------------
The rule of §11.5 and §23-§26, unchanged, applied to linear_physical against
shipped on target R² meaned over the five blocks:
  1. Δ target R² > 0,
  2. it wins in at least 4 of 5 folds,
  3. the paired t across folds is |t| ≥ 2.0.
REAL means groundwater ships as this linear model (the same deployment path §27
built for GRACE). NULL means the estimator was not the floor and the index itself
is — see groundwater_diagnosis.py — and the roadmap moves to the per-well panel
and the ADWR acquisition. One run; no feature-set edits after the numbers are seen.

THE SECOND RUN — ON THE COCHISE TARGET — DECLARED BEFORE IT WAS RUN (§29)
-------------------------------------------------------------------------
The first run (blended index) was NULL: +0.0175, 4/5, t = 0.83, and §28 showed the
target was the floor. Option A made `depth_to_water_anomaly_ft` the Cochise
sub-index (groundwater_levels.py). This run is the same script on the new target:
same arms, same blocks, same three-part rule, judged arm still linear_physical.

Two things are declared here in addition, so the deployment cannot be argued
after the numbers are seen:
  - TIE-BREAK. If linear_physical is REAL and linear_climate's target R² is within
    0.03 of it, linear_climate SHIPS, not linear_physical: a climate-only Layer 1 is
    the architecture (§13), and a pinned irrigation column buys nothing a slider can
    reach. If linear_physical beats linear_climate by more than 0.03, linear_physical
    ships with irrigation pinned at climatology, as the note above describes.
  - If NULL, the shipped in-fold competition is retrained on the Cochise target and
    ships; no third arm is added.

Run:
    python -m scripts.phase2.experiment_groundwater_linear
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

from scripts.phase2.experiment_grace_nested import paired_t
from scripts.phase2.experiment_human_block import (
    COMMON_END,
    COMMON_START,
    N_SPLITS,
    build_variants,
    deseasonalize,
)
from scripts.phase2.features import _engineer_monthly
from scripts.phase2.merge import build_monthly_panel
from scripts.phase2.model_groundwater import _fit_predict as shipped_fit_predict

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = ROOT / "model" / "experiment_groundwater_linear.json"
OUTPUT_FILE_COCHISE = ROOT / "model" / "experiment_groundwater_linear_cochise.json"

MODEL_KEY = "groundwater"
WIN_FOLDS_REQUIRED = 4
T_REQUIRED = 2.0

PHYSICAL = [
    "irrigation_total_withdrawal_mgd",
    "gldas_tws_proxy_delta",
    "precipitation_mm_day",
    "precipitation_mm_day_lag1",
    "temperature_2m_c_anomaly",
]
DESEASON = ["irrigation_total_withdrawal_mgd"]
CLIMATE = [c for c in PHYSICAL if c not in DESEASON]
ALPHAS = np.logspace(-3, 3, 13)


def ridge_fit_predict(x_tr, y_tr, x_te):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=3))).fit(x_tr, y_tr).predict(x_te)


def ols_fit_predict(x_tr, y_tr, x_te):
    return LinearRegression().fit(x_tr, y_tr).predict(x_te)


def run_arm(frame, y, lag1, y_target, common, blocks, fit_predict, deseason_columns) -> dict:
    folds = []
    for i, block in enumerate(blocks):
        train_index = frame.index[
            frame.notna().all(axis=1) & (frame.index < block[0]) & y_target.reindex(frame.index).notna()
        ].intersection(common)
        if len(train_index) < 24:  # noqa: PLR2004
            continue
        x_tr, x_te = frame.loc[train_index], frame.loc[block]
        if deseason_columns:
            x_tr, x_te = deseasonalize(x_tr, x_te, deseason_columns)
        prediction = fit_predict(x_tr, y_target.loc[train_index], x_te)
        level = prediction + lag1.loc[block].to_numpy()
        folds.append(
            {
                "fold": i, "test_start": str(block[0]), "test_end": str(block[-1]),
                "n_train": int(len(train_index)), "n_test": int(len(block)),
                "target_r2": float(r2_score(y_target.loc[block], prediction)),
                "level_r2": float(r2_score(y.loc[block], level)),
            }
        )
    persistence = [float(r2_score(y.loc[b], lag1.loc[b])) for b in blocks]
    return {
        "n_features": int(frame.shape[1]),
        "features": list(frame.columns) if frame.shape[1] <= 12 else None,  # noqa: PLR2004
        "deseasonalized_columns": deseason_columns,
        "target_r2": float(np.mean([f["target_r2"] for f in folds])),
        "level_r2": float(np.mean([f["level_r2"] for f in folds])),
        "skill_vs_persistence": float(np.mean([f["level_r2"] for f in folds]) - np.mean(persistence)),
        "persistence_level_r2": float(np.mean(persistence)),
        "folds": folds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-shipped", action="store_true", help="wiring check: linear arms only — NOT a result")
    args = ap.parse_args()

    monthly = _engineer_monthly(build_monthly_panel())
    variants, y, lag1, common = build_variants(monthly, MODEL_KEY)
    shipped = variants["shipped"]
    y_target = y - lag1
    physical, climate, gldas1 = monthly[PHYSICAL], monthly[CLIMATE], monthly[["gldas_tws_proxy_delta"]]
    for name, frame in (("physical", physical), ("shipped", shipped)):
        miss = frame.reindex(common).isna().any(axis=1)
        if miss.any():
            raise RuntimeError(f"{name} frame incomplete on {int(miss.sum())} scored rows")

    blocks = [common[idx] for _, idx in TimeSeriesSplit(n_splits=N_SPLITS).split(np.arange(len(common)))]

    print("=== groundwater model class: linear residual model vs the shipped competition ===\n")
    print(f"  rows scored        {len(common)}  ({common[0]}..{common[-1]})")
    print(f"  window             {COMMON_START}..{COMMON_END}")
    print(f"  folds              {len(blocks)} shared blocks, identical for every arm")
    print(f"  judged arm         linear_physical = ridge on {PHYSICAL}, {DESEASON} deseasonalized in-fold")
    print(f"  target sd          {float(y_target.loc[common].std()):.4f} ft\n")

    arms = {}
    if args.skip_shipped:
        print("  shipped arm skipped (wiring check)")
    else:
        print("  running shipped (in-fold competition) ...", flush=True)
        arms["shipped"] = run_arm(shipped, y, lag1, y_target, common, blocks, shipped_fit_predict, [])
    for name, frame, fp, des in [
        ("linear_physical", physical, ridge_fit_predict, DESEASON),
        ("linear_climate", climate, ridge_fit_predict, []),
        ("linear_gldas1", gldas1, ols_fit_predict, []),
    ]:
        print(f"  running {name} ...", flush=True)
        arms[name] = run_arm(frame, y, lag1, y_target, common, blocks, fp, des)

    print(f"\n{'arm':18s} {'features':>9s} {'target R2':>11s} {'level R2':>10s} {'skill':>9s}")
    for name, r in arms.items():
        print(f"{name:18s} {r['n_features']:9d} {r['target_r2']:+11.4f} {r['level_r2']:+10.4f} {r['skill_vs_persistence']:+9.4f}")
    print(f"  (persistence level R2 on these blocks: {arms['linear_physical']['persistence_level_r2']:+.4f})")

    if args.skip_shipped:
        print("\n  (--skip-shipped: NOT a result)")
        return

    a, b = arms["shipped"], arms["linear_physical"]
    diffs = [fb["target_r2"] - fa["target_r2"] for fa, fb in zip(a["folds"], b["folds"], strict=True)]
    wins = sum(d > 0 for d in diffs)
    t_stat = paired_t(diffs)
    d_target = b["target_r2"] - a["target_r2"]
    print("\n  per-fold target R2 (shipped -> linear_physical):")
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
    print(f"\n  verdict: {verdict}" + ("" if real else " — the shipped model class stands"))

    out_path = OUTPUT_FILE_COCHISE if "depth_to_water_anomaly_ft_allwells" in monthly.columns else OUTPUT_FILE
    out_path.write_text(json.dumps({
        "question": "does a linear residual model on five physical inputs beat the shipped in-fold competition for groundwater?",
        "rule": {"delta_target_r2_gt_0": True, "wins_required": WIN_FOLDS_REQUIRED, "t_required": T_REQUIRED, "judged_arm": "linear_physical"},
        "rows_scored": int(len(common)), "rows": [str(common[0]), str(common[-1])],
        "physical_features": PHYSICAL, "deseasonalized": DESEASON,
        "arms": arms, "delta_target_r2": d_target, "wins": int(wins), "paired_t": t_stat, "verdict": verdict,
    }, indent=2))
    print(f"\n  wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
