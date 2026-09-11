"""
experiment_grace_nested.py
--------------------------
PHASE3_PLAN.md §11.5 — settle the one question the §10 panel left open.

§10 measured that GRACE scored **better** with the deseasonalized human block than
with its shipped feature set, on both the residual it is trained on
(−0.0119 → +0.0539) and the reconstructed level (+0.4121 → +0.4462), on identical
test rows. If that survives, §8's "no model is retrained for deployment" is leaving
something on the table for the one model that has no skill.

It cannot be claimed from §10, for a reason §10 states about itself: every variant
there used **one fixed XGBoost config** (`XGB_KWARGS`: 300 trees, depth 3, lr 0.05)
rather than each model's own nested tuning. That makes the comparison internally
valid — the feature set is the only thing moving — but it means the "shipped" column
is NOT the shipped model's real score. `grace_cv_results.json` puts the deployed
model's target R² at **+0.0302** under `model_grace.py`'s real nested search, where
§10's fixed config scored the same feature set at **−0.0119**.

The hypothesis this script was written to test was therefore that the advantage is an
artifact of the undertuned shipped arm, and would shrink once both arms got the real
search. **IT DOES NOT, and that prediction is recorded here because the run refuted
it.** Real tuning lifts BOTH arms by a similar amount (+0.0513 shipped, +0.0420 human
block), so the gap barely moves: +0.0658 → +0.0566. The verdict below is still a NULL,
but it is a null on *significance*, not on tuning — see §11.5 in PHASE3_PLAN.md.

So this script re-runs BOTH arms under `model_grace.py`'s actual pipeline —
`RandomizedSearchCV(n_iter=60, cv=TimeSeriesSplit(3))` tuned inside every training
fold — on the rows `build_variants` already guarantees are identical.

What it deliberately does NOT change, so that the feature set stays the only moving
part: the test blocks, the residual-over-lag1 formulation, the fold-local
deseasonalization, and GRACE's `require_real_target` filter (both the target and its
anchor must be measurements — PHASE2_REPORT.md found 33 zero-filled months inside
the window across 23 blocks).

THE DECISION RULE, FIXED BEFORE THE FIRST RUN
---------------------------------------------
This is a model that PROBLEMS.md Part 3 says in bold: **do not tune GRACE.** The
point of a pre-declared rule is that this probe cannot become a search. Primary
metric is **target R²** — the residual the model is actually trained on — meaned
over the five shared blocks. The human block is called REAL only if ALL THREE hold:

  1. Δ target R² > 0,
  2. it wins in **at least 4 of 5** folds,
  3. the paired t across folds is **|t| ≥ 2.0**.

Anything else is a NULL and the shipped feature set stands. Rule 3 is the one that
matters and it is the standard PROBLEMS.md item #3 already used to call the
irrigation trade null (`+0.0098 target R², 2/5 folds, t=0.57` — "≈0.1x the
fold-to-fold spread"). GRACE's fold spread is enormous (`cv_std_r2` 0.60 against a
`cv_mean_r2` of 0.33), which is exactly why a mean difference is not evidence on its
own.

A NULL here is a useful result, not a wasted run: it closes §11.5 and converts
§8's "no model is retrained" from an assumption into a measurement.

Run:
    python -m scripts.phase2.experiment_grace_nested
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from scripts.phase2.features import _engineer_monthly
from scripts.phase2.experiment_human_block import (
    COMMON_END,
    COMMON_START,
    N_SPLITS,
    build_variants,
    deseasonalize,
    is_human,
)
from scripts.phase2.merge import build_monthly_panel
from scripts.phase2.model_grace import _make_search

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = ROOT / "model" / "experiment_grace_nested.json"

MODEL_KEY = "grace"
WIN_FOLDS_REQUIRED = 4
T_REQUIRED = 2.0


def nested_fit_predict(
    x_tr: pd.DataFrame, y_tr: pd.Series, x_te: pd.DataFrame
) -> np.ndarray:
    """model_grace.py's real tuner, fit on the training fold only."""
    return _make_search().fit(x_tr, y_tr).best_estimator_.predict(x_te)


def run_arm(
    frame: pd.DataFrame,
    y: pd.Series,
    lag1: pd.Series,
    y_target: pd.Series,
    common: pd.Index,
    blocks: list[pd.Index],
    deseason: bool,
) -> dict:
    """One feature set, scored on the shared blocks with the nested tuner."""
    human_columns = [c for c in frame.columns if is_human(c)]
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
        if deseason:
            x_tr, x_te = deseasonalize(x_tr, x_te, human_columns)

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
        "n_human_features": int(sum(is_human(c) for c in frame.columns)),
        "deseasonalized": deseason,
        "target_r2": float(np.mean([f["target_r2"] for f in folds])),
        "level_r2": float(np.mean([f["level_r2"] for f in folds])),
        "skill_vs_persistence": float(
            np.mean([f["level_r2"] for f in folds]) - np.mean(persistence)
        ),
        "folds": folds,
    }


def paired_t(diffs: list[float]) -> float:
    """Paired t across folds. 0.0 when the differences carry no spread."""
    d = np.asarray(diffs, dtype=float)
    if len(d) < 2 or d.std(ddof=1) == 0:  # noqa: PLR2004
        return 0.0
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quick",
        action="store_true",
        help="smaller search, for wiring checks only — NOT a result",
    )
    args = ap.parse_args()

    monthly = _engineer_monthly(build_monthly_panel())
    variants, y, lag1, common = build_variants(monthly, MODEL_KEY)
    y_target = y - lag1

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    blocks = [common[idx] for _, idx in splitter.split(np.arange(len(common)))]

    print(f"=== §11.5: GRACE under model_grace.py's real nested pipeline ===\n")
    print(f"  rows scored        {len(common)}  ({common[0]}..{common[-1]})")
    print(f"  window             {COMMON_START}..{COMMON_END} (where the human block exists)")
    print(f"  folds              {len(blocks)} shared blocks, identical for both arms")
    print(f"  tuner              RandomizedSearchCV(n_iter=60, inner TimeSeriesSplit(3))")
    print(f"  target sd          {float(y_target.loc[common].std()):.6f}\n")

    arms = {}
    for name, source, deseason in [
        ("shipped", "shipped", False),
        ("with_human_deseason", "with_human", True),
    ]:
        print(f"  running {name} ...", flush=True)
        arms[name] = run_arm(
            variants[source], y, lag1, y_target, common, blocks, deseason
        )

    a, b = arms["shipped"], arms["with_human_deseason"]
    diffs = [
        fb["target_r2"] - fa["target_r2"]
        for fa, fb in zip(a["folds"], b["folds"], strict=True)
    ]
    wins = sum(d > 0 for d in diffs)
    t_stat = paired_t(diffs)
    d_target = b["target_r2"] - a["target_r2"]

    print(f"\n{'arm':26s} {'features':>9s} {'human':>6s} {'target R2':>11s} {'level R2':>10s} {'skill':>9s}")
    for name in ("shipped", "with_human_deseason"):
        r = arms[name]
        print(
            f"{name:26s} {r['n_features']:9d} {r['n_human_features']:6d} "
            f"{r['target_r2']:+11.4f} {r['level_r2']:+10.4f} {r['skill_vs_persistence']:+9.4f}"
        )

    print(f"\n  per-fold target R2 difference (human block - shipped):")
    for fa, fb, d in zip(a["folds"], b["folds"], diffs, strict=True):
        mark = "win " if d > 0 else "loss"
        print(
            f"    fold {fa['fold']}  {fa['test_start']}..{fa['test_end']}  "
            f"{fa['target_r2']:+8.4f} -> {fb['target_r2']:+8.4f}   {d:+8.4f}  {mark}"
        )
    print(f"\n  fold-to-fold spread of the difference: sd {np.std(diffs, ddof=1):.4f}")

    print(f"\n=== verdict, against the rule fixed in the docstring ===")
    checks = [
        ("Δ target R² > 0", d_target > 0, f"{d_target:+.4f}"),
        (f"wins ≥ {WIN_FOLDS_REQUIRED} of {len(diffs)} folds", wins >= WIN_FOLDS_REQUIRED, f"{wins}/{len(diffs)}"),
        (f"|paired t| ≥ {T_REQUIRED}", abs(t_stat) >= T_REQUIRED, f"t = {t_stat:+.2f}"),
    ]
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label:28s} {detail}")

    real = all(ok for _, ok, _ in checks)
    verdict = "REAL" if real else "NULL"
    print(f"\n  VERDICT: {verdict}")
    if real:
        print(
            "  The human block survives the real tuner. §8's 'no model is retrained'\n"
            "  is leaving skill on the table for GRACE; retraining is now a live option."
        )
    else:
        print(
            "  The shipped feature set stands and nothing is retrained. Note WHICH\n"
            "  criterion failed before concluding anything about tuning: if only the\n"
            "  t-bar failed, the advantage is real in the mean and indistinguishable\n"
            "  from fold noise, which is a different finding from 'tuning closed it'.\n"
            "  Read the per-fold table above — a difference carried by one fold is a\n"
            "  small-sample effect, not a feature-set effect."
        )

    payload = {
        "question": "PHASE3_PLAN.md §11.5",
        "tuner": "RandomizedSearchCV(n_iter=60, inner TimeSeriesSplit(3)) per fold",
        "window": f"{COMMON_START}..{COMMON_END}",
        "n_rows_scored": int(len(common)),
        "test_span": f"{common[0]}..{common[-1]}",
        "decision_rule": {
            "primary_metric": "target_r2",
            "delta_positive": True,
            "min_fold_wins": WIN_FOLDS_REQUIRED,
            "min_abs_paired_t": T_REQUIRED,
        },
        "arms": arms,
        "comparison": {
            "delta_target_r2": d_target,
            "delta_level_r2": b["level_r2"] - a["level_r2"],
            "per_fold_delta": diffs,
            "fold_wins": int(wins),
            "paired_t": t_stat,
            "verdict": verdict,
        },
        "fixed_config_reference": {
            "note": "model/experiment_human_block.json, §10's one-config panel",
            "shipped_target_r2": -0.011853144723506782,
            "with_human_deseason_target_r2": 0.05394495516770885,
        },
        "deployed_reference": {
            "note": "model/grace_cv_results.json — full 2002-10..2023-12 window, 204 rows",
            "cv_target_r2": 0.030237012250611463,
            "skill_r2": -0.03494308191758516,
        },
    }
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
