"""
export.py
---------
Collect all model artifacts and produce a unified comparison summary.

Reads each *_cv_results.json and baselines.json, produces:
  - model/model_comparison.json
  - Printed leaderboard to stdout
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"

# Every trained model must appear here. groundwater and surface_water were previously
# missing, so their rows in the report were hand-written with a blank baseline column —
# which is how a groundwater model that loses to persistence went unnoticed.
MODEL_IDS = [
    "ndvi",
    "grace",
    "groundwater",
    "surface_water",
    "wildfire_monthly",
    "wildlife",
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def build_comparison() -> dict:
    """
    Build model_comparison.json from individual CV results and baselines.
    """
    baselines = load_json(MODEL_DIR / "baselines.json") or {}
    comparison = {}

    for model_id in MODEL_IDS:
        cv_path = MODEL_DIR / f"{model_id}_cv_results.json"
        cv = load_json(cv_path)
        if cv is None:
            comparison[model_id] = {"status": "not_trained"}
            continue

        entry = {
            "status": "trained",
            "formulation": cv.get("formulation"),
            "window": f"{cv.get('window_start')} → {cv.get('window_end')}",
            "n_rows": cv.get("n_rows"),
            "n_features": cv.get("n_features"),
            "cv_method": cv.get("cv_method"),
            "train_r2": cv.get("train_r2"),
            "train_mae": cv.get("train_mae"),
            # R² on the target the model was actually trained on. For a residual model
            # this is far lower than the level R², and it is the number saying
            # whether the model learned anything.
            "cv_target_r2": cv.get("cv_target_r2"),
        }

        # Model-specific metrics
        if model_id == "wildlife":
            entry["loo_r2"] = cv.get("loo_r2")
            entry["loo_mae"] = cv.get("loo_mae")
            entry["spearman_corr"] = cv.get("spearman_corr")
            entry["spearman_p"] = cv.get("spearman_p")
        else:
            entry["cv_mean_r2"] = cv.get("cv_mean_r2")
            entry["cv_std_r2"] = cv.get("cv_std_r2")
            entry["cv_mean_mae"] = cv.get("cv_mean_mae")
            entry["cv_std_mae"] = cv.get("cv_std_mae")

        # Baseline comparison. Prefer the baseline the model script measured on its own
        # folds; fall back to baselines.json only if the model didn't record one.
        bl = baselines.get(model_id, {})
        lag1 = bl.get("lag1_persistence", {})
        baseline_r2 = cv.get("baseline_lag1_r2")
        if baseline_r2 is None:
            baseline_r2 = lag1.get("mean_r2")
        entry["baseline_lag1_r2"] = baseline_r2
        entry["baseline_lag1_mae"] = lag1.get("mean_mae")

        model_r2 = entry.get("cv_mean_r2")
        if model_r2 is None:
            model_r2 = entry.get("loo_r2")

        skill = cv.get("skill_r2")
        if (
            skill is None
            and model_r2 is not None
            and baseline_r2 is not None
            and not math.isnan(baseline_r2)
        ):
            skill = model_r2 - baseline_r2

        if skill is not None and not _is_nan(skill):
            entry["skill_r2"] = round(skill, 4)
            entry["beats_persistence"] = skill > 0
        else:
            entry["skill_r2"] = None
            entry["beats_persistence"] = None

        # A model has to clear TWO baselines, and neither alone is sufficient:
        #   - the mean (R² > 0). R² is already defined against the test-fold mean, so
        #     R² <= 0 means the model is worse than a flat line.
        #   - persistence (skill > 0), which is the hard one for autoregressive targets.
        # Groundwater is why both are needed: it posts a healthy level R² of ~0.54 and
        # still loses to persistence, because the lag1 anchor is doing all the work.
        entry["beats_mean"] = None if model_r2 is None else model_r2 > 0
        entry["beats_baseline"] = bool(
            entry.get("beats_persistence") and entry.get("beats_mean")
        )

        onnx_path = MODEL_DIR / f"{model_id}.onnx"
        entry["onnx_exported"] = onnx_path.exists()

        if cv.get("note"):
            entry["note"] = cv["note"]

        comparison[model_id] = entry

    out_path = MODEL_DIR / "model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison


def _fmt(value: float | None, spec: str = "+.4f") -> str:
    if value is None or _is_nan(value):
        return "—"
    return format(value, spec)


def print_leaderboard(comparison: dict) -> None:
    """
    Print the leaderboard, ranked by skill over persistence.

    Skill, not R², is the headline. For an autoregressive target, a high R² can mean
    nothing more than "last month's value is a good guess" — which is free. Skill is
    what the model adds on top of that, and it is the only column that separates a
    working model from an expensive copy of `lag1`.
    """
    print(f"\n{'='*86}")
    print("  Model Comparison — Phase 2 Results  (ranked by skill over persistence)")
    print(f"{'='*86}\n")

    header = (
        f"{'Model':<17} {'VERDICT':>9} {'SKILL':>8} {'R² lvl':>8} "
        f"{'persist':>8} "
        f"{'R² tgt':>8} {'ONNX':>5}"
    )
    print(header)
    print("-" * len(header))

    def sort_key(item: tuple) -> float:
        skill = item[1].get("skill_r2")
        return skill if skill is not None and not _is_nan(skill) else -math.inf

    failures = []
    for model_id, entry in sorted(comparison.items(), key=sort_key, reverse=True):
        if entry.get("status") == "not_trained":
            dash = "—"
            print(
                f"{model_id:<17} {dash:>9} {dash:>8} {dash:>8} "
                f"{dash:>8} {dash:>8} {dash:>5}"
            )
            continue

        level_r2 = entry.get("cv_mean_r2")
        if level_r2 is None:
            level_r2 = entry.get("loo_r2")

        if entry.get("beats_baseline"):
            verdict = "OK"
        elif entry.get("beats_mean") is False:
            verdict = "< MEAN"
            failures.append(f"{model_id} (worse than a flat line)")
        else:
            verdict = "< PERSIST"
            failures.append(f"{model_id} (no better than copying last month)")

        onnx_str = "✓" if entry.get("onnx_exported") else "✗"
        print(
            f"{model_id:<17} {verdict:>9} {_fmt(entry.get('skill_r2')):>8} "
            f"{_fmt(level_r2, '.4f'):>8} "
            f"{_fmt(entry.get('baseline_lag1_r2'), '.4f'):>8} "
            f"{_fmt(entry.get('cv_target_r2'), '.4f'):>8} {onnx_str:>5}"
        )

    print()
    print("  A model must clear BOTH baselines to read OK:")
    print("    R² lvl  > 0  — beats predicting the test-fold mean (a flat line).")
    print("    SKILL   > 0  — beats lag1 persistence (R² lvl minus persist).")
    print("  Neither alone is enough. Groundwater posts a healthy R² lvl (~0.54) and")
    print("  still loses to persistence: the lag1 anchor is doing all of the work.")
    print("  R² tgt = R² on the target actually trained on (the residual, for a")
    print("  residual model). The level R² beside it is inflated by the lag1 anchor.")

    if failures:
        print("\n  *** NOT PREDICTIVE ***")
        for f in failures:
            print(f"      {f}")

    wl = comparison.get("wildlife", {})
    if wl.get("spearman_corr") is not None:
        print(
            f"\n  [wildlife] Spearman ρ = {wl['spearman_corr']:.4f} "
            f"(p = {wl['spearman_p']:.4f})"
        )

    print(f"\n  Artifacts directory: {MODEL_DIR}/")


def _is_nan(x: any) -> bool:
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False


def main() -> None:
    comparison = build_comparison()
    print_leaderboard(comparison)


if __name__ == "__main__":
    main()
