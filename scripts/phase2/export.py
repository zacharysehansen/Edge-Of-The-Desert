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

MODEL_IDS = ["ndvi", "grace", "wildfire_monthly", "wildlife"]


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
            "train_r2": cv.get("train_r2"),
            "train_mae": cv.get("train_mae"),
        }

        # Model-specific metrics
        if model_id == "wildlife":
            entry["loo_r2"] = cv.get("loo_r2")
            entry["loo_mae"] = cv.get("loo_mae")
            entry["spearman_corr"] = cv.get("spearman_corr")
            entry["spearman_p"] = cv.get("spearman_p")
            entry["cv_method"] = "LeaveOneOut"
        else:
            entry["cv_mean_r2"] = cv.get("cv_mean_r2")
            entry["cv_std_r2"] = cv.get("cv_std_r2")
            entry["cv_mean_mae"] = cv.get("cv_mean_mae")
            entry["cv_std_mae"] = cv.get("cv_std_mae")
            entry["cv_method"] = "TimeSeriesSplit"

        # Baseline comparison
        bl = baselines.get(model_id, {})
        lag1 = bl.get("lag1_persistence", {})
        entry["baseline_lag1_r2"] = lag1.get("mean_r2")
        entry["baseline_lag1_mae"] = lag1.get("mean_mae")

        # Improvement over baseline
        model_r2 = entry.get("cv_mean_r2") or entry.get("loo_r2")
        baseline_r2 = lag1.get("mean_r2")
        if model_r2 is not None and baseline_r2 is not None:

            if not math.isnan(baseline_r2):
                entry["improvement_r2"] = round(model_r2 - baseline_r2, 4)
                entry["beats_baseline"] = model_r2 > baseline_r2
            else:
                entry["improvement_r2"] = None
                entry["beats_baseline"] = None
        else:
            entry["improvement_r2"] = None
            entry["beats_baseline"] = None

        onnx_path = MODEL_DIR / f"{model_id}.onnx"
        entry["onnx_exported"] = onnx_path.exists()

        if cv.get("note"):
            entry["note"] = cv["note"]

        comparison[model_id] = entry

    out_path = MODEL_DIR / "model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison


def print_leaderboard(comparison: dict) -> None:
    """Print a formatted leaderboard to stdout."""
    print(f"\n{'='*72}")
    print("  Model Comparison — Phase 2 Results")
    print(f"{'='*72}\n")

    header = f"{'Model':<12} {'CV R²':>10} {'CV MAE':>10}"
    f" {'BL R²':>10} {'Δ R²':>8} {'Beat?':>6} {'ONNX':>5}"
    print(header)
    print("-" * len(header))

    for model_id, entry in comparison.items():
        if entry.get("status") == "not_trained":
            print(
                f"{model_id:<12} {'—':>10} {'—':>10} "
                f"{'—':>10} {'—':>8} {'—':>6} {'—':>5}"
            )
            continue

        # Get the appropriate R² and MAE
        if model_id == "wildlife":
            r2_str = (
                f"{entry['loo_r2']:.4f}" if entry.get("loo_r2") is not None else "—"
            )
            mae_str = (
                f"{entry['loo_mae']:.4f}" if entry.get("loo_mae") is not None else "—"
            )
        else:
            r2_str = (
                f"{entry['cv_mean_r2']:.4f}"
                if entry.get("cv_mean_r2") is not None
                else "—"
            )
            mae_str = (
                f"{entry['cv_mean_mae']:.6f}"
                if entry.get("cv_mean_mae") is not None
                else "—"
            )

        bl_r2 = entry.get("baseline_lag1_r2")
        bl_str = f"{bl_r2:.4f}" if bl_r2 is not None and not _is_nan(bl_r2) else "—"

        imp = entry.get("improvement_r2")
        imp_str = f"{imp:+.4f}" if imp is not None else "—"

        beats = entry.get("beats_baseline")
        beats_str = "YES" if beats else ("NO" if beats is False else "—")

        onnx_str = "✓" if entry.get("onnx_exported") else "✗"

        print(
            f"{model_id:<12} {r2_str:>10} {mae_str:>10} "
            f"{bl_str:>10} {imp_str:>8} {beats_str:>6} {onnx_str:>5}"
        )

    print()
    for model_id, entry in comparison.items():
        if entry.get("note"):
            print(f"  [{model_id}] {entry['note']}")

    # Wildlife-specific stats
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
