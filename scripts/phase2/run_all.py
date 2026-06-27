"""
run_all.py
----------
Top-level Phase 2 runner. Executes the full pipeline:

  1. merge.py      — build monthly + annual panels
  2. features.py   — engineer features per model
  3. baselines.py  — compute lag1/roll3 baselines
  4. model_*.py    — train, evaluate, export each model
  5. export.py     — build comparison summary

Usage
-----
    python -m scripts.phase2.run_all
    python -m scripts.phase2.run_all --models ndvi grace
    python -m scripts.phase2.run_all --skip-models   # only merge/features/baselines/export
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 pipeline runner")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["ndvi", "grace", "groundwater", "surface_water", "wildfire", "wildfire_monthly", "wildlife"],
        help="Which models to train (default: all seven)",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model training; only run merge/features/baselines/export",
    )
    args = parser.parse_args()

    t0 = time.time()

    # --- Step 1: Merge ---
    print("\n[1/5] Building panels (merge.py)...")
    from scripts.phase2.merge import run as merge_run
    monthly, annual = merge_run()
    print(f"       Monthly: {monthly.shape}  |  Annual: {annual.shape}")

    # --- Step 2: Features ---
    print("\n[2/5] Engineering features (features.py)...")
    from scripts.phase2.features import build_all
    datasets = build_all(monthly_raw=monthly, annual_raw=annual)
    for name, (X, y) in datasets.items():
        print(f"       {name:<10} X={X.shape}  y={y.shape}")

    # --- Step 3: Baselines ---
    print("\n[3/5] Computing baselines (baselines.py)...")
    from scripts.phase2.baselines import compute_baselines
    baselines = compute_baselines(datasets)
    for model_id, v in baselines.items():
        l1 = v["lag1_persistence"]
        r2_str = f"{l1['mean_r2']:.4f}" if not _is_nan(l1.get("mean_r2")) else "n/a"
        print(f"       {model_id:<10} lag1 R² = {r2_str}")

    # --- Step 4: Model training ---
    if not args.skip_models:
        print("\n[4/5] Training models...")
        model_funcs = {}

        if "ndvi" in args.models:
            from scripts.phase2.model_ndvi import train_and_evaluate as train_ndvi
            model_funcs["ndvi"] = train_ndvi
        if "grace" in args.models:
            from scripts.phase2.model_grace import train_and_evaluate as train_grace
            model_funcs["grace"] = train_grace
        if "groundwater" in args.models:
            from scripts.phase2.model_groundwater import train_and_evaluate as train_groundwater
            model_funcs["groundwater"] = train_groundwater
        if "surface_water" in args.models:
            from scripts.phase2.model_surface_water import train_and_evaluate as train_surface_water
            model_funcs["surface_water"] = train_surface_water
        if "wildfire" in args.models:
            from scripts.phase2.model_wildfire import train_and_evaluate as train_wildfire
            model_funcs["wildfire"] = train_wildfire
        if "wildfire_monthly" in args.models:
            from scripts.phase2.model_wildfire_monthly import train_and_evaluate as train_wildfire_monthly
            model_funcs["wildfire_monthly"] = train_wildfire_monthly
        if "wildlife" in args.models:
            from scripts.phase2.model_wildlife import train_and_evaluate as train_wildlife
            model_funcs["wildlife"] = train_wildlife

        for model_id, func in model_funcs.items():
            t1 = time.time()
            print(f"\n       --- {model_id} ---")
            results = func()
            elapsed = time.time() - t1

            if model_id == "wildlife":
                r2_str = f"{results.get('loo_r2', 0):.4f}"
            else:
                r2_str = f"{results.get('cv_mean_r2', 0):.4f}"
            print(f"       {model_id}: R² = {r2_str}  ({elapsed:.1f}s)")
    else:
        print("\n[4/5] Skipped (--skip-models)")

    # --- Step 5: Export comparison ---
    print("\n[5/5] Building comparison (export.py)...")
    from scripts.phase2.export import build_comparison, print_leaderboard
    comparison = build_comparison()
    print_leaderboard(comparison)

    elapsed_total = time.time() - t0
    print(f"\n  Total elapsed: {elapsed_total:.1f}s")


def _is_nan(x) -> bool:
    import math
    try:
        return x is None or math.isnan(x)
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
