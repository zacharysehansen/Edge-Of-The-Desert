"""
experiment_monotone.py
----------------------
PHASE3_PLAN.md §30 → §31: do the streamflow and NDVI models keep their skill when
their rain and drought responses are constrained to the physical sign?

§30's climate-sign gate found the two XGBoost models wrong-signed in some months:
streamflow says more rain means less flow in six months of the year (up to −6.9
points, in months where the rain slider's whole swing is under 0.5 mm/day) and
less flow under a wetter PDSI in every month; NDVI is wrong-signed under rain,
heat and PDSI in a scattered 2–5 months each. Both models have real out-of-fold
skill (+0.68, +0.26), earned on a record where rain, PDSI and temperature move
together and the monsoon dominates. The app decouples them. The wrong-signed
months are the model extrapolating where the data never tested it.

THIS SCRIPT CHANGES NOTHING THAT SHIPS. It re-runs each model's REAL training
procedure — surface water's in-fold feature selection + three-candidate
competition + tuning, NDVI's 60-draw RandomizedSearch — on the model's own
window and its own five outer folds (exactly `train_and_evaluate`'s nested
evaluation, reproducing the shipped scores), and then runs it again with ONE
change: every XGBoost the procedure builds carries `monotone_constraints` on the
rain and drought features. Nothing else moves: not the feature set, not the
candidates, not the search, not the folds.

The constraint, declared here:
  surface_water  every `nclimdiv_precipitation_*` and `nclimdiv_pdsi*` feature
                 (levels, lags, rolls, anomalies) is monotone NON-DECREASING (+1).
                 `nclimdiv_precip_x_temperature` is left free (0): its sign is not
                 physical on its own.
  ndvi           every `precipitation_mm_day*` feature is +1; every `usdm_dsci*`
                 feature is −1 (a higher DSCI is MORE drought); `precip_x_impervious`
                 and `precip_x_temperature` are left free (0).
  Temperature and everything else are unconstrained in both. Heat → NDVI, which
  also fails the gate, is deliberately NOT constrained here: the fuel/greening
  physics of winter warmth in a desert is not one-signed, and the gate's failure
  there is small.

THE DECISION RULE, FIXED BEFORE THE FIRST RUN — in reverse
-----------------------------------------------------------
The constraint is a prior, not a feature, so the burden is on the loss, not the
gain. For each model, on target R² meaned over the five folds, the constrained
model SHIPS unless it loses skill:
  it does NOT ship if  Δ target R² < −0.03           (a material loss, any t)
  or                   Δ target R² < 0 and |t| ≥ 2.0  (a significant loss)
  otherwise it ships, and the climate-sign gate is re-run on the deployed model:
  if the rain/PDSI pairs still fail after deployment, the constraint was not the
  fix, and that is reported rather than iterated on.
One run per model. No change to the constraint set after the numbers are seen.

Run:
    python -m scripts.phase2.experiment_monotone
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from scripts.phase2 import model_ndvi, model_surface_water
from scripts.phase2.experiment_grace_nested import paired_t
from scripts.phase2.features import build_all
from scripts.phase2.metrics import nested_cv_evaluate, persistence_r2

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = ROOT / "model" / "experiment_monotone.json"

MATERIAL_LOSS = -0.03
T_REQUIRED = 2.0


def sign_for(model_key: str, feature: str) -> int:
    if model_key == "surface_water":
        if feature == "nclimdiv_precip_x_temperature":
            return 0
        if "nclimdiv_precipitation" in feature or "nclimdiv_pdsi" in feature:
            return +1
        return 0
    if model_key == "ndvi":
        if feature in ("precip_x_impervious", "precip_x_temperature"):
            return 0
        if feature.startswith("precipitation_mm_day"):
            return +1
        if feature.startswith("usdm_dsci"):
            return -1
        return 0
    raise ValueError(model_key)


# The column order of the frame the current XGBoost will be fit on. Surface water's
# pipeline hands XGBoost a numpy array after the scaler, so the names are captured
# from the feature-selection step that precedes it.
_STATE: dict = {"model_key": None, "columns": None}


class MonotoneXGB(XGBRegressor):
    """XGBRegressor that sets monotone_constraints from the feature names at fit."""

    def fit(self, X, y=None, **kwargs):  # noqa: N803
        columns = list(X.columns) if hasattr(X, "columns") else _STATE["columns"]
        if columns is None or len(columns) != np.asarray(X).shape[1]:
            raise RuntimeError("MonotoneXGB: cannot recover feature names for the constraint")
        cons = tuple(sign_for(_STATE["model_key"], c) for c in columns)
        self.set_params(monotone_constraints=cons)
        return super().fit(X, y, **kwargs)


def with_constraints(module, model_key: str):
    """Context: every XGBRegressor `module` constructs is a MonotoneXGB."""

    class _Ctx:
        def __enter__(self):
            self.saved = module.XGBRegressor
            self.saved_select = getattr(module, "_select_features", None)
            module.XGBRegressor = MonotoneXGB
            _STATE["model_key"] = model_key
            if self.saved_select is not None:

                def recording_select(x, y, threshold):
                    selected = self.saved_select(x, y, threshold)
                    _STATE["columns"] = list(selected)
                    return selected

                module._select_features = recording_select
            return self

        def __exit__(self, *exc):
            module.XGBRegressor = self.saved
            if self.saved_select is not None:
                module._select_features = self.saved_select
            _STATE["model_key"] = None
            _STATE["columns"] = None

    return _Ctx()


def run_model(model_key: str, module, x: pd.DataFrame, y: pd.Series, lag1_col: str) -> dict:
    lag1 = x[lag1_col].copy()
    x_train = x.drop(columns=[lag1_col])
    y_target = y - lag1
    cv = TimeSeriesSplit(n_splits=5)
    n_constrained = sum(sign_for(model_key, c) != 0 for c in x_train.columns)

    def evaluate() -> dict:
        s = nested_cv_evaluate(
            fit_predict=module._fit_predict, x=x_train, y_level=y, y_target=y_target, cv=cv, anchor=lag1
        )
        return {
            "target_r2": float(s["cv_target_r2"]),
            "level_r2": float(s["cv_mean_r2"]),
            "folds": [float(f.get("target_r2", f.get("r2_target"))) for f in s["folds"]],
        }

    print(f"\n=== {model_key}: {len(y)} rows {y.index[0]}..{y.index[-1]}, {x_train.shape[1]} features, "
          f"{n_constrained} constrained ===", flush=True)
    print("  running shipped ...", flush=True)
    shipped = evaluate()
    print("  running constrained ...", flush=True)
    with with_constraints(module, model_key):
        constrained = evaluate()
    persistence = float(persistence_r2(y_level=y, lag1_level=lag1, cv=cv))

    diffs = [b - a for a, b in zip(shipped["folds"], constrained["folds"], strict=True)]
    d = constrained["target_r2"] - shipped["target_r2"]
    t = paired_t(diffs)
    material_loss = d < MATERIAL_LOSS
    significant_loss = d < 0 and abs(t) >= T_REQUIRED
    ships = not (material_loss or significant_loss)

    print(f"\n  {'arm':12s} {'target R2':>11s} {'level R2':>10s} {'skill':>9s}")
    for name, r in (("shipped", shipped), ("constrained", constrained)):
        print(f"  {name:12s} {r['target_r2']:+11.4f} {r['level_r2']:+10.4f} {r['level_r2'] - persistence:+9.4f}")
    print("  per-fold target R2 (shipped -> constrained): "
          + "  ".join(f"{a:+.3f}->{b:+.3f}" for a, b in zip(shipped["folds"], constrained["folds"], strict=True)))
    print(f"  Δ target R² {d:+.4f}   wins {sum(x > 0 for x in diffs)}/5   paired t {t:+.2f}")
    print(f"  [{'FAIL' if material_loss else 'ok  '}]  material loss (Δ < {MATERIAL_LOSS})")
    print(f"  [{'FAIL' if significant_loss else 'ok  '}]  significant loss (Δ < 0 and |t| ≥ {T_REQUIRED})")
    print(f"  verdict: {'SHIPS' if ships else 'DOES NOT SHIP — the unconstrained model stands'}")
    return {
        "rows": int(len(y)), "window": [str(y.index[0]), str(y.index[-1])],
        "n_features": int(x_train.shape[1]), "n_constrained": int(n_constrained),
        "constraints": {c: sign_for(model_key, c) for c in x_train.columns if sign_for(model_key, c) != 0},
        "persistence_level_r2": persistence,
        "shipped": shipped, "constrained": constrained,
        "delta_target_r2": d, "wins": int(sum(x > 0 for x in diffs)), "paired_t": t,
        "material_loss": bool(material_loss), "significant_loss": bool(significant_loss),
        "verdict": "SHIPS" if ships else "DOES NOT SHIP",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["surface_water", "ndvi"])
    args = ap.parse_args()
    datasets = build_all()
    results = {}
    for key in args.models:
        module = {"surface_water": model_surface_water, "ndvi": model_ndvi}[key]
        lag1_col = {"surface_water": "discharge_log_anomaly_lag1", "ndvi": "ndvi_lag1"}[key]
        x, y = datasets[key]
        results[key] = run_model(key, module, x, y, lag1_col)
    OUTPUT_FILE.write_text(json.dumps({
        "rule": {"material_loss": MATERIAL_LOSS, "t_required": T_REQUIRED, "direction": "constrained ships unless it loses"},
        "results": results,
    }, indent=2))
    print(f"\n  wrote {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
