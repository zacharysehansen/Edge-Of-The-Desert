"""
model_wildfire_monthly.py
-------------------------
Model 5b: Wildfire Risk Index — monthly regression.

Multi-model competition with two formulations:
  - Direct: predict wildfire_risk_index directly
  - Residual: predict (wildfire_risk_index - lag1), reconstruct at eval

Models: Ridge, ElasticNet, XGBoost (constrained)
TimeSeriesSplit(n_splits=5) cross-validation.
Export: ONNX + artifacts to model/.

Usage
-----
    python -m scripts.phase2.model_wildfire_monthly
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from scripts.phase2.features import build_all

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "wildfire_monthly"
N_SPLITS = 5
CV = TimeSeriesSplit(n_splits=N_SPLITS)
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Model families
# ---------------------------------------------------------------------------

def _build_ridge() -> tuple[str, Pipeline, dict, bool]:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=RANDOM_STATE)),
    ])
    param_grid = {
        "model__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
    }
    return "Ridge", pipe, param_grid, False


def _build_elasticnet() -> tuple[str, Pipeline, dict, bool]:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(random_state=RANDOM_STATE, max_iter=10000)),
    ])
    param_grid = {
        "model__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    return "ElasticNet", pipe, param_grid, False


def _build_xgboost() -> tuple[str, Pipeline, dict, bool]:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            verbosity=0,
        )),
    ])
    param_grid = {
        "model__n_estimators":     [100, 200, 300],
        "model__max_depth":        [2, 3],
        "model__learning_rate":    [0.01, 0.03, 0.05],
        "model__subsample":        [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.5, 0.6, 0.7],
        "model__reg_alpha":        [0.1, 0.5, 1.0, 5.0],
        "model__reg_lambda":       [1, 5, 10],
        "model__min_child_weight": [3, 5, 10],
    }
    return "XGBoost", pipe, param_grid, True


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_candidate(
    model_name: str,
    formulation: str,
    pipe: Pipeline,
    param_grid: dict,
    use_random: bool,
    X: pd.DataFrame,
    y: pd.Series,
    lag1: pd.Series,
) -> dict:
    """Train and evaluate a candidate model under a given formulation."""
    if formulation == "residual":
        y_train_target = y - lag1
    else:
        y_train_target = y

    if use_random:
        search = RandomizedSearchCV(
            pipe, param_grid, n_iter=40, cv=CV, scoring="r2",
            random_state=RANDOM_STATE, n_jobs=-1, refit=True,
        )
    else:
        search = GridSearchCV(
            pipe, param_grid, cv=CV, scoring="r2", n_jobs=-1, refit=True,
        )

    search.fit(X, y_train_target)
    best_pipe = search.best_estimator_

    # Fold-level evaluation (always score in actual scale)
    fold_r2, fold_mae = [], []
    for train_idx, test_idx in CV.split(X):
        X_tr, y_tr = X.iloc[train_idx], y_train_target.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_te_actual = y.iloc[test_idx]

        fold_pipe = clone(best_pipe)
        fold_pipe.fit(X_tr, y_tr)
        pred = fold_pipe.predict(X_te)

        if formulation == "residual":
            pred_actual = pred + lag1.iloc[test_idx].values
        else:
            pred_actual = pred

        fold_r2.append(r2_score(y_te_actual, pred_actual))
        fold_mae.append(mean_absolute_error(y_te_actual, pred_actual))

    return {
        "label": f"{model_name} ({formulation})",
        "model_name": model_name,
        "formulation": formulation,
        "pipe": best_pipe,
        "params": search.best_params_,
        "cv_mean_r2": float(np.mean(fold_r2)),
        "cv_std_r2": float(np.std(fold_r2)),
        "cv_mean_mae": float(np.mean(fold_mae)),
        "cv_std_mae": float(np.std(fold_mae)),
        "n_rows": len(y),
        "n_features": X.shape[1],
    }


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------

def train_and_evaluate() -> dict:
    """Run full competition and export the winner."""
    datasets = build_all()

    if "wildfire_monthly" not in datasets:
        raise RuntimeError(
            "wildfire_monthly dataset not found. "
            "Ensure data/Final/wildfire_monthly.csv exists and merge.py loads it."
        )

    X_full, y = datasets["wildfire_monthly"]

    lag1_col = "wildfire_risk_index_lag1"
    lag1 = X_full[lag1_col].copy()
    X = X_full.drop(columns=[lag1_col])
    feature_names = list(X.columns)

    candidates = []
    model_builders = [_build_ridge, _build_elasticnet, _build_xgboost]
    formulations = ["direct", "residual"]

    for formulation in formulations:
        for builder in model_builders:
            name, pipe, params, use_random = builder()
            print(f"  {name:<12} × {formulation:<10} ({X.shape[0]} rows, {X.shape[1]} features)...")
            result = _evaluate_candidate(name, formulation, pipe, params, use_random, X, y, lag1)
            candidates.append(result)

    # Print comparison table
    print(f"\n  {'Candidate':<30} {'CV R²':>10} {'CV MAE':>10}")
    print(f"  {'-'*52}")
    for c in sorted(candidates, key=lambda x: x["cv_mean_r2"], reverse=True):
        print(f"  {c['label']:<30} {c['cv_mean_r2']:>10.4f} {c['cv_mean_mae']:>10.4f}")

    # Select winner
    winner = max(candidates, key=lambda c: c["cv_mean_r2"])
    print(f"\n  WINNER: {winner['label']} (CV R² = {winner['cv_mean_r2']:.4f})")

    best_pipe = winner["pipe"]
    formulation = winner["formulation"]

    # Train score
    if formulation == "residual":
        train_target = y - lag1
    else:
        train_target = y

    train_pred = best_pipe.predict(X)
    if formulation == "residual":
        train_pred_actual = train_pred + lag1.values
    else:
        train_pred_actual = train_pred

    train_r2 = float(r2_score(y, train_pred_actual))
    train_mae = float(mean_absolute_error(y, train_pred_actual))

    # Historical predictions
    historical = pd.DataFrame({
        "year_month": y.index.astype(str),
        "wildfire_risk_actual": y.values,
        "wildfire_risk_predicted": train_pred_actual,
    })

    # Feature importance
    sorted_importance = _extract_importance(best_pipe, feature_names, winner["model_name"])

    # Feature stats
    feature_stats = {
        col: {"mean": float(X[col].mean(skipna=True)), "std": float(X[col].std(skipna=True))}
        for col in feature_names
    }

    # Export ONNX
    _export_onnx(best_pipe, feature_names, winner["model_name"])

    # Save artifacts
    cv_results = {
        "model_id": MODEL_ID,
        "model_type": winner["model_name"],
        "formulation": formulation,
        "window_start": str(y.index[0]),
        "window_end": str(y.index[-1]),
        "n_rows": len(y),
        "n_features": len(feature_names),
        "best_params": {k: _jsonable(v) for k, v in winner["params"].items()},
        "cv_mean_r2": winner["cv_mean_r2"],
        "cv_std_r2": winner["cv_std_r2"],
        "cv_mean_mae": winner["cv_mean_mae"],
        "cv_std_mae": winner["cv_std_mae"],
        "train_r2": train_r2,
        "train_mae": train_mae,
        "all_candidates": [
            {"label": c["label"], "cv_mean_r2": c["cv_mean_r2"], "cv_mean_mae": c["cv_mean_mae"]}
            for c in sorted(candidates, key=lambda x: x["cv_mean_r2"], reverse=True)
        ],
        "note": (
            f"Monthly wildfire model. Winner: {winner['label']}. "
            f"Competed 3 models × 2 formulations (6 total). "
            f"219 monthly rows from 2002-10 to 2020-12."
        ),
    }

    _save_json(cv_results, MODEL_DIR / f"{MODEL_ID}_cv_results.json")
    _save_json(feature_names, MODEL_DIR / f"{MODEL_ID}_feature_names.json")
    _save_json(feature_stats, MODEL_DIR / f"{MODEL_ID}_feature_stats.json")
    _save_json(sorted_importance, MODEL_DIR / f"{MODEL_ID}_feature_importance.json")
    historical.to_csv(MODEL_DIR / f"historical_{MODEL_ID}.csv", index=False)

    return cv_results


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _extract_importance(pipe: Pipeline, feature_names: list[str], model_name: str) -> dict:
    model = pipe.named_steps["model"]
    if model_name in ("Ridge", "ElasticNet"):
        coefs = np.abs(model.coef_)
        importance = dict(zip(feature_names, coefs.tolist()))
    else:
        importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _export_onnx(pipe: Pipeline, feature_names: list[str], model_name: str) -> None:
    try:
        if model_name in ("Ridge", "ElasticNet"):
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_sklearn(pipe, initial_types=initial_type)
        else:
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType

            xgb_model = pipe.named_steps["model"]
            booster = xgb_model.get_booster()
            numeric_names = [f"f{i}" for i in range(len(feature_names))]
            booster.feature_names = numeric_names

            clone_model = XGBRegressor(**xgb_model.get_params())
            clone_model.fit(np.zeros((2, len(feature_names))), np.zeros(2))
            clone_model.get_booster().load_model(bytearray(booster.save_raw()))
            clone_model.get_booster().feature_names = numeric_names

            initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_xgboost(clone_model, initial_types=initial_type)
    except ImportError as e:
        print(f"  [WARN] ONNX export dependencies missing ({e}) — skipping")
        return
    except Exception as e:
        print(f"  [WARN] ONNX export failed ({e}) — skipping")
        return

    path = MODEL_DIR / f"{MODEL_ID}.onnx"
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  ONNX exported → {path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Model 5b: Wildfire Risk (Monthly, Multi-Model)")
    print(f"{'='*60}\n")

    results = train_and_evaluate()

    print(f"\n  Winner     : {results['model_type']} ({results['formulation']})")
    print(f"  Window     : {results['window_start']} → {results['window_end']}")
    print(f"  Rows       : {results['n_rows']}")
    print(f"  Features   : {results['n_features']}")
    print(f"  CV R²      : {results['cv_mean_r2']:.4f} ± {results['cv_std_r2']:.4f}")
    print(f"  CV MAE     : {results['cv_mean_mae']:.6f} ± {results['cv_std_mae']:.6f}")
    print(f"  Train R²   : {results['train_r2']:.4f}")
    print(f"  Train MAE  : {results['train_mae']:.6f}")
    print(f"\n  Best params: {results['best_params']}")
    print(f"\n  Artifacts saved to {MODEL_DIR}/")
