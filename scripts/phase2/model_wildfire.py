"""
model_wildfire.py
-----------------
Model 5: Wildfire Risk Index — annual regression.

Multi-model × multi-feature-set competition:
  Feature sets:
    - "full" (29 features, 2000-2020, all inputs)
    - "climate" (18 features, 2000-2021, climate + fire-ecology only)
    - "minimal" (7 features, 2000-2021, domain-pruned to physical drivers)
  Model families:
    - Ridge (strong L2 regularization)
    - ElasticNet (mixed L1/L2)
    - XGBoost (constrained tree depth)

The winner across all 9 combinations is exported.

Usage
-----
    python -m scripts.phase2.model_wildfire
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

MODEL_ID = "wildfire"
N_SPLITS = 4
CV = TimeSeriesSplit(n_splits=N_SPLITS)
RANDOM_STATE = 42

# Feature set keys in the datasets dict
FEATURE_SETS = {
    "full":    "wildfire",
    "climate": "wildfire_climate",
    "minimal": "wildfire_minimal",
}


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
        "model__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    }
    return "Ridge", pipe, param_grid, False


def _build_elasticnet() -> tuple[str, Pipeline, dict, bool]:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(random_state=RANDOM_STATE, max_iter=5000)),
    ])
    param_grid = {
        "model__alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
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
        "model__n_estimators":     [50, 100, 150, 200],
        "model__max_depth":        [2, 3],
        "model__learning_rate":    [0.03, 0.05, 0.08, 0.1],
        "model__subsample":        [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.5, 0.6, 0.7],
        "model__reg_alpha":        [0.1, 0.5, 1.0],
        "model__reg_lambda":       [1, 5, 10],
        "model__min_child_weight": [1, 3, 5],
    }
    return "XGBoost", pipe, param_grid, True


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------

def _evaluate_candidate(
    model_name: str,
    feature_set: str,
    pipe: Pipeline,
    param_grid: dict,
    use_random: bool,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Search, evaluate via CV folds, and return results."""
    cv = TimeSeriesSplit(n_splits=min(N_SPLITS, len(y) - 2))

    if use_random:
        search = RandomizedSearchCV(
            pipe, param_grid, n_iter=30, cv=cv, scoring="r2",
            random_state=RANDOM_STATE, n_jobs=-1, refit=True,
        )
    else:
        search = GridSearchCV(
            pipe, param_grid, cv=cv, scoring="r2", n_jobs=-1, refit=True,
        )

    search.fit(X, y)
    best_pipe = search.best_estimator_

    # Fold-level evaluation
    fold_r2, fold_mae = [], []
    for train_idx, test_idx in cv.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        fold_pipe = clone(best_pipe)
        fold_pipe.fit(X_tr, y_tr)
        pred = fold_pipe.predict(X_te)
        fold_r2.append(r2_score(y_te, pred))
        fold_mae.append(mean_absolute_error(y_te, pred))

    return {
        "label": f"{model_name} ({feature_set})",
        "model_name": model_name,
        "feature_set": feature_set,
        "pipe": best_pipe,
        "params": search.best_params_,
        "cv_mean_r2": float(np.mean(fold_r2)),
        "cv_std_r2": float(np.std(fold_r2)),
        "cv_mean_mae": float(np.mean(fold_mae)),
        "cv_std_mae": float(np.std(fold_mae)),
        "n_rows": len(y),
        "n_features": X.shape[1],
    }


def train_and_evaluate() -> dict:
    """Run full competition and export the winner."""
    datasets = build_all()

    candidates = []

    model_builders = [_build_ridge, _build_elasticnet, _build_xgboost]

    for fs_label, ds_key in FEATURE_SETS.items():
        X, y = datasets[ds_key]
        for builder in model_builders:
            name, pipe, params, use_random = builder()
            print(f"  {name:<12} × {fs_label:<10} ({X.shape[0]} rows, {X.shape[1]} features)...")
            result = _evaluate_candidate(name, fs_label, pipe, params, use_random, X, y)
            candidates.append(result)

    # Print comparison table
    print(f"\n  {'Candidate':<30} {'CV R²':>10} {'CV MAE':>10} {'Rows':>6} {'Feats':>6}")
    print(f"  {'-'*66}")
    for c in sorted(candidates, key=lambda x: x["cv_mean_r2"], reverse=True):
        print(f"  {c['label']:<30} {c['cv_mean_r2']:>10.4f} {c['cv_mean_mae']:>10.4f} "
              f"{c['n_rows']:>6} {c['n_features']:>6}")

    # Select winner
    winner = max(candidates, key=lambda c: c["cv_mean_r2"])
    print(f"\n  WINNER: {winner['label']} (CV R² = {winner['cv_mean_r2']:.4f})")

    # Get the data for the winning feature set
    X, y = datasets[FEATURE_SETS[winner["feature_set"]]]
    feature_names = list(X.columns)
    best_pipe = winner["pipe"]

    # Train score
    train_pred = best_pipe.predict(X)
    train_r2 = float(r2_score(y, train_pred))
    train_mae = float(mean_absolute_error(y, train_pred))

    # Historical predictions
    historical = pd.DataFrame({
        "year": y.index.astype(int),
        "wildfire_risk_actual": y.values,
        "wildfire_risk_predicted": train_pred,
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
        "feature_set": winner["feature_set"],
        "formulation": "direct",
        "window_start": int(y.index[0]),
        "window_end": int(y.index[-1]),
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
            {"label": c["label"], "cv_mean_r2": c["cv_mean_r2"],
             "cv_mean_mae": c["cv_mean_mae"], "n_rows": c["n_rows"],
             "n_features": c["n_features"]}
            for c in sorted(candidates, key=lambda x: x["cv_mean_r2"], reverse=True)
        ],
        "note": (
            f"Annual wildfire model. Winner: {winner['label']}. "
            f"Competed 3 models × 3 feature sets (9 total). "
            f"Fire-ecology features: wet-then-dry interaction, consecutive dry years, "
            f"prior 2-year precip sum, JJA temperature anomaly, max drought severity."
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
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_sklearn(pipe, initial_types=initial_type)
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
    print(f"  Model 5: Wildfire Risk (Multi-Model × Multi-Feature)")
    print(f"{'='*60}\n")

    results = train_and_evaluate()

    print(f"\n  Winner     : {results['model_type']} ({results['feature_set']})")
    print(f"  Window     : {results['window_start']} → {results['window_end']}")
    print(f"  Rows       : {results['n_rows']}")
    print(f"  Features   : {results['n_features']}")
    print(f"  CV R²      : {results['cv_mean_r2']:.4f} ± {results['cv_std_r2']:.4f}")
    print(f"  CV MAE     : {results['cv_mean_mae']:.4f} ± {results['cv_std_mae']:.4f}")
    print(f"  Train R²   : {results['train_r2']:.4f}")
    print(f"  Train MAE  : {results['train_mae']:.4f}")
    print(f"\n  Best params: {results['best_params']}")
    print(f"\n  Artifacts saved to {MODEL_DIR}/")
