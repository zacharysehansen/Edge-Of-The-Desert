"""
model_groundwater.py
--------------------
Model 3: Groundwater Well Levels — monthly regression.

Multi-model competition with residual-over-lag1:
  - Formulations: direct vs residual-over-lag1
  - Algorithms: Ridge, ElasticNet, XGBoost (6 candidates)
  - Feature selection: drops low-importance features after initial fit

TimeSeriesSplit(n_splits=5) cross-validation.
Export: ONNX + artifacts to model/.

Usage
-----
    python -m scripts.phase2.model_groundwater
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from scripts.phase2.features import build_all

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "groundwater"
N_SPLITS = 5
CV = TimeSeriesSplit(n_splits=N_SPLITS)
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Hyperparameter spaces
# ---------------------------------------------------------------------------

XGB_PARAM_SPACE = {
    "n_estimators":     [100, 150, 200, 300],
    "max_depth":        [2],
    "learning_rate":    [0.005, 0.01, 0.02],
    "subsample":        [0.3, 0.4, 0.5],
    "colsample_bytree": [0.2, 0.25, 0.3],
    "reg_alpha":        [5.0, 10.0, 15.0],
    "reg_lambda":       [80, 100, 150],
    "min_child_weight": [60, 80, 100, 120],
}

RIDGE_PARAM_SPACE = {
    "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}

ELASTICNET_PARAM_SPACE = {
    "model__alpha": [0.001, 0.01, 0.1, 1.0],
    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

IMPORTANCE_THRESHOLD = 0.015
BLEND_WEIGHT_XGB = 0.6


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def _select_features(X: pd.DataFrame, y: pd.Series, threshold: float) -> list[str]:
    """Fit a quick XGBoost and return features above the importance threshold."""
    model = XGBRegressor(
        n_estimators=300, max_depth=2, learning_rate=0.05,
        min_child_weight=10, reg_lambda=10, subsample=0.7,
        colsample_bytree=0.5, random_state=RANDOM_STATE, verbosity=0,
    )
    model.fit(X, y)
    importances = dict(zip(X.columns, model.feature_importances_))
    selected = [col for col, imp in importances.items() if imp >= threshold]
    return selected if len(selected) >= 5 else list(X.columns)


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def _evaluate_candidate(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_actual: pd.Series,
    lag1: pd.Series | None,
    formulation: str,
) -> dict:
    """Evaluate a single model candidate via CV. Returns scores dict."""
    fold_r2, fold_mae = [], []

    for train_idx, test_idx in CV.split(X_train):
        X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr = y_train.iloc[train_idx]
        y_te_actual = y_actual.iloc[test_idx]

        if "blend" in name:
            xgb_model = XGBRegressor(
                n_estimators=150, max_depth=2, learning_rate=0.02,
                min_child_weight=40, reg_lambda=50, reg_alpha=2.0,
                subsample=0.5, colsample_bytree=0.3,
                random_state=RANDOM_STATE, verbosity=0,
            )
            xgb_model.fit(X_tr, y_tr)
            pred_xgb = xgb_model.predict(X_te)

            en_pipe = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000))])
            en_pipe.fit(X_tr, y_tr)
            pred_en = en_pipe.predict(X_te)

            pred = BLEND_WEIGHT_XGB * pred_xgb + (1 - BLEND_WEIGHT_XGB) * pred_en
        elif "xgb_tight" in name:
            model = XGBRegressor(
                n_estimators=150, max_depth=2, learning_rate=0.01,
                min_child_weight=50, reg_lambda=80, reg_alpha=5.0,
                subsample=0.5, colsample_bytree=0.3,
                random_state=RANDOM_STATE, verbosity=0,
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
        elif "xgb" in name:
            model = XGBRegressor(
                n_estimators=200, max_depth=2, learning_rate=0.02,
                min_child_weight=30, reg_lambda=30, reg_alpha=2.0,
                subsample=0.6, colsample_bytree=0.4,
                random_state=RANDOM_STATE, verbosity=0,
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
        elif "ridge" in name:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
        elif "elastic" in name:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000))])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
        else:
            continue

        if formulation == "residual" and lag1 is not None:
            pred_actual = pred + lag1.iloc[test_idx].values
        else:
            pred_actual = pred

        r2 = r2_score(y_te_actual, pred_actual)
        mae = mean_absolute_error(y_te_actual, pred_actual)
        fold_r2.append(r2)
        fold_mae.append(mae)

    return {
        "name": name,
        "formulation": formulation,
        "mean_r2": float(np.mean(fold_r2)),
        "std_r2": float(np.std(fold_r2)),
        "mean_mae": float(np.mean(fold_mae)),
        "std_mae": float(np.std(fold_mae)),
    }


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------

def train_and_evaluate() -> dict:
    """Build, tune, evaluate, and export the groundwater well level model."""
    datasets = build_all()
    X, y = datasets[MODEL_ID]

    # Separate lag1 column for residual formulation
    lag1_col = "depth_to_water_ft_mean_lag1"
    lag1 = X[lag1_col].copy()
    X_full = X.drop(columns=[lag1_col])

    # Residual target
    y_residual = y - lag1

    # Feature selection
    selected = _select_features(X_full, y_residual, IMPORTANCE_THRESHOLD)
    X_selected = X_full[selected]
    n_dropped = len(X_full.columns) - len(selected)
    print(f"  Feature selection: {len(selected)} kept, {n_dropped} dropped")

    # ----- Multi-model competition -----
    candidates = []

    # Direct formulation candidates
    for model_name in ["xgb_direct", "ridge_direct", "elastic_direct"]:
        result = _evaluate_candidate(
            model_name, X_selected, y, y, None, "direct"
        )
        candidates.append(result)
        print(f"    {model_name:<20} R²={result['mean_r2']:.4f} ± {result['std_r2']:.4f}")

    # Residual formulation candidates
    for model_name in ["xgb_residual", "xgb_tight_residual", "blend_residual", "ridge_residual", "elastic_residual"]:
        result = _evaluate_candidate(
            model_name, X_selected, y_residual, y, lag1, "residual"
        )
        candidates.append(result)
        print(f"    {model_name:<20} R²={result['mean_r2']:.4f} ± {result['std_r2']:.4f}")

    # Pick winner
    best_candidate = max(candidates, key=lambda c: c["mean_r2"])
    print(f"\n  Winner: {best_candidate['name']} (R²={best_candidate['mean_r2']:.4f})")

    # ----- Train final model with tuning -----
    is_residual = best_candidate["formulation"] == "residual"
    is_blend = "blend" in best_candidate["name"]
    y_final = y_residual if is_residual else y
    feature_names = list(X_selected.columns)

    if is_blend:
        best_model, best_params = _tune_blend(X_selected, y_final)
    elif "xgb" in best_candidate["name"]:
        best_model, best_params = _tune_xgboost(X_selected, y_final)
    elif "ridge" in best_candidate["name"]:
        best_model, best_params = _tune_ridge(X_selected, y_final)
    else:
        best_model, best_params = _tune_elasticnet(X_selected, y_final)

    # ----- Fold-level evaluation -----
    fold_r2, fold_mae = [], []
    fold_details = []

    for fold_i, (train_idx, test_idx) in enumerate(CV.split(X_selected)):
        X_tr = X_selected.iloc[train_idx]
        y_tr = y_final.iloc[train_idx]
        X_te = X_selected.iloc[test_idx]
        y_te_actual = y.iloc[test_idx]

        if is_blend:
            xgb_p = best_params["xgb_params"]
            en_p = best_params["en_params"]
            fold_xgb = XGBRegressor(**xgb_p, objective="reg:squarederror",
                                     tree_method="hist", random_state=RANDOM_STATE, verbosity=0)
            fold_xgb.fit(X_tr, y_tr)
            fold_en = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(**en_p, max_iter=5000))])
            fold_en.fit(X_tr, y_tr)
            pred = BLEND_WEIGHT_XGB * fold_xgb.predict(X_te) + (1 - BLEND_WEIGHT_XGB) * fold_en.predict(X_te)
        elif "xgb" in best_candidate["name"]:
            fold_model = XGBRegressor(**best_params, objective="reg:squarederror",
                                       tree_method="hist", random_state=RANDOM_STATE, verbosity=0)
            fold_model.fit(X_tr, y_tr)
            pred = fold_model.predict(X_te)
        elif "ridge" in best_candidate["name"]:
            fold_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(**best_params))])
            fold_model.fit(X_tr, y_tr)
            pred = fold_model.predict(X_te)
        else:
            fold_model = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(**best_params, max_iter=5000))])
            fold_model.fit(X_tr, y_tr)
            pred = fold_model.predict(X_te)

        if is_residual:
            pred_actual = pred + lag1.iloc[test_idx].values
        else:
            pred_actual = pred

        r2 = r2_score(y_te_actual, pred_actual)
        mae = mean_absolute_error(y_te_actual, pred_actual)
        fold_r2.append(r2)
        fold_mae.append(mae)
        fold_details.append({
            "fold": fold_i,
            "test_start": str(y.index[test_idx[0]]),
            "test_end":   str(y.index[test_idx[-1]]),
            "test_size":  len(test_idx),
            "r2":         float(r2),
            "mae":        float(mae),
        })

    mean_r2  = float(np.mean(fold_r2))
    std_r2   = float(np.std(fold_r2))
    mean_mae = float(np.mean(fold_mae))
    std_mae  = float(np.std(fold_mae))

    # ----- Train score -----
    if is_blend:
        xgb_m, en_m = best_model
        train_pred = BLEND_WEIGHT_XGB * xgb_m.predict(X_selected) + (1 - BLEND_WEIGHT_XGB) * en_m.predict(X_selected)
    else:
        train_pred = best_model.predict(X_selected)

    if is_residual:
        train_pred_actual = train_pred + lag1.values
    else:
        train_pred_actual = train_pred
    train_r2 = float(r2_score(y, train_pred_actual))
    train_mae = float(mean_absolute_error(y, train_pred_actual))

    # ----- Historical predictions -----
    historical = pd.DataFrame({
        "year_month": y.index.astype(str),
        "groundwater_actual": y.values,
        "groundwater_predicted": train_pred_actual,
    })

    # ----- Feature importance -----
    if is_blend:
        xgb_m, en_m = best_model
        xgb_imp = xgb_m.feature_importances_
        en_coefs = np.abs(en_m.named_steps["model"].coef_)
        en_total = en_coefs.sum() if en_coefs.sum() > 0 else 1.0
        en_imp = en_coefs / en_total
        blended_imp = BLEND_WEIGHT_XGB * xgb_imp + (1 - BLEND_WEIGHT_XGB) * en_imp
        importance = dict(zip(feature_names, blended_imp.tolist()))
    elif "xgb" in best_candidate["name"]:
        importance = dict(zip(feature_names, best_model.feature_importances_.tolist()))
    else:
        if hasattr(best_model, "named_steps"):
            coefs = np.abs(best_model.named_steps["model"].coef_)
        else:
            coefs = np.abs(best_model.coef_)
        total = coefs.sum() if coefs.sum() > 0 else 1.0
        importance = dict(zip(feature_names, (coefs / total).tolist()))
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # ----- Feature stats -----
    feature_stats = {
        col: {"mean": float(X_selected[col].mean()), "std": float(X_selected[col].std())}
        for col in feature_names
    }

    # ----- Export ONNX -----
    if is_blend:
        xgb_m, _ = best_model
        _export_onnx_xgb(xgb_m, feature_names)
        print("  [INFO] Blend: exported XGBoost component to ONNX (ElasticNet component requires skl2onnx separately)")
    elif "xgb" in best_candidate["name"]:
        _export_onnx_xgb(best_model, feature_names)
    else:
        _export_onnx_sklearn(best_model, feature_names)

    # ----- Save artifacts -----
    cv_results = {
        "model_id": MODEL_ID,
        "formulation": best_candidate["formulation"],
        "winner": best_candidate["name"],
        "competition": candidates,
        "window_start": str(y.index[0]),
        "window_end": str(y.index[-1]),
        "n_rows": len(y),
        "n_features": len(feature_names),
        "n_features_dropped": n_dropped,
        "best_params": best_params,
        "cv_mean_r2": mean_r2,
        "cv_std_r2": std_r2,
        "cv_mean_mae": mean_mae,
        "cv_std_mae": std_mae,
        "train_r2": train_r2,
        "train_mae": train_mae,
        "folds": fold_details,
    }

    _save_json(cv_results, MODEL_DIR / f"{MODEL_ID}_cv_results.json")
    _save_json(feature_names, MODEL_DIR / f"{MODEL_ID}_feature_names.json")
    _save_json(feature_stats, MODEL_DIR / f"{MODEL_ID}_feature_stats.json")
    _save_json(sorted_importance, MODEL_DIR / f"{MODEL_ID}_feature_importance.json")
    historical.to_csv(MODEL_DIR / f"historical_{MODEL_ID}.csv", index=False)

    return cv_results


# ---------------------------------------------------------------------------
# Tuning functions
# ---------------------------------------------------------------------------

def _tune_blend(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Tune both XGBoost and ElasticNet for the blend, return (model_tuple, params_dict)."""
    xgb_model, xgb_params = _tune_xgboost(X, y)
    en_model, en_params = _tune_elasticnet(X, y)
    return (xgb_model, en_model), {"xgb_params": xgb_params, "en_params": en_params, "blend_weight_xgb": BLEND_WEIGHT_XGB}


def _tune_xgboost(X: pd.DataFrame, y: pd.Series) -> tuple:
    base = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_STATE, verbosity=0,
    )
    search = RandomizedSearchCV(
        base, XGB_PARAM_SPACE, n_iter=60, cv=CV, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1, refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def _tune_ridge(X: pd.DataFrame, y: pd.Series) -> tuple:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    search = RandomizedSearchCV(
        pipe, RIDGE_PARAM_SPACE, n_iter=5, cv=CV, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1, refit=True,
    )
    search.fit(X, y)
    best_alpha = search.best_params_["model__alpha"]
    return search.best_estimator_, {"alpha": best_alpha}


def _tune_elasticnet(X: pd.DataFrame, y: pd.Series) -> tuple:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=5000))])
    search = RandomizedSearchCV(
        pipe, ELASTICNET_PARAM_SPACE, n_iter=20, cv=CV, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1, refit=True,
    )
    search.fit(X, y)
    best_params = {
        "alpha": search.best_params_["model__alpha"],
        "l1_ratio": search.best_params_["model__l1_ratio"],
    }
    return search.best_estimator_, best_params


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _export_onnx_xgb(model: XGBRegressor, feature_names: list[str]) -> None:
    """Export XGBoost to ONNX."""
    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        print("  [WARN] onnxmltools not installed — skipping ONNX export")
        return

    booster = model.get_booster()
    numeric_names = [f"f{i}" for i in range(len(feature_names))]
    booster.feature_names = numeric_names

    clone = XGBRegressor(**model.get_params())
    clone.fit(np.zeros((2, len(feature_names))), np.zeros(2))
    clone.get_booster().load_model(bytearray(booster.save_raw()))
    clone.get_booster().feature_names = numeric_names

    initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_xgboost(clone, initial_types=initial_type)

    path = MODEL_DIR / f"{MODEL_ID}.onnx"
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  ONNX exported → {path}")


def _export_onnx_sklearn(model, feature_names: list[str]) -> None:
    """Export sklearn pipeline to ONNX."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("  [WARN] skl2onnx not installed — skipping ONNX export")
        return

    initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Model 3: Groundwater Well Levels")
    print(f"{'='*60}\n")

    results = train_and_evaluate()

    print(f"\n  Winner     : {results['winner']} ({results['formulation']})")
    print(f"  Window     : {results['window_start']} → {results['window_end']}")
    print(f"  Rows       : {results['n_rows']}")
    print(f"  Features   : {results['n_features']} (dropped {results['n_features_dropped']})")
    print(f"  CV R²      : {results['cv_mean_r2']:.4f} ± {results['cv_std_r2']:.4f}")
    print(f"  CV MAE     : {results['cv_mean_mae']:.4f} ± {results['cv_std_mae']:.4f}")
    print(f"  Train R²   : {results['train_r2']:.4f}")
    print(f"  Train MAE  : {results['train_mae']:.4f}")
    print(f"\n  Best params: {results['best_params']}")
    print(f"\n  Artifacts saved to {MODEL_DIR}/")
