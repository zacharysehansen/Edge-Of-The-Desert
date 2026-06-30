"""
model_surface_water.py
----------------------
Model 4: Surface Water Conditions (Discharge) — monthly regression.

Multi-model competition with log-transformed target:
  - Target: log1p(discharge_cfs_mean) to handle right-skew from flood events
  - Formulations: direct vs residual-over-lag1
  - Algorithms: Ridge, ElasticNet, XGBoost (6 candidates)
  - Feature selection: drops low-importance features after initial fit

TimeSeriesSplit(n_splits=5) cross-validation.
Export: ONNX + artifacts to model/.

Note: gage_height_ft_mean is excluded due to missing values in the source data.
Only discharge_cfs_mean (full coverage) is used as the prediction target.

Usage
-----
    python -m scripts.phase2.model_surface_water
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.impute import SimpleImputer
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

MODEL_ID = "surface_water"
N_SPLITS = 5
CV = TimeSeriesSplit(n_splits=N_SPLITS)
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Hyperparameter spaces
# ---------------------------------------------------------------------------

XGB_PARAM_SPACE = {
    "n_estimators": [400, 600, 800],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "reg_alpha": [0.01, 0.05, 0.1, 0.5],
    "reg_lambda": [1, 2, 5, 10],
    "min_child_weight": [5, 7, 10],
}

RIDGE_PARAM_SPACE = {
    "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}

ELASTICNET_PARAM_SPACE = {
    "model__alpha": [0.001, 0.01, 0.1, 1.0],
    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

# Importance threshold for feature selection (drop features below this)
IMPORTANCE_THRESHOLD = 0.005


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def _select_features(x: pd.DataFrame, y: pd.Series, threshold: float) -> list[str]:
    """Fit a quick XGBoost and return features above the importance threshold."""
    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=5,
        reg_lambda=5,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(x, y)
    importances = dict(zip(x.columns, model.feature_importances_, strict=False))
    selected = [col for col, imp in importances.items() if imp >= threshold]
    return selected if len(selected) >= 5 else list(x.columns)  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------


def _evaluate_candidate(  # noqa: PLR0913
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    y_actual: pd.Series,
    lag1: pd.Series | None,
    formulation: str,
) -> dict:
    """Evaluate a single model candidate via CV. Returns scores dict."""
    fold_r2, fold_mae = [], []

    for train_idx, test_idx in CV.split(x_train):
        x_tr, x_te = x_train.iloc[train_idx], x_train.iloc[test_idx]
        y_tr = y_train.iloc[train_idx]
        y_te_actual = y_actual.iloc[test_idx]

        if "xgb" in name:
            model = XGBRegressor(
                n_estimators=600,
                max_depth=3,
                learning_rate=0.03,
                min_child_weight=7,
                reg_lambda=5,
                reg_alpha=0.1,
                subsample=0.8,
                colsample_bytree=0.7,
                random_state=RANDOM_STATE,
                verbosity=0,
            )
            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)
        elif "ridge" in name:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
            pipe.fit(x_tr, y_tr)
            pred = pipe.predict(x_te)
        elif "elastic" in name:
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
                ]
            )
            pipe.fit(x_tr, y_tr)
            pred = pipe.predict(x_te)
        else:
            continue

        # Convert back from log space and add lag1 if residual
        if formulation == "residual" and lag1 is not None:
            pred_log = pred + lag1.iloc[test_idx].values
        else:
            pred_log = pred

        # Convert from log1p space back to original
        pred_actual = np.expm1(pred_log)
        y_te_orig = (
            np.expm1(y_te_actual.values)
            if formulation == "direct"
            else np.expm1(y_actual.iloc[test_idx].values)
        )

        # For residual, y_actual is already in log space
        if formulation == "residual":
            y_te_orig = np.expm1(y_actual.iloc[test_idx].values)

        r2 = r2_score(y_te_orig, pred_actual)
        mae = mean_absolute_error(y_te_orig, pred_actual)
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


def train_and_evaluate() -> dict:  # noqa: C901, PLR0912, PLR0915
    """Build, tune, evaluate, and export the surface water model."""
    datasets = build_all()
    x, y_raw = datasets[MODEL_ID]

    # Separate lag1 column
    lag1_col = "discharge_cfs_mean_lag1"
    lag1_raw = x[lag1_col].copy()
    x_base = x.drop(columns=[lag1_col])

    # Log-transform target and lag1
    y_log = np.log1p(y_raw)
    lag1_log = np.log1p(lag1_raw)

    # Feature selection on log-target (direct formulation)
    selected = _select_features(x_base, y_log, IMPORTANCE_THRESHOLD)
    x_selected = x_base[selected]
    n_dropped = len(x_base.columns) - len(selected)
    print(f"  Feature selection: {len(selected)} kept, {n_dropped} dropped")

    # ----- Multi-model competition -----
    # Direct formulation: predict log1p(discharge) directly
    # Residual formulation: predict log1p(discharge) - log1p(discharge_lag1)
    y_residual_log = y_log - lag1_log

    candidates = []

    # Direct formulation candidates
    for model_name in ["xgb_direct", "ridge_direct", "elastic_direct"]:
        result = _evaluate_candidate(
            model_name, x_selected, y_log, y_log, None, "direct"
        )
        candidates.append(result)
        print(
            f"    {model_name:<20} R²={result['mean_r2']:.4f} ± {result['std_r2']:.4f}"
        )

    # Residual formulation candidates
    for model_name in ["xgb_residual", "ridge_residual", "elastic_residual"]:
        result = _evaluate_candidate(
            model_name, x_selected, y_residual_log, y_log, lag1_log, "residual"
        )
        candidates.append(result)
        print(
            f"    {model_name:<20} R²={result['mean_r2']:.4f} ± {result['std_r2']:.4f}"
        )

    # Pick winner
    best_candidate = max(candidates, key=lambda c: c["mean_r2"])
    print(f"\n  Winner: {best_candidate['name']} (R²={best_candidate['mean_r2']:.4f})")

    # ----- Train final model with tuning -----
    is_residual = best_candidate["formulation"] == "residual"
    y_final = y_residual_log if is_residual else y_log
    feature_names = list(x_selected.columns)

    if "xgb" in best_candidate["name"]:
        best_model, best_params = _tune_xgboost(x_selected, y_final)
    elif "ridge" in best_candidate["name"]:
        best_model, best_params = _tune_ridge(x_selected, y_final)
    else:
        best_model, best_params = _tune_elasticnet(x_selected, y_final)

    # ----- Fold-level evaluation on original scale -----
    fold_r2, fold_mae = [], []
    fold_details = []

    for fold_i, (train_idx, test_idx) in enumerate(CV.split(x_selected)):
        x_tr = x_selected.iloc[train_idx]
        y_tr = y_final.iloc[train_idx]
        x_te = x_selected.iloc[test_idx]
        y_te_orig = y_raw.iloc[test_idx]

        if "xgb" in best_candidate["name"]:
            fold_model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBRegressor(
                            **best_params,
                            objective="reg:squarederror",
                            tree_method="hist",
                            random_state=RANDOM_STATE,
                            verbosity=0,
                        ),
                    ),
                ]
            )
            fold_model.fit(x_tr, y_tr)
            pred_log = fold_model.predict(x_te)
        elif "ridge" in best_candidate["name"]:
            fold_model = Pipeline(
                [("scaler", StandardScaler()), ("model", Ridge(**best_params))]
            )
            fold_model.fit(x_tr, y_tr)
            pred_log = fold_model.predict(x_te)
        else:
            fold_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(**best_params, max_iter=5000)),
                ]
            )
            fold_model.fit(x_tr, y_tr)
            pred_log = fold_model.predict(x_te)

        if is_residual:
            pred_log = pred_log + lag1_log.iloc[test_idx].values

        pred_actual = np.expm1(pred_log)
        r2 = r2_score(y_te_orig, pred_actual)
        mae = mean_absolute_error(y_te_orig, pred_actual)
        fold_r2.append(r2)
        fold_mae.append(mae)
        fold_details.append(
            {
                "fold": fold_i,
                "test_start": str(y_raw.index[test_idx[0]]),
                "test_end": str(y_raw.index[test_idx[-1]]),
                "test_size": len(test_idx),
                "r2": float(r2),
                "mae": float(mae),
            }
        )

    mean_r2 = float(np.mean(fold_r2))
    std_r2 = float(np.std(fold_r2))
    mean_mae = float(np.mean(fold_mae))
    std_mae = float(np.std(fold_mae))

    train_pred_log = best_model.predict(x_selected)

    if is_residual:
        train_pred_log = train_pred_log + lag1_log.values
    train_pred_actual = np.expm1(train_pred_log)
    train_r2 = float(r2_score(y_raw, train_pred_actual))
    train_mae = float(mean_absolute_error(y_raw, train_pred_actual))

    # ----- Historical predictions -----
    historical = pd.DataFrame(
        {
            "year_month": y_raw.index.astype(str),
            "discharge_actual": y_raw.values,
            "discharge_predicted": train_pred_actual,
        }
    )

    # ----- Feature importance -----
    if "xgb" in best_candidate["name"]:
        importance = dict(
            zip(
                feature_names,
                best_model.named_steps["model"].feature_importances_.tolist(),
                strict=False,
            )
        )
    else:
        coefs = np.abs(best_model.named_steps["model"].coef_)
        total = coefs.sum() if coefs.sum() > 0 else 1.0
        importance = dict(zip(feature_names, (coefs / total).tolist(), strict=False))
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )

    # ----- Feature stats -----
    feature_stats = {
        col: {
            "mean": float(x_selected[col].mean()),
            "std": float(x_selected[col].std()),
        }
        for col in feature_names
    }

    _export_onnx(best_model, feature_names)

    # ----- Save artifacts -----
    cv_results = {
        "model_id": MODEL_ID,
        "formulation": best_candidate["formulation"],
        "target_transform": "log1p",
        "winner": best_candidate["name"],
        "competition": candidates,
        "window_start": str(y_raw.index[0]),
        "window_end": str(y_raw.index[-1]),
        "n_rows": len(y_raw),
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


def _tune_xgboost(x: pd.DataFrame, y: pd.Series) -> tuple:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    verbosity=0,
                ),
            ),
        ]
    )
    prefixed = {f"model__{k}": v for k, v in XGB_PARAM_SPACE.items()}
    search = RandomizedSearchCV(
        pipe,
        prefixed,
        n_iter=60,
        cv=CV,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x, y)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params


def _tune_ridge(x: pd.DataFrame, y: pd.Series) -> tuple:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    search = RandomizedSearchCV(
        pipe,
        RIDGE_PARAM_SPACE,
        n_iter=5,
        cv=CV,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x, y)
    best_alpha = search.best_params_["model__alpha"]
    return search.best_estimator_, {"alpha": best_alpha}


def _tune_elasticnet(x: pd.DataFrame, y: pd.Series) -> tuple:
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("model", ElasticNet(max_iter=5000))]
    )
    search = RandomizedSearchCV(
        pipe,
        ELASTICNET_PARAM_SPACE,
        n_iter=20,
        cv=CV,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x, y)
    best_params = {
        "alpha": search.best_params_["model__alpha"],
        "l1_ratio": search.best_params_["model__l1_ratio"],
    }
    return search.best_estimator_, best_params


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def _export_onnx(pipe: Pipeline, feature_names: list[str]) -> None:
    try:
        initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_sklearn(
            pipe, initial_types=initial_type, target_opset={"": 12, "ai.onnx.ml": 1}
        )
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


def _save_json(obj: json, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Model 4: Surface Water Conditions (Discharge)")
    print(f"{'='*60}\n")

    results = train_and_evaluate()

    print(f"\n  Winner     : {results['winner']} ({results['formulation']})")
    print(f"  Transform  : {results['target_transform']}")
    print(f"  Window     : {results['window_start']} → {results['window_end']}")
    print(f"  Rows       : {results['n_rows']}")
    print(
        f"  Features   : {results['n_features']} "
        f" (dropped {results['n_features_dropped']})"
    )
    print(f"  CV R²      : {results['cv_mean_r2']:.4f} ± {results['cv_std_r2']:.4f}")
    print(f"  CV MAE     : {results['cv_mean_mae']:.4f} ± {results['cv_std_mae']:.4f}")
    print(f"  Train R²   : {results['train_r2']:.4f}")
    print(f"  Train MAE  : {results['train_mae']:.4f}")
    print(f"\n  Best params: {results['best_params']}")
    print(f"\n  Artifacts saved to {MODEL_DIR}/")
