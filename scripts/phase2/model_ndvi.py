"""
model_ndvi.py
-------------
Model 1: NDVI (Vegetation Health) — monthly regression.

Residual-over-lag1 formulation:
  - Train target: ndvi - ndvi_lag1
  - Prediction: predicted_residual + ndvi_lag1

TimeSeriesSplit(n_splits=5) cross-validation.
Export: ONNX + artifacts to model/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from scripts.phase2.features import build_all

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "ndvi"
N_SPLITS = 5
CV = TimeSeriesSplit(n_splits=N_SPLITS)
RANDOM_STATE = 42

PARAM_SPACE = {
    "n_estimators": [400, 600, 800, 1000],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.08],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_alpha": [0, 0.01, 0.05, 0.1],
    "reg_lambda": [0.5, 1, 2, 5],
    "min_child_weight": [1, 3, 5],
}


def train_and_evaluate() -> dict:
    """Build, tune, evaluate, and export the NDVI model."""
    datasets = build_all()
    x, y = datasets[MODEL_ID]

    lag1_col = "ndvi_lag1"
    lag1 = x[lag1_col].copy()
    x_train = x.drop(columns=[lag1_col])
    feature_names = list(x_train.columns)

    y_residual = y - lag1

    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    search = RandomizedSearchCV(
        base_model,
        PARAM_SPACE,
        n_iter=60,
        cv=CV,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_residual)
    best_model = search.best_estimator_
    best_params = search.best_params_

    fold_r2, fold_mae = [], []
    fold_details = []

    for fold_i, (train_idx, test_idx) in enumerate(CV.split(x_train)):
        x_tr = x_train.iloc[train_idx]
        y_tr = y_residual.iloc[train_idx]
        x_te = x_train.iloc[test_idx]
        y_te_actual = y.iloc[test_idx]
        lag1_te = lag1.iloc[test_idx]

        fold_model = XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        fold_model.fit(x_tr, y_tr)
        pred_residual = fold_model.predict(x_te)
        pred_actual = pred_residual + lag1_te.values

        r2 = r2_score(y_te_actual, pred_actual)
        mae = mean_absolute_error(y_te_actual, pred_actual)
        fold_r2.append(r2)
        fold_mae.append(mae)
        fold_details.append(
            {
                "fold": fold_i,
                "test_start": str(y.index[test_idx[0]]),
                "test_end": str(y.index[test_idx[-1]]),
                "test_size": len(test_idx),
                "r2": float(r2),
                "mae": float(mae),
            }
        )

    mean_r2 = float(np.mean(fold_r2))
    std_r2 = float(np.std(fold_r2))
    mean_mae = float(np.mean(fold_mae))
    std_mae = float(np.std(fold_mae))

    train_pred_residual = best_model.predict(x_train)
    train_pred_actual = train_pred_residual + lag1.values
    train_r2 = float(r2_score(y, train_pred_actual))
    train_mae = float(mean_absolute_error(y, train_pred_actual))

    hist_pred_residual = best_model.predict(x_train)
    hist_pred_actual = hist_pred_residual + lag1.values
    historical = pd.DataFrame(
        {
            "year_month": y.index.astype(str),
            "ndvi_actual": y.values,
            "ndvi_predicted": hist_pred_actual,
        }
    )

    importance = dict(
        zip(feature_names, best_model.feature_importances_.tolist(), strict=False)
    )
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )

    feature_stats = {
        col: {"mean": float(x_train[col].mean()), "std": float(x_train[col].std())}
        for col in feature_names
    }

    _export_onnx(best_model, feature_names)

    cv_results = {
        "model_id": MODEL_ID,
        "formulation": "residual_over_lag1",
        "window_start": str(y.index[0]),
        "window_end": str(y.index[-1]),
        "n_rows": len(y),
        "n_features": len(feature_names),
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


def _export_onnx(model: XGBRegressor, feature_names: list[str]) -> None:
    """Export to ONNX. Renames features to f0..fN for onnxmltools compatibility."""

    # onnxmltools requires numeric feature names
    #  clone with f0..fN
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


def _save_json(obj: json, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    print(f"\n{'='*60}")
    print("  Model 1: NDVI (Vegetation Health)")
    print(f"{'='*60}\n")

    results = train_and_evaluate()

    print(f"\n  Window     : {results['window_start']} → {results['window_end']}")
    print(f"  Rows       : {results['n_rows']}")
    print(f"  Features   : {results['n_features']}")
    print(f"  CV R²      : {results['cv_mean_r2']:.4f} ± {results['cv_std_r2']:.4f}")
    print(f"  CV MAE     : {results['cv_mean_mae']:.6f} ± {results['cv_std_mae']:.6f}")
    print(f"  Train R²   : {results['train_r2']:.4f}")
    print(f"  Train MAE  : {results['train_mae']:.6f}")
    print(f"\n  Best params: {results['best_params']}")
    print(f"\n  Artifacts saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
