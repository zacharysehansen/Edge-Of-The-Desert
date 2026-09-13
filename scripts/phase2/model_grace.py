"""
model_grace.py
--------------
Model 2: GRACE Groundwater Storage Anomaly — monthly regression.

Residual-over-lag1 formulation:
  - Train target: grace_groundwater_anomaly - grace_groundwater_anomaly_lag1
  - Prediction: predicted_residual + grace_groundwater_anomaly_lag1

THE ESTIMATOR IS A RIDGE REGRESSION ON FOUR PHYSICAL INPUTS, NOT AN XGBOOST.
PHASE3_PLAN.md §25-§26 (2026-09-12): the month-to-month change in GRACE storage is
largely the land-surface storage change GLDAS observes (r = +0.62, slope +0.8 m/m),
and a standardised ridge on {GLDAS storage change, rain, last month's rain,
temperature anomaly} scores +0.2845 target R² / +0.5664 level R² / skill +0.2015
under the same nested folds where the 45-feature XGBoost scored +0.0210 / +0.3372 /
-0.0277 — 4 of 5 folds, t = +2.52, REAL by the rule declared before the run. The
feature list lives in features.py (`fixed_features`) and is not to be edited here.

The XGBoost search is kept below as `_make_xgb_search` (aliased `_make_search`)
ONLY so that the experiment scripts (§11.5, §23-§26) can still reproduce the
historical "shipped" arm. It is not used to train anything.

TimeSeriesSplit(n_splits=5) outer cross-validation; the ridge penalty is chosen by
an inner TimeSeriesSplit(3) inside every training fold.
Export: ONNX (skl2onnx, scaler + ridge in one graph) + artifacts to model/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from scripts.phase2.features import build_all
from scripts.phase2.metrics import nested_cv_evaluate, persistence_r2, skill_score

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "grace"
N_SPLITS = 5
INNER_SPLITS = 3
CV = TimeSeriesSplit(n_splits=N_SPLITS)
RANDOM_STATE = 42

# Ridge penalty grid, searched inside each training fold (§26).
ALPHAS = np.logspace(-3, 3, 13)

# ---------------------------------------------------------------------------
# Legacy XGBoost search — reproduces the pre-§26 shipped arm in the experiments.
# ---------------------------------------------------------------------------
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


def _make_xgb_search() -> RandomizedSearchCV:
    """The XGBoost tuner GRACE shipped with until §26. Experiments only."""
    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    return RandomizedSearchCV(
        base_model,
        PARAM_SPACE,
        n_iter=60,
        cv=TimeSeriesSplit(n_splits=INNER_SPLITS),
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )


_make_search = _make_xgb_search


# ---------------------------------------------------------------------------
# The shipping estimator
# ---------------------------------------------------------------------------
def _make_ridge() -> Pipeline:
    """Standardised ridge; alpha chosen on an INNER time-series split."""
    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=INNER_SPLITS)),
    )


def _fit_predict(
    x_tr: pd.DataFrame, y_tr: pd.Series, x_te: pd.DataFrame
) -> np.ndarray:
    """Fit on the training fold only (alpha included), predict the residual for x_te."""
    return _make_ridge().fit(x_tr, y_tr).predict(x_te)


def train_and_evaluate() -> dict:
    """Build, evaluate, fit and export the GRACE model."""
    datasets = build_all()
    x, y = datasets[MODEL_ID]

    lag1_col = "grace_groundwater_anomaly_lag1"
    lag1 = x[lag1_col].copy()
    x_train = x.drop(columns=[lag1_col])
    feature_names = list(x_train.columns)

    y_residual = y - lag1

    scores = nested_cv_evaluate(
        fit_predict=_fit_predict,
        x=x_train,
        y_level=y,
        y_target=y_residual,
        cv=CV,
        anchor=lag1,
    )
    baseline_r2 = persistence_r2(y_level=y, lag1_level=lag1, cv=CV)

    mean_r2 = scores["cv_mean_r2"]
    std_r2 = scores["cv_std_r2"]
    mean_mae = scores["cv_mean_mae"]
    std_mae = scores["cv_std_mae"]
    fold_details = scores["folds"]

    # The exported model is refit on everything — only the *score* has to be nested.
    best_model = _make_ridge().fit(x_train, y_residual)
    ridge = best_model.named_steps["ridgecv"]
    scaler = best_model.named_steps["standardscaler"]
    best_params = {"alpha": float(ridge.alpha_)}

    # overfit diagnostic
    train_pred_actual = best_model.predict(x_train) + lag1.values
    train_r2 = float(r2_score(y, train_pred_actual))
    train_mae = float(mean_absolute_error(y, train_pred_actual))

    historical = pd.DataFrame(
        {
            "year_month": y.index.astype(str),
            "grace_actual": y.values,
            "grace_predicted": train_pred_actual,
        }
    )

    # Importance for a linear model: |standardised coefficient|, normalised to sum
    # to one, so top_inputs.py reads it the same way it reads tree importances.
    abs_coef = np.abs(ridge.coef_)
    share = abs_coef / abs_coef.sum() if abs_coef.sum() > 0 else abs_coef
    sorted_importance = dict(
        sorted(
            zip(feature_names, share.tolist(), strict=True), key=lambda kv: kv[1], reverse=True
        )
    )
    coefficients = {
        "intercept": float(ridge.intercept_),
        "standardised": dict(zip(feature_names, ridge.coef_.tolist(), strict=True)),
        "raw": dict(
            zip(feature_names, (ridge.coef_ / scaler.scale_).tolist(), strict=True)
        ),
        "scaler_mean": dict(zip(feature_names, scaler.mean_.tolist(), strict=True)),
        "scaler_scale": dict(zip(feature_names, scaler.scale_.tolist(), strict=True)),
    }

    feature_stats = {
        col: {"mean": float(x_train[col].mean()), "std": float(x_train[col].std())}
        for col in feature_names
    }

    _export_onnx(best_model, feature_names)

    cv_results = {
        "model_id": MODEL_ID,
        "estimator": "ridge on four physical inputs (PHASE3_PLAN.md §26)",
        "formulation": "residual_over_lag1",
        "cv_method": (
            f"nested TimeSeriesSplit({N_SPLITS} outer / {INNER_SPLITS} inner)"
        ),
        "window_start": str(y.index[0]),
        "window_end": str(y.index[-1]),
        "n_rows": len(y),
        "n_features": len(feature_names),
        "best_params": best_params,
        "coefficients": coefficients,
        "cv_mean_r2": mean_r2,
        "cv_std_r2": std_r2,
        "cv_mean_mae": mean_mae,
        "cv_std_mae": std_mae,
        "cv_target_r2": scores["cv_target_r2"],
        "baseline_lag1_r2": baseline_r2,
        "skill_r2": skill_score(mean_r2, baseline_r2),
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


def _export_onnx(model: Pipeline, feature_names: list[str]) -> None:
    """Export the scaler + ridge pipeline as one ONNX graph. The frontend feeds RAW
    feature values (models.js does not standardise), so the scaler must be inside."""
    initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=17)
    path = MODEL_DIR / f"{MODEL_ID}.onnx"
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  ONNX exported → {path}")

    # Round-trip check: the graph must reproduce sklearn on a real row.
    import onnxruntime as ort  # noqa: PLC0415

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    probe = np.zeros((1, len(feature_names)), dtype=np.float32)
    got = float(sess.run(None, {sess.get_inputs()[0].name: probe})[0].ravel()[0])
    want = float(model.predict(pd.DataFrame(probe, columns=feature_names))[0])
    if abs(got - want) > 1e-5:  # noqa: PLR2004
        raise RuntimeError(f"ONNX round-trip mismatch: onnx {got:.8f} vs sklearn {want:.8f}")


def _save_json(obj: json, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    print(f"\n{'='*60}")
    print("  Model 2: GRACE Groundwater Storage Anomaly")
    print(f"{'='*60}\n")

    results = train_and_evaluate()

    print(f"\n  Window       : {results['window_start']} → {results['window_end']}")
    print(f"  Rows         : {results['n_rows']}")
    print(f"  Features     : {results['n_features']}")
    print(f"  Estimator    : {results['estimator']}  alpha={results['best_params']['alpha']:g}")
    print(
        f"  CV R² (level): {results['cv_mean_r2']:.4f}"
        f" ± {results['cv_std_r2']:.4f}"
    )
    print(f"  CV R² (target): {results['cv_target_r2']:.4f}")
    print(
        f"  CV MAE       : {results['cv_mean_mae']:.6f}"
        f" ± {results['cv_std_mae']:.6f}"
    )
    print(
        f"  Persistence  : {results['baseline_lag1_r2']:.4f}   (lag1, same folds)"
    )
    print(f"  Skill        : {results['skill_r2']:+.4f}")
    print(f"  Train R²     : {results['train_r2']:.4f}  (overfit diagnostic)")


if __name__ == "__main__":
    main()
