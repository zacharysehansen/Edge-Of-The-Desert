"""
model_surface_water.py
----------------------
Model 4: Surface Water Conditions (Discharge) — monthly regression.

Multi-model competition:
  - Target: discharge_log_anomaly — the per-gage log anomaly index built in Phase 1
  - Formulations: direct vs residual-over-lag1
  - Algorithms: Ridge, ElasticNet, XGBoost (6 candidates)
  - Feature selection: drops low-importance features after initial fit

TimeSeriesSplit(n_splits=5) cross-validation.
Export: ONNX + artifacts to model/.

Note on the target: this script used to predict log1p(discharge_cfs_mean), a mean of raw
discharge across whichever gages reported that month. Two problems with that. Discharge
spans four orders of magnitude across gages, so the raw mean was substantially one gage
(09525503 alone supplied 45% of it); and a mean over a changing roster moves when the
roster moves. Phase 1 now centers each gage on its own long-term mean *in log space*
before averaging, which fixes both. The log therefore already lives inside the target —
there is no log1p here anymore, and the target is signed, so applying one would be a
domain error rather than a transform.

gage_height_ft_mean is excluded due to missing values in the source data.

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
from scripts.phase2.metrics import nested_cv_evaluate, persistence_r2, skill_score

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "surface_water"
N_SPLITS = 5
INNER_SPLITS = 3
CV = TimeSeriesSplit(n_splits=N_SPLITS)
RANDOM_STATE = 42

# Candidates for the in-fold competition, scored in residual (log-anomaly) space.
CANDIDATES = ("xgb", "ridge", "elastic")

COMPETITION_CONFIGS = {
    "xgb": {
        "n_estimators": 600,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 7,
        "reg_lambda": 5,
        "reg_alpha": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
    },
    "ridge": {"alpha": 1.0},
    "elastic": {"alpha": 0.1, "l1_ratio": 0.5},
}

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


def _build(name: str, params: dict | None = None) -> Pipeline:
    """Instantiate one candidate. `params=None` uses the fixed competition config."""
    cfg = params if params is not None else COMPETITION_CONFIGS[name]
    if name == "xgb":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        **cfg,
                        objective="reg:squarederror",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        verbosity=0,
                    ),
                ),
            ]
        )
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(**cfg))])
    if name == "elastic":
        return Pipeline(
            [("scaler", StandardScaler()), ("model", ElasticNet(**cfg, max_iter=5000))]
        )
    raise ValueError(f"Unknown candidate '{name}'")


def _compete(x: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> list[dict]:
    """Score each candidate on `cv` in residual (log-anomaly) space, best-first."""
    results = []
    for name in CANDIDATES:
        scores = []
        for train_idx, test_idx in cv.split(x):
            model = _build(name)
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            scores.append(r2_score(y.iloc[test_idx], model.predict(x.iloc[test_idx])))
        results.append(
            {
                "name": name,
                "mean_r2": float(np.mean(scores)),
                "std_r2": float(np.std(scores)),
            }
        )
    return sorted(results, key=lambda c: c["mean_r2"], reverse=True)


def _tune(name: str, x: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> tuple:
    """Hyperparameter-tune one candidate on `cv`."""
    if name == "xgb":
        return _tune_xgboost(x, y, cv)
    if name == "ridge":
        return _tune_ridge(x, y, cv)
    return _tune_elasticnet(x, y, cv)


def _fit_predict(x_tr: pd.DataFrame, y_tr: pd.Series, x_te: pd.DataFrame) -> np.ndarray:
    """
    The whole model-building procedure, applied to one training fold.

    Feature selection, the candidate competition, and the hyperparameter search all
    happen here, so none of them can see the outer test fold. Works throughout in
    residual space; the caller adds the lag1 anchor back to reconstruct the index.
    """
    inner = TimeSeriesSplit(n_splits=INNER_SPLITS)

    selected = _select_features(x_tr, y_tr, IMPORTANCE_THRESHOLD)
    x_tr_sel = x_tr[selected]

    winner = _compete(x_tr_sel, y_tr, inner)[0]["name"]
    model, _ = _tune(winner, x_tr_sel, y_tr, inner)
    model.fit(x_tr_sel, y_tr)

    return model.predict(x_te[selected])


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------


def train_and_evaluate() -> dict:  # noqa: C901, PLR0912, PLR0915
    """Build, tune, evaluate, and export the surface water model."""
    datasets = build_all()
    x, y = datasets[MODEL_ID]

    # Separate lag1 column
    lag1_col = "discharge_log_anomaly_lag1"
    lag1 = x[lag1_col].copy()
    x_base = x.drop(columns=[lag1_col])

    # Residual-over-lag1. The target is already a log-space anomaly, so this is a
    # month-over-month change in log flow — i.e. a ratio — which is the right way to
    # represent the multiplicative flood dynamics. The direct formulation was tested
    # and lost, so the formulation is a settled design decision rather than something
    # re-picked from the data each fold.
    y_residual = y - lag1

    # ----- Honest score: every data-dependent choice is made inside the fold -----
    # No `inverse`: the anomaly index *is* the level here. There is no log1p to undo.
    scores = nested_cv_evaluate(
        fit_predict=_fit_predict,
        x=x_base,
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

    # ----- Final model for export: refit the same procedure on the full panel -----
    selected = _select_features(x_base, y_residual, IMPORTANCE_THRESHOLD)
    x_selected = x_base[selected]
    n_dropped = len(x_base.columns) - len(selected)
    feature_names = list(x_selected.columns)
    print(f"  Feature selection: {len(selected)} kept, {n_dropped} dropped")

    candidates = _compete(x_selected, y_residual, CV)
    for c in candidates:
        print(
            f"    {c['name']:<10} residual R²={c['mean_r2']:.4f} ± {c['std_r2']:.4f}"
        )

    best_name = candidates[0]["name"]
    print(f"\n  Winner: {best_name}")

    best_model, best_params = _tune(best_name, x_selected, y_residual, CV)
    best_model.fit(x_selected, y_residual)

    train_pred = best_model.predict(x_selected) + lag1.values
    train_r2 = float(r2_score(y, train_pred))
    train_mae = float(mean_absolute_error(y, train_pred))

    # ----- Historical predictions -----
    historical = pd.DataFrame(
        {
            "year_month": y.index.astype(str),
            "discharge_actual": y.values,
            "discharge_predicted": train_pred,
        }
    )

    # ----- Feature importance -----
    if best_name == "xgb":
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
        "formulation": "residual_over_lag1",
        "target_transform": "none",
        "cv_method": f"nested TimeSeriesSplit({N_SPLITS} outer / {INNER_SPLITS} inner)",
        "winner": best_name,
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
        # R² on the residual — the part the model actually predicts.
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


# ---------------------------------------------------------------------------
# Tuning functions
# ---------------------------------------------------------------------------


def _tune_xgboost(x: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> tuple:
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
        cv=cv,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x, y)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params


def _tune_ridge(x: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> tuple:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    search = RandomizedSearchCV(
        pipe,
        RIDGE_PARAM_SPACE,
        n_iter=5,
        cv=cv,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x, y)
    best_alpha = search.best_params_["model__alpha"]
    return search.best_estimator_, {"alpha": best_alpha}


def _tune_elasticnet(x: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> tuple:
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("model", ElasticNet(max_iter=5000))]
    )
    search = RandomizedSearchCV(
        pipe,
        ELASTICNET_PARAM_SPACE,
        n_iter=20,
        cv=cv,
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
    print(f"  CV R² (level): {results['cv_mean_r2']:.4f} ± {results['cv_std_r2']:.4f}")
    print(
        f"  CV MAE       : {results['cv_mean_mae']:.4f}"
        f" ± {results['cv_std_mae']:.4f}"
    )
    print(f"  Persistence  : {results['baseline_lag1_r2']:.4f}   (lag1, same folds)")
    print(f"  SKILL        : {results['skill_r2']:+.4f}   (level R² − persistence)")
    print(
        f"  CV R² resid  : {results['cv_target_r2']:.4f}"
        "   (the part the model predicts)"
    )
    print(f"  Train R²     : {results['train_r2']:.4f}")
    if results["skill_r2"] is not None and results["skill_r2"] <= 0:
        print(
            "\n  *** NO SKILL: this model does not beat copying last month's value. ***"
        )
    print(f"\n  Best params: {results['best_params']}")
    print(f"\n  Artifacts saved to {MODEL_DIR}/")
