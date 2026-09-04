"""
model_wildlife.py
-----------------
Model 6: Wildlife Abundance Index — annual regression.

Multi-model competition: Ridge, ElasticNet, and XGBoost.
With only ~20 rows, linear models with heavy regularization often
outperform trees because they have far fewer effective parameters.

Leave-One-Out cross-validation for final evaluation.
Reports both R² and Spearman correlation as reliability indicators.

Usage
-----
    python -m scripts.phase2.model_wildlife
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase2.features import build_all
from scripts.phase2.metrics import loo_evaluate

MODEL_ID = "wildlife"
RANDOM_STATE = 42

CV_SEARCH = TimeSeriesSplit(n_splits=3)


# ---------------------------------------------------------------------------
# Model families
# ---------------------------------------------------------------------------


def _build_ridge() -> tuple[Pipeline, dict]:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=RANDOM_STATE)),
        ]
    )
    param_grid = {
        "model__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    }
    return pipe, param_grid


def _build_elasticnet() -> tuple[Pipeline, dict]:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(random_state=RANDOM_STATE, max_iter=5000)),
        ]
    )
    param_grid = {
        "model__alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    return pipe, param_grid


def _build_xgboost() -> tuple[Pipeline, dict]:
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
                    max_depth=2,
                ),
            ),
        ]
    )
    param_grid = {
        "model__n_estimators": [30, 50, 80, 100],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.4, 0.5, 0.6, 0.7],
        "model__reg_alpha": [0.1, 0.5, 1.0, 2.0],
        "model__reg_lambda": [1, 3, 5, 10],
        "model__min_child_weight": [1, 3, 5],
    }
    return pipe, param_grid


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------


def _search_best(  # noqa: PLR0913
    name: str,
    pipe: Pipeline,
    param_grid: dict,
    x: pd.DataFrame,
    y: pd.Series,
    use_random: bool = False,
) -> tuple[Pipeline, dict, float]:
    """Run CV search, return (best_pipe, best_params, best_inner_score)."""
    if use_random:
        search = RandomizedSearchCV(
            pipe,
            param_grid,
            n_iter=25,
            cv=CV_SEARCH,
            scoring="r2",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
    else:
        search = GridSearchCV(
            pipe,
            param_grid,
            cv=CV_SEARCH,
            scoring="r2",
            n_jobs=-1,
            refit=True,
        )
    search.fit(x, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def _fit_predict(x_tr: pd.DataFrame, y_tr: pd.Series, x_te: pd.DataFrame) -> np.ndarray:
    """
    The whole model-building procedure, applied to one LOO training set.

    Both the hyperparameter search and the Ridge/ElasticNet/XGBoost choice happen on
    the 19 training years only. Previously both were done on all 20 years and then
    LOO-scored over the same data, so every held-out year had already helped pick the
    model that predicted it.
    """
    scored = []
    for name, builder, use_random in (
        ("Ridge", _build_ridge, False),
        ("ElasticNet", _build_elasticnet, False),
        ("XGBoost", _build_xgboost, True),
    ):
        pipe, grid = builder()
        best_pipe, _, inner_score = _search_best(
            name, pipe, grid, x_tr, y_tr, use_random
        )
        scored.append((inner_score, name, best_pipe))

    _, _, winner_pipe = max(scored, key=lambda t: t[0])
    winner_pipe = clone(winner_pipe)
    winner_pipe.fit(x_tr, y_tr)
    return winner_pipe.predict(x_te)


def train_and_evaluate() -> dict:
    """Run multi-model competition and export the winner."""
    datasets = build_all()
    x, y = datasets[MODEL_ID]
    feature_names = list(x.columns)

    # ----- Honest score: nested LOO. Tuning + model choice happen per training set.
    print("  Nested LOO (tuning + model choice inside each fold)...")
    honest = loo_evaluate(fit_predict=_fit_predict, x=x, y_level=y)
    honest_preds = np.asarray(honest["predictions"])
    honest_actuals = np.asarray(honest["actuals"])
    sp_corr, sp_p = spearmanr(honest_actuals, honest_preds)
    loo_r2 = honest["loo_r2"]
    loo_mae = honest["loo_mae"]
    print(f"  Nested LOO R² = {loo_r2:.4f}   Spearman ρ = {sp_corr:.4f} (p={sp_p:.4f})")

    # ----- Final model for export: same procedure, refit on all 20 years -----
    candidates = []
    for name, builder, use_random in (
        ("Ridge", _build_ridge, False),
        ("ElasticNet", _build_elasticnet, False),
        ("XGBoost", _build_xgboost, True),
    ):
        print(f"  Running {name}...")
        pipe, params = builder()
        best_pipe, best_params, inner_score = _search_best(
            name, pipe, params, x, y, use_random
        )
        candidates.append(
            {
                "name": name,
                "pipe": best_pipe,
                "params": best_params,
                "inner_r2": inner_score,
            }
        )

    print(f"\n  {'Model':<14} {'inner CV R²':>12}")
    print(f"  {'-'*28}")
    for c in candidates:
        print(f"  {c['name']:<14} {c['inner_r2']:>12.4f}")

    winner = max(candidates, key=lambda c: c["inner_r2"])
    print(f"\n  Winner: {winner['name']}")

    best_pipe = winner["pipe"]

    # Train score
    train_pred = best_pipe.predict(x)
    train_r2 = float(r2_score(y, train_pred))
    train_mae = float(mean_absolute_error(y, train_pred))

    # Historical predictions — nested-LOO out-of-sample values, not the in-sample fit.
    historical = pd.DataFrame(
        {
            "year": y.index.astype(int),
            "abundance_actual": honest_actuals,
            "abundance_predicted_loo": honest_preds,
        }
    )

    # Feature importance
    sorted_importance = _extract_importance(best_pipe, feature_names, winner["name"])

    # Feature stats
    feature_stats = {
        col: {
            "mean": float(x[col].mean(skipna=True)),
            "std": float(x[col].std(skipna=True)),
        }
        for col in feature_names
    }

    # Export ONNX
    _export_onnx(best_pipe, feature_names, winner["name"])

    cv_results = {
        "model_id": MODEL_ID,
        "model_type": winner["name"],
        "formulation": "direct",
        "cv_method": "nested LeaveOneOut (tuning + model choice inside each fold)",
        "window_start": int(y.index[0]),
        "window_end": int(y.index[-1]),
        "n_rows": len(y),
        "n_features": len(feature_names),
        "best_params": {k: _jsonable(v) for k, v in winner["params"].items()},
        "loo_r2": loo_r2,
        "loo_mae": loo_mae,
        "spearman_corr": float(sp_corr),
        "spearman_p": float(sp_p),
        "train_r2": train_r2,
        "train_mae": train_mae,
        "all_candidates": [
            {"name": c["name"], "inner_cv_r2": c["inner_r2"]} for c in candidates
        ],
        "note": (
            f"Wildlife model trained on {len(y)} rows. Ridge/ElasticNet/XGBoost "
            "competition, with both the tuning and the model choice made inside each "
            "LOO fold. Treat Spearman rank correlation as the primary reliability "
            "indicator: at this sample size R² is dominated by a handful of points."
        ),
    }

    _save_json(cv_results, MODEL_DIR / f"{MODEL_ID}_cv_results.json")
    _save_json(feature_names, MODEL_DIR / f"{MODEL_ID}_feature_names.json")
    _save_json(feature_stats, MODEL_DIR / f"{MODEL_ID}_feature_stats.json")
    _save_json(sorted_importance, MODEL_DIR / f"{MODEL_ID}_feature_importance.json")
    historical.to_csv(MODEL_DIR / f"historical_{MODEL_ID}.csv", index=False)

    return cv_results


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------


def _extract_importance(
    pipe: Pipeline, feature_names: list[str], model_name: str
) -> dict:
    model = pipe.named_steps["model"]
    if model_name in ("Ridge", "ElasticNet"):
        coefs = np.abs(model.coef_)
        importance = dict(zip(feature_names, coefs.tolist(), strict=False))
    else:
        importance = dict(
            zip(feature_names, model.feature_importances_.tolist(), strict=False)
        )
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def _export_onnx(pipe: Pipeline, feature_names: list[str], model_name: str) -> None:
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


def _jsonable(v: any) -> int | float | any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def main() -> None:
    print(f"\n{'='*60}")
    print("  Model 6: Wildlife Abundance (Annual, Multi-Model, LOO-CV)")
    print(f"{'='*60}\n")

    results = train_and_evaluate()

    print(f"\n  Winner     : {results['model_type']}")
    print(f"  Window     : {results['window_start']} → {results['window_end']}")
    print(f"  Rows       : {results['n_rows']}")
    print(f"  Features   : {results['n_features']}")
    print(f"  LOO R²     : {results['loo_r2']:.4f}   (nested)")
    print(f"  LOO MAE    : {results['loo_mae']:.4f}")
    print(
        f"  Spearman   : {results['spearman_corr']:.4f}"
        f"  (p={results['spearman_p']:.4f})"
    )
    print(f"  Train R²   : {results['train_r2']:.4f}")
    print(f"  Train MAE  : {results['train_mae']:.4f}")
    print(f"\n  Best params: {results['best_params']}")
    print(f"\n  Artifacts saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
