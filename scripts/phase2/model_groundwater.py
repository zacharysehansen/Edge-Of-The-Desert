"""
model_groundwater.py
--------------------
Model 3: Groundwater Well Levels — monthly regression.

Residual-over-lag1, with an in-fold competition between Ridge, ElasticNet, XGBoost,
a tightly-regularised XGBoost, and an XGB+EN blend.

Scoring is **nested** (see scripts/phase2/metrics.py): feature selection, the choice
of candidate, and the hyperparameter search all happen inside `_fit_predict`, which
only ever sees the training fold. An earlier version made all three choices on the
full panel and then cross-validated over the same folds, which reports a max over
noisy draws rather than an out-of-sample score.

The direct formulation is not a candidate. It was tested and lost decisively (CV R²
of -2.2 to -20 versus +0.6 for residual), so it is treated as a settled design
decision rather than something to re-pick from the data on every fold.

Read `cv_target_r2` (R² on the residual) before `cv_mean_r2` (R² on the reconstructed
level). The level score is dominated by the lag1 anchor, which this model does not
predict; the residual score is the part it is responsible for.

Usage
-----
    python -m scripts.phase2.model_groundwater
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase2.features import build_all
from scripts.phase2.metrics import nested_cv_evaluate, persistence_r2, skill_score

MODEL_DIR = REPO_ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "groundwater"
N_SPLITS = 5
INNER_SPLITS = 3
CV = TimeSeriesSplit(n_splits=N_SPLITS)
RANDOM_STATE = 42

# Candidates for the in-fold competition. Fixed configs are used to pick a winner;
# the winner is then hyperparameter-tuned on the same inner split.
CANDIDATES = ("xgb", "xgb_tight", "blend", "ridge", "elastic")

COMPETITION_CONFIGS = {
    "xgb": {
        "n_estimators": 200,
        "max_depth": 2,
        "learning_rate": 0.02,
        "min_child_weight": 30,
        "reg_lambda": 30,
        "reg_alpha": 2.0,
        "subsample": 0.6,
        "colsample_bytree": 0.4,
    },
    "xgb_tight": {
        "n_estimators": 150,
        "max_depth": 2,
        "learning_rate": 0.02,
        "min_child_weight": 20,
        "reg_lambda": 20,
        "reg_alpha": 1.0,
        "subsample": 0.6,
        "colsample_bytree": 0.5,
    },
    "ridge": {"alpha": 1.0},
    "elastic": {"alpha": 0.05, "l1_ratio": 0.5},
}

XGB_PARAM_SPACE = {
    "n_estimators": [100, 150, 200],
    "max_depth": [2],
    "learning_rate": [0.01, 0.02, 0.05],
    "subsample": [0.6, 0.7, 0.8],
    "colsample_bytree": [0.4, 0.5, 0.7],
    "reg_alpha": [0.0, 1.0, 2.0],
    "reg_lambda": [10, 20, 50],
    "min_child_weight": [10, 20, 30],
}

RIDGE_PARAM_SPACE = {
    "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}

ELASTICNET_PARAM_SPACE = {
    "model__alpha": [0.001, 0.01, 0.1, 1.0],
    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

IMPORTANCE_THRESHOLD = 0.015
MAX_FEATURES = 16
BLEND_WEIGHT_XGB = 0.6


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def _select_features(x: pd.DataFrame, y: pd.Series, threshold: float) -> list[str]:
    """Fit a quick XGBoost and return the most important features.

    Keeps features above ``threshold``, then caps the set at ``MAX_FEATURES``
    (highest importance first) to keep the rows/feature ratio sane on a small
    dataset. Falls back to the top ``MAX_FEATURES`` if too few clear the bar.
    """
    model = XGBRegressor(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=10,
        reg_lambda=10,
        subsample=0.7,
        colsample_bytree=0.5,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(x, y)
    importances = dict(zip(x.columns, model.feature_importances_, strict=False))
    ranked = sorted(importances, key=lambda c: importances[c], reverse=True)
    selected = [c for c in ranked if importances[c] >= threshold]
    if len(selected) < 5:  # noqa: PLR2004
        selected = ranked[:MAX_FEATURES]
    return selected[:MAX_FEATURES]


class _Blend:
    """XGBoost + ElasticNet convex blend, with the surface of a single estimator."""

    def __init__(
        self,
        xgb_params: dict,
        en_params: dict,
        weight: float = BLEND_WEIGHT_XGB,
    ) -> None:
        self.weight = weight
        self.xgb = XGBRegressor(
            **xgb_params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        self.en = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(**en_params, max_iter=5000)),
            ]
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> _Blend:
        self.xgb.fit(x, y)
        self.en.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        xgb_pred = self.xgb.predict(x)
        en_pred = self.en.predict(x)
        return self.weight * xgb_pred + (1 - self.weight) * en_pred


def _build(name: str, params: dict | None = None) -> XGBRegressor | Pipeline | _Blend:
    """Instantiate one candidate. `params=None` uses the fixed competition config."""
    if name == "blend":
        if params is None:
            return _Blend(
                COMPETITION_CONFIGS["xgb_tight"], COMPETITION_CONFIGS["elastic"]
            )
        return _Blend(params["xgb_params"], params["en_params"])

    cfg = params if params is not None else COMPETITION_CONFIGS[name]
    if name in ("xgb", "xgb_tight"):
        return XGBRegressor(
            **cfg,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(**cfg))])
    if name == "elastic":
        return Pipeline(
            [("scaler", StandardScaler()), ("model", ElasticNet(**cfg, max_iter=5000))]
        )
    raise ValueError(f"Unknown candidate '{name}'")


def _compete(x: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> list[dict]:
    """Score every candidate on `cv` in residual space. Returns results best-first."""
    results = []
    for name in CANDIDATES:
        scores = []
        for train_idx, test_idx in cv.split(x):
            model = _build(name)
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(x.iloc[test_idx])
            scores.append(r2_score(y.iloc[test_idx], pred))
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
    if name == "blend":
        return _tune_blend(x, y, cv)
    if name in ("xgb", "xgb_tight"):
        return _tune_xgboost(x, y, cv)
    if name == "ridge":
        return _tune_ridge(x, y, cv)
    return _tune_elasticnet(x, y, cv)


def _fit_predict(x_tr: pd.DataFrame, y_tr: pd.Series, x_te: pd.DataFrame) -> np.ndarray:
    """
    The whole model-building procedure, applied to one training fold.

    Everything that looks at data happens here — feature selection, the candidate
    competition, and the hyperparameter search — so none of it can see the outer
    test fold. Inner scoring is in residual space, which is the target being fit.
    """
    inner = TimeSeriesSplit(n_splits=INNER_SPLITS)

    selected = _select_features(x_tr, y_tr, IMPORTANCE_THRESHOLD)
    x_tr_sel = x_tr[selected]

    winner = _compete(x_tr_sel, y_tr, inner)[0]["name"]
    model, _ = _tune(winner, x_tr_sel, y_tr, inner)
    model.fit(x_tr_sel, y_tr)

    return model.predict(x_te[selected])


def train_and_evaluate() -> dict:  # noqa: C901, PLR0912, PLR0915
    """Build, tune, evaluate, and export the groundwater well level model."""
    datasets = build_all()
    x, y = datasets[MODEL_ID]

    lag1_col = "depth_to_water_anomaly_ft_lag1"
    lag1 = x[lag1_col].copy()
    x_full = x.drop(columns=[lag1_col])

    y_residual = y - lag1

    # ----- Honest score: every data-dependent choice is made inside the fold -----
    scores = nested_cv_evaluate(
        fit_predict=_fit_predict,
        x=x_full,
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
    # Selecting/competing/tuning on everything is correct here (we want the best model
    # to ship); the cost of doing so is already priced into the nested score above.
    selected = _select_features(x_full, y_residual, IMPORTANCE_THRESHOLD)
    x_selected = x_full[selected]
    n_dropped = len(x_full.columns) - len(selected)
    feature_names = list(x_selected.columns)
    print(f"  Feature selection: {len(selected)} kept, {n_dropped} dropped")

    candidates = _compete(x_selected, y_residual, CV)
    for c in candidates:
        print(f"    {c['name']:<12} residual R²={c['mean_r2']:.4f} ± {c['std_r2']:.4f}")

    best_name = candidates[0]["name"]
    is_blend = best_name == "blend"
    print(f"\n  Winner: {best_name}")

    best_model, best_params = _tune(best_name, x_selected, y_residual, CV)
    best_model.fit(x_selected, y_residual)

    train_pred = best_model.predict(x_selected)
    train_pred_actual = train_pred + lag1.values
    train_r2 = float(r2_score(y, train_pred_actual))
    train_mae = float(mean_absolute_error(y, train_pred_actual))

    historical = pd.DataFrame(
        {
            "year_month": y.index.astype(str),
            "groundwater_actual": y.values,
            "groundwater_predicted": train_pred_actual,
        }
    )

    if is_blend:
        xgb_imp = best_model.xgb.feature_importances_
        en_coefs = np.abs(best_model.en.named_steps["model"].coef_)
        en_total = en_coefs.sum() if en_coefs.sum() > 0 else 1.0
        en_imp = en_coefs / en_total
        blended_imp = BLEND_WEIGHT_XGB * xgb_imp + (1 - BLEND_WEIGHT_XGB) * en_imp
        importance = dict(zip(feature_names, blended_imp.tolist(), strict=False))
    elif best_name.startswith("xgb"):
        importance = dict(
            zip(feature_names, best_model.feature_importances_.tolist(), strict=False)
        )
    else:
        if hasattr(best_model, "named_steps"):
            coefs = np.abs(best_model.named_steps["model"].coef_)
        else:
            coefs = np.abs(best_model.coef_)
        total = coefs.sum() if coefs.sum() > 0 else 1.0
        importance = dict(zip(feature_names, (coefs / total).tolist(), strict=False))
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )

    feature_stats = {
        col: {
            "mean": float(x_selected[col].mean()),
            "std": float(x_selected[col].std()),
        }
        for col in feature_names
    }

    if is_blend:
        _export_onnx_xgb(best_model.xgb, feature_names)
        print(
            "  [INFO] Blend: exported XGBoost component to ONNX"
            " (ElasticNet component requires skl2onnx separately)"
        )
    elif best_name.startswith("xgb"):
        _export_onnx_xgb(best_model, feature_names)
    else:
        _export_onnx_sklearn(best_model, feature_names)

    _verify_onnx_not_constant(x_selected)

    cv_results = {
        "model_id": MODEL_ID,
        "formulation": "residual_over_lag1",
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


def _tune_blend(x: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> tuple:
    """Tune both arms of the blend, then hand back a single fitted-able _Blend."""
    _, xgb_params = _tune_xgboost(x, y, cv)
    _, en_params = _tune_elasticnet(x, y, cv)
    params = {
        "xgb_params": xgb_params,
        "en_params": en_params,
        "blend_weight_xgb": BLEND_WEIGHT_XGB,
    }
    return _Blend(xgb_params, en_params), params


def _tune_xgboost(x: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> tuple:
    base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    search = RandomizedSearchCV(
        base,
        XGB_PARAM_SPACE,
        n_iter=60,
        cv=cv,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x, y)
    return search.best_estimator_, search.best_params_


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


def _verify_onnx_not_constant(x_selected: pd.DataFrame, min_std: float = 1e-4) -> None:
    """Fail loudly if the exported ONNX returns a (near-)constant prediction.

    A degenerate model (single constant leaf) silently breaks the frontend:
    the output never responds to slider changes. We re-load the model we just
    wrote and check that its predictions vary across the training rows.
    """
    path = MODEL_DIR / f"{MODEL_ID}.onnx"
    sess = ort.InferenceSession(str(path))
    input_name = sess.get_inputs()[0].name
    feed = {input_name: x_selected.to_numpy(dtype=np.float32)}
    preds = sess.run(None, feed)[0].ravel()
    pred_std = float(np.std(preds))
    pred_range = float(np.ptp(preds))
    print(
        f"  ONNX sanity check: pred std={pred_std:.6g}, range={pred_range:.6g}"
        f" over {len(preds)} training rows"
    )
    if pred_std < min_std:
        msg = (
            f"Exported ONNX is effectively constant (std={pred_std:.6g} <"
            f" {min_std}). The model learned no signal and will not respond to"
            " inputs. Loosen regularization in XGB_PARAM_SPACE / the candidate"
            " configs and retrain."
        )
        raise RuntimeError(msg)


def _export_onnx_xgb(model: XGBRegressor, feature_names: list[str]) -> None:
    """Export XGBoost to ONNX."""

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


def _export_onnx_sklearn(model: XGBRegressor, feature_names: list[str]) -> None:
    """Export sklearn pipeline to ONNX."""
    initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    path = MODEL_DIR / f"{MODEL_ID}.onnx"
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  ONNX exported → {path}")


def _save_json(obj: json, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    print(f"\n{'='*60}")
    print("  Model 3: Groundwater Well Levels")
    print(f"{'='*60}\n")

    results = train_and_evaluate()

    print(f"\n  Winner       : {results['winner']} ({results['formulation']})")
    print(f"  Window       : {results['window_start']} → {results['window_end']}")
    print(f"  Rows         : {results['n_rows']}")
    print(
        f"  Features     : {results['n_features']}"
        f" (dropped {results['n_features_dropped']})"
    )
    print(f"  CV R² (level): {results['cv_mean_r2']:.4f} ± {results['cv_std_r2']:.4f}")
    print(
        f"  CV MAE       : {results['cv_mean_mae']:.4f}"
        f" ± {results['cv_std_mae']:.4f}"
    )
    print(
        f"  Persistence  : {results['baseline_lag1_r2']:.4f}   (lag1, same folds)"
    )
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


if __name__ == "__main__":
    main()
