"""
metrics.py
----------
Shared, honest evaluation for every Phase 2 model.

This module exists to enforce two rules that the per-model scripts kept getting
wrong in different ways.

**1. A residual model must be scored twice.**
The residual models predict ``y - lag1`` and reconstruct via ``pred + lag1``.
Reconstructed-level R² is therefore dominated by the lag1 anchor, which the model
does not predict. A model with *zero* skill still posts a high level R² simply
because persistence is a good forecast. Scoring the target the model was actually
trained on (the residual) is the only way to see that. Groundwater scored 0.61 at
the level and -1.34 on the residual: it had no skill and the level score hid it.

**2. Every choice made from the data must be made inside the fold.**
Feature selection, candidate-model choice, and hyperparameter search all peek at
the data. Doing any of them on the full panel and then reporting CV over the same
folds turns the score into a max over many noisy draws. ``nested_cv_evaluate``
takes a single ``fit_predict`` callable that receives *only* the training fold and
must do all of its own selecting, competing, and tuning in there.

The headline metric is the **skill score**: R² minus the lag1-persistence R² on the
same folds. For an autoregressive target that is the only number that answers the
question "is this model doing anything a copy of last month wouldn't?"
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import BaseCrossValidator, LeaveOneOut

# fit_predict(x_train, y_train_target, x_test) -> predictions in TARGET space.
# Must perform ALL data-dependent choices (feature selection, model competition,
# hyperparameter search) using only the arguments it is given.
FitPredict = Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray]


def skill_score(model_r2: float, baseline_r2: float) -> float | None:
    """
    R² gained over the lag1-persistence baseline.

    Negative means the model is worse than doing nothing.
    """
    if model_r2 is None or baseline_r2 is None:
        return None
    if not np.isfinite(model_r2) or not np.isfinite(baseline_r2):
        return None
    return float(model_r2 - baseline_r2)


def persistence_r2(
    *,
    y_level: pd.Series,
    lag1_level: pd.Series,
    cv: BaseCrossValidator,
) -> float:
    """
    Lag1-persistence R² on the *same* folds the model is scored on.

    Computed here rather than read from baselines.json so a model can never be
    compared against a baseline measured on a different split.
    """
    scores = []
    for _, test_idx in cv.split(y_level.to_frame()):
        y_te = y_level.iloc[test_idx]
        p_te = lag1_level.iloc[test_idx]
        valid = y_te.notna() & p_te.notna()
        if valid.sum() < 2:  # noqa: PLR2004
            continue
        scores.append(r2_score(y_te[valid], p_te[valid]))
    return float(np.mean(scores)) if scores else float("nan")


def nested_cv_evaluate(  # noqa: PLR0913
    *,
    fit_predict: FitPredict,
    x: pd.DataFrame,
    y_level: pd.Series,
    y_target: pd.Series,
    cv: BaseCrossValidator,
    anchor: pd.Series | None = None,
    inverse: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """
    Cross-validate a model without letting any fold see its own test data.

    Parameters
    ----------
    fit_predict
        Fits on (x_train, y_train_target) and predicts x_test, in target space.
        All data-dependent choices must happen inside this call.
    x
        Feature matrix. Must NOT contain the anchor column.
    y_level
        The real-world quantity being predicted (ft, cfs, NDVI, …). Scores that get
        reported to a user are computed against this.
    y_target
        What the model is actually trained on: ``y_level`` for a direct model, or
        ``y_level - anchor`` for a residual one (transformed space if applicable).
    anchor
        The lag1 series to add back when reconstructing the level. None if direct.
    inverse
        Maps reconstructed target space back to the level, e.g. ``np.expm1`` when the
        target was log1p-transformed. Applied after the anchor is added back.

    Returns
    -------
    dict with both the level scores (what a user sees) and the target scores (what the
    model is responsible for), plus per-fold detail.
    """
    level_r2, level_mae, target_r2, folds = [], [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(x)):
        x_tr, x_te = x.iloc[train_idx], x.iloc[test_idx]
        y_tr_target = y_target.iloc[train_idx]

        pred_target = np.asarray(fit_predict(x_tr, y_tr_target, x_te), dtype=float)

        # Score the target the model was trained on — the part it is responsible for.
        y_te_target = y_target.iloc[test_idx]
        fold_target_r2 = float(r2_score(y_te_target, pred_target))

        # Reconstruct the level for the user-facing score.
        reconstructed = pred_target
        if anchor is not None:
            reconstructed = reconstructed + anchor.iloc[test_idx].to_numpy()
        if inverse is not None:
            reconstructed = inverse(reconstructed)

        y_te_level = y_level.iloc[test_idx]
        fold_level_r2 = float(r2_score(y_te_level, reconstructed))
        fold_level_mae = float(mean_absolute_error(y_te_level, reconstructed))

        level_r2.append(fold_level_r2)
        level_mae.append(fold_level_mae)
        target_r2.append(fold_target_r2)
        folds.append(
            {
                "fold": fold_i,
                "test_start": str(y_level.index[test_idx[0]]),
                "test_end": str(y_level.index[test_idx[-1]]),
                "test_size": len(test_idx),
                "r2": fold_level_r2,
                "mae": fold_level_mae,
                "target_r2": fold_target_r2,
            }
        )

    return {
        "cv_mean_r2": float(np.mean(level_r2)),
        "cv_std_r2": float(np.std(level_r2)),
        "cv_mean_mae": float(np.mean(level_mae)),
        "cv_std_mae": float(np.std(level_mae)),
        "cv_target_r2": float(np.mean(target_r2)),
        "cv_target_std_r2": float(np.std(target_r2)),
        "folds": folds,
    }


def loo_evaluate(
    *,
    fit_predict: FitPredict,
    x: pd.DataFrame,
    y_level: pd.Series,
) -> dict:
    """
    Leave-one-out variant for the annual models, where n is too small to hold out
    a contiguous block.

    R² is pooled across all held-out points rather than averaged per fold: with one
    point per fold a per-fold R² is undefined.
    """
    preds, actuals = [], []
    for train_idx, test_idx in LeaveOneOut().split(x):
        pred = fit_predict(x.iloc[train_idx], y_level.iloc[train_idx], x.iloc[test_idx])
        preds.append(float(np.asarray(pred, dtype=float).ravel()[0]))
        actuals.append(float(y_level.iloc[test_idx[0]]))

    preds_arr, actuals_arr = np.asarray(preds), np.asarray(actuals)
    return {
        "loo_r2": float(r2_score(actuals_arr, preds_arr)),
        "loo_mae": float(mean_absolute_error(actuals_arr, preds_arr)),
        "n_points": len(preds),
        "predictions": preds,
        "actuals": actuals,
    }
