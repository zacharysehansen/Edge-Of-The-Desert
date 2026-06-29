"""
baselines.py
------------
Compute lag1-persistence and 3-period rolling-mean baselines for every
model using the same cross-validation splitter that the learned models use.

Results are saved to model/baselines.json and also returned as a dict
so model scripts can compare against them directly.

Usage
-----
    python scripts/phase2/baselines.py        # run standalone, prints table
    from scripts.phase2.baselines import compute_baselines
    baselines = compute_baselines(datasets)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, TimeSeriesSplit

from scripts.phase2.features import build_all

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
BASELINES_PATH = MODEL_DIR / "baselines.json"

# CV splitters — must match the model scripts
CV_SPLITTERS = {
    "ndvi": TimeSeriesSplit(n_splits=5),
    "grace": TimeSeriesSplit(n_splits=5),
    "groundwater": TimeSeriesSplit(n_splits=5),
    "surface_water": TimeSeriesSplit(n_splits=5),
    "wildfire_monthly": TimeSeriesSplit(n_splits=5),
    "wildlife": LeaveOneOut(),
}

LAG1_COL = {
    "ndvi": "ndvi_lag1",
    "grace": "grace_groundwater_anomaly_lag1",
    "groundwater": "depth_to_water_ft_mean_lag1",
    "surface_water": "discharge_cfs_mean_lag1",
    "wildfire_monthly": "wildfire_risk_index_lag1",
    "wildlife": "bbs_abundance_index_lag1",
}


def _cv_score_baseline(
    x: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    pred_col: str,
) -> dict[str, float]:
    """
    Run cross-validation using a named column in X as the prediction.
    Returns mean/std of R² and MAE across folds.
    """
    r2_scores, mae_scores = [], []

    for _, test_idx in cv.split(x):
        y_test = y.iloc[test_idx]
        if pred_col not in x.columns:
            # Column not available for this model — skip
            return {
                "mean_r2": float("nan"),
                "std_r2": float("nan"),
                "mean_mae": float("nan"),
                "std_mae": float("nan"),
                "note": f"{pred_col} not in features",
            }
        y_pred = x.iloc[test_idx][pred_col]
        # Drop rows where the lag prediction itself is NaN
        valid = y_test.notna() & y_pred.notna()
        if valid.sum() < 2:  # noqa: PLR2004
            continue
        r2_scores.append(r2_score(y_test[valid], y_pred[valid]))
        mae_scores.append(mean_absolute_error(y_test[valid], y_pred[valid]))

    if not r2_scores:
        return {
            "mean_r2": float("nan"),
            "std_r2": float("nan"),
            "mean_mae": float("nan"),
            "std_mae": float("nan"),
            "note": "no valid folds",
        }

    return {
        "mean_r2": float(np.mean(r2_scores)),
        "std_r2": float(np.std(r2_scores)),
        "mean_mae": float(np.mean(mae_scores)),
        "std_mae": float(np.std(mae_scores)),
        "n_folds": len(r2_scores),
    }


def _loo_score_baseline(
    x: pd.DataFrame,
    y: pd.Series,
    pred_col: str,
) -> dict[str, float]:
    if pred_col not in x.columns:
        return {
            "mean_r2": float("nan"),
            "std_r2": float("nan"),
            "mean_mae": float("nan"),
            "std_mae": float("nan"),
            "note": f"{pred_col} not in features",
        }

    loo = LeaveOneOut()
    all_y, all_pred = [], []

    for _, test_idx in loo.split(x):
        y_test = y.iloc[test_idx[0]]
        y_pred = x.iloc[test_idx[0]][pred_col]
        if pd.isna(y_test) or pd.isna(y_pred):
            continue
        all_y.append(y_test)
        all_pred.append(y_pred)

    if len(all_y) < 2:  # noqa: PLR2004
        return {
            "mean_r2": float("nan"),
            "std_r2": float("nan"),
            "mean_mae": float("nan"),
            "std_mae": float("nan"),
            "note": "no valid folds",
        }

    return {
        "mean_r2": float(r2_score(all_y, all_pred)),
        "std_r2": float("nan"),
        "mean_mae": float(mean_absolute_error(all_y, all_pred)),
        "std_mae": float("nan"),
        "n_folds": len(all_y),
    }


def _roll3_predictions(x: pd.DataFrame, y: pd.Series, lag1_col: str) -> pd.Series:
    """
    Build a 3-period rolling mean baseline series aligned to y.
    Uses the lag1 column and two additional shifts to avoid look-ahead.
    """
    lag1 = (
        x[lag1_col].copy()
        if lag1_col in x.columns
        else pd.Series(np.nan, index=x.index)
    )
    lag2 = lag1.shift(1)
    lag3 = lag1.shift(2)
    roll3 = (lag1 + lag2 + lag3) / 3.0
    return roll3


def _cv_score_roll3(
    x: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    lag1_col: str,
) -> dict[str, float]:
    """Cross-validate the rolling-3 mean baseline."""
    roll3 = _roll3_predictions(x, y, lag1_col)

    r2_scores, mae_scores = [], []
    for _, test_idx in cv.split(x):
        y_test = y.iloc[test_idx]
        y_pred = roll3.iloc[test_idx]
        valid = y_test.notna() & y_pred.notna()
        if valid.sum() < 2:  # noqa: PLR2004
            continue
        r2_scores.append(r2_score(y_test[valid], y_pred[valid]))
        mae_scores.append(mean_absolute_error(y_test[valid], y_pred[valid]))

    if not r2_scores:
        return {
            "mean_r2": float("nan"),
            "std_r2": float("nan"),
            "mean_mae": float("nan"),
            "std_mae": float("nan"),
            "note": "no valid folds",
        }

    return {
        "mean_r2": float(np.mean(r2_scores)),
        "std_r2": float(np.std(r2_scores)),
        "mean_mae": float(np.mean(mae_scores)),
        "std_mae": float(np.std(mae_scores)),
        "n_folds": len(r2_scores),
    }


def compute_baselines(
    datasets: dict[str, tuple[pd.DataFrame, pd.Series]] | None = None,
) -> dict[str, dict]:
    """
    Compute lag1-persistence and roll3-mean baselines for all models.

    Parameters
    ----------
    datasets : optional
        Pre-built dict from features.build_all(). Built fresh if None.

    Returns
    -------
    dict  model_id -> {"lag1_persistence": {...}, "roll3_mean": {...}}
    Saved to model/baselines.json.
    """
    if datasets is None:
        datasets = build_all()

    results = {}

    for model_id, (x, y) in datasets.items():
        if model_id not in CV_SPLITTERS:
            continue
        cv = CV_SPLITTERS[model_id]
        lag1_col = LAG1_COL[model_id]

        if isinstance(cv, LeaveOneOut):
            lag1_result = _loo_score_baseline(x, y, pred_col=lag1_col)
            roll3 = _roll3_predictions(x, y, lag1_col)
            x_roll3 = x.copy()
            x_roll3["_roll3"] = roll3
            roll3_result = _loo_score_baseline(x_roll3, y, pred_col="_roll3")
        else:
            lag1_result = _cv_score_baseline(x, y, cv, pred_col=lag1_col)
            roll3_result = _cv_score_roll3(x, y, cv, lag1_col=lag1_col)

        results[model_id] = {
            "lag1_persistence": lag1_result,
            "roll3_mean": roll3_result,
            "n_rows": len(y),
            "index_start": str(y.index[0]),
            "index_end": str(y.index[-1]),
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINES_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    baselines = compute_baselines()

    print(f"\nBaselines saved → {BASELINES_PATH}\n")
    header = (
        f"{'Model':<12} {'Lag1 R²':>10} {'Lag1 MAE':>10} {'Roll3 R²':>10} "
        + f"{'Roll3 MAE':>10} {'Rows':>6}"
    )
    print(header)
    print("-" * len(header))

    for model_id, v in baselines.items():
        l1 = v["lag1_persistence"]
        r3 = v["roll3_mean"]
        print(
            f"{model_id:<12} "
            f"{l1['mean_r2']:>10.4f} "
            f"{l1['mean_mae']:>10.4f} "
            f"{r3['mean_r2']:>10.4f} "
            f"{r3['mean_mae']:>10.4f} "
            f"{v['n_rows']:>6}"
        )
