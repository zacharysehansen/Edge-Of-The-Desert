import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_of_the_desert.phase2 import create_historical_sustainability_plot

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / "data" / "Final"
MODEL_DIR = REPO_ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

SOURCES = {
    "snotel":      "snotel_swe.csv",
    "irrigation":  "irrigation_huc12_monthly_az_2000_2020.csv",
    "public":      "nwaa_public_supply_az_monthly.csv",
    "powell":      "powell_combined.csv",
    "population":  "azpop_monthly.csv",
    "ndvi":        "modis_ndvi.csv",
    "streamflow":  "usgs_streamflow.csv",
    "usdm":        "usdm_sustainability.csv",
}

frames = {}
for key, filename in SOURCES.items():
    df = pd.read_csv(DATA_DIR / filename)
    df["year_month"] = pd.to_datetime(df["year_month"]).dt.to_period("M")
    df = df.set_index("year_month").sort_index()
    frames[key] = df

# =============================================================================
# SECTION 2: MERGE AND CLEAN
# =============================================================================

DROP_COLS = {
    "irrigation": [
        "irrigation_total_withdrawal_gallons_per_day",
        "irrigation_total_withdrawal_acre_feet_month",
        "huc12_count",
    ],
    "public": [
        "public_supply_groundwater_gallons_per_day",
        "public_supply_groundwater_acre_feet_month",
        "huc12_count",
    ],
}

for key, cols in DROP_COLS.items():
    frames[key] = frames[key].drop(columns=cols)

base_index = pd.period_range(start="2000-10", end="2020-12", freq="M")
merged = pd.DataFrame(index=base_index)
merged.index.name = "year_month"

for key, df in frames.items():
    merged = merged.join(df, how="left")

# =============================================================================
# SECTION 3: IMPUTATION
# =============================================================================

snotel_col = "snow_water_equivalent_in"
snotel_observed = merged[snotel_col].dropna()

monthly_climatology = (
    snotel_observed
    .groupby(snotel_observed.index.month)
    .median()
    .clip(lower=0)
)

gap_mask = merged[snotel_col].isnull()
merged.loc[gap_mask, snotel_col] = merged.loc[gap_mask].index.month.map(monthly_climatology)

merged = merged.interpolate(method="linear", limit_direction="both")
merged = merged[merged["usdm_sustainability"].notna()]

merged[snotel_col] = merged[snotel_col].clip(lower=0)

# =============================================================================
# SECTION 4: FEATURE ENGINEERING
# =============================================================================

months = merged.index.month
merged["month_sin"] = np.sin(2 * np.pi * months / 12)
merged["month_cos"] = np.cos(2 * np.pi * months / 12)

lag_targets = {
    "usdm_sustainability_lag1": ("usdm_sustainability", 1),
    "usdm_sustainability_lag3": ("usdm_sustainability", 3),
}
lag_features = {
    "streamflow_cfs_lag1":           ("streamflow_cfs", 1),
    "snow_water_equivalent_in_lag1": ("snow_water_equivalent_in", 1),
    "powell_storage_lag1":           ("powell_storage", 1),
    "ndvi_lag1":                     ("ndvi", 1),
}
for name, (col, k) in {**lag_targets, **lag_features}.items():
    merged[name] = merged[col].shift(k)

merged["usdm_sustainability_roll3"] = (
    merged["usdm_sustainability"].shift(1).rolling(3).mean()
)
merged["usdm_sustainability_roll6"] = (
    merged["usdm_sustainability"].shift(1).rolling(6).mean()
)
merged["streamflow_cfs_roll3"] = (
    merged["streamflow_cfs"].shift(1).rolling(3).mean()
)

merged["AZPOP_pct_change"] = merged["AZPOP"].pct_change()
merged = merged.drop(columns=["AZPOP"])

merged = merged.dropna()

feature_cols = [c for c in merged.columns if c not in ("usdm_dsci", "usdm_sustainability")]

print("=" * 60)
print("READY FOR MODELING")
print("=" * 60)
print(f"\n  Shape:    {merged.shape}")
print(f"  Features: {len(feature_cols)}")
print(f"  Target range: {merged['usdm_sustainability'].min():.2f} - {merged['usdm_sustainability'].max():.2f}")
print(f"\n  Feature list:")
for col in feature_cols:
    print(f"    {col:<48}  min={merged[col].min():>12.4f}  max={merged[col].max():>12.4f}")

# =============================================================================
# SECTION 5: MODELING
# =============================================================================

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, ElasticNet
import json

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    XGBRegressor = None

X_df = merged[feature_cols].copy()
X = X_df.to_numpy()
y = merged["usdm_sustainability"].to_numpy()
year_months = merged.index.astype(str).to_numpy()

baseline_predictions = {
    "lag1_persistence": merged["usdm_sustainability_lag1"].to_numpy(),
    "roll3_mean": merged["usdm_sustainability_roll3"].to_numpy(),
}

tscv = TimeSeriesSplit(n_splits=5)

compact_feature_cols = [
    "month_sin",
    "month_cos",
    "usdm_sustainability_lag1",
    "usdm_sustainability_lag3",
    "usdm_sustainability_roll3",
    "usdm_sustainability_roll6",
    "streamflow_cfs",
    "streamflow_cfs_lag1",
    "streamflow_cfs_roll3",
    "snow_water_equivalent_in",
    "snow_water_equivalent_in_lag1",
    "powell_pool_elevation",
    "ndvi",
    "ndvi_lag1",
]

missing_compact_features = [col for col in compact_feature_cols if col not in feature_cols]
assert not missing_compact_features, f"Missing compact features: {missing_compact_features}"

feature_sets = {
    "full": feature_cols,
    "compact": compact_feature_cols,
}
X_by_set = {
    name: merged[cols].copy()
    for name, cols in feature_sets.items()
}


def summarize_folds(fold_records):
    fold_r2 = [record["r2"] for record in fold_records]
    fold_mae = [record["mae"] for record in fold_records]
    return {
        "folds": fold_records,
        "fold_r2": fold_r2,
        "fold_mae": fold_mae,
        "mean_r2": float(np.mean(fold_r2)),
        "std_r2": float(np.std(fold_r2)),
        "mean_mae": float(np.mean(fold_mae)),
        "std_mae": float(np.std(fold_mae)),
    }


def evaluate_estimator_cv(estimator, X, y, year_months, cv):
    fold_records = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        model = clone(estimator)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fold_records.append({
            "fold": fold,
            "train_start": year_months[train_idx[0]],
            "train_end": year_months[train_idx[-1]],
            "test_start": year_months[test_idx[0]],
            "test_end": year_months[test_idx[-1]],
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
        })
    return summarize_folds(fold_records)


def evaluate_baseline_cv(predictions, y, year_months, cv):
    fold_records = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(predictions), 1):
        y_test = y[test_idx]
        preds = predictions[test_idx]
        fold_records.append({
            "fold": fold,
            "train_start": year_months[train_idx[0]],
            "train_end": year_months[train_idx[-1]],
            "test_start": year_months[test_idx[0]],
            "test_end": year_months[test_idx[-1]],
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
        })
    return summarize_folds(fold_records)


def print_cv_summary(title, summary):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for record in summary["folds"]:
        print(
            f"  Fold {record['fold']}  |  R2: {record['r2']:.4f}  |  MAE: {record['mae']:.4f}  "
            f"|  n_train={record['n_train']}  n_test={record['n_test']}  "
            f"|  test={record['test_start']}->{record['test_end']}"
        )
    print(f"\n  Mean R2:  {summary['mean_r2']:.4f}  (+/- {summary['std_r2']:.4f})")
    print(f"  Mean MAE: {summary['mean_mae']:.4f}  (+/- {summary['std_mae']:.4f})")


def make_serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [make_serializable(v) for v in value]
    if isinstance(value, list):
        return [make_serializable(v) for v in value]
    return value


def prefixed_params(params, prefix):
    return {
        f"{prefix}{key}": value
        for key, value in params.items()
    }


def build_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


class ResidualLag1Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator, lag_feature_index):
        self.base_estimator = base_estimator
        self.lag_feature_index = lag_feature_index

    def fit(self, X, y):
        self.base_estimator_ = clone(self.base_estimator)
        lag1 = np.asarray(X[:, self.lag_feature_index], dtype=float)
        residual_target = np.asarray(y, dtype=float) - lag1
        self.base_estimator_.fit(X, residual_target)
        return self

    def predict(self, X):
        lag1 = np.asarray(X[:, self.lag_feature_index], dtype=float)
        residual_pred = self.base_estimator_.predict(X)
        return residual_pred + lag1


def build_xgb_pipeline():
    assert XGBRegressor is not None, "xgboost is not installed."
    return build_pipeline(
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            # Let the outer CV/search parallelize work to avoid nested CPU oversubscription.
            n_jobs=1,
        )
    )


XGB_PARAM_DIST = {
    "model__n_estimators":      [100, 300, 500, 800],
    "model__max_depth":         [3, 4, 5, 6],
    "model__learning_rate":     [0.01, 0.05, 0.1, 0.2],
    "model__subsample":         [0.6, 0.8, 1.0],
    "model__colsample_bytree":  [0.6, 0.8, 1.0],
    "model__min_child_weight":  [1, 3, 5],
    "model__reg_alpha":         [0, 0.1, 0.5],
    "model__reg_lambda":        [1, 2, 5],
}

RIDGE_PARAM_GRID = {
    "model__alpha": np.logspace(-3, 3, 13),
}

ELASTICNET_PARAM_GRID = {
    "model__alpha": np.logspace(-3, 1, 9),
    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}


def build_xgb_search(n_iter):
    return RandomizedSearchCV(
        build_xgb_pipeline(),
        param_distributions=XGB_PARAM_DIST,
        n_iter=n_iter,
        scoring="r2",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )


def build_ridge_search():
    return GridSearchCV(
        build_pipeline(Ridge()),
        param_grid=RIDGE_PARAM_GRID,
        scoring="r2",
        cv=tscv,
        n_jobs=-1,
        verbose=0,
    )


def build_elasticnet_search():
    return GridSearchCV(
        build_pipeline(ElasticNet(max_iter=20000, random_state=42)),
        param_grid=ELASTICNET_PARAM_GRID,
        scoring="r2",
        cv=tscv,
        n_jobs=-1,
        verbose=0,
    )


def build_residual_xgb_search(lag_feature_index, n_iter):
    return RandomizedSearchCV(
        ResidualLag1Regressor(
            base_estimator=build_xgb_pipeline(),
            lag_feature_index=lag_feature_index,
        ),
        param_distributions=prefixed_params(XGB_PARAM_DIST, "base_estimator__"),
        n_iter=n_iter,
        scoring="r2",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )


def extract_feature_importance(estimator, candidate_feature_cols):
    if isinstance(estimator, ResidualLag1Regressor):
        model = estimator.base_estimator_.named_steps["model"]
    else:
        model = estimator.named_steps["model"]

    if hasattr(model, "get_booster"):
        raw_importances = model.get_booster().get_score(importance_type="gain")
        named_importances = {
            candidate_feature_cols[int(name.replace("f", ""))]: float(score)
            for name, score in raw_importances.items()
        }
        items = sorted(named_importances.items(), key=lambda item: item[1], reverse=True)
        return {
            "label": "Gain",
            "kind": "gain",
            "items": items,
        }

    if hasattr(model, "coef_"):
        coef_values = np.abs(np.ravel(model.coef_))
        items = sorted(
            [
                (candidate_feature_cols[i], float(coef_values[i]))
                for i in range(len(candidate_feature_cols))
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        return {
            "label": "|Coefficient|",
            "kind": "abs_coef",
            "items": items,
        }

    return {
        "label": "Importance",
        "kind": "unknown",
        "items": [],
    }


def run_candidate(candidate):
    candidate_feature_cols = feature_sets[candidate["feature_set"]]
    X_candidate_df = X_by_set[candidate["feature_set"]]
    X_candidate = X_candidate_df.to_numpy()

    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {candidate['label']}")
    print("=" * 60)
    print(
        f"  Feature set: {candidate['feature_set']} "
        f"({len(candidate_feature_cols)} features)"
    )
    print(f"  Search:      {candidate['search_note']}")

    search = candidate["build_search"]()
    search.fit(X_candidate, y)

    best_estimator = search.best_estimator_
    best_params = {
        key: make_serializable(value)
        for key, value in search.best_params_.items()
    }

    print(f"\n  Best CV R2 (search): {search.best_score_:.4f}")
    print("  Best params:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    cv_summary = evaluate_estimator_cv(best_estimator, X_candidate, y, year_months, tscv)
    print_cv_summary("Cross-validation", cv_summary)

    train_preds = best_estimator.predict(X_candidate)
    train_summary = {
        "r2": float(r2_score(y, train_preds)),
        "mae": float(mean_absolute_error(y, train_preds)),
    }

    print("\n  Train fit:")
    print(f"    R2:  {train_summary['r2']:.4f}")
    print(f"    MAE: {train_summary['mae']:.4f}")

    feature_importance = extract_feature_importance(best_estimator, candidate_feature_cols)
    if feature_importance["items"]:
        print("\n  Top features:")
        for name, score in feature_importance["items"][:10]:
            print(f"    {name:<48}  {score:>10.4f}")

    result = {
        "label": candidate["label"],
        "model_family": candidate["model_family"],
        "feature_set": candidate["feature_set"],
        "prediction_mode": candidate["prediction_mode"],
        "export_compatible": candidate["export_compatible"],
        "feature_names": candidate_feature_cols,
        "n_features": len(candidate_feature_cols),
        "search_note": candidate["search_note"],
        "search_best_score": float(search.best_score_),
        "best_params": best_params,
        "cv": cv_summary,
        "train": train_summary,
        "feature_importance": {
            "label": feature_importance["label"],
            "kind": feature_importance["kind"],
            "top_items": [
                {"feature": name, "score": float(score)}
                for name, score in feature_importance["items"][:15]
            ],
        },
    }

    return result, best_estimator


candidate_configs = []

if XGBRegressor is not None:
    candidate_configs.extend([
        {
            "name": "xgb_full",
            "label": "XGBoost (all features)",
            "model_family": "xgboost",
            "feature_set": "full",
            "prediction_mode": "level",
            "export_compatible": True,
            "search_note": "RandomizedSearchCV, 40 iterations",
            "build_search": lambda: build_xgb_search(n_iter=40),
        },
        {
            "name": "xgb_compact",
            "label": "XGBoost (compact features)",
            "model_family": "xgboost",
            "feature_set": "compact",
            "prediction_mode": "level",
            "export_compatible": True,
            "search_note": "RandomizedSearchCV, 40 iterations",
            "build_search": lambda: build_xgb_search(n_iter=40),
        },
        {
            "name": "xgb_compact_residual_lag1",
            "label": "XGBoost (compact, residual over lag1)",
            "model_family": "xgboost",
            "feature_set": "compact",
            "prediction_mode": "residual_lag1",
            "export_compatible": True,
            "search_note": "RandomizedSearchCV, 40 iterations on residual target",
            "build_search": lambda: build_residual_xgb_search(
                lag_feature_index=feature_sets["compact"].index("usdm_sustainability_lag1"),
                n_iter=40,
            ),
        },
    ])
else:
    print("\n  Note: xgboost is not installed in this Python environment; skipping XGBoost experiments.")

candidate_configs.extend([
    {
        "name": "ridge_compact",
        "label": "Ridge (compact features)",
        "model_family": "ridge",
        "feature_set": "compact",
        "prediction_mode": "level",
        "export_compatible": True,
        "search_note": "GridSearchCV over alpha",
        "build_search": build_ridge_search,
    },
    {
        "name": "elasticnet_compact",
        "label": "ElasticNet (compact features)",
        "model_family": "elasticnet",
        "feature_set": "compact",
        "prediction_mode": "level",
        "export_compatible": True,
        "search_note": "GridSearchCV over alpha and l1_ratio",
        "build_search": build_elasticnet_search,
    },
])

baseline_cv = {
    name: evaluate_baseline_cv(preds, y, year_months, tscv)
    for name, preds in baseline_predictions.items()
}

print("\n" + "=" * 60)
print("BASELINES")
print("=" * 60)
for name, summary in baseline_cv.items():
    print_cv_summary(f"Baseline: {name}", summary)

candidate_results = {}
fitted_estimators = {}
for candidate in candidate_configs:
    result, estimator = run_candidate(candidate)
    candidate_results[candidate["name"]] = result
    fitted_estimators[candidate["name"]] = estimator

leaderboard = []
for name, result in candidate_results.items():
    leaderboard.append({
        "name": name,
        "label": result["label"],
        "kind": "candidate",
        "model_family": result["model_family"],
        "feature_set": result["feature_set"],
        "prediction_mode": result["prediction_mode"],
        "export_compatible": result["export_compatible"],
        "n_features": result["n_features"],
        "mean_r2": result["cv"]["mean_r2"],
        "mean_mae": result["cv"]["mean_mae"],
    })

for name, summary in baseline_cv.items():
    leaderboard.append({
        "name": name,
        "label": name,
        "kind": "baseline",
        "model_family": "baseline",
        "feature_set": "n/a",
        "prediction_mode": "level",
        "export_compatible": False,
        "n_features": 1,
        "mean_r2": summary["mean_r2"],
        "mean_mae": summary["mean_mae"],
    })

leaderboard = sorted(
    leaderboard,
    key=lambda row: (-row["mean_r2"], row["mean_mae"], row["label"]),
)

best_candidate_name = min(
    candidate_results,
    key=lambda name: (
        -candidate_results[name]["cv"]["mean_r2"],
        candidate_results[name]["cv"]["mean_mae"],
        candidate_results[name]["label"],
    ),
)
export_candidate_names = [
    name
    for name, result in candidate_results.items()
    if result["export_compatible"]
]
assert export_candidate_names, "At least one export-compatible candidate is required."

selected_export_candidate_name = min(
    export_candidate_names,
    key=lambda name: (
        -candidate_results[name]["cv"]["mean_r2"],
        candidate_results[name]["cv"]["mean_mae"],
        candidate_results[name]["label"],
    ),
)

best_result = candidate_results[best_candidate_name]
selected_result = candidate_results[selected_export_candidate_name]
selected_pipeline = fitted_estimators[selected_export_candidate_name]
selected_feature_cols = selected_result["feature_names"]
selected_X_df = X_by_set[selected_result["feature_set"]]
selected_X = selected_X_df.to_numpy()

print("\n" + "=" * 60)
print("LEADERBOARD")
print("=" * 60)
for rank, row in enumerate(leaderboard, 1):
    print(
        f"  {rank:>2}. {row['label']:<28}  "
        f"R2={row['mean_r2']:.4f}  MAE={row['mean_mae']:.4f}  "
        f"[{row['kind']}]"
    )

print("\n" + "=" * 60)
print("SELECTED MODEL")
print("=" * 60)
print(f"  Best learned model: {best_result['label']}")
print(f"  Mean CV R2:         {best_result['cv']['mean_r2']:.4f}")
print(f"  Mean CV MAE:        {best_result['cv']['mean_mae']:.4f}")
print(f"  Prediction mode:    {best_result['prediction_mode']}")
print(f"  Selected export:    {selected_result['label']}")
print(f"  Export feature set: {selected_result['feature_set']} ({selected_result['n_features']} features)")
if leaderboard[0]["kind"] == "baseline":
    print(f"  Benchmark leader:   {leaderboard[0]['label']} (still beats every learned model)")
else:
    print(f"  Benchmark leader:   {leaderboard[0]['label']} (learned model)")
if best_candidate_name != selected_export_candidate_name:
    print("  Export note:        best learned model is comparison-only and is not exported to ONNX.")

report_feature_importance = {
    "model_name": best_candidate_name,
    "model_label": best_result["label"],
    "label": best_result["feature_importance"]["label"],
    "kind": best_result["feature_importance"]["kind"],
    "top_items": best_result["feature_importance"]["top_items"],
}

report_importance_items = report_feature_importance["top_items"]
if report_importance_items:
    print(f"\n  Best-model top features ({report_feature_importance['label']}):")
    for item in report_importance_items[:10]:
        print(
            f"    {item['feature']:<48}  "
            f"{item['score']:>10.4f}"
        )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if report_importance_items:
    top_items = report_importance_items[:15][::-1]
    top_names = [item["feature"] for item in top_items]
    top_scores = [item["score"] for item in top_items]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_names, top_scores, color="#1f5aa6", alpha=0.9)
    ax.set_xlabel(report_feature_importance["label"])
    ax.set_title("Best Model: Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"\n  Saved feature_importance.png")

comparison_rows = [
    {
        **row,
        "rank": rank,
    }
    for rank, row in enumerate(leaderboard, 1)
]

cv_results = {
    "best_learned_model_name": best_candidate_name,
    "best_learned_model_label": best_result["label"],
    "best_learned_model_prediction_mode": best_result["prediction_mode"],
    "selected_model_name": selected_export_candidate_name,
    "selected_model_label": selected_result["label"],
    "selected_model_family": selected_result["model_family"],
    "selected_feature_set": selected_result["feature_set"],
    "report_feature_importance": report_feature_importance,
    "fold_r2": selected_result["cv"]["fold_r2"],
    "fold_mae": selected_result["cv"]["fold_mae"],
    "mean_r2": selected_result["cv"]["mean_r2"],
    "std_r2": selected_result["cv"]["std_r2"],
    "mean_mae": selected_result["cv"]["mean_mae"],
    "std_mae": selected_result["cv"]["std_mae"],
    "best_cv_r2": selected_result["search_best_score"],
    "best_params": selected_result["best_params"],
    "leaderboard": comparison_rows,
    "experiments": candidate_results,
    "baselines": baseline_cv,
}
with open(MODEL_DIR / "cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)
print("  Saved cv_results.json")

with open(MODEL_DIR / "model_comparison.json", "w") as f:
    json.dump(
        {
            "best_learned_model_name": best_candidate_name,
            "selected_model_name": selected_export_candidate_name,
            "report_feature_importance": report_feature_importance,
            "leaderboard": comparison_rows,
            "experiments": candidate_results,
            "baselines": baseline_cv,
        },
        f,
        indent=2,
    )
print("  Saved model_comparison.json")

# =============================================================================
# SECTION 6: EXPORT AND ARTIFACTS
# =============================================================================

import onnx
import onnxruntime as rt


def convert_estimator_to_onnx(model, n_features):
    if hasattr(model, "get_booster"):
        from onnxmltools import convert_xgboost as convert_xgboost_model
        from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType

        return convert_xgboost_model(
            model,
            initial_types=[("float_input", OnnxFloatTensorType([None, n_features]))],
            target_opset=15,
        )

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "skl2onnx is required to export non-XGBoost models to ONNX."
        ) from exc

    return convert_sklearn(
        model,
        initial_types=[("float_input", SklFloatTensorType([None, n_features]))],
        target_opset=15,
    )


def get_preprocessing_estimator(estimator):
    if isinstance(estimator, ResidualLag1Regressor):
        return estimator.base_estimator_
    return estimator


def get_core_model(estimator):
    return get_preprocessing_estimator(estimator).named_steps["model"]


def ensure_opset_import(model, domain, version):
    for opset in model.opset_import:
        if opset.domain == domain:
            opset.version = max(opset.version, version)
            return model

    opset = model.opset_import.add()
    opset.domain = domain
    opset.version = version
    return model


def add_lag1_back_to_scaled_input_onnx(model, lag_feature_index, lag_mean, lag_std):
    model = ensure_opset_import(model, "", 15)

    graph = model.graph
    input_name = graph.input[0].name
    final_output_name = graph.output[0].name
    residual_output_name = f"{final_output_name}_residual"

    for node in graph.node:
        for output_index, output_name in enumerate(node.output):
            if output_name == final_output_name:
                node.output[output_index] = residual_output_name

    original_output = onnx.ValueInfoProto()
    original_output.CopyFrom(graph.output[0])
    graph.output.clear()
    graph.output.extend([original_output])

    lag_index_name = f"{final_output_name}_lag1_index"
    lag_mean_name = f"{final_output_name}_lag1_mean"
    lag_std_name = f"{final_output_name}_lag1_std"
    lag_scaled_name = f"{final_output_name}_lag1_scaled"
    lag_raw_name = f"{final_output_name}_lag1_raw"
    residual_shape_name = f"{final_output_name}_residual_shape"
    lag_match_shape_name = f"{final_output_name}_lag1_match_shape"

    graph.initializer.extend([
        onnx.numpy_helper.from_array(
            np.array([lag_feature_index], dtype=np.int64),
            name=lag_index_name,
        ),
        onnx.numpy_helper.from_array(
            np.array([lag_mean], dtype=np.float32),
            name=lag_mean_name,
        ),
        onnx.numpy_helper.from_array(
            np.array([lag_std], dtype=np.float32),
            name=lag_std_name,
        ),
    ])

    graph.node.extend([
        onnx.helper.make_node(
            "Gather",
            inputs=[input_name, lag_index_name],
            outputs=[lag_scaled_name],
            axis=1,
        ),
        onnx.helper.make_node(
            "Mul",
            inputs=[lag_scaled_name, lag_std_name],
            outputs=[f"{lag_raw_name}_scaled"],
        ),
        onnx.helper.make_node(
            "Add",
            inputs=[f"{lag_raw_name}_scaled", lag_mean_name],
            outputs=[lag_raw_name],
        ),
        onnx.helper.make_node(
            "Shape",
            inputs=[residual_output_name],
            outputs=[residual_shape_name],
        ),
        onnx.helper.make_node(
            "Reshape",
            inputs=[lag_raw_name, residual_shape_name],
            outputs=[lag_match_shape_name],
        ),
        onnx.helper.make_node(
            "Add",
            inputs=[residual_output_name, lag_match_shape_name],
            outputs=[final_output_name],
        ),
    ])

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


selected_preprocessing_estimator = get_preprocessing_estimator(selected_pipeline)
selected_model = get_core_model(selected_pipeline)
scaler = selected_preprocessing_estimator.named_steps["scaler"]
imputer = selected_preprocessing_estimator.named_steps["imputer"]

X_imputed = imputer.transform(selected_X)
X_scaled = scaler.transform(X_imputed)

scaler_params = {
    col: {"mean": float(scaler.mean_[i]), "std": float(scaler.scale_[i])}
    for i, col in enumerate(selected_feature_cols)
}

onnx_model = convert_estimator_to_onnx(selected_model, X_scaled.shape[1])
if isinstance(selected_pipeline, ResidualLag1Regressor):
    lag_feature_name = "usdm_sustainability_lag1"
    lag_feature_index = selected_feature_cols.index(lag_feature_name)
    onnx_model = add_lag1_back_to_scaled_input_onnx(
        onnx_model,
        lag_feature_index=lag_feature_index,
        lag_mean=scaler_params[lag_feature_name]["mean"],
        lag_std=scaler_params[lag_feature_name]["std"],
    )

onnx_path = MODEL_DIR / "water_sustainability.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

sess        = rt.InferenceSession(str(onnx_path))
input_name  = sess.get_inputs()[0].name
onnx_preds  = sess.run(None, {input_name: X_scaled.astype(np.float32)})[0].flatten()
sklearn_preds = selected_pipeline.predict(selected_X)

max_diff = np.max(np.abs(onnx_preds - sklearn_preds))
assert max_diff < 1e-3, f"ONNX round-trip failed: max diff = {max_diff:.6f}"

print("=" * 60)
print("EXPORT")
print("=" * 60)
print(f"  Selected model: {selected_result['label']}")
print(f"\n  ONNX round-trip max diff: {max_diff:.6f}  (pass)")

with open(MODEL_DIR / "feature_names.json", "w") as f:
    json.dump(selected_feature_cols, f, indent=2)

feature_stats = {}
for col in selected_feature_cols:
    feature_stats[col] = {
        "min":    float(selected_X_df[col].min()),
        "max":    float(selected_X_df[col].max()),
        "median": float(selected_X_df[col].median()),
        "p5":     float(selected_X_df[col].quantile(0.05)),
        "p95":    float(selected_X_df[col].quantile(0.95)),
        "mean":   scaler_params[col]["mean"],
        "std":    scaler_params[col]["std"],
    }

with open(MODEL_DIR / "feature_stats.json", "w") as f:
    json.dump(feature_stats, f, indent=2)

historical_preds = selected_pipeline.predict(selected_X)
historical = pd.DataFrame({
    "year_month": selected_X_df.index.astype(str),
    "usdm_sustainability": historical_preds.astype(float),
    "actual_usdm_sustainability": merged.loc[selected_X_df.index, "usdm_sustainability"].astype(float).to_numpy(),
})
historical["residual"] = (
    historical["usdm_sustainability"] - historical["actual_usdm_sustainability"]
)
historical.to_csv(MODEL_DIR / "historical_sustainability.csv", index=False)

create_historical_sustainability_plot(
    MODEL_DIR / "historical_sustainability.csv",
    MODEL_DIR / "historical_sustainability_plot.png",
    model_label=selected_result["label"],
)

print("  Saved water_sustainability.onnx")
print("  Saved feature_names.json")
print("  Saved feature_stats.json")
print("  Saved historical_sustainability.csv")
print("  Saved historical_sustainability_plot.png")

median_vals = np.array([[feature_stats[c]["median"] for c in selected_feature_cols]], dtype=np.float32)
median_scaled = ((median_vals - np.array([feature_stats[c]["mean"] for c in selected_feature_cols]))
                 / np.array([feature_stats[c]["std"] for c in selected_feature_cols])).astype(np.float32)
smoke_pred = sess.run(None, {input_name: median_scaled})[0].flatten()[0]
assert 0 <= smoke_pred <= 100, f"Smoke test out of range: {smoke_pred:.2f}"
print(f"\n  Smoke test passed: predicted sustainability = {smoke_pred:.2f}")
