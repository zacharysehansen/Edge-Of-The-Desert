"""§5 cross-check: can the panel identify a human effect at all?

This is a MEASUREMENT, not a candidate model. Nothing it produces is exported and
no shipped artifact changes. It exists to answer one question that PHASE3_PLAN.md
§5 says has to be answered in writing before Layer 2 is built:

    If you force the human block back into every model, deseasonalize it so the
    estimator sees pumping anomalies rather than the calendar, and constrain the
    signs to what water balance says they must be — does the panel find the
    effect, and what does it cost in out-of-fold R²?

Three outcomes, all useful (PHASE3_PLAN.md §5):

  1. constrained fit agrees in sign, costs little R²  -> corroborates Layer 2's betas
  2. agrees in sign, costs real R²                    -> keep Layer 2, report the trade
  3. cannot find the sign at all                      -> the panel genuinely cannot
                                                         identify these effects, and
                                                         Layer 2 is the only route

Wildlife is deliberately absent. PHASE2_REPORT.md already measured the short-window
variant it would need: n=23 drops LOO R² from +0.0956 to -0.1702 with rho below its
own permutation null. That answer is in; it goes straight to Layer 2.

WHAT IS HELD FIXED
------------------
All variants are trained and scored on the SAME test rows, with the SAME fixed
hyperparameters. The shipped models each run their own nested tuning and feature
selection; reproducing that here would mean the variants differed in three ways at
once. The only thing that varies between variants is the feature set and the sign
constraint, which is the whole point.

For the two long-record models (surface water, wildfire) a fourth variant trains on
the full pre-2000 record while scoring the same test rows, because that extra
training data is a real advantage the human-block variants cannot have — the human
series do not exist before 2000. Without it the comparison would flatter the human
block by hiding what `long_record` actually bought.

    python -m scripts.phase2.experiment_human_block
    python -m scripts.phase2.experiment_human_block --models groundwater grace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from scripts.phase2.features import (
    _engineer_monthly,
    _monthly_feature_cols,
    _MONTHLY_MODEL_SPECS,
)
from scripts.phase2.merge import build_monthly_panel

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = ROOT / "model" / "experiment_human_block.json"

RANDOM_STATE = 42
N_SPLITS = 5

# The window where every human series exists. Irrigation and public supply stop at
# 2020-12; GRACE is zero-filled before 2002-10. Every variant is SCORED here.
COMMON_START, COMMON_END = "2002-10", "2020-12"

# One config for every variant, so the feature set is the only thing that moves.
# Shallow, because the common window is 219 rows.
XGB_KWARGS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    tree_method="hist",
    n_jobs=-1,
)

# Base columns that are a human lever. Their lags and rolls come along with them.
# Wider than the D1 count in PHASE3_PLAN.md, which counts only bases a slider can
# reach: `mead_total_release` has no control in the frontend, and
# `precip_x_impervious` is half climate. Both are in the block that gets
# deseasonalized, and both are left sign-unconstrained.
HUMAN_BASES = [
    "population",
    "irrigation_total_withdrawal_mgd",
    "public_supply_groundwater_mgd",
    "impervious_pct",
    "mead_pool_elevation",
    "mead_total_release",
    "precip_x_impervious",
]

# Expected sign of each lever's effect on each TARGET AS MODELLED, from water
# balance and land-cover arithmetic — not fitted. For the four residual models the
# target is a monthly change, and a rate-driven process has the same sign on the
# change as on the level. 0 means no defensible prior, so the feature is included
# but left unconstrained.
#
# Targets, and which direction is "more water":
#   grace          storage anomaly            + = more water
#   ndvi           vegetation greenness       + = greener
#   groundwater    depth to water             + = DEEPER = less water
#   surface_water  log discharge anomaly      + = more flow
#   wildfire       risk index                 + = more fire
EXPECTED_SIGN: dict[str, dict[str, int]] = {
    "grace": {
        "irrigation_total_withdrawal_mgd": -1,   # pumping depletes storage
        "public_supply_groundwater_mgd": -1,
        "population": -1,                        # more people, more municipal draw
        "impervious_pct": -1,                    # pavement suppresses recharge
        "mead_pool_elevation": +1,               # full reservoir = CAP water = less pumping
        "mead_total_release": 0,
        "precip_x_impervious": 0,
    },
    "ndvi": {
        # Positive on irrigated pixels and negative on groundwater: the tension
        # PHASE3_PLAN.md §4 wants shown rather than hidden.
        "irrigation_total_withdrawal_mgd": +1,
        "public_supply_groundwater_mgd": 0,
        "population": -1,                        # development displaces vegetation
        "impervious_pct": -1,                    # pavement is not green
        "mead_pool_elevation": +1,
        "mead_total_release": 0,
        "precip_x_impervious": 0,
    },
    "groundwater": {
        # Depth to water: pumping makes the number BIGGER.
        "irrigation_total_withdrawal_mgd": +1,
        "public_supply_groundwater_mgd": +1,
        "population": +1,
        "impervious_pct": +1,                    # less recharge
        "mead_pool_elevation": -1,               # CAP water substitutes for pumping
        "mead_total_release": 0,
        "precip_x_impervious": 0,
    },
    "surface_water": {
        "irrigation_total_withdrawal_mgd": -1,   # diversion and baseflow loss
        "public_supply_groundwater_mgd": -1,
        "population": -1,
        "impervious_pct": +1,                    # runoff: the dominant urban-desert signal
        "mead_pool_elevation": 0,                # only the Colorado gages see it
        "mead_total_release": 0,
        "precip_x_impervious": +1,
    },
    # Weaker priors than the water-balance ones above, and flagged as such in the
    # report. More people means more ignitions; more pavement means less fuel.
    "wildfire": {
        "population": +1,
        "impervious_pct": -1,
        "irrigation_total_withdrawal_mgd": 0,
        "public_supply_groundwater_mgd": 0,
        "mead_pool_elevation": 0,
        "mead_total_release": 0,
        "precip_x_impervious": 0,
    },
}

# UI key -> the spec key in features.py.
MODELS = {
    "grace": "grace",
    "ndvi": "ndvi",
    "groundwater": "groundwater",
    "surface_water": "surface_water",
    "wildfire": "wildfire_monthly",
}

# Wildfire predicts the level directly; the other four predict a residual over lag1.
DIRECT_MODELS = {"wildfire"}


def base_of(column: str) -> str:
    """Strip the engineered suffixes to get the underlying panel column."""
    for base in HUMAN_BASES:
        if column == base or column.startswith(base + "_"):
            return base
    return column


def is_human(column: str) -> bool:
    return base_of(column) in HUMAN_BASES


def deseasonalize(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subtract each column's month-of-year mean, learned on the TRAINING fold only.

    This is the step PHASE3_PLAN.md §5 calls "the one that gives the coefficient a
    chance to mean something": irrigation is 95.2% a repeating seasonal template, so
    an estimator handed the raw column alongside month_sin/month_cos and temperature
    is being asked to split three collinear encodings of "it is July". Removing the
    template leaves the pumping anomaly, which is the only part that could carry a
    policy signal.

    Learned inside the fold because a month-of-year mean taken over the full panel
    is a look-ahead: the test rows would have contributed to their own centring.
    """
    train, test = train.copy(), test.copy()
    train_months = train.index.month
    test_months = test.index.month
    for column in columns:
        by_month = train[column].groupby(train_months).mean()
        overall = float(train[column].mean())
        train[column] = train[column] - by_month.reindex(train_months).to_numpy()
        # A month absent from the training fold falls back to the fold mean rather
        # than to NaN. With TimeSeriesSplit's first fold this does happen.
        test[column] = test[column] - by_month.reindex(test_months).fillna(overall).to_numpy()
    return train, test


def constraints_for(columns: list[str], model_key: str) -> tuple[int, ...]:
    signs = EXPECTED_SIGN[model_key]
    return tuple(signs.get(base_of(c), 0) if is_human(c) else 0 for c in columns)


def fit_predict(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    monotone: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, XGBRegressor]:
    kwargs = dict(XGB_KWARGS)
    if monotone is not None and any(monotone):
        kwargs["monotone_constraints"] = "(" + ",".join(str(s) for s in monotone) + ")"
    model = XGBRegressor(**kwargs)
    model.fit(x_train, y_train)
    return model.predict(x_test), model


def summarize_directions(
    directions: dict[str, list[float]], model_key: str, target_sd: float
) -> dict[str, dict]:
    """Per lever: the expected sign, and how often the folds actually found it.

    `folds_agreeing` out of `n_folds` is the number that matters. A lever whose
    fitted direction flips between folds has not been identified, it has been
    fitted to noise — the same failure D2 measured across months.

    `effect_in_sd` scales the p10->p90 response by the target's own standard
    deviation, which is what separates "the panel found a real effect pointing the
    wrong way" from "the panel found nothing and the sign is the sign of nothing".
    """
    signs = EXPECTED_SIGN[model_key]
    result = {}
    for base, values in directions.items():
        finite = [v for v in values if np.isfinite(v)]
        expected = signs.get(base, 0)
        if not finite or expected == 0:
            continue
        agreeing = sum(1 for v in finite if v != 0 and np.sign(v) == expected)
        result[base] = {
            "expected_sign": expected,
            "n_folds": len(finite),
            "folds_agreeing": agreeing,
            "mean_effect": float(np.mean(finite)),
            "effect_in_sd": float(np.mean(finite) / target_sd) if target_sd else 0.0,
            "flat_folds": sum(1 for v in finite if v == 0),
        }
    return result


def partial_direction(
    model: XGBRegressor, x_train: pd.DataFrame, base: str
) -> float:
    """Which way does the fitted model move when this lever alone is raised?

    A synthetic row at the training median, with every column belonging to `base`
    (the level, its lags, its rolls) swung from its p10 to its p90 together. The
    return is the change in predicted output, so its SIGN is the direction the
    panel actually fitted — which is the thing PHASE3_PLAN.md §5 asks about and
    that a monotone constraint would otherwise hide by imposing.
    """
    columns = [c for c in x_train.columns if base_of(c) == base]
    if not columns:
        return float("nan")
    row = x_train.median(axis=0)
    low, high = row.copy(), row.copy()
    for column in columns:
        low[column] = float(x_train[column].quantile(0.10))
        high[column] = float(x_train[column].quantile(0.90))
    frame = pd.DataFrame([low, high], columns=x_train.columns)
    predictions = model.predict(frame)
    return float(predictions[1] - predictions[0])


def build_variants(
    monthly: pd.DataFrame, model_key: str
) -> tuple[dict[str, pd.DataFrame], pd.Series, pd.Series, pd.DatetimeIndex]:
    """Build every feature-set variant for one model, plus the shared target.

    Returns (variants, y, lag1, common_index). Every variant is a frame indexed by
    the whole span it can train on; `common_index` is the rows all of them are
    SCORED on.
    """
    spec = _MONTHLY_MODEL_SPECS[MODELS[model_key]]
    target = spec["target"]
    lag1_col = f"{target}_lag1"
    exclude = [target, *spec.get("also_exclude", [])]

    # The shipped feature set is read from the exported artifact, not rebuilt from
    # the spec: three of these models run their own feature selection inside
    # model_*.py, so the spec's list is a superset of what actually deployed
    # (groundwater ships 16 of 63).
    #
    # That selection was done on the full panel rather than inside a fold, so
    # reusing it here gives the shipped variant a small look-ahead advantage. It is
    # left in deliberately: it can only flatter the shipped baseline, which makes
    # any finding of "the human block costs little" conservative rather than
    # optimistic.
    shipped = json.loads(
        (ROOT / "model" / f"{MODELS[model_key]}_feature_names.json").read_text()
    )

    # The human-inclusive set: the full short-window catalogue, with neither
    # `long_record` nor `extended` applied. This is the variant PHASE3_PLAN.md §5
    # asks for, and it is not a shipping candidate.
    with_human = _monthly_feature_cols(exclude_target_base=exclude)

    shipped = [f for f in shipped if f in monthly.columns]
    with_human = [f for f in with_human if f in monthly.columns]

    y = monthly[target]
    lag1 = monthly[lag1_col]

    variants: dict[str, pd.DataFrame] = {}
    variants["shipped"] = monthly[shipped]
    variants["with_human"] = monthly[with_human]

    # Rows every variant can be scored on: inside the common window, with the
    # human-inclusive feature set complete (the binding constraint) and a real
    # target and anchor.
    window = pd.period_range(COMMON_START, COMMON_END, freq="M")
    usable = monthly.index.intersection(window)
    complete = (
        variants["with_human"].loc[usable].notna().all(axis=1)
        & variants["shipped"].loc[usable].notna().all(axis=1)
        & y.loc[usable].notna()
        & lag1.loc[usable].notna()
    )

    flag = spec.get("require_real_target")
    if flag is not None:
        # GRACE: both the target and the anchor must be measurements, not fill.
        real = monthly[flag] == 1
        complete &= (real & real.shift(1, fill_value=False)).loc[usable]

    common_index = usable[complete.to_numpy()]
    return variants, y, lag1, common_index


def evaluate(model_key: str, monthly: pd.DataFrame) -> dict:
    variants, y, lag1, common = build_variants(monthly, model_key)
    direct = model_key in DIRECT_MODELS
    y_target = y if direct else (y - lag1)
    target_sd = float(y_target.loc[common].std())

    # Test blocks are defined once, on the common rows, and reused by every variant
    # so that "identical test rows" is enforced rather than hoped for. Each variant
    # then trains on whatever rows IT has that fall strictly before its test block —
    # which is how the long-record variant gets to use its pre-2000 months without
    # ever seeing a test row.
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    blocks = [common[test_idx] for _, test_idx in splitter.split(np.arange(len(common)))]

    human_columns = [c for c in variants["with_human"].columns if is_human(c)]

    runs: dict[str, dict] = {}
    definitions = [
        ("shipped", "shipped", False, False, False),
        ("with_human", "with_human", False, False, False),
        ("with_human_deseason", "with_human", True, False, False),
        ("with_human_constrained", "with_human", True, True, False),
    ]
    if _MONTHLY_MODEL_SPECS[MODELS[model_key]].get("long_record"):
        definitions.append(("shipped_long_record", "shipped", False, False, True))

    for name, source, deseason, constrain, long_record in definitions:
        frame = variants[source]
        target_r2, level_r2, importances = [], [], []
        directions: dict[str, list[float]] = {b: [] for b in HUMAN_BASES}

        for block in blocks:
            train_index = frame.index[
                frame.notna().all(axis=1)
                & (frame.index < block[0])
                & y_target.reindex(frame.index).notna()
            ]
            if not long_record:
                train_index = train_index.intersection(common)
            if len(train_index) < 24:  # noqa: PLR2004
                continue

            x_tr, x_te = frame.loc[train_index], frame.loc[block]
            if deseason:
                x_tr, x_te = deseasonalize(x_tr, x_te, human_columns)

            monotone = constraints_for(list(frame.columns), model_key) if constrain else None
            prediction, model = fit_predict(
                x_tr, y_target.loc[train_index], x_te, monotone
            )

            target_r2.append(float(r2_score(y_target.loc[block], prediction)))
            level = prediction if direct else prediction + lag1.loc[block].to_numpy()
            level_r2.append(float(r2_score(y.loc[block], level)))

            gains = dict(zip(frame.columns, model.feature_importances_, strict=False))
            total = sum(gains.values()) or 1.0
            importances.append(
                sum(v for k, v in gains.items() if is_human(k)) / total
            )

            # Only the unconstrained variants can answer "which sign did the panel
            # find?" — a constrained fit was told the answer.
            if not constrain:
                for base in HUMAN_BASES:
                    directions[base].append(partial_direction(model, x_tr, base))

        persistence = [
            float(r2_score(y.loc[block], lag1.loc[block])) for block in blocks
        ]
        runs[name] = {
            "n_train_last_fold": int(len(train_index)),
            "n_features": int(frame.shape[1]),
            "n_human_features": sum(is_human(c) for c in frame.columns),
            "target_r2": float(np.mean(target_r2)),
            "level_r2": float(np.mean(level_r2)),
            "skill_vs_persistence": float(np.mean(level_r2) - np.mean(persistence)),
            "human_importance_share": float(np.mean(importances)),
        }
        if not constrain:
            runs[name]["fitted_directions"] = summarize_directions(
                directions, model_key, target_sd
            )

    return {
        "n_test_rows": int(len(common)),
        "target_sd": target_sd,
        "test_span": f"{common[0]}..{common[-1]}",
        "n_folds": len(blocks),
        "persistence_level_r2": float(
            np.mean([float(r2_score(y.loc[b], lag1.loc[b])) for b in blocks])
        ),
        "variants": runs,
    }


VARIANT_LABELS = {
    "shipped": "shipped feature set",
    "shipped_long_record": "shipped + pre-2000 training rows",
    "with_human": "human block forced in",
    "with_human_deseason": "  + deseasonalized",
    "with_human_constrained": "  + sign-constrained",
}


def report(results: dict[str, dict]) -> None:
    for model_key, result in results.items():
        print(f"\n{'=' * 78}")
        print(
            f"{model_key}   {result['n_test_rows']} scored rows "
            f"({result['test_span']}), {result['n_folds']} folds, "
            f"persistence level R2 = {result['persistence_level_r2']:+.4f}"
        )
        print(f"{'-' * 78}")
        print(
            f"{'variant':34s}{'feat':>6s}{'hum':>5s}"
            f"{'target R2':>12s}{'level R2':>11s}{'human imp':>11s}"
        )
        for name, run in result["variants"].items():
            print(
                f"{VARIANT_LABELS[name]:34s}{run['n_features']:6d}"
                f"{run['n_human_features']:5d}"
                f"{run['target_r2']:+12.4f}{run['level_r2']:+11.4f}"
                f"{run['human_importance_share'] * 100:10.1f}%"
            )

        directions = result["variants"]["with_human_deseason"].get(
            "fitted_directions", {}
        )
        if directions:
            print(f"\n  sign recovery (deseasonalized, UNconstrained):")
            for base, entry in directions.items():
                expected = "+" if entry["expected_sign"] > 0 else "-"
                verdict = (
                    "found"
                    if entry["folds_agreeing"] == entry["n_folds"]
                    else "NOT FOUND"
                    if entry["folds_agreeing"] == 0
                    else "unstable"
                )
                # An effect smaller than 5% of the target's own sd is not a
                # wrong sign, it is an absence.
                if abs(entry["effect_in_sd"]) < 0.05:  # noqa: PLR2004
                    verdict = "no effect"
                print(
                    f"    {base:34s} expect {expected}  "
                    f"{entry['folds_agreeing']}/{entry['n_folds']} folds agree  "
                    f"effect {entry['effect_in_sd']:+.3f} sd   {verdict}"
                )


def verdict(results: dict[str, dict]) -> None:
    """Score the three outcomes PHASE3_PLAN.md §5 laid out, per lever."""
    print(f"\n{'=' * 78}")
    print("VERDICT (PHASE3_PLAN.md §5)")
    print(f"{'=' * 78}")

    found = unstable = missing = negligible = 0
    for model_key, result in results.items():
        for entry in (
            result["variants"]["with_human_deseason"]
            .get("fitted_directions", {})
            .values()
        ):
            if abs(entry["effect_in_sd"]) < 0.05:  # noqa: PLR2004
                negligible += 1
            elif entry["folds_agreeing"] == entry["n_folds"]:
                found += 1
            elif entry["folds_agreeing"] == 0:
                missing += 1
            else:
                unstable += 1

        shipped = result["variants"]["shipped"]
        constrained = result["variants"]["with_human_constrained"]
        cost = shipped["target_r2"] - constrained["target_r2"]
        print(
            f"  {model_key:14s} sign-constrained costs "
            f"{cost:+.4f} target R2 vs shipped "
            f"({shipped['target_r2']:+.4f} -> {constrained['target_r2']:+.4f})"
        )

    total = found + unstable + missing + negligible
    print(
        f"\n  lever/target pairs with a declared sign: {total}\n"
        f"    sign found in every fold          : {found}\n"
        f"    sign unstable across folds        : {unstable}\n"
        f"    sign never found                  : {missing}\n"
        f"    effect < 0.05 sd (no effect found) : {negligible}"
    )
    if found <= total // 4:
        print(
            "\n  -> PHASE3_PLAN.md §5 outcome 3: the panel cannot identify these\n"
            "     effects. Layer 2 is the only route for the levers that failed."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=None)
    args = parser.parse_args()

    print("Building the monthly panel...")
    monthly = _engineer_monthly(build_monthly_panel())

    results = {}
    for model_key in args.models or MODELS:
        print(f"  evaluating {model_key} ...")
        results[model_key] = evaluate(model_key, monthly)

    report(results)
    verdict(results)

    with OUTPUT_FILE.open("w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nWritten to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
