"""
experiment_feature_blocks.py
----------------------------
PHASE3_PLAN.md §33: do VPD and GLDAS runoff / soil moisture give wildfire, NDVI
and streamflow any skill? Three feature blocks, three models, one harness.

THIS SCRIPT CHANGES NOTHING THAT SHIPS. For each model it re-runs the model's
REAL training procedure (`_fit_predict`: wildfire's four-candidate competition,
NDVI's 60-draw search, streamflow's selection + competition) as an arm per block,
scored on five shared TimeSeriesSplit test blocks, training strictly before each.

THE ROW RULE, declared: the shipped arm is trained on ITS OWN FULL HISTORY before
each test block (wildfire and streamflow reach back to 1984 and 1980; the new
blocks start in 2000), and every arm is scored on the same blocks, which lie
inside 2000-2023 where the new features exist. That gives the shipped model its
row advantage, so a challenger that wins wins against the model as it actually
ships. (§10 and M4 did the opposite for the human block, to be conservative
toward "the block costs little"; here the question is "the block helps", so the
conservative direction is the reverse.)

Blocks, declared:
  vpd       vpd_kpa, vpd_kpa_lag1, vpd_kpa_roll3        (humidity.py)
  runoff    gldas_runoff_mm_day, _lag1, _roll3           (gldas.py, §33 re-pull)
  soil      gldas_root_zone_mm, _lag1, and its month-to-month change (gldas.py)
Models and the blocks each is offered:
  wildfire_monthly   vpd, soil          (dryness of air and fuel; runoff is not a fire driver)
  ndvi               vpd, soil          (plant water stress; root-zone water)
  surface_water      runoff, soil       (what the land model says reaches the channel)
Each block is judged on its own; a model may take more than one.

THE DECISION RULE, FIXED BEFORE THE FIRST RUN
---------------------------------------------
The rule of §11.5 and §23-§26 per (model, block), on target R² over the five
shared blocks: REAL only if Δ > 0, wins ≥ 4 of 5, and paired |t| ≥ 2.0. REAL
blocks deploy (features.py, the frontend gets a derived driver for the new input
from the sliders it depends on, retrain, gates). NULL blocks stay acquired and
unused. No block is edited after the numbers are seen.

Run:
    python -m scripts.phase2.experiment_feature_blocks
    python -m scripts.phase2.experiment_feature_blocks --models ndvi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from scripts.phase2 import model_ndvi, model_surface_water, model_wildfire_monthly
from scripts.phase2.experiment_grace_nested import paired_t
from scripts.phase2.features import build_all

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "data" / "Final"
OUTPUT_FILE = ROOT / "model" / "experiment_feature_blocks.json"

WIN_FOLDS_REQUIRED, T_REQUIRED, N_SPLITS = 4, 2.0, 5
BLOCK_START, BLOCK_END = "2000-01", "2023-12"

MODULES = {"wildfire_monthly": model_wildfire_monthly, "ndvi": model_ndvi, "surface_water": model_surface_water}
LAG1 = {"wildfire_monthly": None, "ndvi": "ndvi_lag1", "surface_water": "discharge_log_anomaly_lag1"}
OFFERED = {"wildfire_monthly": ["vpd", "soil"], "ndvi": ["vpd", "soil"], "surface_water": ["runoff", "soil"]}


def _monthly(path: str, col: str) -> pd.Series:
    d = pd.read_csv(FINAL / path)
    s = d.set_index(pd.PeriodIndex(d["year_month"], freq="M"))[col].astype(float)
    return s


def blocks() -> dict[str, pd.DataFrame]:
    vpd = _monthly("humidity_monthly.csv", "vpd_kpa")
    runoff = _monthly("gldas_monthly.csv", "gldas_runoff_mm_day")
    soil = _monthly("gldas_monthly.csv", "gldas_root_zone_mm")
    return {
        "vpd": pd.DataFrame({"vpd_kpa": vpd, "vpd_kpa_lag1": vpd.shift(1), "vpd_kpa_roll3": vpd.shift(1).rolling(3, min_periods=2).mean()}),
        "runoff": pd.DataFrame({"gldas_runoff_mm_day": runoff, "gldas_runoff_mm_day_lag1": runoff.shift(1), "gldas_runoff_mm_day_roll3": runoff.shift(1).rolling(3, min_periods=2).mean()}),
        "soil": pd.DataFrame({"gldas_root_zone_mm": soil, "gldas_root_zone_mm_lag1": soil.shift(1), "gldas_root_zone_delta": soil.diff()}),
    }


def align(block: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    if isinstance(index, pd.DatetimeIndex):
        block = block.copy()
        block.index = block.index.to_timestamp()
    return block.reindex(index)


def run_arm(module, x: pd.DataFrame, y: pd.Series, lag1: pd.Series | None, test_blocks: list[pd.Index], train_rows: pd.Index) -> dict:
    y_target = y - lag1 if lag1 is not None else y
    folds = []
    for i, b in enumerate(test_blocks):
        tr = train_rows[(train_rows < b[0])]
        tr = tr[x.loc[tr].notna().all(axis=1) & y_target.loc[tr].notna()]
        pred = module._fit_predict(x.loc[tr], y_target.loc[tr], x.loc[b])
        level = pred + (lag1.loc[b].to_numpy() if lag1 is not None else 0.0)
        folds.append({"fold": i, "test_start": str(b[0]), "test_end": str(b[-1]), "n_train": int(len(tr)),
                      "target_r2": float(r2_score(y_target.loc[b], pred)), "level_r2": float(r2_score(y.loc[b], level))})
    return {"n_features": int(x.shape[1]), "target_r2": float(np.mean([f["target_r2"] for f in folds])),
            "level_r2": float(np.mean([f["level_r2"] for f in folds])), "folds": folds}


def evaluate_model(key: str, datasets: dict, all_blocks: dict) -> dict:
    module = MODULES[key]
    x, y = datasets[key]
    lag1 = x[LAG1[key]].copy() if LAG1[key] else None
    x_train = x.drop(columns=[LAG1[key]]) if LAG1[key] else x
    # Scored rows: inside 2000-2023 where every offered block is complete.
    offered = {name: align(all_blocks[name], x_train.index) for name in OFFERED[key]}
    window = x_train.index[(x_train.index >= pd.Period(BLOCK_START, "M").to_timestamp() if isinstance(x_train.index, pd.DatetimeIndex) else x_train.index >= pd.Period(BLOCK_START, "M"))]
    complete = pd.Series(True, index=window)
    for frame in offered.values():
        complete &= frame.loc[window].notna().all(axis=1)
    common = window[complete.to_numpy()]
    test_blocks = [common[idx] for _, idx in TimeSeriesSplit(n_splits=N_SPLITS).split(np.arange(len(common)))]

    print(f"\n=== {key}: scored on {len(common)} rows {common[0]}..{common[-1]}; shipped trains on its full {len(x_train)}-row history ===", flush=True)
    print("  running shipped ...", flush=True)
    shipped = run_arm(module, x_train, y, lag1, test_blocks, x_train.index)
    out = {"rows_scored": int(len(common)), "scored": [str(common[0]), str(common[-1])], "shipped": shipped, "blocks": {}}
    for name, frame in offered.items():
        print(f"  running +{name} ...", flush=True)
        xb = pd.concat([x_train, frame], axis=1)
        arm = run_arm(module, xb, y, lag1, test_blocks, common)  # the block exists on `common` rows only
        diffs = [fb["target_r2"] - fa["target_r2"] for fa, fb in zip(shipped["folds"], arm["folds"], strict=True)]
        wins, t, d = int(sum(v > 0 for v in diffs)), paired_t(diffs), arm["target_r2"] - shipped["target_r2"]
        real = d > 0 and wins >= WIN_FOLDS_REQUIRED and abs(t) >= T_REQUIRED
        out["blocks"][name] = {"arm": arm, "delta_target_r2": d, "wins": wins, "paired_t": t, "verdict": "REAL" if real else "NULL"}
        print(f"    {name:8s} target R² {shipped['target_r2']:+.4f} -> {arm['target_r2']:+.4f}  Δ {d:+.4f}  wins {wins}/5  t {t:+.2f}  "
              f"[{'PASS' if d > 0 else 'FAIL'} Δ>0] [{'PASS' if wins >= WIN_FOLDS_REQUIRED else 'FAIL'} wins] [{'PASS' if abs(t) >= T_REQUIRED else 'FAIL'} t]  -> {'REAL' if real else 'NULL'}")
        print("      folds: " + "  ".join(f"{fa['target_r2']:+.3f}->{fb['target_r2']:+.3f}" for fa, fb in zip(shipped["folds"], arm["folds"], strict=True)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODULES))
    args = ap.parse_args()
    datasets = build_all()
    all_blocks = blocks()
    results = {key: evaluate_model(key, datasets, all_blocks) for key in args.models}
    prior = json.loads(OUTPUT_FILE.read_text()) if OUTPUT_FILE.exists() else {"results": {}}
    prior["results"].update(results)
    prior["rule"] = {"wins_required": WIN_FOLDS_REQUIRED, "t_required": T_REQUIRED, "shipped_trains_on_full_history": True}
    OUTPUT_FILE.write_text(json.dumps(prior, indent=2))
    print(f"\n  wrote {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
