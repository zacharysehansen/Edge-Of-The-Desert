"""Measure what the frontend sliders actually do to the exported models.

PHASE2_REPORT.md scores forecast accuracy. Nothing in Phase 2 ever scored the
question the interface actually asks: *move a human-pressure slider, does the
output respond, and in the right direction?* This script answers that, offline,
against the same `model/*.onnx` files the browser loads.

Three modes:

  --mode sweep     One-step inference, as `models.js` runs today. Each slider is
                   swung min -> max with the others at their default; the number
                   reported is the change in the 0-100 score the UI renders.

  --mode rollout   The Layer 3 experiment (PHASE3_PLAN.md §4). Iterates the four
                   residual models forward month-by-month for `--months`, feeding
                   each model's own output back into its lag/roll features, so a
                   sustained scenario accumulates instead of being charged once.
                   `--revert` adds the empirical mean-reversion term below.

  --mode lambda    Fit the per-target mean-reversion rate from the observed panel.

  --mode acceptance
                   The PHASE3_PLAN.md §7 guard, restructured per §11.4. Sign
                   stability across all 12 months is the HARD GATE, because that is
                   what D2 violates and what Layer 2 satisfies by construction.
                   Magnitude is reported per lever but not gated: §10 measured that
                   12 of 18 lever/target pairs have no effect at all, so demanding
                   >= 5 score points everywhere would be demanding the app display
                   something the data says is absent.

WHY --revert EXISTS, AND WHY THE NAIVE ROLLOUT IS INVALID
---------------------------------------------------------
Every residual model drops its own lagged target before fitting
(`x.drop(columns=[lag1_col])` in each `model_*.py`). That is a defensible choice
for one-step forecasting -- it stops the anchor doubling as a predictor -- but it
means the exported artifact is NOT a dynamical system:

    dy_t = f(exogenous_t)            with no dependence on y_t

There is no recession, no restoring force, no equilibrium. Integrating it D steps
is therefore just multiplication by D: a pure linear ramp. Measured, at sustained
maximum precipitation, surface water reaches a log anomaly of +17.3 by month 36
against a historical p95 of +1.06 -- roughly 3e7 times normal flow.

`--revert` restores the missing term empirically. Regressing dy on (y_lag1 - ybar)
over the observed panel gives a per-target reversion rate lambda; the rollout then
integrates

    y_t = y_{t-1} + f(x_t) - lambda * (y_{t-1} - y_eq)

which converges to a finite displacement f/lambda under sustained forcing instead
of ramping without bound. See PHASE3_PLAN.md §4 for the measured verdict on
whether that is sufficient. It is not, on its own.

Run from the repository root:

    python scripts/phase3/slider_sensitivity.py --mode sweep
    python scripts/phase3/slider_sensitivity.py --mode rollout --months 36 --revert
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structural import Structural  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = ROOT / "frontend" / "computed_stats.json"
MODEL_DIR = ROOT / "model"
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.csv"

# UI key -> exported filename stem. Mirrors MODEL_FILENAMES in models.js.
MODELS = {
    "grace": "grace",
    "ndvi": "ndvi",
    "groundwater": "groundwater",
    "surface_water": "surface_water",
    "wildfire": "wildfire_monthly",
    "wildlife": "wildlife",
}

# The four residual-over-lag1 models and the panel column each one reconstructs.
RESIDUAL_TARGETS = {
    "grace": "grace_groundwater_anomaly",
    "ndvi": "ndvi",
    "groundwater": "depth_to_water_anomaly_ft",
    "surface_water": "discharge_log_anomaly",
}

HUMAN_SLIDERS = [
    "population",
    "irrigation_total_withdrawal_mgd",
    "public_supply_groundwater_mgd",
    "impervious_pct",
    "mead_pool_elevation",
]
CLIMATE_SLIDERS = ["precipitation_mm_day", "temperature_2m_c", "nclimdiv_pdsi"]

# models.js holds Mead releases fixed; there is no slider for it.
MEAD_TOTAL_RELEASE = 12657.0


def load_stats() -> tuple[dict, dict]:
    stats = json.loads(STATS_PATH.read_text())
    return stats["SLIDER_STATS"], stats["OUTPUT_STATS"]


def load_models() -> tuple[dict, dict]:
    sessions, names = {}, {}
    for key, stem in MODELS.items():
        sessions[key] = ort.InferenceSession(str(MODEL_DIR / f"{stem}.onnx"))
        names[key] = json.loads(
            (MODEL_DIR / f"{stem}_feature_names.json").read_text()
        )
    return sessions, names


def dsci_from_pdsi(pdsi: float) -> float:
    """OLS fit of USDM DSCI on PDSI over their 288-month overlap. Mirrors models.js."""
    return max(0.0, min(500.0, 114.109 - 37.231 * pdsi))


def wrap_month(month: int) -> int:
    """A lag of 3 from January is October, and carries October's climatology."""
    return (int(round(month)) - 1) % 12 + 1


# Base feature name -> the slider whose delta drives it. Mirrors DRIVER_SOURCE in
# models.js: the nClimDiv series are the same physical quantities as the MERRA-2
# ones (r = +0.999 / +0.920 over the overlap), so one slider drives both.
DRIVER_SOURCE = {
    "population": "population",
    "irrigation_total_withdrawal_mgd": "irrigation_total_withdrawal_mgd",
    "public_supply_groundwater_mgd": "public_supply_groundwater_mgd",
    "impervious_pct": "impervious_pct",
    "mead_pool_elevation": "mead_pool_elevation",
    "precipitation_mm_day": "precipitation_mm_day",
    "temperature_2m_c": "temperature_2m_c",
    "nclimdiv_pdsi": "nclimdiv_pdsi",
    "nclimdiv_temperature_c": "temperature_2m_c",
    "nclimdiv_precipitation_mm_day": "precipitation_mm_day",
}

TEMPORAL_DRIVERS = [*DRIVER_SOURCE, "usdm_dsci", "mead_total_release"]

# features.py builds _anomaly / _anomaly_lag1 / _anomaly_roll3 for exactly these.
ANOMALY_DRIVERS = [
    "temperature_2m_c",
    "precipitation_mm_day",
    "nclimdiv_temperature_c",
    "nclimdiv_precipitation_mm_day",
]


class Catalog:
    """Rebuilds the feature catalog `models.js` assembles for each inference pass.

    Sliders carry POLICY DELTAS (PHASE3_PLAN.md D5), not raw values. A raw model
    input is the delta plus that month's climatology:

        mode "scale"   raw(m, d) = baseline * (1 + d/100) * seasonal[m]
        mode "offset"  raw(m, d) = baseline +  d          + seasonal[m]

    `elapsed` is how long the scenario has been held, so month offset j is under
    the scenario iff `j < elapsed` and anything older sits at its own month's
    normal. Lag/roll/anomaly definitions mirror features.py:

        X_lagK    = X at offset K
        X_rollW   = mean of X over offsets 1..W      (shift(1).rolling(W).mean())
        X_anomaly = X(0) - X_roll12
    """

    def __init__(self, slider_stats: dict):
        self.policy = {
            k: v["policy"] for k, v in slider_stats.items() if "policy" in v
        }
        self.zero = {k: 0.0 for k in self.policy}

    def slider_raw(self, key: str, delta: float, month: int) -> float:
        p = self.policy[key]
        seasonal = p["seasonal"][wrap_month(month) - 1]
        if p["mode"] == "scale":
            return p["baseline"] * (1 + delta / 100.0) * seasonal
        return p["baseline"] + delta + seasonal

    def driver_at(self, name: str, deltas: dict, month: int) -> float:
        if name == "mead_total_release":
            return MEAD_TOTAL_RELEASE
        # usdm_dsci is not a control; it is regressed off the PDSI slider, and its
        # lag/roll features have to follow the same way.
        if name == "usdm_dsci":
            return dsci_from_pdsi(
                self.slider_raw("nclimdiv_pdsi", deltas.get("nclimdiv_pdsi", 0.0), month)
            )
        source = DRIVER_SOURCE[name]
        return self.slider_raw(source, deltas.get(source, 0.0), month)

    def _series(self, name: str, deltas: dict, month: int, elapsed: int):
        def at(offset: int) -> float:
            applied = deltas if offset < elapsed else self.zero
            return self.driver_at(name, applied, month - offset)

        return at

    @staticmethod
    def _mean(at, first: int, count: int) -> float:
        return sum(at(j) for j in range(first, first + count)) / count

    def build(self, sv: dict, month: int, elapsed: int) -> dict:
        angle = 2 * math.pi * (month - 1) / 12
        cat = {
            "month_sin": math.sin(angle),
            "month_cos": math.cos(angle),
            "grace_available": 1.0,
            "year_linear": 19.0,
        }

        series = {n: self._series(n, sv, month, elapsed) for n in TEMPORAL_DRIVERS}
        for name, at in series.items():
            cat[name] = at(0)
            cat[f"{name}_lag1"] = at(1)
            cat[f"{name}_lag3"] = at(3)
            cat[f"{name}_lag6"] = at(6)
            cat[f"{name}_roll3"] = self._mean(at, 1, 3)
            cat[f"{name}_roll6"] = self._mean(at, 1, 6)
            cat[f"{name}_roll12"] = self._mean(at, 1, 12)

        for name in ANOMALY_DRIVERS:
            at = series[name]
            anom = lambda j, at=at: at(j) - self._mean(at, j + 1, 12)  # noqa: E731
            cat[f"{name}_anomaly"] = anom(0)
            cat[f"{name}_anomaly_lag1"] = anom(1)
            cat[f"{name}_anomaly_roll3"] = (anom(1) + anom(2) + anom(3)) / 3

        cat["precip_x_impervious"] = cat["precipitation_mm_day"] * cat["impervious_pct"]
        cat["precip_x_temperature"] = (
            cat["precipitation_mm_day"] * cat["temperature_2m_c"]
        )
        cat["nclimdiv_precip_x_temperature"] = (
            cat["nclimdiv_precipitation_mm_day"] * cat["nclimdiv_temperature_c"]
        )

        # Annual aggregates over the 12 months ending on `month`; _lag1 is the 12
        # before that. Only the wildlife model consumes these.
        def jja(at) -> float:
            vals = [at(j) for j in range(12) if 6 <= wrap_month(month - j) <= 8]
            return sum(vals) / len(vals)

        pdsi, temp = series["nclimdiv_pdsi"], series["nclimdiv_temperature_c"]
        precip = series["nclimdiv_precipitation_mm_day"]
        precip_annual = self._mean(precip, 0, 12) * 12
        cat["nclimdiv_pdsi_annual_mean"] = self._mean(pdsi, 0, 12)
        cat["nclimdiv_pdsi_annual_mean_lag1"] = self._mean(pdsi, 12, 12)
        cat["nclimdiv_pdsi_jja_mean"] = jja(pdsi)
        cat["nclimdiv_temperature_c_annual_mean"] = self._mean(temp, 0, 12)
        cat["nclimdiv_temperature_c_jja_mean"] = jja(temp)
        cat["nclimdiv_precipitation_mm_day_annual_sum"] = precip_annual
        cat["nclimdiv_precipitation_mm_day_annual_sum_lag1"] = (
            self._mean(precip, 12, 12) * 12
        )
        cat["nclimdiv_log_precip_annual"] = math.log1p(max(0.0, precip_annual))
        return cat


def fill_response_features(cat: dict, history: dict[str, list[float]]) -> dict:
    """Write each response variable's lag/roll features from its own rollout history."""
    for key, base in RESIDUAL_TARGETS.items():
        h = history[key]
        cat[base] = h[-1]
        cat[f"{base}_lag1"] = h[-1]
        cat[f"{base}_lag3"] = h[-3]
        cat[f"{base}_roll3"] = float(np.mean(h[-3:]))
        cat[f"{base}_roll6"] = float(np.mean(h[-6:]))
    cat["grace_groundwater_anomaly_annual_mean"] = float(np.mean(history["grace"][-12:]))
    cat["ndvi_annual_mean"] = float(np.mean(history["ndvi"][-12:]))
    cat["ndvi_jja_mean"] = cat["ndvi_annual_mean"]
    cat["ndvi_annual_mean_lag1"] = float(np.mean(history["ndvi"][-24:-12]))
    return cat


class Runner:
    def __init__(self):
        self.slider_stats, self.output_stats = load_stats()
        self.sessions, self.names = load_models()
        self.catalog = Catalog(self.slider_stats)
        self.seed = {k: self.output_stats[k]["baseline"] for k in self.output_stats}
        # Every policy delta at zero: the climatological normal for the month.
        self.default = dict(self.catalog.zero)
        self.structural = Structural()
        # Each residual model's output at the default scenario. Layer 3 integrates the
        # deviation from this, never the raw residual — see finalizePrediction() in
        # frontend/models.js for why.
        self._default_residuals: dict[str, float] | None = None

    def raw(self, key: str, cat: dict) -> float:
        vec = np.array(
            [[float(cat.get(n, 0.0)) for n in self.names[key]]], dtype=np.float32
        )
        sess = self.sessions[key]
        return float(sess.run(None, {sess.get_inputs()[0].name: vec})[0].ravel()[0])

    def score(self, key: str, value: float) -> float:
        o = self.output_stats[key]
        return (value - o["min"]) / (o["max"] - o["min"]) * 100.0

    def climate_only(self, sv: dict) -> dict:
        """Human levers back to their climatological normal. Mirrors state.js."""
        return {k: (0.0 if k in HUMAN_SLIDERS else v) for k, v in sv.items()}

    def _raw_residuals(self, sv: dict, month: int, elapsed: int) -> dict:
        """One inference pass per model; returns each model's own raw output.

        Layer 1 is a climate model (PHASE3_PLAN.md §4), so the human deltas are held
        at zero here and Layer 2 supplies the entire human response.
        """
        history = {k: [self.seed[k]] * 24 for k in RESIDUAL_TARGETS}
        cat = fill_response_features(
            self.catalog.build(self.climate_only(sv), month, elapsed), history
        )
        return {key: self.raw(key, cat) for key in MODELS}

    def default_residuals(self, month: int, elapsed: int) -> dict:
        if self._default_residuals is None:
            self._default_residuals = self._raw_residuals(self.default, month, elapsed)
        return self._default_residuals

    def one_step(self, sv: dict, month: int = 7, elapsed: int = 12) -> dict:
        """The full shipped stack: Layer 1 (learned) + Layer 3 (integration) + Layer 2.

        Mirrors runPipeline/finalizePrediction in frontend/models.js.
        """
        bias = self.default_residuals(month, elapsed)
        outputs = self._raw_residuals(sv, month, elapsed)
        structural = self.structural.response(sv, month, elapsed)

        out = {}
        for key, model_output in outputs.items():
            value = model_output
            if key in RESIDUAL_TARGETS:
                forcing = model_output - bias[key]
                value = (
                    self.seed[key]
                    + bias[key]
                    + self.structural.integrate_learned_residual(key, forcing, elapsed)
                )
            out[key] = value + structural.get(key, {}).get("total", 0.0)
        return out

    def layers(self, sv: dict, month: int = 7, elapsed: int = 12) -> dict:
        """Same as one_step, split into its layers, for the provenance table."""
        bias = self.default_residuals(month, elapsed)
        outputs = self._raw_residuals(sv, month, elapsed)
        base = self.default_residuals(month, elapsed)
        structural = self.structural.response(sv, month, elapsed)
        split = {}
        for key in MODELS:
            if key in RESIDUAL_TARGETS:
                learned = self.structural.integrate_learned_residual(
                    key, outputs[key] - bias[key], elapsed
                )
            else:
                learned = outputs[key] - base[key]
            split[key] = {
                "learned": learned,
                "structural": structural.get(key, {}).get("total", 0.0),
            }
        return split

    def rollout(
        self,
        sv: dict,
        month: int = 7,
        months: int = 12,
        lam: dict | None = None,
    ) -> tuple[dict, dict]:
        """Integrate forward. The trajectory ENDS on `month`, so step i is month
        (month - months + i), letting seasonality cycle correctly across the run."""
        history = {k: [self.seed[k]] * 24 for k in RESIDUAL_TARGETS}
        wildlife_hist = [0.0] * 3
        traj: dict[str, list[float]] = {k: [] for k in MODELS}
        cat: dict = {}

        for i in range(1, months + 1):
            m = ((month - 1 - months + i) % 12) + 1
            cat = fill_response_features(self.catalog.build(sv, m, i), history)
            for key in ("grace", "ndvi", "groundwater", "surface_water"):
                prev = history[key][-1]
                level = prev + self.raw(key, cat)
                if lam:
                    level -= lam[key] * (prev - self.seed[key])
                history[key].append(level)
                traj[key].append(level)
                # Downstream models in the same step see the updated upstream state,
                # matching the GRACE -> NDVI ordering in runPipeline().
                cat = fill_response_features(cat, history)
            traj["wildfire"].append(self.raw("wildfire", cat))
            if i % 12 == 0:  # wildlife is annual and autoregressive
                cat["bbs_abundance_anomaly_lag1"] = wildlife_hist[-1]
                cat["bbs_abundance_anomaly_lag2"] = wildlife_hist[-2]
                cat["bbs_abundance_anomaly_roll3"] = float(np.mean(wildlife_hist[-3:]))
                value = self.raw("wildlife", cat)
                wildlife_hist.append(value)
                traj["wildlife"].append(value)

        out = {k: (traj[k][-1] if traj[k] else None) for k in MODELS}
        if out["wildlife"] is None:  # scenario shorter than a year
            for f in (
                "bbs_abundance_anomaly_lag1",
                "bbs_abundance_anomaly_lag2",
                "bbs_abundance_anomaly_roll3",
            ):
                cat[f] = 0.0
            out["wildlife"] = self.raw("wildlife", cat)
        return out, traj


def fit_lambda() -> dict[str, float]:
    """dy_t = a + b*(y_{t-1} - ybar). lambda = -b is the monthly reversion rate."""
    panel = pd.read_csv(PANEL_PATH)
    lam = {}
    print(f"{'target':16s}{'n':>5s}{'lambda':>10s}{'1/lambda (mo)':>15s}{'t':>8s}")
    for key, col in RESIDUAL_TARGETS.items():
        s = panel[col].dropna()
        prev = s.shift(1).dropna()
        dy = (s.loc[prev.index] - prev).to_numpy()
        x = (prev - prev.mean()).to_numpy()
        slope, intercept = np.polyfit(x, dy, 1)
        resid = dy - (slope * x + intercept)
        se = math.sqrt(
            float(np.sum(resid**2)) / (len(dy) - 2) / float(np.sum((x - x.mean()) ** 2))
        )
        lam[key] = -float(slope)
        efold = 1 / -slope if slope < 0 else float("inf")
        print(f"{key:16s}{len(dy):5d}{-slope:10.4f}{efold:15.1f}{slope / se:8.2f}")
    return lam


def print_table(runner: Runner, predict, title: str) -> None:
    """Swing each slider across its full POLICY range with the others at normal."""
    print(f"\n=== {title} ===")
    print(f"{'slider (policy min -> max)':32s}" + "".join(f"{k[:9]:>11s}" for k in MODELS))
    for slider in HUMAN_SLIDERS + CLIMATE_SLIDERS:
        policy = runner.slider_stats[slider]["policy"]
        lo, hi = dict(runner.default), dict(runner.default)
        lo[slider] = policy["min"]
        hi[slider] = policy["max"]
        a, b = predict(lo), predict(hi)
        row = "".join(
            f"{runner.score(k, b[k]) - runner.score(k, a[k]):+11.2f}" for k in MODELS
        )
        span = f"{policy['min']:g}..{policy['max']:g}"
        print(f"{slider[:24]:24s}{span:>8s}{row}" + ("   <-- human" if slider in HUMAN_SLIDERS else ""))
    print("\nUnits: change in the 0-100 output score for a full min -> max slider swing.")


# The direction each human slider must move each output, aggregated from the
# structural layer's own declared lever signs and propagated across transfer edges.
#
# A (slider, output) pair can have MORE THAN ONE lever, and they can disagree: more
# people means more effluent (+) and more municipal pumping (-) on the same stream.
# Where they disagree the net sign is an OUTPUT of the model rather than an
# assumption, so the gate checks stability only and reports which way it resolved.
def expected_signs(runner: "Runner") -> dict[tuple[str, str], int | None]:
    declared: dict[tuple[str, str], set[int]] = {}
    for lever in runner.structural.params["levers"]:
        declared.setdefault((lever["slider"], lever["output"]), set()).add(lever["sign"])

    # Every slider reaching a transfer's source also reaches its destination.
    for transfer in runner.structural.params["transfers"]:
        for (slider, output), signs in list(declared.items()):
            if output != transfer["from"]:
                continue
            propagated = {s * transfer["sign"] for s in signs}
            declared.setdefault((slider, transfer["to"]), set()).update(propagated)

    return {
        key: (next(iter(signs)) if len(signs) == 1 else None)
        for key, signs in declared.items()
    }


def acceptance(runner: Runner, months: int) -> int:
    """Sign stability across all 12 months. Returns the number of failures."""
    signs = expected_signs(runner)
    output_span = {
        k: runner.output_stats[k]["max"] - runner.output_stats[k]["min"]
        for k in runner.output_stats
    }

    print(f"\n=== acceptance: sign stability across 12 months (duration {months} mo) ===")
    print(f"{'lever -> output':52s}{'expect':>7s}{'months ok':>11s}{'min pts':>9s}{'max pts':>9s}  verdict")

    failures = 0
    for (slider, output), sign in signs.items():
        policy = runner.slider_stats[slider]["policy"]
        effects = []
        for month in range(1, 13):
            lo, hi = dict(runner.default), dict(runner.default)
            lo[slider], hi[slider] = policy["min"], policy["max"]
            a = runner.one_step(lo, month, months)
            b = runner.one_step(hi, month, months)
            effects.append(runner.score(output, b[output]) - runner.score(output, a[output]))

        magnitudes = [abs(e) for e in effects]
        nonzero = [e for e in effects if e != 0]

        if sign is None:
            # Opposing levers on the same pair: only stability is gated.
            stable = len(nonzero) == 12 and len({e > 0 for e in nonzero}) == 1
            agreeing = 12 if stable else 0
            resolved = "net +" if nonzero and nonzero[0] > 0 else "net -"
            label = resolved
            ok = stable
        else:
            agreeing = sum(1 for e in effects if e != 0 and (e > 0) == (sign > 0))
            label = "+" if sign > 0 else "-"
            ok = agreeing == 12

        if not ok:
            failures += 1
        print(
            f"{slider[:30]:30s} -> {output:16s}{label:>7s}"
            f"{agreeing:>8d}/12{min(magnitudes):9.2f}{max(magnitudes):9.2f}"
            f"  {'PASS' if ok else 'FAIL'}"
        )

    print(
        "\nMagnitude is reported, not gated (PHASE3_PLAN.md §11.4). A lever whose real\n"
        "physical effect is small should say so, not be inflated to clear a threshold."
    )
    return failures


def no_double_count(runner: "Runner", months: int) -> int:
    """PHASE3_PLAN.md §11.3 — assert no human lever reaches Layer 1.

    §4 splits the model as `ML_climate(...) + Σ β_j (lever_j − baseline_j)`, so a human
    lever that still moved the learned output would be counted twice: once by the fitted
    coefficient and once by the structural β. `groundwater` is the path that matters —
    it is the only model carrying `irrigation_total_withdrawal_mgd`, which is also
    Layer 2's strongest corroborated lever.

    This is a gate rather than a report because the exposure is wider than the feature
    lists suggest and it is reachable by ordinary edits: `climateOnly()` derives its key
    list from `panel: 'human'` in SLIDER_DEFS, so a lever added to the climate panel for
    layout reasons would silently start double-counting. The assertion is exact equality,
    not a tolerance — Layer 1 does not see these values at all, so any nonzero difference
    is a wiring bug and not a numerical one.
    """
    print("=== §11.3 gate: no human lever reaches Layer 1 ===\n")
    pol = {k: runner.slider_stats[k]["policy"] for k in HUMAN_SLIDERS}
    history = {k: [runner.seed[k]] * 24 for k in RESIDUAL_TARGETS}

    def layer1(sv: dict, month: int) -> dict:
        cat = fill_response_features(
            runner.catalog.build(runner.climate_only(sv), month, months), history
        )
        return {k: runner.raw(k, cat) for k in MODELS}

    print(f"{'lever (policy min -> max)':34s} " + " ".join(f"{m[:9]:>10s}" for m in MODELS))
    worst, failures = 0.0, 0
    for lev in HUMAN_SLIDERS:
        cells = []
        for key in MODELS:
            dev = 0.0
            for month in range(1, 13):
                lo, hi = dict(runner.default), dict(runner.default)
                lo[lev], hi[lev] = pol[lev]["min"], pol[lev]["max"]
                dev = max(dev, abs(layer1(hi, month)[key] - layer1(lo, month)[key]))
            cells.append(dev)
            worst = max(worst, dev)
            if dev != 0.0:
                failures += 1
        print(f"{lev:34s} " + " ".join(f"{c:10.1e}" for c in cells))

    print(
        f"\nworst deviation across {len(HUMAN_SLIDERS)}x{len(MODELS)}x12 = "
        f"{len(HUMAN_SLIDERS) * len(MODELS) * 12} comparisons: {worst:.1e}"
    )
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("sweep", "rollout", "lambda", "acceptance", "no-double-count"),
        default="sweep",
    )
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--month-of-year", type=int, default=7)
    ap.add_argument(
        "--revert",
        action="store_true",
        help="add the empirical mean-reversion term (rollout mode only)",
    )
    args = ap.parse_args()

    if args.mode == "lambda":
        fit_lambda()
        return

    runner = Runner()
    if args.mode == "no-double-count":
        failures = no_double_count(runner, args.months)
        if failures:
            raise SystemExit(
                f"\n{failures} lever/output pair(s) still reach Layer 1. The human "
                f"response is counted twice — see PHASE3_PLAN.md §11.3."
            )
        print("\nLayer 1 is bit-identical across every human lever. No double-count.")
        return

    if args.mode == "acceptance":
        failures = acceptance(runner, args.months)
        if failures:
            raise SystemExit(f"\n{failures} lever(s) failed the sign-stability gate.")
        print("\nEvery structural lever holds its declared sign in all 12 months.")
        return

    if args.mode == "sweep":
        print_table(
            runner,
            lambda sv: runner.one_step(sv, args.month_of_year, args.months),
            f"one-step inference, as shipped (elapsed={args.months} mo)",
        )
        return

    lam = fit_lambda() if args.revert else None
    label = "mean-reverting" if args.revert else "naive (NO restoring force)"
    print_table(
        runner,
        lambda sv: runner.rollout(sv, args.month_of_year, args.months, lam)[0],
        f"forward rollout, {label}, {args.months} months",
    )

    # Stability check: with nothing perturbed the state must not wander.
    _, traj = runner.rollout(runner.default, args.month_of_year, args.months, lam)
    print("\nStability at DEFAULT sliders (nothing perturbed):")
    for key in RESIDUAL_TARGETS:
        o = runner.output_stats[key]
        end = traj[key][-1]
        span = abs(end - runner.seed[key]) / (o["max"] - o["min"])
        print(
            f"  {key:14s} seed {runner.seed[key]:+9.4f} -> t={args.months} {end:+11.4f}"
            f"   drift = {span:5.2f}x the full historical range"
        )


if __name__ == "__main__":
    main()
