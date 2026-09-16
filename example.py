"""

This is the integration surface for a prototype that is NOT this repo's frontend.
It is the Python equivalent of calling `runAll()` in `frontend/models.js`, and it
returns what the output cards render, already assembled:

    from example import run_scenario

    out = run_scenario(temperature_2m_c=0.79, precipitation_mm_day=-32.11)
    out["wildfire"].value    # 0.4919 raw index
    out["wildfire"].score    # 91.7   on the 0-100 historical scale (NOT clamped)
    out["wildfire"].delta    # +21.7  score points vs. the same month at normal

Nothing else needs to be called first. No ONNX handling, no feature catalog, no
Layer 2 bookkeeping, no clamping.

WHY THIS WRAPPER EXISTS
-----------------------
Loading `model/*.onnx` and feeding it a scenario is NOT enough to get the numbers
the app shows,the human sliders come back flat and the
model looks broken. The shipped answer is three layers (PHASE3_PLAN.md §4):

    output = ML_climate(climate, season, lag1)      <- Layer 1, the ONNX models
           + integrated over the scenario duration  <- Layer 3
           + sum of beta * (lever - baseline)       <- Layer 2, structural.py

Layer 1 is a CLIMATE model. The human levers are forced to their climatological
normal before inference (`climate_only()` in slider_sensitivity.py, mirroring
state.js), because four of the six models carry no human feature at all. Every bit
of the human-lever response comes from Layer 2, which is water balance and published
shortage tiers rather than a fit. Feeding the human deltas into the ONNX models as
well would double-count eight lever/output paths — that is what
`slider_sensitivity.py --mode no-double-count` exists to forbid.

So Layer 2 cannot be folded into the model INPUTS. It is added to the OUTPUT, in the
output's own units, which is what `run_scenario` does for you. The per-lever
breakdown is on each result if a prototype wants to show attribution.

RELATION TO THE REST OF THE REPO
--------------------------------
Everything here delegates to `scripts/phase3/slider_sensitivity.py::Runner`, which is
the headless mirror of the browser stack and is held to it by
`scripts/phase3/check_catalog_parity.py`. This file adds no arithmetic of its own —
only the argument handling, the baseline, the scoring and the metadata that the
browser keeps in `ui.js` and `state.js`.

Run it directly for a worked demo:

    python example.py                # a table of three scenarios
    python example.py --json         # the same thing as JSON, for a non-Python app
    python example.py --describe     # the input/output contract
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts" / "phase3"))

from slider_sensitivity import Runner  # noqa: E402

# ── Output metadata ──────────────────────────────────────────────────────────
#
# Mirrors OUTPUT_DEFS in frontend/state.js. Duplicated for the same reason
# structural.py duplicates structural.js: a prototype in another language should not
# have to parse an ES module to learn that a RISING groundwater number is bad news.
#
# `rising` is the plain-English sentence to put next to an up arrow. Drive colour
# off `higher_is_better`, not off the sign of the delta.

OUTPUT_META = {
    "grace": {
        "label": "GRACE Groundwater Anomaly",
        "unit": "m",
        "higher_is_better": True,
        "rising": "more water stored",
        "note": "Liquid-water-equivalent thickness, straight from the source. An "
        "anomaly centred near zero, not a level.",
    },
    "ndvi": {
        "label": "NDVI Vegetation Health",
        "unit": "NDVI",
        "higher_is_better": True,
        "rising": "greener",
        "note": "A level, roughly 0.19-0.28 over the historical range.",
    },
    "groundwater": {
        "label": "Well Depth vs Normal (Cochise basins)",
        "unit": "ft",
        "higher_is_better": False,
        "rising": "water table DEEPER - less water",
        "note": "Depth to water, so POSITIVE IS WORSE. A per-station anomaly index "
        "for the Cochise County wells, not an eight-county blend (§28-§29).",
    },
    "surface_water": {
        "label": "Streamflow vs Normal",
        "unit": "log ratio",
        "higher_is_better": True,
        "rising": "more flow past the gages",
        "note": "Per-gage log anomaly: 0 is normal flow, +0.7 is about double.",
    },
    "wildfire": {
        "label": "Wildfire Risk Index",
        "unit": "",
        "higher_is_better": False,
        "rising": "more burned area",
        "note": "Not a residual model - it predicts the level directly.",
    },
    "wildlife": {
        "label": "Bird Abundance vs Normal",
        "unit": "log ratio",
        "higher_is_better": True,
        "rising": "more birds",
        "note": "Per-route log-abundance anomaly: 0 is an average year. Annual, so a "
        "scenario shorter than 12 months barely moves it.",
    },
}

VALID_MONTHS = range(1, 13)

# What the app's dropdown offers. Other values still work — a duration is just how
# many months back the scenario is treated as having been held — but these are the
# ones the acceptance gates are run at.
SUGGESTED_DURATIONS = (1, 3, 6, 12, 24, 36)


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Lever:
    """One structural (Layer 2) path's contribution to one output."""

    id: str
    displacement: float  # in the output's own units
    points: float  # the same thing in 0-100 score points
    kind: str  # "rate" | "level" | "transfer"
    tier: str  # "corroborated" | "structural-only"
    slider: str | None = None
    source_output: str | None = None  # set when kind == "transfer"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "displacement": self.displacement,
            "points": self.points,
            "kind": self.kind,
            "tier": self.tier,
            "slider": self.slider,
            "source_output": self.source_output,
        }


@dataclass(frozen=True)
class Output:
    """One environmental response under one scenario.

    `value` is the number in the variable's own units and is what a prototype should
    plot. `score` rescales it onto the 0-100 historical range and is NOT clamped, so
    it can leave 0-100 when the scenario does; `score_display` is the clamped version
    for a progress bar, and `out_of_range` says when the two differ.
    """

    key: str
    label: str
    value: float
    unit: str
    # NOT clamped. The 0-100 scale is the historical min-max range, so a score
    # outside it is a real finding — this scenario leaves anything the region has
    # experienced — and clamping would hide it behind a number that looks in-range.
    # Clamping is a rendering concern: `normalizeOutput` in state.js clamps because
    # its result becomes a bar's CSS width, while Runner.score() in
    # slider_sensitivity.py does not clamp at all. Use `score_display` for a bar and
    # `score` for anything you compute with.
    score: float
    score_display: float  # `score` clamped to 0-100, ready for a progress bar
    delta: float  # score points vs. the same month/duration with every lever normal
    baseline_value: float  # the raw value that delta is measured from
    higher_is_better: bool
    rising: str
    direction: str  # "better" | "worse" | "unchanged", already sign-corrected
    climate: float  # Layer 1 + Layer 3, in raw units, as a change from baseline
    human: float  # Layer 2, in raw units
    out_of_range: bool  # score left 0-100: beyond the historical record, not an error
    levers: tuple[Lever, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "score": self.score,
            "score_display": self.score_display,
            "delta": self.delta,
            "baseline_value": self.baseline_value,
            "higher_is_better": self.higher_is_better,
            "rising": self.rising,
            "direction": self.direction,
            "climate": self.climate,
            "human": self.human,
            "out_of_range": self.out_of_range,
            "levers": [lever.to_dict() for lever in self.levers],
        }


# ── The runner, loaded once ──────────────────────────────────────────────────
#
# Building a Runner opens six ONNX sessions and reads the stats files, which takes a
# second or two. A prototype calling run_scenario() on every slider drag must not pay
# that each time, so it is cached here and the caches below key off it.

_runner: Runner | None = None
_bias_cache: dict[tuple[int, int], dict] = {}
_baseline_cache: dict[tuple[int, int], dict] = {}


def _get_runner() -> Runner:
    global _runner
    if _runner is None:
        _runner = Runner()
    return _runner


def _prime_bias(runner: Runner, month: int, duration: int) -> None:
    """Pin the residual-bias term to THIS month and duration.

    Runner caches `_default_residuals` on the instance for whichever scenario asked
    first, which is faithful to the browser (calibrateBaselines runs once, at load,
    for July/12mo, and never re-runs when the month dropdown changes). That is fine
    for an app where the bias is a fixed display anchor, and wrong for a library
    someone will call across all twelve months. Each (month, duration) therefore gets
    its own bias, computed once and reused.
    """
    key = (month, duration)
    if key not in _bias_cache:
        runner._default_residuals = None
        _bias_cache[key] = runner._raw_residuals(runner.default, month, duration)
    runner._default_residuals = _bias_cache[key]


def _baseline(runner: Runner, month: int, duration: int) -> dict:
    """Every lever at its climatological normal — what `delta` is measured against."""
    key = (month, duration)
    if key not in _baseline_cache:
        _prime_bias(runner, month, duration)
        _baseline_cache[key] = runner.one_step(dict(runner.default), month, duration)
    return _baseline_cache[key]


# ── Input handling ───────────────────────────────────────────────────────────


def slider_specs() -> dict[str, dict]:
    """The eight policy sliders: units, range, and which panel they belong to.

    Read from frontend/computed_stats.json via the Runner, so a Phase 3 re-run cannot
    leave a prototype's UI pointing at a range that no longer exists.
    """
    runner = _get_runner()
    human = {
        "population",
        "irrigation_total_withdrawal_mgd",
        "public_supply_groundwater_mgd",
        "impervious_pct",
        "mead_pool_elevation",
    }
    specs = {}
    for key, policy in runner.catalog.policy.items():
        specs[key] = {
            "unit": policy["unit"],
            "min": policy["min"],
            "max": policy["max"],
            "step": policy["step"],
            "default": policy["default"],
            "panel": "human" if key in human else "climate",
        }
    return specs


def _resolve_deltas(
    deltas: dict | None, kwargs: dict, strict: bool
) -> tuple[dict, list[str]]:
    """Merge the dict and keyword forms, reject unknown names, clamp to range.

    Clamping rather than raising is deliberate: a prototype wiring a slider to a
    rounded range will otherwise crash on the last pixel. The clamps are returned so
    a caller that wants to be strict can surface them.
    """
    runner = _get_runner()
    policy = runner.catalog.policy

    merged = dict(deltas or {})
    merged.update(kwargs)

    unknown = sorted(set(merged) - set(policy))
    if unknown:
        raise ValueError(
            f"Unknown slider(s): {', '.join(unknown)}. "
            f"Valid sliders are: {', '.join(sorted(policy))}."
        )

    values = dict(runner.catalog.zero)
    clamped: list[str] = []
    for key, raw in merged.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Slider '{key}' must be a number, got {raw!r}.") from None
        low, high = policy[key]["min"], policy[key]["max"]
        if value < low or value > high:
            if strict:
                raise ValueError(
                    f"Slider '{key}' = {value} is outside its policy range "
                    f"[{low}, {high}] ({policy[key]['unit']})."
                )
            clamped.append(f"{key}={value} -> {min(max(value, low), high)}")
            value = min(max(value, low), high)
        values[key] = value
    return values, clamped


# ── The public entry point ───────────────────────────────────────────────────


def run_scenario(
    deltas: dict | None = None,
    *,
    month: int = 7,
    duration_months: int = 12,
    strict: bool = False,
    **kwargs,
) -> dict[str, Output]:
    """Run one scenario through all three layers and return the six responses.

    Arguments
    ---------
    deltas
        Policy DELTAS keyed by slider name, or pass them as keywords. Every slider
        omitted sits at 0, which means "normal for this month" — NOT zero irrigation.
        A delta is a departure from a deseasonalized baseline in units a person can
        defend (people added, % of annual withdrawal, points of impervious cover,
        feet of reservoir elevation), because a raw percentile range is not a policy
        axis: irrigation's raw maximum only ever meant "June" (PHASE3_PLAN.md D5).
        Call `slider_specs()` for the names, units and ranges.
    month
        1-12. This is what supplies seasonality — the sliders do not.
    duration_months
        How long the scenario is treated as having been held. Drives both the lag
        features and the Layer 2/3 integration, so 1 and 36 are genuinely different
        questions, not a smoothing knob.
    strict
        Raise on an out-of-range delta instead of clamping it to the policy range.

    Returns
    -------
    A dict of six `Output` objects keyed by 'grace', 'ndvi', 'groundwater',
    'surface_water', 'wildfire', 'wildlife'. Use `.value` to plot in real units and
    `.score` for a 0-100 bar. `.climate` and `.human` split the change from baseline
    into the learned and structural layers and sum exactly to `value - baseline_value`.

    Examples
    --------
    >>> out = run_scenario(irrigation_total_withdrawal_mgd=40)
    >>> round(out["groundwater"].value, 2)      # feet deeper than normal
    1.56
    >>> out["groundwater"].out_of_range         # past the historical maximum
    True
    """
    if month not in VALID_MONTHS:
        raise ValueError(f"month must be 1-12, got {month}.")
    if duration_months < 1:
        raise ValueError(f"duration_months must be >= 1, got {duration_months}.")

    runner = _get_runner()
    values, clamped_inputs = _resolve_deltas(deltas, kwargs, strict)
    if clamped_inputs:
        # Input clamping is the opposite call from output clamping, and for a reason.
        # A score outside 0-100 is still a meaningful linear rescaling of a number the
        # model really produced. A delta outside its policy range is an extrapolation
        # of Layer 2's betas past anything they were calibrated on, and those are
        # linear — population at 9e9 would return a confident, meaningless answer.
        # So the range is enforced. It is never enforced SILENTLY.
        warnings.warn(
            "Policy delta(s) outside the slider range, clamped: "
            + "; ".join(clamped_inputs)
            + ". Pass strict=True to raise instead.",
            stacklevel=2,
        )

    _prime_bias(runner, month, duration_months)
    raw = runner.one_step(values, month, duration_months)
    split = runner.layers(values, month, duration_months)
    structural = runner.structural.response(values, month, duration_months)
    baseline = _baseline(runner, month, duration_months)

    results = {}
    for key, value in raw.items():
        meta = OUTPUT_META[key]
        stats = runner.output_stats[key]
        span = stats["max"] - stats["min"]

        # Both scores unclamped, so `delta` stays exactly (climate + human) rescaled.
        # Clamping first cost 24 points on the irrigation +40% groundwater card: the
        # true delta is +77.3 and two clamped scores reported +53.6.
        score = runner.score(key, value)
        delta = score - runner.score(key, baseline[key])

        # "Better" is not "up". Groundwater is depth to water, so a rise is a loss.
        if abs(delta) < 0.05:
            direction = "unchanged"
        elif (delta > 0) == meta["higher_is_better"]:
            direction = "better"
        else:
            direction = "worse"

        levers = tuple(
            Lever(
                id=lever_id,
                displacement=detail["displacement"],
                points=(detail["displacement"] / span * 100.0) if span else 0.0,
                kind=detail["kind"],
                tier=detail["tier"],
                slider=detail.get("slider"),
                source_output=detail.get("from"),
            )
            for lever_id, detail in sorted(
                structural.get(key, {}).get("byLever", {}).items(),
                key=lambda kv: -abs(kv[1]["displacement"]),
            )
        )

        results[key] = Output(
            key=key,
            label=meta["label"],
            value=value,
            unit=meta["unit"],
            score=score,
            score_display=min(max(score, 0.0), 100.0),
            delta=delta,
            baseline_value=baseline[key],
            higher_is_better=meta["higher_is_better"],
            rising=meta["rising"],
            direction=direction,
            climate=split[key]["learned"],
            human=split[key]["structural"],
            out_of_range=not (0.0 <= score <= 100.0),
            levers=levers,
        )
    return results


def run_scenario_json(deltas: dict | None = None, **kwargs) -> dict:
    """`run_scenario` as plain JSON-able dicts, for a prototype in another language.

    Shell out to `python example.py --json ...` and parse stdout, or import this from
    a small Flask/FastAPI route.
    """
    return {k: out.to_dict() for k, out in run_scenario(deltas, **kwargs).items()}


def describe() -> str:
    """The input/output contract as readable text."""
    lines = ["INPUTS — policy deltas, 0 = normal for the selected month", ""]
    for key, spec in slider_specs().items():
        lines.append(
            f"  {key:34s} {spec['min']:>10g} .. {spec['max']:<10g} "
            f"{spec['unit']}  [{spec['panel']}]"
        )
    lines += [
        "",
        "  month             1-12, supplies seasonality",
        f"  duration_months   how long the scenario is held; app offers "
        f"{', '.join(str(d) for d in SUGGESTED_DURATIONS)}",
        "",
        "OUTPUTS",
        "",
    ]
    for key, meta in OUTPUT_META.items():
        arrow = "higher is better" if meta["higher_is_better"] else "LOWER is better"
        lines.append(f"  {key:16s} {meta['label']}")
        lines.append(f"  {'':16s}   unit: {meta['unit'] or '(index)'} — {arrow}")
        lines.append(f"  {'':16s}   rising = {meta['rising']}")
        lines.append(f"  {'':16s}   {meta['note']}")
        lines.append("")
    return "\n".join(lines)


# ── Demo ─────────────────────────────────────────────────────────────────────

DEMO_SCENARIOS = {
    "normal (all levers at climatological normal)": {},
    "hot and dry": {"temperature_2m_c": 0.79, "precipitation_mm_day": -32.11},
    "irrigation +40% of annual withdrawal": {"irrigation_total_withdrawal_mgd": 40},
    "Lake Mead down 95 ft": {"mead_pool_elevation": -95},
    "+3M people, +2 pts impervious": {"population": 3_000_000, "impervious_pct": 2.0},
}


def _print_table(name: str, out: dict[str, Output]) -> None:
    print(f"\n{name}")
    print("-" * 78)
    print(
        f"{'output':16s}{'value':>11s}{'unit':>11s}{'score':>8s}{'delta':>8s}"
        f"{'climate':>10s}{'human':>10s}"
    )
    for key, o in out.items():
        flag = " *" if o.out_of_range else ""
        print(
            f"{key:16s}{o.value:>11.4f}{o.unit or '-':>11s}{o.score:>8.1f}"
            f"{o.delta:>+8.1f}{o.climate:>+10.4f}{o.human:>+10.4f}{flag}"
        )
        for lever in o.levers:
            # The lever id, not just the slider: one slider can reach one output by
            # several paths that disagree. More people means more effluent (+) and
            # more municipal pumping (-) on the same stream, and printing both as
            # "population" makes the net look like an arithmetic error.
            via = lever.id if lever.kind != "transfer" else f"{lever.id} (transfer)"
            print(f"{'':18s}via {via:36s}{lever.displacement:>+9.4f}  [{lever.tier}]")
    if any(o.out_of_range for o in out.values()):
        print(
            "  * score is outside 0-100: this scenario leaves the historical record. "
            "Not an error, and not clamped — use .score_display for a bar."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--duration", type=int, default=12)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SLIDER=VALUE",
        help="a policy delta, e.g. --set temperature_2m_c=0.79 (repeatable)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON, not a table")
    parser.add_argument("--describe", action="store_true", help="print the contract")
    args = parser.parse_args()

    if args.describe:
        print(describe())
        return

    if args.set:
        deltas = {}
        for item in args.set:
            if "=" not in item:
                parser.error(f"--set expects SLIDER=VALUE, got {item!r}")
            slider, _, value = item.partition("=")
            deltas[slider.strip()] = float(value)
        out = run_scenario(deltas, month=args.month, duration_months=args.duration)
        if args.json:
            print(json.dumps({k: o.to_dict() for k, o in out.items()}, indent=2))
        else:
            _print_table(
                f"scenario: {deltas}  month={args.month} "
                f"duration={args.duration}mo",
                out,
            )
        return

    if args.json:
        payload = {
            name: {
                k: o.to_dict()
                for k, o in run_scenario(
                    deltas, month=args.month, duration_months=args.duration
                ).items()
            }
            for name, deltas in DEMO_SCENARIOS.items()
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"month={args.month}  duration={args.duration} months")
    print(
        "value/climate/human are in each output's own units; score and delta are "
        "0-100 points."
    )
    for name, deltas in DEMO_SCENARIOS.items():
        out = run_scenario(deltas, month=args.month, duration_months=args.duration)
        _print_table(name, out)


if __name__ == "__main__":
    main()
