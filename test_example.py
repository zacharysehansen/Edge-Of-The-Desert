"""Tests for example.py — the one-call integration surface.

    pytest test_example.py -v

These are contract tests, not accuracy tests. PHASE2_REPORT.md scores whether the
models forecast well; `slider_sensitivity.py --mode acceptance` scores whether the
levers move the outputs in the declared direction. What is tested here is narrower
and nobody else covers it: that a prototype calling `run_scenario()` gets numbers
that are internally consistent, correctly labelled, and identical to what the
browser would show.

The load-bearing tests, in rough order of how much they would cost to get wrong:

  * the layer identity           climate + human == value - baseline_value
  * the delta identity           delta == (climate + human) rescaled
  * no double-counting           a human lever must leave Layer 1 untouched
  * bias-cache order independence  month 7 must not depend on having asked for
                                   month 1 first (example.py diverges from the
                                   browser here deliberately — see _prime_bias)
  * no silent clamping           a score past 100 must survive as a score past 100

The last one is a regression test with a specific history: `delta` was briefly
computed by subtracting two scores that had each been clamped to 0-100, which
reported +53.6 for a response whose true value is +77.3.
"""

from __future__ import annotations

import json
import math

import pytest

import example
from example import (
    OUTPUT_META,
    Output,
    run_scenario,
    run_scenario_json,
    slider_specs,
)

OUTPUT_KEYS = ("grace", "ndvi", "groundwater", "surface_water", "wildfire", "wildlife")

HUMAN_SLIDERS = (
    "population",
    "irrigation_total_withdrawal_mgd",
    "public_supply_groundwater_mgd",
    "impervious_pct",
    "mead_pool_elevation",
)
CLIMATE_SLIDERS = ("precipitation_mm_day", "temperature_2m_c", "nclimdiv_pdsi")

# Scenarios that between them touch every lever, both signs, and the extremes of
# each policy range.
SCENARIOS = {
    "normal": {},
    "hot_dry": {"temperature_2m_c": 0.79, "precipitation_mm_day": -32.11},
    "wet_cool": {"temperature_2m_c": -0.71, "precipitation_mm_day": 44.57},
    "drought": {"nclimdiv_pdsi": -1.96},
    "irrigation_up": {"irrigation_total_withdrawal_mgd": 40},
    "irrigation_cut": {"irrigation_total_withdrawal_mgd": -60},
    "mead_low": {"mead_pool_elevation": -95},
    "mead_full": {"mead_pool_elevation": 130},
    "growth": {"population": 3_000_000, "impervious_pct": 2.0},
    "shrink": {"population": -500_000, "impervious_pct": -0.4},
    "public_supply_up": {"public_supply_groundwater_mgd": 60},
    "combined": {
        "population": 1_500_000,
        "irrigation_total_withdrawal_mgd": -30,
        "mead_pool_elevation": -50,
        "temperature_2m_c": 0.4,
        "precipitation_mm_day": -15.0,
    },
}

DURATIONS = (1, 6, 12, 36)


def _span(key: str) -> float:
    stats = example._get_runner().output_stats[key]
    return stats["max"] - stats["min"]


@pytest.fixture(scope="module")
def normal():
    return run_scenario()


# ── Contract ─────────────────────────────────────────────────────────────────


class TestContract:
    def test_returns_all_six_outputs(self, normal):
        assert set(normal) == set(OUTPUT_KEYS)
        assert all(isinstance(o, Output) for o in normal.values())

    def test_every_field_is_finite(self, normal):
        for key, o in normal.items():
            for field in ("value", "score", "score_display", "delta",
                          "baseline_value", "climate", "human"):
                assert math.isfinite(getattr(o, field)), f"{key}.{field} not finite"

    def test_metadata_matches_the_frontend_defs(self, normal):
        """Label, unit and direction must agree with OUTPUT_META, which mirrors
        OUTPUT_DEFS in frontend/state.js. A prototype colours its cards off these."""
        for key, o in normal.items():
            assert o.label == OUTPUT_META[key]["label"]
            assert o.unit == OUTPUT_META[key]["unit"]
            assert o.higher_is_better == OUTPUT_META[key]["higher_is_better"]

    def test_groundwater_is_declared_lower_is_better(self, normal):
        """The sign trap. Depth to water rising means LESS water, and a prototype
        that assumes 'up is good' across the board will colour it backwards."""
        assert normal["groundwater"].higher_is_better is False
        assert normal["wildfire"].higher_is_better is False
        for key in ("grace", "ndvi", "surface_water", "wildlife"):
            assert normal[key].higher_is_better is True

    def test_slider_specs_cover_all_eight_sliders(self):
        specs = slider_specs()
        assert set(specs) == set(HUMAN_SLIDERS) | set(CLIMATE_SLIDERS)
        for key, spec in specs.items():
            assert spec["min"] < spec["max"]
            assert spec["min"] <= spec["default"] <= spec["max"]
            assert spec["panel"] == ("human" if key in HUMAN_SLIDERS else "climate")

    def test_omitted_sliders_default_to_normal(self):
        """run_scenario() with nothing set must equal every delta explicitly zero.
        A delta of 0 means 'normal for this month', NOT zero irrigation."""
        explicit = run_scenario({k: 0.0 for k in slider_specs()})
        for key, o in run_scenario().items():
            assert o.value == pytest.approx(explicit[key].value, rel=1e-12)

    def test_dict_and_keyword_forms_agree(self):
        by_dict = run_scenario({"temperature_2m_c": 0.5})
        by_kwarg = run_scenario(temperature_2m_c=0.5)
        for key in OUTPUT_KEYS:
            assert by_dict[key].value == pytest.approx(by_kwarg[key].value, rel=1e-12)

    def test_deterministic(self):
        first = run_scenario(SCENARIOS["combined"], month=4, duration_months=6)
        second = run_scenario(SCENARIOS["combined"], month=4, duration_months=6)
        for key in OUTPUT_KEYS:
            assert first[key].value == second[key].value


# ── The layer identities ─────────────────────────────────────────────────────


class TestLayerIdentities:
    """Layer 1 + Layer 3 (`climate`) and Layer 2 (`human`) must account for the
    entire departure from baseline, with nothing unexplained in between. This is
    what lets a prototype show attribution and have it add up on screen."""

    @pytest.mark.parametrize("name", list(SCENARIOS))
    @pytest.mark.parametrize("duration", DURATIONS)
    def test_climate_plus_human_equals_change_from_baseline(self, name, duration):
        out = run_scenario(SCENARIOS[name], month=7, duration_months=duration)
        for key, o in out.items():
            assert o.climate + o.human == pytest.approx(
                o.value - o.baseline_value, abs=1e-9
            ), f"{name}/{duration}mo/{key}: layers do not sum to the change"

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_delta_is_the_layer_sum_rescaled(self, name):
        """`delta` must be the same quantity as climate+human, in score points.
        This identity is exactly what clamping used to break."""
        out = run_scenario(SCENARIOS[name], month=7, duration_months=12)
        for key, o in out.items():
            expected = (o.climate + o.human) / _span(key) * 100.0
            assert o.delta == pytest.approx(expected, abs=1e-8), f"{name}/{key}"

    @pytest.mark.parametrize("month", range(1, 13))
    def test_identities_hold_in_every_month(self, month):
        out = run_scenario(SCENARIOS["combined"], month=month, duration_months=12)
        for key, o in out.items():
            assert o.climate + o.human == pytest.approx(
                o.value - o.baseline_value, abs=1e-9
            ), f"month {month}/{key}"

    def test_baseline_scenario_is_at_rest(self):
        """Every delta at zero is the definition of the baseline, so all three of
        delta, climate and human must be exactly zero — not merely small. A nonzero
        value here means the baseline and the scenario disagree about what 'normal'
        is, which is how PHASE3_PLAN.md §4b's rollout used to manufacture drift."""
        for duration in DURATIONS:
            out = run_scenario(month=7, duration_months=duration)
            for key, o in out.items():
                assert o.climate == 0.0, f"{duration}mo/{key} climate"
                assert o.human == 0.0, f"{duration}mo/{key} human"
                assert o.delta == pytest.approx(0.0, abs=1e-12)
                assert o.value == pytest.approx(o.baseline_value, abs=1e-12)

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_levers_sum_to_the_human_term(self, name):
        """The per-lever breakdown is what a prototype shows as attribution. If the
        parts do not sum to the whole, the card is lying about where the number
        came from."""
        out = run_scenario(SCENARIOS[name], month=7, duration_months=12)
        for key, o in out.items():
            total = sum(lever.displacement for lever in o.levers)
            assert total == pytest.approx(o.human, abs=1e-12), f"{name}/{key}"


# ── Layer separation ─────────────────────────────────────────────────────────


class TestNoDoubleCounting:
    """The mirror of `slider_sensitivity.py --mode no-double-count`.

    Layer 1 is a climate model: the human deltas are forced to their climatological
    normal before inference. If a human lever ever moves the learned term, its
    response is being charged twice — once to a fitted coefficient and once to the
    structural beta — over eight lever/output paths (PHASE3_PLAN.md §11.3).
    """

    @pytest.mark.parametrize("slider", HUMAN_SLIDERS)
    def test_human_lever_leaves_layer_1_untouched(self, slider):
        spec = slider_specs()[slider]
        for value in (spec["min"], spec["max"]):
            out = run_scenario({slider: value}, month=7, duration_months=12)
            for key, o in out.items():
                assert o.climate == 0.0, (
                    f"{slider}={value} moved the LEARNED term for {key} "
                    f"by {o.climate} — that is a double-count"
                )

    @pytest.mark.parametrize("slider", CLIMATE_SLIDERS)
    def test_climate_lever_leaves_layer_2_untouched(self, slider):
        """The converse: Layer 2's levers are driven by human sliders only, so a
        climate slider must not manufacture a structural response."""
        spec = slider_specs()[slider]
        for value in (spec["min"], spec["max"]):
            out = run_scenario({slider: value}, month=7, duration_months=12)
            for key, o in out.items():
                assert o.human == 0.0, f"{slider}={value} moved Layer 2 for {key}"

    @pytest.mark.parametrize("slider", HUMAN_SLIDERS)
    def test_human_lever_does_move_something(self, slider):
        """The guard against the tests above passing for the wrong reason: a lever
        wired to nothing would satisfy 'Layer 1 untouched' trivially."""
        spec = slider_specs()[slider]
        out = run_scenario({slider: spec["max"]}, month=7, duration_months=12)
        assert any(o.human != 0.0 for o in out.values()), (
            f"{slider} at its maximum moved no output at all"
        )


# ── Clamping ─────────────────────────────────────────────────────────────────


class TestClamping:
    """A score outside 0-100 means the scenario left the historical record. That is
    a finding, so `score` keeps it and only `score_display` is clamped."""

    def test_score_is_not_clamped(self):
        out = run_scenario(irrigation_total_withdrawal_mgd=40)
        assert out["groundwater"].score > 100.0
        assert out["groundwater"].out_of_range is True

    def test_score_display_is_clamped(self):
        out = run_scenario(irrigation_total_withdrawal_mgd=40)
        assert out["groundwater"].score_display == 100.0

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_score_display_is_score_clamped(self, name):
        for o in run_scenario(SCENARIOS[name], month=7).values():
            assert o.score_display == min(max(o.score, 0.0), 100.0)

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_out_of_range_flags_exactly_the_excursions(self, name):
        for o in run_scenario(SCENARIOS[name], month=7).values():
            assert o.out_of_range == (not 0.0 <= o.score <= 100.0)

    def test_delta_is_not_computed_from_clamped_scores(self):
        """Regression. `delta` was briefly the difference of two scores that had
        each been clamped to 0-100, which reported +53.6 for a response whose true
        value is +77.3 — a 24-point error in a field callers do arithmetic on."""
        o = run_scenario(irrigation_total_withdrawal_mgd=40)["groundwater"]
        assert o.delta == pytest.approx(77.3, abs=0.1)
        assert o.delta != pytest.approx(53.6, abs=0.1)

    def test_scores_stay_on_the_declared_scale(self, normal):
        """At the baseline nothing should be off-scale; if it is, the historical
        min/max in computed_stats.json no longer bracket the models' own normal."""
        for key, o in normal.items():
            assert 0.0 <= o.score <= 100.0, f"{key} is off-scale at baseline"


# ── Direction ────────────────────────────────────────────────────────────────


class TestDirection:
    def test_deeper_water_table_reads_as_worse(self):
        """groundwater is depth to water, so a POSITIVE delta is bad news. This is
        the one a prototype is most likely to render backwards."""
        o = run_scenario(irrigation_total_withdrawal_mgd=40)["groundwater"]
        assert o.delta > 0
        assert o.direction == "worse"

    def test_more_fire_reads_as_worse(self):
        o = run_scenario(SCENARIOS["hot_dry"])["wildfire"]
        assert o.delta > 0
        assert o.direction == "worse"

    def test_more_birds_reads_as_better(self):
        o = run_scenario(precipitation_mm_day=44.57)["wildlife"]
        assert o.delta > 0
        assert o.direction == "better"

    def test_shallower_water_table_reads_as_better(self):
        """The inverted output in the good direction: rain makes depth to water go
        DOWN, and that has to read as an improvement."""
        o = run_scenario(precipitation_mm_day=44.57)["groundwater"]
        assert o.delta < 0
        assert o.direction == "better"

    def test_baseline_reads_as_unchanged(self, normal):
        assert all(o.direction == "unchanged" for o in normal.values())

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_direction_is_consistent_with_delta_and_polarity(self, name):
        for o in run_scenario(SCENARIOS[name], month=7).values():
            if o.direction == "unchanged":
                assert abs(o.delta) < 0.05
            else:
                improving = (o.delta > 0) == o.higher_is_better
                assert o.direction == ("better" if improving else "worse")


# ── Known wrong-signed climate responses ─────────────────────────────────────


class TestKnownClimateSignFailures:
    """Pairs where the SHIPPED model disagrees with physics, pinned as xfail.

    These are not wrapper bugs and nothing here can fix them — they live in the
    learned models. `slider_sensitivity.py --mode climate-signs` is the gate that
    scores them and it currently fails 5 of 18 sustained pairs
    (PHASE3_PLAN.md §30-§31, and the README says so before the check is run).

    They are written down as tests for one reason: a professor wiring a prototype
    WILL drag the rain slider, WILL see vegetation go the wrong way, and will
    otherwise assume the integration is broken. An xfail that names the gate is a
    cheaper answer than that conversation.

    strict=False on purpose: if retraining fixes one, these report XPASS rather
    than failing the suite, which is the signal to re-run the gate and promote the
    pair into TestDirection.
    """

    @pytest.mark.xfail(
        reason="rain -> NDVI is wrong-signed in 4 of 12 months; climate-signs FAIL",
        strict=False,
    )
    def test_rain_should_green_the_vegetation(self):
        assert run_scenario(precipitation_mm_day=44.57)["ndvi"].delta > 0

    @pytest.mark.xfail(
        reason="a wetter PDSI lowers streamflow in every month; climate-signs FAIL "
               "0/12. PDSI is held for the whole scenario and reaches the model "
               "through 12-month rolls",
        strict=False,
    )
    def test_wetter_drought_index_should_raise_streamflow(self):
        assert run_scenario(nclimdiv_pdsi=2.42)["surface_water"].delta > 0

    def test_the_gate_that_owns_these_still_exists(self):
        """If this import breaks, the xfails above have lost their referee and
        somebody should find out why before deleting them."""
        from slider_sensitivity import climate_signs  # noqa: F401


# ── Input handling ───────────────────────────────────────────────────────────


class TestInputValidation:
    def test_unknown_slider_raises_and_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="Unknown slider"):
            run_scenario(popluation=1)
        try:
            run_scenario(nonsense=1)
        except ValueError as err:
            assert "population" in str(err)

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            run_scenario(population="lots")

    def test_out_of_range_warns_and_clamps(self):
        """Clamped, because Layer 2's betas are linear and extrapolating them past
        their calibration returns a confident meaningless number. Never silently."""
        with pytest.warns(UserWarning, match="outside the slider range"):
            out = run_scenario(population=9e9)
        capped = run_scenario(population=3_000_000)
        for key in OUTPUT_KEYS:
            assert out[key].value == pytest.approx(capped[key].value, rel=1e-12)

    def test_strict_raises_instead_of_clamping(self):
        with pytest.raises(ValueError, match="outside its policy range"):
            run_scenario(population=9e9, strict=True)

    def test_in_range_values_do_not_warn(self, recwarn):
        run_scenario(population=3_000_000, strict=True)
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    @pytest.mark.parametrize("month", [0, 13, -1])
    def test_bad_month_raises(self, month):
        with pytest.raises(ValueError, match="month must be 1-12"):
            run_scenario(month=month)

    @pytest.mark.parametrize("duration", [0, -5])
    def test_bad_duration_raises(self, duration):
        with pytest.raises(ValueError, match="duration_months must be"):
            run_scenario(duration_months=duration)


# ── Caching ──────────────────────────────────────────────────────────────────


class TestBiasCache:
    """example.py gives each (month, duration) its own residual-bias term, which the
    browser does not — calibrateBaselines() runs once at load for July/12mo and never
    re-runs when the month dropdown changes (see _prime_bias). That divergence is
    deliberate and it pokes a private attribute on Runner, so it is pinned here.
    """

    @staticmethod
    def _clear():
        example._bias_cache.clear()
        example._baseline_cache.clear()
        if example._runner is not None:
            example._runner._default_residuals = None

    def test_result_does_not_depend_on_what_was_asked_first(self):
        self._clear()
        fresh = run_scenario(temperature_2m_c=0.5, month=7)

        for primer in (1, 3, 11):
            self._clear()
            run_scenario(temperature_2m_c=0.5, month=primer)
            after = run_scenario(temperature_2m_c=0.5, month=7)
            for key in OUTPUT_KEYS:
                assert after[key].value == pytest.approx(fresh[key].value, rel=1e-12), (
                    f"month 7 changed after asking for month {primer} first — "
                    f"the residual bias is leaking between months"
                )

    def test_duration_does_not_leak_between_calls(self):
        self._clear()
        fresh = run_scenario(temperature_2m_c=0.5, month=7, duration_months=36)
        self._clear()
        run_scenario(temperature_2m_c=0.5, month=7, duration_months=1)
        after = run_scenario(temperature_2m_c=0.5, month=7, duration_months=36)
        for key in OUTPUT_KEYS:
            assert after[key].value == pytest.approx(fresh[key].value, rel=1e-12)

    def test_different_months_give_different_answers(self):
        """The guard against the cache being order-independent because it is inert."""
        january = run_scenario(month=1)
        july = run_scenario(month=7)
        assert january["ndvi"].value != july["ndvi"].value


# ── Agreement with the rest of the repo ──────────────────────────────────────


class TestAgreesWithRunner:
    """example.py must add no arithmetic of its own. Runner is the headless mirror
    that `scripts/phase3/check_catalog_parity.py` holds to the browser, so matching
    Runner is what makes these numbers the app's numbers."""

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_values_match_runner_one_step(self, name):
        runner = example._get_runner()
        deltas = dict(runner.default)
        deltas.update(SCENARIOS[name])

        example._prime_bias(runner, 7, 12)
        expected = runner.one_step(deltas, 7, 12)

        for key, o in run_scenario(SCENARIOS[name], month=7, duration_months=12).items():
            assert o.value == pytest.approx(expected[key], rel=1e-12), f"{name}/{key}"

    def test_score_matches_runner_score(self):
        runner = example._get_runner()
        for key, o in run_scenario(SCENARIOS["hot_dry"], month=7).items():
            assert o.score == pytest.approx(runner.score(key, o.value), rel=1e-12)


# ── JSON surface ─────────────────────────────────────────────────────────────


class TestJsonSurface:
    def test_round_trips_through_json(self):
        payload = run_scenario_json(SCENARIOS["combined"], month=5, duration_months=6)
        restored = json.loads(json.dumps(payload))
        assert set(restored) == set(OUTPUT_KEYS)

    def test_json_carries_the_same_numbers(self):
        typed = run_scenario(SCENARIOS["growth"], month=7)
        payload = run_scenario_json(SCENARIOS["growth"], month=7)
        for key in OUTPUT_KEYS:
            assert payload[key]["value"] == typed[key].value
            assert payload[key]["score"] == typed[key].score
            assert payload[key]["out_of_range"] == typed[key].out_of_range

    def test_levers_survive_serialization(self):
        payload = run_scenario_json(SCENARIOS["growth"], month=7)
        levers = payload["surface_water"]["levers"]
        assert levers, "growth should reach streamflow through Layer 2"
        for lever in levers:
            assert set(lever) >= {"id", "displacement", "points", "kind", "tier"}
            assert lever["tier"] in ("corroborated", "structural-only")

    def test_lever_ids_are_distinct_within_an_output(self):
        """One slider can reach one output by several paths that disagree — more
        people means more effluent (+) and more municipal pumping (-) on the same
        stream. Keying attribution by slider would collapse them."""
        payload = run_scenario_json({"population": 3_000_000}, month=7)
        ids = [lever["id"] for lever in payload["surface_water"]["levers"]]
        assert len(ids) == len(set(ids))


# ── Coverage sweep ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("duration", DURATIONS)
@pytest.mark.parametrize("month", range(1, 13))
def test_every_month_and_duration_runs_clean(month, duration):
    out = run_scenario(SCENARIOS["combined"], month=month, duration_months=duration)
    assert set(out) == set(OUTPUT_KEYS)
    for key, o in out.items():
        assert math.isfinite(o.value), f"{month}/{duration}mo/{key} is not finite"
        assert math.isfinite(o.score)


# ── Golden values ────────────────────────────────────────────────────────────

# A canary, not a specification. These are what the currently exported models
# produce; re-running Phase 2 or Phase 3 SHOULD change them. If this is the only
# failing test after a retrain, read the new numbers, satisfy yourself that the
# change is the one you intended, and update them here.
GOLDEN = {
    "hot_dry": {
        "grace": -0.05113563498407602,
        "ndvi": 0.23685245147049427,
        "groundwater": 0.11590774030685425,
        "surface_water": 0.17590468401908876,
        "wildfire": 0.4919384717941284,
        "wildlife": -0.02030629850924015,
    },
    "irrigation_up": {
        "grace": -0.05707963251279222,
        "ndvi": 0.25399934668622576,
        "groundwater": 1.5553101185406815,
        "surface_water": 0.45123234574988197,
        "wildfire": 0.3755187690258026,
        "wildlife": -0.005495370806374972,
    },
}


@pytest.mark.parametrize("name", list(GOLDEN))
def test_golden_values(name):
    out = run_scenario(SCENARIOS[name], month=7, duration_months=12)
    for key, expected in GOLDEN[name].items():
        assert out[key].value == pytest.approx(expected, rel=1e-9), (
            f"{name}/{key} changed. If you re-ran Phase 2 or Phase 3 this is "
            f"expected — verify the change is intended and update GOLDEN."
        )
