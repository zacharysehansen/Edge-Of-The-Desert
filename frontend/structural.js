// Layer 2 — the structural response layer (PHASE3_PLAN.md §4)
// Layer 3 — mean-reverting integration over the scenario duration (§4b)
//
// The learned models are climate forecasters. Four of the six carry no human-lever
// input at all, and where the features do exist PHASE3_PLAN.md §10 measured that the
// panel cannot identify them: of 21 lever/target pairs with a declared physical sign,
// 3 were found and 12 had no measurable effect. So the human response is supplied
// here instead, with signs and magnitudes fixed by water balance, land-cover
// arithmetic and published policy.
//
// Every number lives in structural_params.json with a source and a status tag. This
// file holds only arithmetic, so the Python mirror in slider_sensitivity.py can
// duplicate ~40 lines of formula and never a value. `node frontend/structural.js
// --dump` prints a fixed scenario set for scripts/phase3/check_catalog_parity.py to
// compare against.
//
// WHY THE INTEGRATION IS NOT OPTIONAL
// -----------------------------------
// A pumping change moves a storage balance, so its effect accumulates: that is the
// entire physical story, and charging it once (as the app does today) discards it.
// But naive accumulation is unbounded — §4a measured surface water reaching 3e7 times
// normal flow — so the forcing is integrated against the empirical mean-reversion
// rate of §4b:
//
//     z_t = (1 - λ)·z_{t-1} + s        =>      z_n = (s/λ)·(1 - (1-λ)^n)
//
// which is exact for a constant forcing, needs no loop, and converges to s/λ instead
// of ramping. At n = 1 it returns exactly s, so a one-month scenario is unchanged
// from the current behaviour.

import params from './structural_params.json' with { type: 'json' };
import { SLIDER_POLICY, ZERO_DELTAS, sliderRaw } from './state.js';

const C = params.constants;

// Outputs Layer 2 can reach. wildfire is absent on purpose and permanently: ignition
// is not the limiting factor for large-fire extent in the Southwest, which is what the
// MTBS target measures, and no coefficient from population or impervious cover to that
// could be sourced. It ships as climate-only and the interface says so.
const STRUCTURAL_OUTPUTS = ['groundwater', 'grace', 'ndvi', 'surface_water', 'wildlife'];

function value(entry) {
    return entry.value;
}

// ── The pumping chain ─────────────────────────────────────────────────────────

// Extra groundwater pumped, in AF/month, attributable to one slider's policy delta.
// Irrigation and public supply are a % of their own baseline withdrawal; population
// converts through the per-capita municipal draw derived from the project's own data.
function pumpingAfPerMonth(sliderKey, delta) {
    const perUnit = params.pumping_mgd_per_slider_unit[sliderKey];
    if (!perUnit) return 0;
    return perUnit.value * delta * value(C.af_per_mgd_month);
}

// Arizona's combined 2007 Interim Guidelines shortage plus DCP contribution, in
// thousand acre-feet per year, for a given Lake Mead elevation. A descending-threshold
// step function transcribed from LBOps Table 1 — this is law, not inference.
function meadReductionKafPerYear(elevationFt) {
    for (const tier of params.mead_tiers) {
        if (elevationFt > tier.above) return tier.az_combined_kaf;
    }
    return params.mead_tiers[params.mead_tiers.length - 1].az_combined_kaf;
}

// A lower reservoir means a larger CAP cut, part of which lands in-region and part of
// THAT is replaced by pumping rather than by fallowing. Both shares are assumptions
// with stated bands; they scale this lever linearly.
function meadSubstitutedPumpingAfPerMonth(delta) {
    const baseline = params.mead_baseline_elevation;
    const cutKaf =
        meadReductionKafPerYear(baseline + delta) - meadReductionKafPerYear(baseline);
    return (
        (cutKaf * 1000 *
            value(C.region_share_of_az_reduction) *
            value(C.groundwater_substitution_fraction)) /
        12
    );
}

// A flow change, as the mean-over-gages log anomaly the surface water model predicts.
// The target is mean_g ln(Q_g / Q_g_normal), so a change distributed in proportion to
// existing flow converts exactly: Δanomaly = ln(1 + ΔQ / Q_regional). Effluent is NOT
// distributed that way — it lands on a few reaches — so this overstates that one lever
// and the overstatement is stated rather than damped by an invented factor.
function flowToLogAnomaly(deltaCfs) {
    const regional = value(C.regional_baseline_cfs);
    return Math.log(Math.max(1e-6, 1 + deltaCfs / regional));
}

// Storm runoff added by converting desert to pavement, in cfs.
function runoffCfs(deltaImperviousPoints) {
    const areaFraction = deltaImperviousPoints / 100;
    const contrast =
        value(C.runoff_coefficient_impervious) - value(C.runoff_coefficient_natural);
    const regionM2 = value(C.region_acres) * SQ_M_PER_ACRE;
    const depthMetresPerSecond = value(C.mean_precipitation_mm_day) / 1000 / 86400;
    return areaFraction * contrast * depthMetresPerSecond * regionM2 * CFS_PER_CMS;
}

const SQ_M_PER_ACRE = 4046.8564224;
const CFS_PER_CMS = 35.3147;

// ── Per-lever forcing ─────────────────────────────────────────────────────────

// Returns the lever's contribution in the OUTPUT's own units: ft/month for
// groundwater depth, metres of equivalent water height per month for GRACE, NDVI
// index for the two land-cover levers.
function leverContribution(lever, deltas, month) {
    const delta = deltas[lever.slider] ?? 0;
    if (delta === 0) return 0;

    switch (lever.path) {
        case 'pumping_to_depth':
            return pumpingAfPerMonth(lever.slider, delta) / value(C.storage_af_per_ft);

        case 'mead_tier_to_depth':
            return meadSubstitutedPumpingAfPerMonth(delta) / value(C.storage_af_per_ft);

        case 'pumping_to_storage':
            return -pumpingAfPerMonth(lever.slider, delta) * value(C.grace_units_per_af);

        case 'mead_tier_to_storage':
            return -meadSubstitutedPumpingAfPerMonth(delta) * value(C.grace_units_per_af);

        case 'effluent_to_flow': {
            // Municipal supply that reaches the sewer comes back as treated effluent,
            // and on the Santa Cruz that effluent IS the perennial flow.
            const mgd = delta * (value(C.gpcd_groundwater) / 1e6)
                * value(C.effluent_return_fraction);
            return flowToLogAnomaly(mgd * value(C.cfs_per_mgd));
        }

        case 'runoff_to_flow':
            return flowToLogAnomaly(runoffCfs(delta));

        case 'capture_to_flow': {
            // Part of what is pumped would otherwise have reached a stream, so more
            // pumping means less flow.
            const captured = pumpingAfPerMonth(lever.slider, delta)
                * value(C.stream_capture_fraction);
            return flowToLogAnomaly(-captured * value(C.cfs_per_af_month));
        }

        case 'mead_capture_to_flow': {
            const captured = meadSubstitutedPumpingAfPerMonth(delta)
                * value(C.stream_capture_fraction);
            return flowToLogAnomaly(-captured * value(C.cfs_per_af_month));
        }

        case 'landcover_impervious':
            // Δimpervious is in points of cover, so /100 is the area fraction converted.
            return (delta / 100) * (value(C.ndvi_impervious) - value(C.ndvi_natural));

        case 'landcover_irrigated': {
            // The greenness the irrigated fraction contributes, scaled by how much of
            // the baseline withdrawal remains. Positive on NDVI while the same slider
            // is negative on groundwater — the tension §4 wants shown, not hidden.
            const fraction = delta / 100;
            return (
                fraction *
                value(C.irrigated_fraction) *
                (value(C.ndvi_irrigated_crop) - value(C.ndvi_natural))
            );
        }

        default:
            throw new Error(`Unknown structural path "${lever.path}"`);
    }
}

// ── Layer 3 ───────────────────────────────────────────────────────────────────

// Displacement after `months` of a constant per-month forcing under mean reversion.
// Exact solution of z_t = (1-λ)z_{t-1} + s, so no rollout loop and no drift from
// repeated inference. λ = 0 degenerates to the naive ramp, which §4a showed must not
// ship — it is rejected rather than silently allowed.
function integrateRate(perMonth, lambda, months) {
    if (!(months > 0)) return 0;
    if (!(lambda > 0)) {
        throw new Error(
            'Refusing to integrate without a reversion rate: naive accumulation is ' +
            'unbounded (PHASE3_PLAN.md §4a).',
        );
    }
    return (perMonth / lambda) * (1 - Math.pow(1 - lambda, months));
}

function reversionFor(output) {
    const entry = params.reversion_per_month[output];
    return entry ? entry.value : 0;
}

// ── Public entry point ────────────────────────────────────────────────────────

/**
 * Structural displacement per output for one scenario, in the output's own units,
 * plus a per-lever breakdown for the provenance UI.
 */
function structuralResponse(deltas, month, durationMonths) {
    const result = {};
    for (const output of STRUCTURAL_OUTPUTS) {
        result[output] = { total: 0, byLever: {} };
    }

    for (const lever of params.levers) {
        const contribution = leverContribution(lever, deltas, month);
        if (contribution === 0) continue;

        const displacement =
            lever.kind === 'rate'
                ? integrateRate(contribution, reversionFor(lever.output), durationMonths)
                // A level lever is a persistent offset, not a rate: converting desert
                // to pavement does not accumulate month over month.
                : contribution;

        const bucket = result[lever.output];
        bucket.total += displacement;
        bucket.byLever[lever.id] = {
            displacement,
            perMonth: lever.kind === 'rate' ? contribution : null,
            kind: lever.kind,
            tier: lever.tier,
            slider: lever.slider,
        };
    }

    // Transfer edges, applied after the direct levers so the source output's total is
    // final. This is what gives surface water's levers a route to wildlife, and it is
    // the only route wildlife has — one coefficient carrying every lever upstream of
    // it, rather than a new lever set invented per output.
    for (const transfer of params.transfers) {
        const source = result[transfer.from];
        if (!source || source.total === 0) continue;
        const coefficient = value(C[transfer.constant]);
        const displacement = coefficient * source.total;
        const bucket = result[transfer.to];
        bucket.total += displacement;
        bucket.byLever[transfer.id] = {
            displacement,
            perMonth: null,
            kind: 'transfer',
            tier: transfer.tier,
            from: transfer.from,
        };
    }
    return result;
}

/**
 * Layer 3 for the learned residual models. `residual` is the one-step change the ONNX
 * model predicts for this scenario; this returns the displacement it accumulates to
 * over the scenario duration. At durationMonths = 1 it returns `residual` unchanged.
 */
function integrateLearnedResidual(output, residual, durationMonths) {
    // Off by default, and the reasoning is in structural_params.json's note plus
    // PHASE3_PLAN.md §13: the learned models carry no own-target feedback (§4a), so
    // there are no dynamics in them to integrate, and applying an externally fitted λ
    // multiplies their known-wrong human coefficients (D2) by up to 4.3x. The
    // structural layer is integrated regardless — a storage balance genuinely does
    // accumulate.
    if (!params.integrate_learned_residual.value) return residual;
    const lambda = reversionFor(output);
    if (!(lambda > 0)) return residual;
    return integrateRate(residual, lambda, durationMonths);
}

const PARITY_SCENARIOS = [
    { name: 'normal',            deltas: {},                                          month: 7,  duration: 12 },
    { name: 'irrigation-cut',    deltas: { irrigation_total_withdrawal_mgd: -60 },     month: 7,  duration: 12 },
    { name: 'irrigation-cut-36', deltas: { irrigation_total_withdrawal_mgd: -60 },     month: 7,  duration: 36 },
    { name: 'irrigation-up',     deltas: { irrigation_total_withdrawal_mgd: 40 },      month: 1,  duration: 1  },
    { name: 'mead-tier3',        deltas: { mead_pool_elevation: -64 },                 month: 7,  duration: 12 },
    { name: 'mead-deepest',      deltas: { mead_pool_elevation: -95 },                 month: 7,  duration: 36 },
    { name: 'mead-full',         deltas: { mead_pool_elevation: 130 },                 month: 7,  duration: 12 },
    { name: 'population-high',   deltas: { population: 3000000 },                      month: 5,  duration: 24 },
    { name: 'public-supply-up',  deltas: { public_supply_groundwater_mgd: 60 },        month: 3,  duration: 12 },
    { name: 'urbanized',         deltas: { impervious_pct: 2.0 },                      month: 7,  duration: 12 },
    { name: 'combined',          deltas: { irrigation_total_withdrawal_mgd: -60, mead_pool_elevation: -95, population: 3000000, impervious_pct: 2.0 }, month: 9, duration: 24 },
];

if (process?.argv?.includes('--dump')) {
    const dump = PARITY_SCENARIOS.map(({ name, deltas, month, duration }) => ({
        name,
        deltas,
        month,
        duration,
        response: structuralResponse({ ...ZERO_DELTAS, ...deltas }, month, duration),
        responseBand: structuralResponseBand({ ...ZERO_DELTAS, ...deltas }, month, duration),
        meadElevation: params.mead_baseline_elevation + (deltas.mead_pool_elevation ?? 0),
        meadReductionKaf: meadReductionKafPerYear(
            params.mead_baseline_elevation + (deltas.mead_pool_elevation ?? 0),
        ),
    }));
    console.log(JSON.stringify(dump, null, 2));
}

// ── Parameter bands ───────────────────────────────────────────────────────────
//
// Every Layer 2 number rendered on a card is a product of constants, and twelve of
// those ship with a declared `band` — eight UNTESTED, and since PHASE3_PLAN.md §12a
// the dominant one (`storage_af_per_ft`, 184,023..289,336) MEASURED but with
// t = +1.78 at the horizon it is used. Showing only the point estimate claims a
// precision the parameters do not have, which is the same failure this whole layer
// exists to fix, in miniature.
//
// The envelope is taken over the CORNERS of the relevant bands rather than by
// propagating derivatives: every path is a product, quotient or difference of
// positive quantities composed with a log, so each is monotone in each constant
// across its band and the corner extremes are the true extremes. Corners are also
// the only version that stays right when two levers share a constant — all three
// pumping levers divide by `storage_af_per_ft`, so they move together, and summing
// independent per-lever minima would understate the width.
//
// "Relevant" is detected by perturbation rather than from a hardcoded
// path -> constant map, for the same reason climateOnly() derives its keys from
// `panel`: a map would drift the first time a path gained a factor.

function bandedConstants() {
    return Object.keys(C).filter(k => Array.isArray(C[k]?.band) && C[k].band.length === 2);
}

function evalWith(overrides, deltas, month, durationMonths) {
    const saved = {};
    for (const [k, v] of Object.entries(overrides)) {
        saved[k] = C[k].value;
        C[k].value = v;
    }
    try {
        return structuralResponse(deltas, month, durationMonths);
    } finally {
        for (const [k, v] of Object.entries(saved)) C[k].value = v;
    }
}

function cornersOf(names) {
    let out = [[]];
    for (const name of names) {
        const next = [];
        for (const prefix of out) for (const end of C[name].band) next.push([...prefix, end]);
        out = next;
    }
    return out;
}

/**
 * Low/high envelope of every total and lever displacement over the declared bands.
 * Same shape as structuralResponse, with `total` as [lo, hi] and each lever carrying
 * `displacementBand` plus `drivers` — which banded constants actually move it, so
 * the card can say why the range is wide.
 */
function structuralResponseBand(deltas, month, durationMonths) {
    const base = structuralResponse(deltas, month, durationMonths);
    const EPS = 1e-12;

    const relevant = bandedConstants().filter(name => {
        const [lo, hi] = C[name].band;
        const probe = hi !== C[name].value ? hi : lo;
        const alt = evalWith({ [name]: probe }, deltas, month, durationMonths);
        return STRUCTURAL_OUTPUTS.some(
            out => Math.abs(alt[out].total - base[out].total) > EPS,
        );
    });

    const bands = {};
    for (const out of STRUCTURAL_OUTPUTS) {
        bands[out] = { total: [base[out].total, base[out].total], byLever: {} };
        for (const [id, entry] of Object.entries(base[out].byLever)) {
            bands[out].byLever[id] = {
                displacementBand: [entry.displacement, entry.displacement],
                drivers: [],
            };
        }
    }

    for (const corner of cornersOf(relevant)) {
        const overrides = Object.fromEntries(relevant.map((n, i) => [n, corner[i]]));
        const trial = evalWith(overrides, deltas, month, durationMonths);
        for (const out of STRUCTURAL_OUTPUTS) {
            const span = bands[out].total;
            span[0] = Math.min(span[0], trial[out].total);
            span[1] = Math.max(span[1], trial[out].total);
            for (const [id, entry] of Object.entries(trial[out].byLever)) {
                const target = bands[out].byLever[id];
                if (!target) continue;
                target.displacementBand[0] = Math.min(target.displacementBand[0], entry.displacement);
                target.displacementBand[1] = Math.max(target.displacementBand[1], entry.displacement);
            }
        }
    }

    // Attribute the width: which relevant constant moves THIS lever.
    for (const name of relevant) {
        const [lo, hi] = C[name].band;
        const probe = hi !== C[name].value ? hi : lo;
        const alt = evalWith({ [name]: probe }, deltas, month, durationMonths);
        for (const out of STRUCTURAL_OUTPUTS) {
            for (const [id, entry] of Object.entries(base[out].byLever)) {
                const moved = alt[out].byLever[id]?.displacement;
                if (moved !== undefined && Math.abs(moved - entry.displacement) > EPS) {
                    bands[out].byLever[id].drivers.push(name);
                }
            }
        }
    }

    return { outputs: bands, relevantConstants: relevant };
}

// Lever/transfer metadata by id, for the provenance UI: mechanism, evidence, tier.
const LEVER_INFO = Object.fromEntries([
    ...params.levers.map(l => [l.id, l]),
    ...params.transfers.map(t => [t.id, t]),
]);

const CLIMATE_ONLY_OUTPUTS = params.climate_only_outputs ?? [];

// The banded constants' own metadata, for the tooltips that explain a range. Read
// straight from structural_params.json so the numbers in the interface are the
// numbers the arithmetic used — there is no second copy to fall out of step.
const PARAM_BANDS = Object.fromEntries(
    bandedConstants().map(name => [name, C[name]]),
);

export {
    LEVER_INFO,
    CLIMATE_ONLY_OUTPUTS,
    STRUCTURAL_OUTPUTS,
    PARITY_SCENARIOS,
    meadReductionKafPerYear,
    meadSubstitutedPumpingAfPerMonth,
    pumpingAfPerMonth,
    integrateRate,
    integrateLearnedResidual,
    structuralResponse,
    structuralResponseBand,
    bandedConstants,
    PARAM_BANDS,
};
