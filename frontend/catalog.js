// Scenario -> model-input reconstruction, factored out of models.js so it can run
// under plain Node with no onnxruntime import. That is what lets
// scripts/phase3/check_catalog_parity.py compare this implementation against the
// Python mirror in scripts/phase3/slider_sensitivity.py on every scenario it
// tests — the two have to agree or the headless acceptance sweep is measuring a
// different app from the one that ships.
//
//     node frontend/catalog.js --dump > catalog.json

import {
    ZERO_DELTAS,
    state,
    wrapMonth,
    sliderRaw,
    getMonthEncoding,
    OUTPUT_STATS,
} from './state.js';
import { DERIVED } from './state.js';

// Residual models predict (value - lag1) and reconstruct against these anchors.
const SEED_BASELINES = {
    grace:         OUTPUT_STATS.grace.baseline,
    ndvi:          OUTPUT_STATS.ndvi.baseline,
    groundwater:   OUTPUT_STATS.groundwater.baseline,
    surface_water: OUTPUT_STATS.surface_water.baseline,
};

// The drought slider is PDSI. NDVI, GRACE and wildfire were trained on the USDM DSCI, which
// only exists from 2000 and correlates just -0.66 with PDSI (R² = 0.44). This is the OLS fit
// of DSCI on PDSI over their 288-month overlap; DSCI is bounded [0, 500] so the result is
// clamped. It is an approximation, and a lossy one — stated here rather than hidden.
function dsciFromPdsi(pdsi) {
    return Math.max(0, Math.min(500, 114.109 - 37.231 * pdsi));
}

// ── Inference ─────────────────────────────────────────────────────────────────
//
// SCENARIO RECONSTRUCTION (PHASE3_PLAN.md D5)
//
// `state.sliders` holds POLICY DELTAS, not raw model inputs. `sliderRaw(key,
// delta, month)` in state.js turns a delta back into the raw value the models
// were trained on by adding that month's climatology, so delta = 0 is "normal for
// this month" and the month dropdown — not the slider — supplies seasonality.
//
// `durationMonths` is how long the scenario has been held. Month offset j (j
// months before the selected month) is under the scenario iff `j < duration`;
// anything older sits at its own month's climatological normal. That single rule
// replaces the old lagByDuration / rollByDuration / annualMeanByDuration /
// annualSumByDuration blends, and it makes every lag and rolling feature carry
// the seasonal shape of its OWN month instead of the current month's.
//
// The lag/roll/anomaly definitions now mirror features.py exactly:
//
//     X_lagK    = X at offset K
//     X_rollW   = mean of X over offsets 1..W       (shift(1).rolling(W).mean())
//     X_anomaly = X(0) - X_roll12                   (departure from trailing year)
//
// The old catalog got two of those wrong: its rolling means included the current
// month, and its "anomaly" was the distance from the pooled p50 rather than from
// the trailing 12-month mean. Both fed grace, ndvi, surface_water and wildfire.

// No slider; Mead releases are held at their historical mean.
const CONSTANT_DRIVERS = {
    mead_total_release: 12657,
};

// Base feature name -> the slider whose delta drives it. Several model inputs are
// a second encoding of a quantity that already has a control: the nClimDiv series
// are the same physical temperature and precipitation as the MERRA-2 ones (they
// correlate +0.999 and +0.920 over the overlap), so one slider drives both.
const DRIVER_SOURCE = {
    population:                      'population',
    irrigation_total_withdrawal_mgd: 'irrigation_total_withdrawal_mgd',
    public_supply_groundwater_mgd:   'public_supply_groundwater_mgd',
    impervious_pct:                  'impervious_pct',
    mead_pool_elevation:             'mead_pool_elevation',
    precipitation_mm_day:            'precipitation_mm_day',
    temperature_2m_c:                'temperature_2m_c',
    nclimdiv_pdsi:                   'nclimdiv_pdsi',
    nclimdiv_temperature_c:          'temperature_2m_c',
    nclimdiv_precipitation_mm_day:   'precipitation_mm_day',
};

// Every driver that needs the lag/roll family built for it.
const TEMPORAL_DRIVERS = [
    ...Object.keys(DRIVER_SOURCE),
    'usdm_dsci',
    'mead_total_release',
];

// features.py builds `_anomaly`, `_anomaly_lag1` and `_anomaly_roll3` for exactly
// these four.
const ANOMALY_DRIVERS = [
    'temperature_2m_c',
    'precipitation_mm_day',
    'nclimdiv_temperature_c',
    'nclimdiv_precipitation_mm_day',
];

function driverAt(name, deltas, month) {
    if (name in CONSTANT_DRIVERS) return CONSTANT_DRIVERS[name];
    // usdm_dsci is not a control. It is regressed off the PDSI slider, so its lag
    // and rolling features have to be derived the same way — the old catalog left
    // all five of them pinned to a static p50, which made the drought slider
    // invisible to every DSCI lag in grace, ndvi and groundwater.
    if (name === 'usdm_dsci') {
        return dsciFromPdsi(sliderRaw('nclimdiv_pdsi', deltas.nclimdiv_pdsi, month));
    }
    const source = DRIVER_SOURCE[name];
    if (!source) throw new Error(`No driver defined for feature "${name}"`);
    return sliderRaw(source, deltas[source], month);
}

// Value of `name` `offset` months before the selected month. Offsets inside the
// scenario window carry the slider deltas; older ones fall back to climatology.
function driverAtOffset(name, deltas, month, durationMonths, offset) {
    const applied = offset < durationMonths ? deltas : ZERO_DELTAS;
    return driverAt(name, applied, month - offset);
}

function meanOfOffsets(at, firstOffset, count) {
    let total = 0;
    for (let j = firstOffset; j < firstOffset + count; j++) total += at(j);
    return total / count;
}

function addTemporalFeatures(catalog, name, at) {
    catalog[name] = at(0);
    catalog[`${name}_lag1`] = at(1);
    catalog[`${name}_lag3`] = at(3);
    catalog[`${name}_lag6`] = at(6);
    catalog[`${name}_roll3`]  = meanOfOffsets(at, 1, 3);
    catalog[`${name}_roll6`]  = meanOfOffsets(at, 1, 6);
    catalog[`${name}_roll12`] = meanOfOffsets(at, 1, 12);
}

function addAnomalyFeatures(catalog, name, at) {
    const anomalyAt = (j) => at(j) - meanOfOffsets(at, j + 1, 12);
    catalog[`${name}_anomaly`] = anomalyAt(0);
    catalog[`${name}_anomaly_lag1`] = anomalyAt(1);
    catalog[`${name}_anomaly_roll3`] = (anomalyAt(1) + anomalyAt(2) + anomalyAt(3)) / 3;
}

// The annual panel aggregates a calendar year; the scenario's analogue is the 12
// months ending on the selected month, and `_lag1` is the 12 before that.
function jjaMean(at, month, startOffset) {
    const values = [];
    for (let j = startOffset; j < startOffset + 12; j++) {
        const m = wrapMonth(month - j);
        if (m >= 6 && m <= 8) values.push(at(j));
    }
    return values.reduce((a, b) => a + b, 0) / values.length;
}

function buildFeatureCatalog(sliderDeltas, month, durationMonths = state.scenarioDurationMonths) {
    const { month_sin, month_cos } = getMonthEncoding(month);
    const at = (name) => (offset) =>
        driverAtOffset(name, sliderDeltas, month, durationMonths, offset);

    const graceBaseline = SEED_BASELINES.grace;
    const annualNdviBaseline = SEED_BASELINES.ndvi;
    const groundwaterBaseline = SEED_BASELINES.groundwater;
    const surfaceWaterBaseline = SEED_BASELINES.surface_water;
    const wildlifeBaseline = 0.0;   // the target is an anomaly: 0 == an average year

    const catalog = {
        month_sin,
        month_cos,
        grace_available: 1,
        // The annual panel's linear year index. Held at the reference year.
        year_linear: 19,

        // Response-variable state. Residual models predict (value − lag1) and
        // reconstruct against these anchors in finalizePrediction(); runPipeline()
        // overwrites them once each model has actually run.
        grace_groundwater_anomaly:       graceBaseline,
        grace_groundwater_anomaly_lag1:  graceBaseline,
        grace_groundwater_anomaly_lag3:  graceBaseline,
        grace_groundwater_anomaly_roll3: graceBaseline,
        grace_groundwater_anomaly_roll6: graceBaseline,
        ndvi:                            annualNdviBaseline,
        ndvi_lag1:                       annualNdviBaseline,
        ndvi_lag3:                       annualNdviBaseline,
        ndvi_roll3:                      annualNdviBaseline,
        ndvi_roll6:                      annualNdviBaseline,
        depth_to_water_anomaly_ft:       groundwaterBaseline,
        depth_to_water_anomaly_ft_lag1:  groundwaterBaseline,
        depth_to_water_anomaly_ft_lag3:  groundwaterBaseline,
        depth_to_water_anomaly_ft_roll3: groundwaterBaseline,
        depth_to_water_anomaly_ft_roll6: groundwaterBaseline,
        discharge_log_anomaly:           surfaceWaterBaseline,
        discharge_log_anomaly_lag1:      surfaceWaterBaseline,
        discharge_log_anomaly_lag3:      surfaceWaterBaseline,
        discharge_log_anomaly_roll3:     surfaceWaterBaseline,
        discharge_log_anomaly_roll6:     surfaceWaterBaseline,
        bbs_abundance_anomaly_lag1:      wildlifeBaseline,
        bbs_abundance_anomaly_lag2:      wildlifeBaseline,
        bbs_abundance_anomaly_roll3:     wildlifeBaseline,
    };

    for (const name of TEMPORAL_DRIVERS) addTemporalFeatures(catalog, name, at(name));
    for (const name of ANOMALY_DRIVERS)  addAnomalyFeatures(catalog, name, at(name));

    // Interactions, built from the reconstructed raw values so they stay consistent
    // with the lag family above.
    catalog.precip_x_impervious  = catalog.precipitation_mm_day * catalog.impervious_pct;
    catalog.precip_x_temperature = catalog.precipitation_mm_day * catalog.temperature_2m_c;
    catalog.nclimdiv_precip_x_temperature =
        catalog.nclimdiv_precipitation_mm_day * catalog.nclimdiv_temperature_c;

    // GRACE's storage-change input (PHASE3_PLAN.md §26). GLDAS has no slider: its
    // monthly change is derived from the rain and temperature the sliders set, by
    // the OLS in generate_stats.py (R² stated in computed_stats.json). It uses the
    // same reconstructed values as the lag family above, so it is consistent with
    // them, and it carries no human input, so Layer 1 stays climate-only.
    const g = DERIVED.gldas_tws_proxy_delta;
    if (g) {
        const c = g.coefficients;
        catalog.gldas_tws_proxy_delta = g.intercept
            + c.precipitation_mm_day      * catalog.precipitation_mm_day
            + c.precipitation_mm_day_lag1 * catalog.precipitation_mm_day_lag1
            + c.temperature_2m_c          * catalog.temperature_2m_c
            + c.month_sin * month_sin
            + c.month_cos * month_cos;
    }

    // Annual aggregates. The wildlife model is the only consumer.
    const pdsiAt = at('nclimdiv_pdsi');
    const ncTempAt = at('nclimdiv_temperature_c');
    const ncPrecipAt = at('nclimdiv_precipitation_mm_day');
    const precipAnnualSum = meanOfOffsets(ncPrecipAt, 0, 12) * 12;

    catalog.nclimdiv_pdsi_annual_mean = meanOfOffsets(pdsiAt, 0, 12);
    catalog.nclimdiv_pdsi_annual_mean_lag1 = meanOfOffsets(pdsiAt, 12, 12);
    catalog.nclimdiv_pdsi_jja_mean = jjaMean(pdsiAt, month, 0);
    catalog.nclimdiv_temperature_c_annual_mean = meanOfOffsets(ncTempAt, 0, 12);
    catalog.nclimdiv_temperature_c_jja_mean = jjaMean(ncTempAt, month, 0);
    catalog.nclimdiv_precipitation_mm_day_annual_sum = precipAnnualSum;
    catalog.nclimdiv_precipitation_mm_day_annual_sum_lag1 =
        meanOfOffsets(ncPrecipAt, 12, 12) * 12;
    catalog.nclimdiv_log_precip_annual = Math.log1p(Math.max(0, precipAnnualSum));

    console.log('[EotD]   catalog =', catalog);
    return catalog;
}


// A fixed scenario set, dumped so the Python mirror can be checked against it.
// Kept in sync by scripts/phase3/check_catalog_parity.py, which reads this list.
const PARITY_SCENARIOS = [
    { name: 'normal-july',        deltas: {},                                        month: 7,  duration: 12 },
    { name: 'normal-january',     deltas: {},                                        month: 1,  duration: 12 },
    { name: 'irrigation-max',     deltas: { irrigation_total_withdrawal_mgd: 40 },    month: 7,  duration: 12 },
    { name: 'irrigation-min-1mo', deltas: { irrigation_total_withdrawal_mgd: -60 },   month: 3,  duration: 1  },
    { name: 'mead-low',           deltas: { mead_pool_elevation: -95 },               month: 10, duration: 36 },
    { name: 'population-high',    deltas: { population: 3000000 },                    month: 5,  duration: 24 },
    { name: 'hot-dry',            deltas: { temperature_2m_c: 0.79, precipitation_mm_day: -32.11, nclimdiv_pdsi: -1.96 }, month: 8, duration: 12 },
    { name: 'wet',                deltas: { precipitation_mm_day: 44.57 },            month: 12, duration: 6  },
];

// `typeof process` and not `process?.` — optional chaining short-circuits a property
// that is null or undefined, but it does NOT protect an identifier that was never
// declared, and in a browser `process` is undeclared. `process?.argv` therefore throws
// ReferenceError at module load, which took this whole module down and every module
// that imports it with it. Every check in the repo runs under Node, where `process`
// exists, so nothing caught it; `--mode browser-safety` in check_catalog_parity.py now
// does, by loading each module with the Node-only globals deleted.
if (typeof process !== 'undefined' && process.argv?.includes('--dump')) {
    const quiet = console.log;
    console.log = () => {};
    // The scenario spec travels with its output, so the Python side never has to
    // parse this file — it replays exactly what was dumped.
    const dump = PARITY_SCENARIOS.map(({ name, deltas, month, duration }) => ({
        name,
        deltas,
        month,
        duration,
        catalog: buildFeatureCatalog({ ...ZERO_DELTAS, ...deltas }, month, duration),
    }));
    console.log = quiet;
    console.log(JSON.stringify(dump, null, 2));
}

export {
    CONSTANT_DRIVERS,
    DRIVER_SOURCE,
    TEMPORAL_DRIVERS,
    ANOMALY_DRIVERS,
    PARITY_SCENARIOS,
    SEED_BASELINES,
    dsciFromPdsi,
    buildFeatureCatalog,
};
