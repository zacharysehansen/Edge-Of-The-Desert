import * as ort from "onnxruntime-web/wasm";
import {
    OUTPUT_STATS,
    SLIDER_STATS,
    state,
    getMonthEncoding,
    normalizeOutput,
    computeDelta,
} from './state.js';

// Tell the WASM runtime where to find its binary files.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/';

// Disable multi-threading to avoid requiring crossOriginIsolated mode.
ort.env.wasm.numThreads = 1;

console.log('[EotD] models.js loaded, numThreads set to 1');

const MODEL_KEYS = ["grace", "ndvi", "groundwater", "surface_water", "wildfire", "wildlife"];

const MODEL_FILENAMES = {
    grace:         "grace",
    ndvi:          "ndvi",
    groundwater:   "groundwater",
    surface_water: "surface_water",
    wildfire:      "wildfire_monthly",
    wildlife:      "wildlife",
};

// Per-model { session, inputName, outputName, featureNames, featureStats, predictScore }
const models = {};

// Set once loadModels() kicks off calibration. runAll() awaits this so the
// first live inference never races the calibration pass over the same
// shared state.featureCatalog.
let calibrationPromise = null;

const RESIDUAL_LAG_FEATURES = {
    grace: "grace_groundwater_anomaly_lag1",
    ndvi: "ndvi_lag1",
    groundwater: "depth_to_water_ft_mean_lag1",
    surface_water: "discharge_cfs_mean_lag1",
};

const TARGET_TRANSFORMS = {
    surface_water: "log1p",
};

const STATIC_FEATURE_BASELINES = {
    mead_total_release: 12657,
};

// ── Tensor helpers ────────────────────────────────────────────────────────────

function getStatsDefault(name, featureStats) {
    const stats = featureStats?.[name];
    if (!stats) return 0;
    return Number.isFinite(stats.mean) ? stats.mean : 0;
}

function getBaseFeatureName(name) {
    const lagMatch  = name.match(/^(.+)_lag\d+$/);
    const rollMatch = name.match(/^(.+)_roll\d+$/);
    return lagMatch?.[1] ?? rollMatch?.[1] ?? null;
}

function resolveRawFeatureValue(name, featureRow, featureCatalog, featureStats) {
    if (name in featureRow) return { value: featureRow[name], source: 'featureRow' };
    if (name in featureCatalog) return { value: featureCatalog[name], source: 'featureCatalog' };

    const base = getBaseFeatureName(name);
    if (base) {
        if (base in featureRow) {
            return { value: featureRow[base], source: `featureRow ${base}` };
        }
        if (base in featureCatalog) {
            return { value: featureCatalog[base], source: `featureCatalog ${base}` };
        }
    }

    if (name in featureStats) {
        return { value: getStatsDefault(name, featureStats), source: 'featureStats mean' };
    }

    if (base && base in featureStats) {
        return { value: getStatsDefault(base, featureStats), source: `featureStats mean ${base}` };
    }

    return { value: 0, source: 'zero default' };
}

function createInputFloat32Array(featureNames, featureRow, featureCatalog, featureStats) {
    const arr = new Float32Array(featureNames.length);
    const missingFeatures = [];
    const staleFeatures = [];
    const foundFeatures = [];

    for (let i = 0; i < featureNames.length; i++) {
        const name = featureNames[i];
        const { value, source } = resolveRawFeatureValue(
            name,
            featureRow,
            featureCatalog,
            featureStats,
        );

        arr[i] = Number.isFinite(value) ? value : getStatsDefault(name, featureStats);
        foundFeatures.push({ name, source, value: arr[i] });
        if (source === 'zero default') {
            missingFeatures.push(name);
        } else if (source.startsWith('featureStats mean')) {
            staleFeatures.push(name);
        }
    }

    if (missingFeatures.length > 0) {
        console.warn(`[EotD] Missing features defaulted to 0:`, missingFeatures);
    }
    if (staleFeatures.length > 0) {
        console.warn(`[EotD] Features not found in the current scenario, using a static training-stat mean instead (these will NOT respond to slider changes):`, staleFeatures);
    }

    return arr;
}

function extractScalarScore(outputTensor) {
    const value = outputTensor.data[0];
    return value;
}

// ── Per-model predictScore ────────────────────────────────────────────────────

function makePredictScore(modelKey) {
    return async function predictScore(featureRow) {
        console.log(`[EotD] predictScore called for "${modelKey}"`);
        const { session, inputName, outputName, featureNames, featureStats } = models[modelKey];

        if (!session || !inputName || !outputName) {
            throw new Error(`The ONNX session for ${modelKey} has not been initialized yet.`);
        }

        const inputArray = createInputFloat32Array(
            featureNames,
            featureRow,
            state.featureCatalog,
            featureStats,
        );

        const feeds = {
            [inputName]: new ort.Tensor(
                "float32",
                inputArray,
                [1, featureNames.length],
            ),
        };

        console.log(`[EotD] inputArray for "${modelKey}": ` + featureNames.map((n, i) => `${n}=${inputArray[i]}`).join(' | '));

        const startTime = performance.now();
        const outputs = await session.run(feeds);
        const elapsed = performance.now() - startTime;

        const result = extractScalarScore(outputs[outputName]);
        return result;
    };
}

// ── Model loading ─────────────────────────────────────────────────────────────

async function loadModels(onModelReady) {
    console.log('[EotD] loadModels() starting...');
    for (const key of MODEL_KEYS) {
        const filename = MODEL_FILENAMES[key];
        console.log(`[EotD] Loading model "${key}" from /model/${filename}.onnx ...`);
        try {
            const [session, featureNames, featureStats, cvResults] = await Promise.all([
                ort.InferenceSession.create(`/model/${filename}.onnx`),
                fetch(`/model/${filename}_feature_names.json`).then(r => r.json()),
                fetch(`/model/${filename}_feature_stats.json`).then(r => r.json()),
                fetch(`/model/${filename}_cv_results.json`).then(r => r.json()).catch(() => ({})),
            ]);

            console.log(`[EotD] Model "${key}" loaded. Features (${featureNames.length}):`, featureNames);
            console.log(`[EotD]   inputNames:`, session.inputNames);
            console.log(`[EotD]   outputNames:`, session.outputNames);

            models[key] = {
                session,
                inputName:    session.inputNames[0],
                outputName:   session.outputNames[0],
                featureNames: featureNames,
                featureStats: featureStats,
                cvResults:    cvResults,
                predictScore: makePredictScore(key),
            };

            onModelReady(key, true);
        } catch (err) {
            console.error(`[EotD] Failed to load model: ${key}`, err?.message ?? err);
            models[key] = {
                session: null, inputName: null, outputName: null,
                featureNames: [], featureStats: {}, cvResults: {}, predictScore: null,
            };
            onModelReady(key, false);
        }
    }
    console.log('[EotD] loadModels() complete. Models state:', Object.keys(models).map(k => `${k}: ${models[k].session ? 'OK' : 'FAILED'}`));

    calibrationPromise = calibrateBaselines();
    await calibrationPromise;
}

function canRun(modelKey) {
    const result = !!(models[modelKey]?.session);
    console.log(`[EotD] canRun("${modelKey}") = ${result}`);
    return result;
}

// ── Inference ─────────────────────────────────────────────────────────────────

function durationFraction(durationMonths, fullEffectMonths) {
    return Math.max(0, Math.min(1, durationMonths / fullEffectMonths));
}

function blendByDuration(current, baseline, durationMonths, fullEffectMonths) {
    return baseline + (current - baseline) * durationFraction(durationMonths, fullEffectMonths);
}

function lagByDuration(current, baseline, durationMonths, lagMonths) {
    return durationMonths >= lagMonths ? current : baseline;
}

function rollByDuration(current, baseline, durationMonths, windowMonths) {
    return blendByDuration(current, baseline, durationMonths, windowMonths);
}

function annualMeanByDuration(current, baseline, durationMonths) {
    return blendByDuration(current, baseline, durationMonths, 12);
}

function annualSumByDuration(currentMonthly, baselineMonthly, durationMonths) {
    return baselineMonthly * 12
        + (currentMonthly - baselineMonthly) * Math.min(Math.max(durationMonths, 0), 12);
}

function priorAnnualByDuration(current, baseline, durationMonths) {
    if (durationMonths <= 12) return baseline;
    return blendByDuration(current, baseline, durationMonths - 12, 12);
}

function sliderBaseline(key) {
    return SLIDER_STATS[key]?.default ?? 0;
}

function addMonthlyTemporalFeatures(catalog, baseName, current, baseline, durationMonths) {
    catalog[`${baseName}_lag1`] = lagByDuration(current, baseline, durationMonths, 1);
    catalog[`${baseName}_lag3`] = lagByDuration(current, baseline, durationMonths, 3);
    catalog[`${baseName}_roll3`] = rollByDuration(current, baseline, durationMonths, 3);
    catalog[`${baseName}_roll6`] = rollByDuration(current, baseline, durationMonths, 6);
    catalog[`${baseName}_roll12`] = rollByDuration(current, baseline, durationMonths, 12);
}

function buildFeatureCatalog(sliderValues, month, durationMonths = state.scenarioDurationMonths) {

    console.log('[EotD] buildFeatureCatalog called with sliderValues =', JSON.stringify(sliderValues), 'month =', month, 'durationMonths =', durationMonths);

    const { month_sin, month_cos } = getMonthEncoding(month);
    const baselinePrecip = sliderBaseline('precipitation_mm_day');
    const baselineTemperature = sliderBaseline('temperature_2m_c');
    const baselineIrrigation = sliderBaseline('irrigation_total_withdrawal_mgd');
    const baselinePublicSupply = sliderBaseline('public_supply_groundwater_mgd');
    const currentPrecipAnnual = annualSumByDuration(
        sliderValues.precipitation_mm_day,
        baselinePrecip,
        durationMonths,
    );
    const annualNdviBaseline = OUTPUT_STATS.ndvi.baseline;
    const graceBaseline = OUTPUT_STATS.grace.baseline;
    const groundwaterBaseline = OUTPUT_STATS.groundwater.baseline;
    const surfaceWaterBaseline = OUTPUT_STATS.surface_water.baseline;
    const wildlifeBaseline = 0.75;

    const catalog = {
        population:                      sliderValues.population,
        irrigation_total_withdrawal_mgd: sliderValues.irrigation_total_withdrawal_mgd,
        public_supply_groundwater_mgd:   sliderValues.public_supply_groundwater_mgd,
        impervious_pct:                  sliderValues.impervious_pct,
        mead_pool_elevation:             sliderValues.mead_pool_elevation,
        mead_total_release:              STATIC_FEATURE_BASELINES.mead_total_release,
        precipitation_mm_day:            sliderValues.precipitation_mm_day,
        temperature_2m_c:                sliderValues.temperature_2m_c,
        usdm_dsci:                       sliderValues.usdm_dsci,
        grace_available:                 1,
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
        depth_to_water_ft_mean:          groundwaterBaseline,
        depth_to_water_ft_mean_lag1:     groundwaterBaseline,
        depth_to_water_ft_mean_lag3:     groundwaterBaseline,
        depth_to_water_ft_mean_roll3:    groundwaterBaseline,
        depth_to_water_ft_mean_roll6:    groundwaterBaseline,
        discharge_cfs_mean:              surfaceWaterBaseline,
        discharge_cfs_mean_lag1:         surfaceWaterBaseline,
        discharge_cfs_mean_lag3:         surfaceWaterBaseline,
        discharge_cfs_mean_roll3:        surfaceWaterBaseline,
        discharge_cfs_mean_roll6:        surfaceWaterBaseline,
        population_annual_mean:                      annualMeanByDuration(sliderValues.population, sliderBaseline('population'), durationMonths),
        irrigation_total_withdrawal_mgd_annual_sum: annualSumByDuration(sliderValues.irrigation_total_withdrawal_mgd, baselineIrrigation, durationMonths),
        public_supply_groundwater_mgd_annual_sum:   annualSumByDuration(sliderValues.public_supply_groundwater_mgd, baselinePublicSupply, durationMonths),
        mead_pool_elevation_annual_mean:             annualMeanByDuration(sliderValues.mead_pool_elevation, sliderBaseline('mead_pool_elevation'), durationMonths),
        mead_pool_elevation_june:                    durationMonths >= 6 ? sliderValues.mead_pool_elevation : sliderBaseline('mead_pool_elevation'),
        mead_total_release_annual_sum:               STATIC_FEATURE_BASELINES.mead_total_release * 12,
        usdm_dsci_annual_mean:                       annualMeanByDuration(sliderValues.usdm_dsci, sliderBaseline('usdm_dsci'), durationMonths),
        temperature_2m_c_annual_mean:                annualMeanByDuration(sliderValues.temperature_2m_c, baselineTemperature, durationMonths),
        temperature_2m_c_jja_mean:                   durationMonths >= 3 ? sliderValues.temperature_2m_c : baselineTemperature,
        precipitation_mm_day_annual_sum:             currentPrecipAnnual,
        log_precip_annual:                           Math.log1p(Math.max(0, currentPrecipAnnual)),
        ndvi_annual_mean:                            annualNdviBaseline,
        ndvi_jja_mean:                               annualNdviBaseline,
        ndvi_annual_mean_lag1:                       annualNdviBaseline,
        grace_groundwater_anomaly_annual_mean:       graceBaseline,
        impervious_pct_annual_mean:                  annualMeanByDuration(sliderValues.impervious_pct, sliderBaseline('impervious_pct'), durationMonths),
        bbs_abundance_index_lag1:                    wildlifeBaseline,
        bbs_abundance_index_lag2:                    wildlifeBaseline,
        bbs_abundance_index_roll3:                   wildlifeBaseline,
        year_linear:                                 19,
        precipitation_mm_day_annual_sum_lag1:        priorAnnualByDuration(currentPrecipAnnual, baselinePrecip * 12, durationMonths),
        usdm_dsci_annual_mean_lag1:                  priorAnnualByDuration(sliderValues.usdm_dsci, sliderBaseline('usdm_dsci'), durationMonths),
        month_sin,
        month_cos,
        precip_x_impervious:  sliderValues.precipitation_mm_day * sliderValues.impervious_pct,
        precip_x_temperature: sliderValues.precipitation_mm_day * sliderValues.temperature_2m_c,
    };

            // Add temporal features for groundwater baseline (required for residual lag)
        addMonthlyTemporalFeatures(
            catalog,
            'depth_to_water_ft_mean',
            groundwaterBaseline,
            groundwaterBaseline,
            durationMonths,
        );
        for (const key of Object.keys(SLIDER_STATS)) {
        addMonthlyTemporalFeatures(
            catalog,
            key,
            sliderValues[key],
            sliderBaseline(key),
            durationMonths,
        );
    }

    const temperatureAnomaly = sliderValues.temperature_2m_c - baselineTemperature;
    const precipitationAnomaly = sliderValues.precipitation_mm_day - baselinePrecip;
    catalog.temperature_2m_c_anomaly = temperatureAnomaly;
    catalog.precipitation_mm_day_anomaly = precipitationAnomaly;
    addMonthlyTemporalFeatures(catalog, 'temperature_2m_c_anomaly', temperatureAnomaly, 0, durationMonths);
    addMonthlyTemporalFeatures(catalog, 'precipitation_mm_day_anomaly', precipitationAnomaly, 0, durationMonths);
    addMonthlyTemporalFeatures(
        catalog,
        'mead_total_release',
        STATIC_FEATURE_BASELINES.mead_total_release,
        STATIC_FEATURE_BASELINES.mead_total_release,
        durationMonths,
    );

    console.log('[EotD]   catalog =', catalog);
    return catalog;
}

function getFeatureValueForModel(modelKey, name) {
    const model = models[modelKey];
    const { value } = resolveRawFeatureValue(
        name,
        {},
        state.featureCatalog,
        model?.featureStats ?? {},
    );
    return value;
}

function finalizePrediction(modelKey, modelOutput) {
    let value = modelOutput;
    const residualLagFeature = RESIDUAL_LAG_FEATURES[modelKey];

    if (residualLagFeature) {
        const lagValue = getFeatureValueForModel(modelKey, residualLagFeature);
        if (TARGET_TRANSFORMS[modelKey] === 'log1p') {
            value = Math.expm1(modelOutput + Math.log1p(Math.max(0, lagValue)));
        } else {
            value = modelOutput + lagValue;
        }
    } else if (TARGET_TRANSFORMS[modelKey] === 'log1p') {
        value = Math.expm1(modelOutput);
    }

    return value;
}

async function safePredict(modelKey, featureRow) {
    console.log(`[EotD] safePredict("${modelKey}") called with featureRow keys:`, Object.keys(featureRow));
    try {
        const modelOutput = await models[modelKey].predictScore(featureRow);
        const result = finalizePrediction(modelKey, modelOutput);
        console.log(`[EotD] safePredict("${modelKey}") returned:`, result, '(model output:', modelOutput, ')');
        return result;
    } catch (err) {
        console.error(`[EotD] ${modelKey} inference failed:`, err?.message ?? err);
        console.error(`[EotD]   Full error:`, err);
        return null;
    }
}

async function runPipeline(sliderValues, month, durationMonths) {

    // Build the shared catalog once
    state.featureCatalog = buildFeatureCatalog(sliderValues, month, durationMonths);
    const graceBaseline = OUTPUT_STATS.grace.baseline;
    const annualNdviBaseline = OUTPUT_STATS.ndvi.baseline;

    const graceWarmup = canRun('grace')
        ? await safePredict('grace', {})
        : null;
    if (graceWarmup !== null) {
        state.featureCatalog.grace_groundwater_anomaly_lag1 = graceWarmup;
    }

    // GRACE runs first.
    console.log('[EotD] --- Running GRACE ---');
    const graceRaw = canRun('grace')
        ? await safePredict('grace', {})
        : null;
    console.log('[EotD] graceRaw =', graceRaw);
    if (graceRaw !== null) {
        Object.assign(state.featureCatalog, {
            grace_groundwater_anomaly:       graceRaw,
            grace_groundwater_anomaly_lag1:  lagByDuration(graceRaw, graceBaseline, durationMonths, 1),
            grace_groundwater_anomaly_lag3:  lagByDuration(graceRaw, graceBaseline, durationMonths, 3),
            grace_groundwater_anomaly_roll3: rollByDuration(graceRaw, graceBaseline, durationMonths, 3),
            grace_groundwater_anomaly_roll6: rollByDuration(graceRaw, graceBaseline, durationMonths, 6),
            grace_groundwater_anomaly_annual_mean: annualMeanByDuration(graceRaw, graceBaseline, durationMonths),
        });
    }

    // NDVI runs second
    console.log('[EotD] --- Running NDVI ---');
    const ndviRaw = canRun('ndvi')
        ? await safePredict('ndvi', { grace_groundwater_anomaly: graceRaw ?? 0 })
        : null;
    console.log('[EotD] ndviRaw =', ndviRaw);
    if (ndviRaw !== null) {
        Object.assign(state.featureCatalog, {
            ndvi:              ndviRaw,
            ndvi_lag1:         lagByDuration(ndviRaw, annualNdviBaseline, durationMonths, 1),
            ndvi_lag3:         lagByDuration(ndviRaw, annualNdviBaseline, durationMonths, 3),
            ndvi_roll3:        rollByDuration(ndviRaw, annualNdviBaseline, durationMonths, 3),
            ndvi_roll6:        rollByDuration(ndviRaw, annualNdviBaseline, durationMonths, 6),
            ndvi_annual_mean:  annualMeanByDuration(ndviRaw, annualNdviBaseline, durationMonths),
            ndvi_jja_mean:     durationMonths >= 3 ? ndviRaw : annualNdviBaseline,
            ndvi_annual_mean_lag1: priorAnnualByDuration(ndviRaw, annualNdviBaseline, durationMonths),
        });
    }

    // Remaining four run in parallel
    console.log('[EotD] --- Running remaining models in parallel ---');
    const [gwRaw, swRaw, wfRaw, wlRaw] = await Promise.all([
        canRun('groundwater')   ? safePredict('groundwater',   {})                     : null,
        canRun('surface_water') ? safePredict('surface_water', {})                     : null,
        canRun('wildfire')      ? safePredict('wildfire',      {})                     : null,
        canRun('wildlife')      ? safePredict('wildlife',      { ndvi: ndviRaw ?? 0 }) : null,
    ]);

    console.log('[EotD] gwRaw =', gwRaw);
    console.log('[EotD] swRaw =', swRaw);
    console.log('[EotD] wfRaw =', wfRaw);
    console.log('[EotD] wlRaw =', wlRaw);

    const rawResults = {
        grace:         graceRaw,
        ndvi:          ndviRaw,
        groundwater:   gwRaw,
        surface_water: swRaw,
        wildfire:      wfRaw,
        wildlife:      wlRaw,
    };

    return rawResults;
}

async function runAll(sliderValues, month, durationMonths = state.scenarioDurationMonths) {
    if (calibrationPromise) await calibrationPromise;

    const rawResults = await runPipeline(sliderValues, month, durationMonths);

    for (const [key, rawValue] of Object.entries(rawResults)) {
        if (rawValue === null) {
            continue;
        }
        const score = normalizeOutput(key, rawValue);
                    const delta = computeDelta(key, score);
            state.outputs[key] = { score, rawValue, delta, loading: false, error: false };
            }

    return state.outputs;
}

// ── Baseline calibration ────────────────────────────────────────────────────
// The red "default" line used to come from OUTPUT_STATS[key].baseline, a p50
// stat precomputed offline. That stat doesn't necessarily match what the model
// itself predicts when every slider sits at its default value, so the line and
// the bar could disagree even when nothing had changed.
//
// This runs the same pipeline once with every slider at its default, and uses
// the resulting raw model outputs as the new baseline. normalizeOutput and
// computeDelta read OUTPUT_STATS[key].baseline directly, so updating it here
// is enough to move the red line, no other code needs to change.
async function calibrateBaselines(month = state.month, durationMonths = state.scenarioDurationMonths) {
    console.log('[EotD] Calibrating baselines from default slider values...');

    const defaultSliderValues = Object.fromEntries(
        Object.keys(SLIDER_STATS).map(key => [key, SLIDER_STATS[key].default]),
    );

    const rawResults = await runPipeline(defaultSliderValues, month, durationMonths);

    for (const [key, rawValue] of Object.entries(rawResults)) {
        if (rawValue === null) continue;
        OUTPUT_STATS[key].baseline = rawValue;
    }

    console.log('[EotD] Baseline calibration complete:', OUTPUT_STATS);
    return OUTPUT_STATS;
}

export { loadModels, runAll, calibrateBaselines };