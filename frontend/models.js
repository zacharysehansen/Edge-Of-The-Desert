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
    groundwater: "depth_to_water_anomaly_ft_lag1",
    surface_water: "discharge_log_anomaly_lag1",
};

// Surface water used to carry a log1p target transform. It no longer does: the target
// is now a per-gage log anomaly index, so the log lives inside the target itself and
// the value is signed. Applying expm1 to it would be a domain error, not an inverse.
const TARGET_TRANSFORMS = {};

// The drought slider is PDSI. NDVI, GRACE and wildfire were trained on the USDM DSCI, which
// only exists from 2000 and correlates just -0.66 with PDSI (R² = 0.44). This is the OLS fit
// of DSCI on PDSI over their 288-month overlap; DSCI is bounded [0, 500] so the result is
// clamped. It is an approximation, and a lossy one — stated here rather than hidden.
function dsciFromPdsi(pdsi) {
    return Math.max(0, Math.min(500, 114.109 - 37.231 * pdsi));
}

const STATIC_FEATURE_BASELINES = {
    mead_total_release: 12657,
};

// Snapshot of the historical (pre-calibration) baselines, captured at module
// load. Residual models predict (value − lag1) and reconstruct via
// `residual + lag1` in finalizePrediction(), where lag1 is seeded from these.
// The seed must stay fixed at the historical reference value: calibrateBaselines()
// overwrites OUTPUT_STATS[key].baseline with the model's own default-slider
// prediction (used only to position the red baseline line / compute deltas), and
// reusing that mutated value as the lag1 seed would add the residual twice.
const SEED_BASELINES = {
    grace:         OUTPUT_STATS.grace.baseline,
    ndvi:          OUTPUT_STATS.ndvi.baseline,
    groundwater:   OUTPUT_STATS.groundwater.baseline,
    surface_water: OUTPUT_STATS.surface_water.baseline,
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


        const startTime = performance.now();
        const outputs = await session.run(feeds);
        const elapsed = performance.now() - startTime;

        const result = extractScalarScore(outputs[outputName]);
        return result;
    };
}

// ── Model loading ─────────────────────────────────────────────────────────────

// Models are fetched from the absolute path /model/, so the HTTP server must be rooted at
// the REPOSITORY root — model/ and frontend/ are siblings. Serving from inside frontend/ is
// the easy mistake: every /model/ request 404s, the 404 body is an HTML page, and .json()
// then dies on "<" with "unexpected character at line 1 column 1", which says nothing at all
// about the actual problem. Check r.ok first and say what is really wrong.
async function fetchJson(url) {
    const r = await fetch(url);
    if (!r.ok) {
        throw new Error(
            `HTTP ${r.status} fetching ${url}. ` +
            `Serve from the repository root (model/ and frontend/ are siblings): ` +
            `run "python -m http.server 8000" in the repo root, then open ` +
            `http://localhost:8000/frontend/index.html`
        );
    }
    return r.json();
}

async function loadModels(onModelReady) {
    console.log('[EotD] loadModels() starting...');
    for (const key of MODEL_KEYS) {
        const filename = MODEL_FILENAMES[key];
        console.log(`[EotD] Loading model "${key}" from /model/${filename}.onnx ...`);
        try {
            const [session, featureNames, featureStats, cvResults] = await Promise.all([
                ort.InferenceSession.create(`/model/${filename}.onnx`),
                fetchJson(`/model/${filename}_feature_names.json`),
                fetchJson(`/model/${filename}_feature_stats.json`),
                fetchJson(`/model/${filename}_cv_results.json`).catch(() => ({})),
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
    const annualNdviBaseline = SEED_BASELINES.ndvi;
    const graceBaseline = SEED_BASELINES.grace;
    const groundwaterBaseline = SEED_BASELINES.groundwater;
    const surfaceWaterBaseline = SEED_BASELINES.surface_water;
    const wildlifeBaseline = 0.0;   // the target is an anomaly: 0 == an average year

    const catalog = {
        population:                      sliderValues.population,
        irrigation_total_withdrawal_mgd: sliderValues.irrigation_total_withdrawal_mgd,
        public_supply_groundwater_mgd:   sliderValues.public_supply_groundwater_mgd,
        impervious_pct:                  sliderValues.impervious_pct,
        mead_pool_elevation:             sliderValues.mead_pool_elevation,
        mead_total_release:              STATIC_FEATURE_BASELINES.mead_total_release,
        precipitation_mm_day:            sliderValues.precipitation_mm_day,
        temperature_2m_c:                sliderValues.temperature_2m_c,
        usdm_dsci:                       dsciFromPdsi(sliderValues.nclimdiv_pdsi),
        // nClimDiv climate block — what surface water actually runs on. Temperature and
        // precipitation are the same physical quantities as the MERRA-2 sliders (they
        // correlate +0.999 and +0.920 over the overlap), so the sliders drive both.
        nclimdiv_pdsi:                   sliderValues.nclimdiv_pdsi,
        nclimdiv_temperature_c:          sliderValues.temperature_2m_c,
        nclimdiv_precipitation_mm_day:   sliderValues.precipitation_mm_day,
        nclimdiv_precip_x_temperature:   sliderValues.precipitation_mm_day * sliderValues.temperature_2m_c,
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
        population_annual_mean:                      annualMeanByDuration(sliderValues.population, sliderBaseline('population'), durationMonths),
        irrigation_total_withdrawal_mgd_annual_sum: annualSumByDuration(sliderValues.irrigation_total_withdrawal_mgd, baselineIrrigation, durationMonths),
        public_supply_groundwater_mgd_annual_sum:   annualSumByDuration(sliderValues.public_supply_groundwater_mgd, baselinePublicSupply, durationMonths),
        mead_pool_elevation_annual_mean:             annualMeanByDuration(sliderValues.mead_pool_elevation, sliderBaseline('mead_pool_elevation'), durationMonths),
        mead_pool_elevation_june:                    durationMonths >= 6 ? sliderValues.mead_pool_elevation : sliderBaseline('mead_pool_elevation'),
        mead_total_release_annual_sum:               STATIC_FEATURE_BASELINES.mead_total_release * 12,
        usdm_dsci_annual_mean:                       annualMeanByDuration(dsciFromPdsi(sliderValues.nclimdiv_pdsi), dsciFromPdsi(sliderBaseline('nclimdiv_pdsi')), durationMonths),
        nclimdiv_pdsi_annual_mean:                   annualMeanByDuration(sliderValues.nclimdiv_pdsi, sliderBaseline('nclimdiv_pdsi'), durationMonths),
        nclimdiv_pdsi_jja_mean:                      sliderValues.nclimdiv_pdsi,
        nclimdiv_temperature_c_annual_mean:          annualMeanByDuration(sliderValues.temperature_2m_c, sliderBaseline('temperature_2m_c'), durationMonths),
        nclimdiv_temperature_c_jja_mean:             sliderValues.temperature_2m_c,
        nclimdiv_precipitation_mm_day_annual_sum:    annualSumByDuration(sliderValues.precipitation_mm_day, sliderBaseline('precipitation_mm_day'), durationMonths),
        nclimdiv_log_precip_annual:                  Math.log1p(Math.max(0, annualSumByDuration(sliderValues.precipitation_mm_day, sliderBaseline('precipitation_mm_day'), durationMonths))),
        temperature_2m_c_annual_mean:                annualMeanByDuration(sliderValues.temperature_2m_c, baselineTemperature, durationMonths),
        temperature_2m_c_jja_mean:                   durationMonths >= 3 ? sliderValues.temperature_2m_c : baselineTemperature,
        precipitation_mm_day_annual_sum:             currentPrecipAnnual,
        log_precip_annual:                           Math.log1p(Math.max(0, currentPrecipAnnual)),
        ndvi_annual_mean:                            annualNdviBaseline,
        ndvi_jja_mean:                               annualNdviBaseline,
        ndvi_annual_mean_lag1:                       annualNdviBaseline,
        grace_groundwater_anomaly_annual_mean:       graceBaseline,
        impervious_pct_annual_mean:                  annualMeanByDuration(sliderValues.impervious_pct, sliderBaseline('impervious_pct'), durationMonths),
        bbs_abundance_anomaly_lag1:                  wildlifeBaseline,
        bbs_abundance_anomaly_lag2:                 wildlifeBaseline,
        bbs_abundance_anomaly_roll3:                wildlifeBaseline,
        year_linear:                                 19,
        precipitation_mm_day_annual_sum_lag1:        priorAnnualByDuration(currentPrecipAnnual, baselinePrecip * 12, durationMonths),
        usdm_dsci_annual_mean_lag1:                  priorAnnualByDuration(dsciFromPdsi(sliderValues.nclimdiv_pdsi), dsciFromPdsi(sliderBaseline('nclimdiv_pdsi')), durationMonths),
        nclimdiv_pdsi_annual_mean_lag1:              priorAnnualByDuration(sliderValues.nclimdiv_pdsi, sliderBaseline('nclimdiv_pdsi'), durationMonths),
        nclimdiv_precipitation_mm_day_annual_sum_lag1: priorAnnualByDuration(sliderValues.precipitation_mm_day, sliderBaseline('precipitation_mm_day'), durationMonths),
        month_sin,
        month_cos,
        precip_x_impervious:  sliderValues.precipitation_mm_day * sliderValues.impervious_pct,
        precip_x_temperature: sliderValues.precipitation_mm_day * sliderValues.temperature_2m_c,
    };

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
    const graceBaseline = SEED_BASELINES.grace;
    const annualNdviBaseline = SEED_BASELINES.ndvi;
    const groundwaterBaseline = SEED_BASELINES.groundwater;
    const surfaceWaterBaseline = SEED_BASELINES.surface_water;

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

    if (gwRaw !== null) {
        Object.assign(state.featureCatalog, {
            depth_to_water_anomaly_ft:       gwRaw,
            depth_to_water_anomaly_ft_lag1:  lagByDuration(gwRaw, groundwaterBaseline, durationMonths, 1),
            depth_to_water_anomaly_ft_lag3:  lagByDuration(gwRaw, groundwaterBaseline, durationMonths, 3),
            depth_to_water_anomaly_ft_roll3: rollByDuration(gwRaw, groundwaterBaseline, durationMonths, 3),
            depth_to_water_anomaly_ft_roll6: rollByDuration(gwRaw, groundwaterBaseline, durationMonths, 6),
        });
    }

    if (swRaw !== null) {
        Object.assign(state.featureCatalog, {
            discharge_log_anomaly:       swRaw,
            discharge_log_anomaly_lag1:  lagByDuration(swRaw, surfaceWaterBaseline, durationMonths, 1),
            discharge_log_anomaly_lag3:  lagByDuration(swRaw, surfaceWaterBaseline, durationMonths, 3),
            discharge_log_anomaly_roll3: rollByDuration(swRaw, surfaceWaterBaseline, durationMonths, 3),
            discharge_log_anomaly_roll6: rollByDuration(swRaw, surfaceWaterBaseline, durationMonths, 6),
        });
    }

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