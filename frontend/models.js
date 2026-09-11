import * as ort from "onnxruntime-web/wasm";
import {
    OUTPUT_STATS,
    ZERO_DELTAS,
    state,
    climateOnly,
    normalizeOutput,
    computeDelta,
} from './state.js';
import { SEED_BASELINES, buildFeatureCatalog } from './catalog.js';
import { structuralResponse, integrateLearnedResidual } from './structural.js';

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

// Each residual model's own output at the DEFAULT scenario — every policy delta at
// zero, i.e. the climatological normal. Captured by calibrateBaselines().
//
// Layer 3 integrates the deviation from this, not the raw residual. That distinction
// is what stops the integration manufacturing drift: at the default scenario the
// system is by definition at rest, so a nonzero residual there is model bias, and
// dividing a bias by a small λ is how PHASE3_PLAN.md §4b's mean-reverting rollout
// still crept for GRACE and groundwater. Only the scenario-INDUCED change in the
// residual is a forcing that has any business accumulating.
//
// null until calibration finishes, which also means calibration itself runs
// un-integrated — correct, because its forcing is zero by construction.
let defaultResiduals = null;
let lastModelOutputs = {};

const RESIDUAL_LAG_FEATURES = {
    grace: "grace_groundwater_anomaly_lag1",
    ndvi: "ndvi_lag1",
    groundwater: "depth_to_water_anomaly_ft_lag1",
    surface_water: "discharge_log_anomaly_lag1",
};

// Surface water used to carry a log1p target transform. It no longer does: the target
// is now a per-gage log anomaly index, so the log lives inside the target itself and
// the value is signed. Applying expm1 to it would be a domain error, not an inverse.
// The mechanism is gone rather than left empty, so nobody re-enables it by accident.

// SEED_BASELINES is defined in catalog.js, which owns the scenario reconstruction.
// It must stay pinned to the historical reference value: calibrateBaselines()
// overwrites OUTPUT_STATS[key].baseline with the model's own normal-scenario
// prediction (used only to position the red baseline line and compute deltas),
// and reusing that mutated value as the lag1 seed would add the residual twice.

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

// Response-variable feedback only. The four residual models predict a one-step
// change against a fixed anchor, so after a model runs its own lag/roll features
// are pulled part-way toward the fresh prediction rather than snapping to it.
// This is the placeholder Layer 3 (PHASE3_PLAN.md §4b) replaces with a real
// mean-reverting integration; it is NOT how the driver features are built.
function lagByDuration(current, baseline, durationMonths, lagMonths) {
    return durationMonths >= lagMonths ? current : baseline;
}

function rollByDuration(current, baseline, durationMonths, windowMonths) {
    const fraction = Math.max(0, Math.min(1, durationMonths / windowMonths));
    return baseline + (current - baseline) * fraction;
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

function finalizePrediction(modelKey, modelOutput, durationMonths, structural) {
    let value = modelOutput;
    const residualLagFeature = RESIDUAL_LAG_FEATURES[modelKey];

    if (residualLagFeature) {
        const lagValue = getFeatureValueForModel(modelKey, residualLagFeature);

        if (defaultResiduals) {
            // Layer 3 (PHASE3_PLAN.md §4b). The bias term is added once and never
            // integrated; only the forcing accumulates, and it does so against the
            // empirical mean-reversion rate so it converges instead of ramping.
            // At durationMonths = 1 this is arithmetically identical to the old
            // single-step reconstruction.
            const bias = defaultResiduals[modelKey] ?? 0;
            const forcing = modelOutput - bias;
            value = lagValue + bias
                + integrateLearnedResidual(modelKey, forcing, durationMonths);
        } else {
            value = modelOutput + lagValue;
        }

    }

    // Layer 2 (PHASE3_PLAN.md §4). The human-lever response the learned models cannot
    // carry — four of six have no human feature at all — added in the output's own
    // units, already integrated over the scenario duration by structural.js.
    const contribution = structural?.[modelKey]?.total ?? 0;
    return value + contribution;
}

async function safePredict(modelKey, featureRow, durationMonths, structural) {
    console.log(`[EotD] safePredict("${modelKey}") called with featureRow keys:`, Object.keys(featureRow));
    try {
        const modelOutput = await models[modelKey].predictScore(featureRow);
        lastModelOutputs[modelKey] = modelOutput;
        const result = finalizePrediction(modelKey, modelOutput, durationMonths, structural);
        console.log(`[EotD] safePredict("${modelKey}") returned:`, result, '(model output:', modelOutput, ')');
        return result;
    } catch (err) {
        console.error(`[EotD] ${modelKey} inference failed:`, err?.message ?? err);
        console.error(`[EotD]   Full error:`, err);
        return null;
    }
}

async function runPipeline(sliderDeltas, month, durationMonths) {

    // Layer 1 sees climate only — the human levers are held at their climatological
    // normal, because Layer 2 owns that response (see climateOnly() in state.js).
    state.featureCatalog = buildFeatureCatalog(
        climateOnly(sliderDeltas), month, durationMonths,
    );

    // Layer 2, computed once for the whole pipeline and exposed for the provenance UI.
    const structural = structuralResponse(sliderDeltas, month, durationMonths);
    state.structural = structural;
    const graceBaseline = SEED_BASELINES.grace;
    const annualNdviBaseline = SEED_BASELINES.ndvi;
    const groundwaterBaseline = SEED_BASELINES.groundwater;
    const surfaceWaterBaseline = SEED_BASELINES.surface_water;

    const graceWarmup = canRun('grace')
        ? await safePredict('grace', {}, durationMonths, structural)
        : null;
    if (graceWarmup !== null) {
        state.featureCatalog.grace_groundwater_anomaly_lag1 = graceWarmup;
    }

    // GRACE runs first.
    console.log('[EotD] --- Running GRACE ---');
    const graceRaw = canRun('grace')
        ? await safePredict('grace', {}, durationMonths, structural)
        : null;
    console.log('[EotD] graceRaw =', graceRaw);
    if (graceRaw !== null) {
        Object.assign(state.featureCatalog, {
            grace_groundwater_anomaly:       graceRaw,
            grace_groundwater_anomaly_lag1:  lagByDuration(graceRaw, graceBaseline, durationMonths, 1),
            grace_groundwater_anomaly_lag3:  lagByDuration(graceRaw, graceBaseline, durationMonths, 3),
            grace_groundwater_anomaly_roll3: rollByDuration(graceRaw, graceBaseline, durationMonths, 3),
            grace_groundwater_anomaly_roll6: rollByDuration(graceRaw, graceBaseline, durationMonths, 6),
        });
    }

    // NDVI runs second
    console.log('[EotD] --- Running NDVI ---');
    const ndviRaw = canRun('ndvi')
        ? await safePredict('ndvi', { grace_groundwater_anomaly: graceRaw ?? 0 }, durationMonths, structural)
        : null;
    console.log('[EotD] ndviRaw =', ndviRaw);
    if (ndviRaw !== null) {
        Object.assign(state.featureCatalog, {
            ndvi:              ndviRaw,
            ndvi_lag1:         lagByDuration(ndviRaw, annualNdviBaseline, durationMonths, 1),
            ndvi_lag3:         lagByDuration(ndviRaw, annualNdviBaseline, durationMonths, 3),
            ndvi_roll3:        rollByDuration(ndviRaw, annualNdviBaseline, durationMonths, 3),
            ndvi_roll6:        rollByDuration(ndviRaw, annualNdviBaseline, durationMonths, 6),
        });
    }

    // Remaining four run in parallel
    console.log('[EotD] --- Running remaining models in parallel ---');
    const [gwRaw, swRaw, wfRaw, wlRaw] = await Promise.all([
        canRun('groundwater')   ? safePredict('groundwater', {}, durationMonths, structural)                     : null,
        canRun('surface_water') ? safePredict('surface_water', {}, durationMonths, structural)                     : null,
        canRun('wildfire')      ? safePredict('wildfire', {}, durationMonths, structural)                     : null,
        // No ndvi override: the wildlife model has no ndvi feature, so passing one only
        // looked like a dependency. Its structural link is the riparian transfer edge.
        canRun('wildlife')      ? safePredict('wildlife', {}, durationMonths, structural) : null,
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

async function runAll(sliderDeltas, month, durationMonths = state.scenarioDurationMonths) {
    if (calibrationPromise) await calibrationPromise;

    const rawResults = await runPipeline(sliderDeltas, month, durationMonths);

    for (const [key, rawValue] of Object.entries(rawResults)) {
        if (rawValue === null) {
            continue;
        }
        const score = normalizeOutput(key, rawValue);
        const delta = computeDelta(key, score);
        state.outputs[key] = {
            score,
            rawValue,
            delta,
            provenance: splitProvenance(key, rawValue),
            loading: false,
            error: false,
        };
    }

    return state.outputs;
}

// Layer 4 (PHASE3_PLAN.md §4). Splits the change from baseline into the part the
// learned climate model produced and the part the structural levers did, in score
// points, with a per-lever breakdown.
//
// Computed from raw values rather than from the rendered score, because
// normalizeOutput clamps to 0-100 and a clamped total would not equal the sum of its
// parts. The card can then say WHY the number moved, which the black box never could.
function splitProvenance(modelKey, rawValue) {
    const stats = OUTPUT_STATS[modelKey];
    const span = stats.max - stats.min;
    const toPoints = (raw) => (span ? (raw / span) * 100 : 0);

    const structural = state.structural?.[modelKey];
    const humanRaw = structural?.total ?? 0;
    const totalRaw = rawValue - stats.baseline;

    const levers = Object.entries(structural?.byLever ?? {})
        .map(([id, detail]) => ({
            id,
            points: toPoints(detail.displacement),
            kind: detail.kind,
            tier: detail.tier,
            slider: detail.slider ?? null,
            from: detail.from ?? null,
        }))
        .filter(lever => Math.abs(lever.points) >= 0.005)
        .sort((a, b) => Math.abs(b.points) - Math.abs(a.points));

    return {
        climatePoints: toPoints(totalRaw - humanRaw),
        humanPoints: toPoints(humanRaw),
        levers,
    };
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

    // Every policy delta at 0: the climatological normal for this month, which is
    // exactly what the red baseline line on each output card should mark.
    // defaultResiduals stays null through this pass, so the pipeline runs
    // un-integrated and the captured residuals are the true one-step values.
    defaultResiduals = null;
    lastModelOutputs = {};
    const rawResults = await runPipeline({ ...ZERO_DELTAS }, month, durationMonths);
    defaultResiduals = { ...lastModelOutputs };

    for (const [key, rawValue] of Object.entries(rawResults)) {
        if (rawValue === null) continue;
        OUTPUT_STATS[key].baseline = rawValue;
    }

    console.log('[EotD] Baseline calibration complete:', OUTPUT_STATS);
    return OUTPUT_STATS;
}

export { loadModels, runAll, calibrateBaselines };