import { state, getMonthEncoding, normalizeOutput, computeDelta } from './state.js';

const MODEL_KEYS = ["grace", "ndvi", "groundwater", "surface_water", "wildfire", "wildlife"];

const MODEL_FILENAMES = {
  grace:         "grace",
  ndvi:          "ndvi",
  groundwater:   "groundwater",
  surface_water: "surface_water",
  wildfire:      "wildfire_monthly",
  wildlife:      "wildlife",
};

const sessions = {};
const featureNames = {};

async function loadModels(onModelReady) {
  for (const key of MODEL_KEYS) {
    const filename = MODEL_FILENAMES[key];
    try {
      const [session, importanceData] = await Promise.all([
        ort.InferenceSession.create(`/model/${filename}.onnx`),
        fetch(`/model/${filename}_feature_importance.json`).then(r => r.json()),
      ]);
      sessions[key] = session;
      featureNames[key] = Object.keys(importanceData);
      onModelReady(key, true);
    } catch (err) {
      console.error(`Failed to load model: ${key}`, err?.message ?? err, err?.stack ?? '');
      onModelReady(key, false);
    }
  }
}

function canRun(modelKey) {
  return !!sessions[modelKey] && !!featureNames[modelKey];
}

function buildInputVector(modelKey, sliderValues, month, upstreamOutputs) {
  const { month_sin, month_cos } = getMonthEncoding(month);

  const baseValues = {
    population:                      sliderValues.population,
    irrigation_total_withdrawal_mgd: sliderValues.irrigation_total_withdrawal_mgd,
    public_supply_groundwater_mgd:   sliderValues.public_supply_groundwater_mgd,
    impervious_pct:                  sliderValues.impervious_pct,
    mead_pool_elevation:             sliderValues.mead_pool_elevation,
    mead_total_release:              sliderValues.mead_pool_elevation,
    precipitation_mm_day:            sliderValues.precipitation_mm_day,
    temperature_2m_c:                sliderValues.temperature_2m_c,
    usdm_dsci:                       sliderValues.usdm_dsci,
    month_sin,
    month_cos,
    precip_x_impervious:             sliderValues.precipitation_mm_day * sliderValues.impervious_pct,
    precip_x_temperature:            sliderValues.precipitation_mm_day * sliderValues.temperature_2m_c,
    grace_groundwater_anomaly:       upstreamOutputs.grace ?? 0,
    ndvi:                            upstreamOutputs.ndvi ?? 0,
  };

  const names = featureNames[modelKey];
  const inputArray = new Float32Array(names.length);

  for (let i = 0; i < names.length; i++) {
    const name = names[i];

    if (name in baseValues) {
      inputArray[i] = baseValues[name];
      continue;
    }

    if (name.includes('_anomaly')) {
      inputArray[i] = 0;
      continue;
    }

    const lagMatch = name.match(/^(.+?)_lag\d+$/);
    const rollMatch = name.match(/^(.+?)_roll\d+$/);

    if (lagMatch && lagMatch[1] in baseValues) {
      inputArray[i] = baseValues[lagMatch[1]];
    } else if (rollMatch && rollMatch[1] in baseValues) {
      inputArray[i] = baseValues[rollMatch[1]];
    } else {
      inputArray[i] = 0;
    }
  }

  return inputArray;
}

async function runModel(modelKey, inputArray) {
  const session = sessions[modelKey];
  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor('float32', inputArray, [1, inputArray.length]);
  const feeds = { [inputName]: tensor };
  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  return results[outputName].data[0];
}

// Runs a single model and returns null on any error instead of throwing.
// This keeps one bad model from aborting the entire inference pass.
async function safeRun(modelKey, inputArray) {
  try {
    return await runModel(modelKey, inputArray);
  } catch (err) {
    console.error(`[EotD] ${modelKey} inference failed:`, err?.message ?? err);
    return null;
  }
}

async function runAll(sliderValues, month) {
  const outputs = {};
  const upstreamOutputs = {};

  // GRACE runs first; its output feeds into NDVI.
  const graceRaw = canRun('grace')
    ? await safeRun('grace', buildInputVector('grace', sliderValues, month, {}))
    : null;
  if (graceRaw !== null) {
    outputs.grace = graceRaw;
    upstreamOutputs.grace = graceRaw;
  }

  // NDVI runs second; its output feeds into Wildlife.
  const ndviRaw = canRun('ndvi')
    ? await safeRun('ndvi', buildInputVector('ndvi', sliderValues, month, upstreamOutputs))
    : null;
  if (ndviRaw !== null) {
    outputs.ndvi = ndviRaw;
    upstreamOutputs.ndvi = ndviRaw;
  }

  // Remaining four run in parallel. canRun check is done before buildInputVector
  // so we never access featureNames for a model that didn't load.
  const [gwRaw, swRaw, wfRaw, wlRaw] = await Promise.all([
    canRun('groundwater')   ? safeRun('groundwater',   buildInputVector('groundwater',   sliderValues, month, {}))              : null,
    canRun('surface_water') ? safeRun('surface_water', buildInputVector('surface_water', sliderValues, month, {}))              : null,
    canRun('wildfire')      ? safeRun('wildfire',      buildInputVector('wildfire',      sliderValues, month, {}))              : null,
    canRun('wildlife')      ? safeRun('wildlife',      buildInputVector('wildlife',      sliderValues, month, upstreamOutputs)) : null,
  ]);

  if (gwRaw !== null) outputs.groundwater   = gwRaw;
  if (swRaw !== null) outputs.surface_water = swRaw;
  if (wfRaw !== null) outputs.wildfire      = wfRaw;
  if (wlRaw !== null) outputs.wildlife      = wlRaw;

  for (const [key, rawValue] of Object.entries(outputs)) {
    const score = normalizeOutput(key, rawValue);
    const delta = computeDelta(key, score);
    state.outputs[key] = { score, rawValue, delta, loading: false, error: false };
  }

  return state.outputs;
}

export { loadModels, runAll };