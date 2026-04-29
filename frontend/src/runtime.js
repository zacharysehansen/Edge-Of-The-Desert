import { csvParse } from "d3";
import * as ort from "onnxruntime-web/wasm";

const INITIAL_STATE = {
  dataStatus: "idle",
  bundleReady: false,
  bundleError: null,
  modelReady: false,
  modelStatus: "idle",
  modelError: null,
  modelInfo: null,
  displayMetadata: null,
  featureCatalog: {},
  controlDefinitions: [],
  controlValues: {},
  historicalSeries: [],
  historicalFeatureVectors: [],
  projectionContext: {
    sourceLastObservedMonth: null,
    projectionStartMonth: null,
    featureRowLastObserved: {},
    populationSeedAbsolute: null,
    fixedModelInputValues: {},
    derivedFeatureValues: {},
    derivedFeatureOverrides: {},
    targetLagValues: {},
    visualSupportValues: {},
  },
  currentPrediction: null,
};

function isDevelopmentMode() {
  return Boolean(import.meta.env?.DEV);
}

function parseYearMonth(yearMonth) {
  if (!yearMonth) return null;
  return new Date(`${yearMonth}-01T00:00:00Z`);
}

function coerceCsvValue(key, value) {
  if (value == null) return null;

  const trimmed = String(value).trim();
  if (trimmed === "") return null;
  if (key === "year_month") return trimmed;

  if (trimmed.toLowerCase() === "true") return true;
  if (trimmed.toLowerCase() === "false") return false;

  const numericValue = Number(trimmed);
  return Number.isFinite(numericValue) ? numericValue : trimmed;
}

function parseCsvRows(text) {
  return csvParse(text, (row) => {
    const normalized = {};

    Object.entries(row).forEach(([key, value]) => {
      normalized[key] = coerceCsvValue(key, value);
    });

    if (normalized.year_month) {
      normalized.yearMonth = normalized.year_month;
      normalized.date = parseYearMonth(normalized.year_month);
    }

    return normalized;
  });
}

function normalizeHistoricalSeries(rows) {
  return rows.map((row) => ({
    yearMonth: row.yearMonth,
    date: row.date,
    predictedScore: row.usdm_sustainability ?? null,
    actualScore: row.actual_usdm_sustainability ?? row.usdm_sustainability ?? null,
    score: row.actual_usdm_sustainability ?? row.usdm_sustainability ?? null,
    residual: row.residual ?? null,
    source: "historical",
  }));
}

function normalizeHistoricalFeatureVectors(rows) {
  return rows.map((row) => ({
    ...row,
    isModelComplete: Boolean(row.model_complete),
    pointType: row.vector_status ?? (row.model_complete ? "model_complete" : "historical_only"),
  }));
}

function pickValues(source, keys) {
  return keys.reduce((accumulator, key) => {
    if (Object.prototype.hasOwnProperty.call(source, key)) {
      accumulator[key] = source[key];
    }
    return accumulator;
  }, {});
}

const DEFAULT_GRACE_GROUNDWATER_ANOMALY = -0.077;

function buildGraceControlDefinition(displayMetadata, projectionSeed) {
  const featureDefinition = displayMetadata.features?.grace_groundwater_anomaly ?? {};
  const featureDomain = featureDefinition.domain ?? {};
  const defaultValue =
    projectionSeed.control_seed_values?.grace_groundwater_anomaly
    ?? DEFAULT_GRACE_GROUNDWATER_ANOMALY;
  const minValue = featureDomain.p5 ?? featureDomain.min ?? defaultValue;
  const maxValue = featureDomain.p95 ?? featureDomain.max ?? defaultValue;

  return {
    id: "grace_groundwater_anomaly",
    label: featureDefinition.label ?? "GRACE Groundwater Anomaly",
    unit: featureDefinition.unit ?? "anomaly",
    decimals: featureDefinition.decimals ?? 3,
    domain: featureDomain,
    knob: {
      default: defaultValue,
      min: minValue,
      max: maxValue,
      step: featureDefinition.knob?.step ?? (maxValue - minValue) / 200,
    },
    model_mapping: {
      type: "direct_feature",
      output_feature: "grace_groundwater_anomaly",
      input_feature: "grace_groundwater_anomaly",
    },
  };
}

function buildControlDefinitions(displayMetadata, projectionSeed) {
  const controlIds = (displayMetadata.knob_controls ?? []).map((id) =>
    id === "population" ? "grace_groundwater_anomaly" : id,
  );

  return controlIds.map((id) => {
    if (id === "grace_groundwater_anomaly" && !displayMetadata.controls?.[id]) {
      return buildGraceControlDefinition(displayMetadata, projectionSeed);
    }

    return {
      id,
      ...displayMetadata.controls[id],
    };
  });
}

function buildInitialControlValues(controlDefinitions, projectionSeed) {
  return controlDefinitions.reduce((accumulator, definition) => {
    const seedValue = projectionSeed.control_seed_values?.[definition.id];
    accumulator[definition.id] = seedValue ?? definition.knob.default;
    return accumulator;
  }, {});
}

function computePopulationChange(absolutePop, previousPop) {
  if (!Number.isFinite(absolutePop) || !Number.isFinite(previousPop) || previousPop === 0) {
    return 0;
  }

  return (absolutePop - previousPop) / previousPop;
}

function buildProjectionContext(displayMetadata, projectionSeed) {
  const projectionPolicy = displayMetadata.projection_policy ?? {};
  const featureRowLastObserved = projectionSeed.feature_row_last_observed ?? {};
  const derivedFeatures = projectionPolicy.derived_features ?? [];
  const fixedRawFeatures = projectionPolicy.fixed_raw_features ?? [];
  const targetLagFeatures = Object.keys(featureRowLastObserved).filter((name) =>
    name.startsWith("usdm_sustainability_"),
  );
  const populationSeedAbsolute = projectionSeed.latest_observed_control_values?.population ?? null;
  const defaultPopulation = projectionSeed.control_seed_values?.population ?? populationSeedAbsolute;

  return {
    sourceLastObservedMonth: projectionSeed.source_last_observed_month,
    projectionStartMonth: projectionSeed.projection_start_month,
    featureRowLastObserved,
    populationSeedAbsolute,
    fixedModelInputValues: pickValues(featureRowLastObserved, fixedRawFeatures),
    derivedFeatureValues: pickValues(featureRowLastObserved, derivedFeatures),
    derivedFeatureOverrides: {
      AZPOP_pct_change: computePopulationChange(defaultPopulation, populationSeedAbsolute),
    },
    targetLagValues: pickValues(featureRowLastObserved, targetLagFeatures),
    visualSupportValues: pickValues(featureRowLastObserved, [
      "grace_groundwater_anomaly",
      "grace_available",
      "powell_pool_elevation",
    ]),
  };
}

function buildCurrentPrediction(projectionSeed, historicalSeries) {
  return (
    projectionSeed.latest_predicted_score
    ?? projectionSeed.latest_observed_score
    ?? historicalSeries.at(-1)?.score
    ?? null
  );
}

function buildHydratedState(displayMetadata, projectionSeed, historicalSeriesRows, historicalFeatureVectorRows) {
  const controlDefinitions = buildControlDefinitions(displayMetadata, projectionSeed);
  const historicalSeries = normalizeHistoricalSeries(historicalSeriesRows);
  const historicalFeatureVectors = normalizeHistoricalFeatureVectors(historicalFeatureVectorRows);

  return {
    displayMetadata,
    featureCatalog: displayMetadata.features ?? {},
    controlDefinitions,
    controlValues: buildInitialControlValues(controlDefinitions, projectionSeed),
    historicalSeries,
    historicalFeatureVectors,
    projectionContext: buildProjectionContext(displayMetadata, projectionSeed),
    currentPrediction: buildCurrentPrediction(projectionSeed, historicalSeries),
  };
}

function assertFiniteNumber(value, message) {
  if (!Number.isFinite(value)) {
    throw new Error(message);
  }
}

function clipScore(value) {
  return Math.max(0, Math.min(100, value));
}

function buildMedianFeatureRow(featureNames, featureCatalog) {
  return featureNames.reduce((row, featureName) => {
    const median = featureCatalog?.[featureName]?.domain?.median;
    if (!Number.isFinite(median)) {
      throw new Error(`Missing median metadata for ${featureName}`);
    }
    row[featureName] = median;
    return row;
  }, {});
}

function createOrderedFloat32Array(featureNames, featureRow) {
  const orderedValues = new Float32Array(featureNames.length);

  featureNames.forEach((featureName, index) => {
    if (!Object.prototype.hasOwnProperty.call(featureRow, featureName)) {
      throw new Error(`Feature row is missing required feature "${featureName}"`);
    }

    const value = Number(featureRow[featureName]);
    assertFiniteNumber(value, `Feature "${featureName}" is not finite.`);
    orderedValues[index] = value;
  });

  return orderedValues;
}

function extractScalarScore(outputValue) {
  if (!outputValue?.data?.length) {
    throw new Error("Model output tensor is empty.");
  }

  const score = Number(outputValue.data[0]);
  assertFiniteNumber(score, "Model output score is not finite.");
  return clipScore(score);
}

async function fetchJson(basePath, fileName) {
  const response = await fetch(`${basePath}/${fileName}`);
  if (!response.ok) {
    throw new Error(`Failed to load ${fileName}`);
  }
  return response.json();
}

async function fetchCsv(basePath, fileName) {
  const response = await fetch(`${basePath}/${fileName}`);
  if (!response.ok) {
    throw new Error(`Failed to load ${fileName}`);
  }

  return parseCsvRows(await response.text());
}

export function formatDateRange(start, end) {
  if (!start || !end) return "--";
  return `${start} to ${end}`;
}

export function formatNumber(value, decimals = 0) {
  if (value == null || Number.isNaN(value)) return "--";

  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatControlValue(value, definition) {
  return formatNumber(value, definition?.decimals ?? 0);
}

export function createRuntime({ basePath = "./model" } = {}) {
  let state = {
    ...INITIAL_STATE,
    projectionContext: { ...INITIAL_STATE.projectionContext },
  };
  const listeners = new Set();

  let featureNames = [];
  let session = null;
  let inputName = null;
  let outputName = null;
  let initPromise = null;
  let latestPredictionRequest = 0;

  function notify() {
    listeners.forEach((listener) => listener(state));
  }

  function setState(partialState) {
    state = {
      ...state,
      ...partialState,
      projectionContext: {
        ...state.projectionContext,
        ...(partialState.projectionContext ?? {}),
      },
    };
    notify();
    return state;
  }

  function getState() {
    return state;
  }

  function subscribe(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  function findControlDefinition(controlId) {
    return state.controlDefinitions.find((control) => control.id === controlId) ?? null;
  }

  function getFeatureNames() {
    if (featureNames.length > 0) {
      return featureNames;
    }

    return Object.keys(state.featureCatalog ?? {});
  }

  function getFeatureFallback(featureName) {
    return state.featureCatalog?.[featureName]?.domain?.median ?? 0;
  }

  function getSourceMonth() {
    const sourceMonth = state.projectionContext?.sourceLastObservedMonth;
    if (!sourceMonth) return null;
    const [, month] = String(sourceMonth).split("-");
    return month ?? null;
  }

  function buildFeatureRow() {
    const projectionContext = state.projectionContext ?? {};
    const featureRow = {
      ...(projectionContext.featureRowLastObserved ?? {}),
      ...(projectionContext.fixedModelInputValues ?? {}),
      ...(projectionContext.targetLagValues ?? {}),
      ...(projectionContext.derivedFeatureValues ?? {}),
      ...(projectionContext.visualSupportValues ?? {}),
      ...(projectionContext.derivedFeatureOverrides ?? {}),
    };

    state.controlDefinitions.forEach((definition) => {
      const value = state.controlValues?.[definition.id];
      if (value == null) return;

      if (definition.model_mapping.type === "direct_feature") {
        featureRow[definition.model_mapping.output_feature] = value;
      }
    });

    if (featureRow.temperature_2m_c != null) {
      const sourceMonth = getSourceMonth();
      const climatology =
        state.displayMetadata?.projection_policy?.temperature_monthly_climatology_c ?? {};
      const baseline = sourceMonth ? climatology[sourceMonth] : null;

      if (baseline != null) {
        featureRow.temperature_2m_c_anomaly = featureRow.temperature_2m_c - baseline;
      }
    }

    return getFeatureNames().reduce((row, featureName) => {
      row[featureName] = featureRow[featureName] ?? getFeatureFallback(featureName);
      return row;
    }, {});
  }

  async function initModel() {
    if (state.modelReady && state.modelInfo) {
      return state.modelInfo;
    }

    if (initPromise) {
      return initPromise;
    }

    initPromise = (async () => {
      featureNames = await fetchJson(basePath, "feature_names.json");

      session = await ort.InferenceSession.create(`${basePath}/water_sustainability.onnx`, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });

      inputName = session.inputNames[0] ?? null;
      outputName = session.outputNames[0] ?? null;

      if (!inputName || !outputName) {
        throw new Error("The ONNX model is missing an input or output name.");
      }

      const medianFeatureRow = buildMedianFeatureRow(featureNames, state.featureCatalog);
      const feeds = {
        [inputName]: new ort.Tensor(
          "float32",
          createOrderedFloat32Array(featureNames, medianFeatureRow),
          [1, featureNames.length],
        ),
      };
      const outputs = await session.run(feeds);
      const smokePrediction = extractScalarScore(outputs[outputName]);

      if (isDevelopmentMode()) {
        console.info(
          `[runtime] ONNX ready via wasm. Smoke prediction = ${smokePrediction.toFixed(2)}`,
        );
      }

      return {
        inputName,
        outputName,
        featureCount: featureNames.length,
        smokePrediction,
      };
    })();

    try {
      return await initPromise;
    } finally {
      initPromise = null;
    }
  }

  async function predictScore(featureRow) {
    if (!session || !inputName || !outputName) {
      throw new Error("The ONNX session has not been initialized yet.");
    }

    const feeds = {
      [inputName]: new ort.Tensor(
        "float32",
        createOrderedFloat32Array(featureNames, featureRow),
        [1, featureNames.length],
      ),
    };
    const outputs = await session.run(feeds);
    return extractScalarScore(outputs[outputName]);
  }

  async function updateCurrentPrediction() {
    if (!state.modelReady) return;

    const featureRow = buildFeatureRow();
    const requestId = ++latestPredictionRequest;

    let prediction;
    try {
      prediction = await predictScore(featureRow);
    } catch (error) {
      console.error("Inference failed after control update.", error);
      return;
    }

    if (requestId !== latestPredictionRequest) {
      return;
    }

    setState({ currentPrediction: prediction });
  }

  async function load() {
    setState({
      dataStatus: "loading",
      bundleReady: false,
      bundleError: null,
      modelReady: false,
      modelStatus: "idle",
      modelError: null,
      modelInfo: null,
    });

    try {
      const [
        displayMetadata,
        projectionSeed,
        historicalSeriesRows,
        historicalFeatureVectorRows,
      ] = await Promise.all([
        fetchJson(basePath, "display_metadata.json"),
        fetchJson(basePath, "projection_seed.json"),
        fetchCsv(basePath, "historical_sustainability.csv"),
        fetchCsv(basePath, "historical_feature_vectors.csv"),
      ]);

      setState({
        ...buildHydratedState(
          displayMetadata,
          projectionSeed,
          historicalSeriesRows,
          historicalFeatureVectorRows,
        ),
        dataStatus: "ready",
        bundleReady: true,
        bundleError: null,
      });
    } catch (error) {
      setState({
        dataStatus: "error",
        bundleReady: false,
        bundleError: error instanceof Error ? error.message : String(error),
      });
      return;
    }

    setState({
      modelReady: false,
      modelStatus: "loading",
      modelError: null,
      modelInfo: null,
    });

    try {
      const modelInfo = await initModel();
      setState({
        modelReady: true,
        modelStatus: "ready",
        modelError: null,
        modelInfo,
      });
      await updateCurrentPrediction();
    } catch (error) {
      setState({
        modelReady: false,
        modelStatus: "error",
        modelError: error instanceof Error ? error.message : String(error),
        modelInfo: null,
      });
    }
  }

  function setControlValue(controlId, value) {
    const definition = findControlDefinition(controlId);
    if (!definition) return;

    const nextControlValues = {
      ...state.controlValues,
      [controlId]: value,
    };
    const nextProjectionContext = {
      ...state.projectionContext,
    };

    if (definition.model_mapping.type === "pct_change_from_absolute_series") {
      nextProjectionContext.derivedFeatureOverrides = {
        ...state.projectionContext.derivedFeatureOverrides,
        AZPOP_pct_change: computePopulationChange(
          value,
          state.projectionContext.populationSeedAbsolute,
        ),
      };
    }

    setState({
      controlValues: nextControlValues,
      projectionContext: nextProjectionContext,
    });

    void updateCurrentPrediction();
  }

  return {
    getState,
    subscribe,
    load,
    setControlValue,
  };
}
