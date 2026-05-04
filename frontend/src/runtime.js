import * as ort from "onnxruntime-web/wasm";
import * as audio from "./components/components.js";
import {
  advanceProjectionPreparation,
  INITIAL_STATE,
  buildHydratedState,
  buildMedianFeatureRow,
  buildProjectionPoint,
  buildProjectionPreparation,
  buildProjectionSeries,
  computePopulationChange,
  createScaledFloat32Array,
  extractScalarScore,
  fetchCsv,
  fetchJson,
} from "./runtime-utils.js";

function isDevelopmentMode() {
  return Boolean(import.meta.env?.DEV);
}

const PROJECTION_STEP_INTERVAL_MS = 1200;

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
  let latestProjectionRun = 0;
  let projectionTimerId = null;
  let ambientAudioPromise = null;

  function getLiveScenarioScore(nextState = state) {
    const projectedPoints = (nextState.projectionSeries ?? []).filter((point) =>
      !point?.isProjectionAnchor && Number.isFinite(point?.score),
    );
    const latestProjectedPoint = projectedPoints.at(-1) ?? null;

    return latestProjectedPoint?.score
      ?? nextState.currentPrediction
      ?? null;
  }

  function syncAudio(nextState = state) {
    const score = getLiveScenarioScore(nextState);
    if (!Number.isFinite(score)) return;
    audio.updateScore(score);
  }

  function notify() {
    listeners.forEach((listener) => listener(state));
  }

  function setState(partialState) {
    const nextState = {
      ...state,
      ...partialState,
      projectionContext: {
        ...state.projectionContext,
        ...(partialState.projectionContext ?? {}),
      },
    };
    state = nextState;
    syncAudio(nextState);
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

  function clearProjectionTimer() {
    if (projectionTimerId != null) {
      clearTimeout(projectionTimerId);
      projectionTimerId = null;
    }
  }

  function cancelProjectionRun() {
    clearProjectionTimer();
    latestProjectionRun += 1;
  }

  function getProjectedPoints() {
    return (state.projectionSeries ?? []).filter((point) => !point?.isProjectionAnchor);
  }

  function buildProjectionPreparationFromState({ seedState = null, status = "ready" } = {}) {
    return buildProjectionPreparation({
      projectionConnections: state.projectionConnections,
      projectionContext: state.projectionContext,
      controlDefinitions: state.controlDefinitions,
      controlValues: state.controlValues,
      currentPrediction: state.currentPrediction,
      displayMetadata: state.displayMetadata,
      featureCatalog: state.featureCatalog,
      seedState,
      status,
    });
  }

  function scheduleProjectionStep(runId) {
    clearProjectionTimer();
    projectionTimerId = window.setTimeout(() => {
      void runProjectionStep(runId);
    }, PROJECTION_STEP_INTERVAL_MS);
  }

  async function runProjectionStep(runId) {
    if (runId !== latestProjectionRun) {
      return;
    }

    const preparation = state.projectionPreparation;
    const horizonMonths = state.projectionHorizonMonths;
    const isFirstProjectedStep = (preparation?.stepIndex ?? 0) === 0;
    const reachedHorizon = horizonMonths !== null && (preparation?.stepIndex ?? 0) >= horizonMonths;

    if (!preparation?.pendingFeatureRow || reachedHorizon) {
      setState({
        animationStatus: "complete",
        projectionPreparation: preparation
          ? {
              ...preparation,
              status: "complete",
            }
          : preparation,
      });
      return;
    }

    let prediction;
    try {
      prediction = await predictScore(preparation.pendingFeatureRow);
    } catch (error) {
      if (runId !== latestProjectionRun) {
        return;
      }

      console.error("Inference failed while advancing the projection engine.", error);
      setState({
        animationStatus: "idle",
        projectionPreparation: preparation
          ? {
              ...preparation,
              status: "error",
            }
          : preparation,
      });
      return;
    }

    if (runId !== latestProjectionRun) {
      return;
    }

    const projectedPoint = buildProjectionPoint({
      yearMonth: preparation.nextProjectionMonth,
      score: prediction,
      monthIndex: (preparation.stepIndex ?? 0) + 1,
      featureRow: preparation.pendingFeatureRow,
      featureSources: preparation.pendingFeatureSources,
      scenarioControls: state.controlValues,
    });
    const projectedPoints = projectedPoint
      ? [...getProjectedPoints(), projectedPoint]
      : getProjectedPoints();
    const nextPreparation = advanceProjectionPreparation({
      projectionPreparation: preparation,
      projectionConnections: state.projectionConnections,
      projectionContext: state.projectionContext,
      controlDefinitions: state.controlDefinitions,
      controlValues: state.controlValues,
      predictedScore: prediction,
      displayMetadata: state.displayMetadata,
      featureCatalog: state.featureCatalog,
    });
    const nextReachedHorizon = horizonMonths !== null && (nextPreparation?.stepIndex ?? 0) >= horizonMonths;
    const isComplete = !nextPreparation?.pendingFeatureRow || nextReachedHorizon;

    setState({
      currentPrediction: isFirstProjectedStep ? prediction : state.currentPrediction,
      projectionSeries: buildProjectionSeries({
        historicalSeries: state.historicalSeries,
        projectionPoints: projectedPoints,
        projectionAnchorYearMonth: state.projectionContext?.sourceLastObservedMonth ?? null,
      }),
      projectionPreparation: nextPreparation
        ? {
            ...nextPreparation,
            status: isComplete ? "complete" : "running",
          }
        : nextPreparation,
      animationStatus: isComplete ? "complete" : "running",
    });

    if (!isComplete) {
      scheduleProjectionStep(runId);
    }
  }

  function startProjectionRun() {
    if (!state.modelReady) {
      return;
    }

    clearProjectionTimer();
    latestProjectionRun += 1;
    const runId = latestProjectionRun;
    const preparation = buildProjectionPreparationFromState({
      seedState: null,
      status: "running",
    });

    setState({
      projectionSeries: buildProjectionSeries({
        historicalSeries: state.historicalSeries,
        projectionPoints: [],
        projectionAnchorYearMonth: state.projectionContext?.sourceLastObservedMonth ?? null,
      }),
      projectionPreparation: preparation,
      currentPrediction: null,
      animationStatus: preparation?.pendingFeatureRow ? "running" : "complete",
    });

    if (!preparation?.pendingFeatureRow) {
      return;
    }

    void runProjectionStep(runId);
  }

  async function initModel() {
    if (state.modelReady && state.modelInfo) {
      return state.modelInfo;
    }

    if (initPromise) {
      return initPromise;
    }

    initPromise = (async () => {
      featureNames = state.modelFeatureNames?.length > 0
        ? state.modelFeatureNames
        : await fetchJson(basePath, "feature_names.json");

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
          createScaledFloat32Array(featureNames, medianFeatureRow, state.featureCatalog),
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
        createScaledFloat32Array(featureNames, featureRow, state.featureCatalog),
        [1, featureNames.length],
      ),
    };
    const outputs = await session.run(feeds);
    return extractScalarScore(outputs[outputName]);
  }

  async function load() {
    cancelProjectionRun();
    setState({
      dataStatus: "loading",
      bundleReady: false,
      bundleError: null,
      modelReady: false,
      modelStatus: "idle",
      modelError: null,
      modelInfo: null,
      animationStatus: "idle",
    });

    try {
      const [
        displayMetadata,
        modelFeatureNames,
        featureStats,
        projectionSeed,
        historicalSeriesRows,
        historicalFeatureVectorRows,
      ] = await Promise.all([
        fetchJson(basePath, "display_metadata.json"),
        fetchJson(basePath, "feature_names.json"),
        fetchJson(basePath, "feature_stats.json"),
        fetchJson(basePath, "projection_seed.json"),
        fetchCsv(basePath, "historical_sustainability.csv"),
        fetchCsv(basePath, "historical_feature_vectors.csv"),
      ]);

      featureNames = modelFeatureNames;

      setState({
        ...buildHydratedState(
          displayMetadata,
          projectionSeed,
          historicalSeriesRows,
          historicalFeatureVectorRows,
          modelFeatureNames,
          featureStats,
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
      startProjectionRun();
    } catch (error) {
      setState({
        modelReady: false,
        modelStatus: "error",
        modelError: error instanceof Error ? error.message : String(error),
        modelInfo: null,
        animationStatus: "idle",
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
    const modelReady = state.modelReady;

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

    if (!modelReady) {
      setState({
        projectionSeries: buildProjectionSeries({
          historicalSeries: state.historicalSeries,
          projectionPoints: [],
          projectionAnchorYearMonth: nextProjectionContext?.sourceLastObservedMonth ?? null,
        }),
        projectionPreparation: buildProjectionPreparationFromState(),
        currentPrediction: null,
        animationStatus: "idle",
      });
      return;
    }

    startProjectionRun();
  }

  function setProjectionHorizon(horizonMonths) {
    setState({ projectionHorizonMonths: horizonMonths });
    startProjectionRun();
  }

  async function enableAmbientAudio() {
    if (audio.isEnabled()) {
      setState({ audioEnabled: true });
      return true;
    }

    if (ambientAudioPromise) {
      return ambientAudioPromise;
    }

    ambientAudioPromise = (async () => {
      await audio.enable();
      setState({ audioEnabled: audio.isEnabled() });
      return audio.isEnabled();
    })();

    try {
      return await ambientAudioPromise;
    } finally {
      ambientAudioPromise = null;
    }
  }

  return {
    getState,
    subscribe,
    load,
    setControlValue,
    setProjectionHorizon,
    enableAmbientAudio,
  };
}
