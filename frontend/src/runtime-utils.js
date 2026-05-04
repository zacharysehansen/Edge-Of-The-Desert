import { csvParse } from "d3";

export const INITIAL_STATE = {
  dataStatus: "idle",
  bundleReady: false,
  bundleError: null,
  modelReady: false,
  modelStatus: "idle",
  modelError: null,
  modelInfo: null,
  displayMetadata: null,
  featureCatalog: {},
  modelFeatureNames: [],
  controlDefinitions: [],
  controlValues: {},
  historicalSeries: [],
  historicalFeatureVectors: [],
  projectionSeries: [],
  projectionHorizonMonths: 24,
  projectionConnections: null,
  projectionPreparation: null,
  animationStatus: "idle",
  audioEnabled: false,
  projectionContext: {
    contextSource: null,
    sourceLastObservedMonth: null,
    projectionStartMonth: null,
    featureRowLastObserved: {},
    latestObservedControlValues: {},
    populationSeedAbsolute: null,
    fixedModelInputValues: {},
    derivedFeatureValues: {},
    derivedFeatureOverrides: {},
    targetLagValues: {},
    visualSupportValues: {},
    targetHistory: [],
    exogenousHistory: {},
  },
  currentPrediction: null,
};

const DEFAULT_PROJECTION_HORIZON_MONTHS = 24;
const DEFAULT_GRACE_GROUNDWATER_ANOMALY = -0.077;
const DEFAULT_PROJECTION_BASELINE_MONTH = "2020-07";
const TARGET_FEATURE_NAME = "usdm_sustainability";
const TEMPERATURE_ANOMALY_FEATURE = "temperature_2m_c_anomaly";
const LAG_FEATURE_PATTERN = /^(.*)_lag(\d+)$/;
const ROLL_FEATURE_PATTERN = /^(.*)_roll(\d+)$/;
const HISTORY_KEY_PATTERN = /^(.*)_last(\d+)$/;

function parseYearMonth(yearMonth) {
  if (!yearMonth) return null;
  return new Date(`${yearMonth}-01T00:00:00Z`);
}

function formatYearMonth(date) {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return null;

  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  return `${year}-${month}`;
}

function addUtcMonths(date, monthOffset) {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return null;

  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth() + monthOffset, 1));
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

function normalizeHistoryRows(rows) {
  return (rows ?? [])
    .map((row) => {
      const yearMonth = row.year_month ?? row.yearMonth ?? null;
      const value = Number(row.value);

      if (!yearMonth || !Number.isFinite(value)) {
        return null;
      }

      return {
        yearMonth,
        date: parseYearMonth(yearMonth),
        value,
      };
    })
    .filter(Boolean)
    .sort((left, right) => left.date - right.date);
}

function normalizeExogenousHistory(historyMap) {
  return Object.entries(historyMap ?? {}).reduce((accumulator, [historyKey, rows]) => {
    const match = HISTORY_KEY_PATTERN.exec(historyKey);
    const baseFeature = match?.[1] ?? historyKey;
    const depth = match ? Number(match[2]) : rows?.length ?? 0;

    accumulator[baseFeature] = {
      historyKey,
      baseFeature,
      depth,
      rows: normalizeHistoryRows(rows),
    };

    return accumulator;
  }, {});
}

function cloneHistoryRows(rows) {
  return (rows ?? []).map((row) => ({
    ...row,
    date: row.date instanceof Date ? new Date(row.date.getTime()) : row.date,
  }));
}

function cloneExogenousHistory(historyMap) {
  return Object.entries(historyMap ?? {}).reduce((accumulator, [baseFeature, history]) => {
    accumulator[baseFeature] = {
      ...history,
      rows: cloneHistoryRows(history.rows),
    };
    return accumulator;
  }, {});
}

function buildHistoryRow(yearMonth, value) {
  const numericValue = Number(value);
  if (!yearMonth || !Number.isFinite(numericValue)) {
    return null;
  }

  return {
    yearMonth,
    date: parseYearMonth(yearMonth),
    value: numericValue,
  };
}

function trimHistoryRows(rows, maxLength) {
  if (!Array.isArray(rows) || !Number.isFinite(maxLength) || maxLength < 1) {
    return rows ?? [];
  }

  return rows.slice(-maxLength);
}

function getLatestHistoryValue(rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return null;
  }

  const value = Number(rows.at(-1)?.value);
  return Number.isFinite(value) ? value : null;
}

function getHistoryStorageFeature(baseFeature) {
  return baseFeature === "AZPOP_pct_change" ? "population_change" : baseFeature;
}

function buildHistoryKey(baseFeature, depth) {
  return `${baseFeature}_last${Math.max(1, Number(depth) || 1)}`;
}

function coerceFiniteNumber(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
}

function appendHistoryValue(historyMap, baseFeature, yearMonth, value, maxLength = 1) {
  const historyRow = buildHistoryRow(yearMonth, value);
  if (!historyRow) {
    return historyMap;
  }

  const storageFeature = getHistoryStorageFeature(baseFeature);
  const existingHistory = historyMap[storageFeature];
  const nextRows = trimHistoryRows(
    [...(existingHistory?.rows ?? []), historyRow],
    Math.max(1, Number(maxLength) || existingHistory?.depth || 1),
  );

  return {
    ...historyMap,
    [storageFeature]: {
      historyKey: existingHistory?.historyKey ?? buildHistoryKey(storageFeature, maxLength),
      baseFeature: storageFeature,
      depth: Math.max(existingHistory?.depth ?? 0, Number(maxLength) || 1),
      rows: nextRows,
    },
  };
}

function pickValues(source, keys) {
  return keys.reduce((accumulator, key) => {
    if (Object.prototype.hasOwnProperty.call(source, key)) {
      accumulator[key] = source[key];
    }
    return accumulator;
  }, {});
}

function pickObservedControlValues(row = {}) {
  return {
    powell_pool_elevation: coerceFiniteNumber(row.powell_pool_elevation),
    snow_water_equivalent_in: coerceFiniteNumber(row.snow_water_equivalent_in),
    precipitation_mm_day: coerceFiniteNumber(row.precipitation_mm_day),
    temperature_2m_c: coerceFiniteNumber(row.temperature_2m_c),
    irrigation_total_withdrawal_mgd: coerceFiniteNumber(row.irrigation_total_withdrawal_mgd),
    public_supply_groundwater_mgd: coerceFiniteNumber(row.public_supply_groundwater_mgd),
    grace_groundwater_anomaly: coerceFiniteNumber(row.grace_groundwater_anomaly),
    population: coerceFiniteNumber(row.AZPOP),
  };
}

function deriveProjectionHistoryColumns(modelFeatureNames = []) {
  const historyColumns = new Set(["AZPOP"]);

  modelFeatureNames.forEach((featureName) => {
    const lagMatch = LAG_FEATURE_PATTERN.exec(featureName);
    if (lagMatch) {
      historyColumns.add(lagMatch[1]);
      return;
    }

    const rollMatch = ROLL_FEATURE_PATTERN.exec(featureName);
    if (rollMatch) {
      historyColumns.add(rollMatch[1]);
      return;
    }

    if (
      featureName === TARGET_FEATURE_NAME
      || featureName === TEMPERATURE_ANOMALY_FEATURE
      || featureName === "month_sin"
      || featureName === "month_cos"
    ) {
      return;
    }

    historyColumns.add(featureName);
  });

  return [...historyColumns];
}

function buildHistoricalExogenousHistory({
  historicalFeatureVectors,
  baselineDate,
  modelFeatureNames,
}) {
  const relevantRows = (historicalFeatureVectors ?? [])
    .filter((row) =>
      row?.date instanceof Date
      && !Number.isNaN(row.date.getTime())
      && row.date <= baselineDate,
    );

  return deriveProjectionHistoryColumns(modelFeatureNames).reduce((historyMap, columnName) => {
    const storageFeature = columnName === "AZPOP"
      ? "population"
      : getHistoryStorageFeature(columnName);
    const rows = relevantRows
      .map((row) => buildHistoryRow(row.yearMonth, row[columnName]))
      .filter(Boolean);

    if (rows.length === 0) {
      return historyMap;
    }

    historyMap[storageFeature] = {
      historyKey: buildHistoryKey(storageFeature, rows.length),
      baseFeature: storageFeature,
      depth: rows.length,
      rows,
    };
    return historyMap;
  }, {});
}

function getMonthNumber(yearMonth) {
  if (!yearMonth) return null;

  const [, month] = String(yearMonth).split("-");
  const monthNumber = Number(month);
  return Number.isInteger(monthNumber) && monthNumber >= 1 && monthNumber <= 12
    ? monthNumber
    : null;
}

function buildCalendarFeatureValues(yearMonth) {
  const monthNumber = getMonthNumber(yearMonth);
  if (!monthNumber) {
    return {
      month_sin: null,
      month_cos: null,
    };
  }

  const angle = (2 * Math.PI * (monthNumber - 1)) / 12;
  return {
    month_sin: Math.sin(angle),
    month_cos: Math.cos(angle),
  };
}

function getTemperatureBaseline(displayMetadata, yearMonth) {
  const monthNumber = getMonthNumber(yearMonth);
  if (!monthNumber) return null;

  const climatology = displayMetadata?.projection_policy?.temperature_monthly_climatology_c ?? {};
  const baseline = climatology[String(monthNumber)] ?? climatology[monthNumber] ?? null;
  return Number.isFinite(baseline) ? Number(baseline) : null;
}

function buildTemperatureAnomalyHistory(temperatureHistoryRows, displayMetadata) {
  return cloneHistoryRows(temperatureHistoryRows)
    .map((row) => {
      const baseline = getTemperatureBaseline(displayMetadata, row.yearMonth);
      if (!Number.isFinite(baseline)) return null;

      return {
        ...row,
        value: row.value - baseline,
      };
    })
    .filter(Boolean);
}

function getLagValue(historyRows, lagDepth) {
  if (!Array.isArray(historyRows) || historyRows.length < lagDepth || lagDepth < 1) {
    return null;
  }

  return historyRows.at(-lagDepth)?.value ?? null;
}

function getRollingAverage(historyRows, windowSize) {
  if (!Array.isArray(historyRows) || historyRows.length < windowSize || windowSize < 1) {
    return null;
  }

  const values = historyRows.slice(-windowSize).map((row) => row.value);
  if (values.some((value) => !Number.isFinite(value))) {
    return null;
  }

  const total = values.reduce((sum, value) => sum + value, 0);
  return total / windowSize;
}

function addUnique(list, value) {
  if (!list.includes(value)) {
    list.push(value);
  }
}

function buildGraceControlDefinition(displayMetadata, projectionSeed, projectionContext = null) {
  const featureDefinition = displayMetadata.features?.grace_groundwater_anomaly ?? {};
  const featureDomain = featureDefinition.domain ?? {};
  const defaultValue =
    projectionContext?.latestObservedControlValues?.grace_groundwater_anomaly
    ?? projectionSeed.control_seed_values?.grace_groundwater_anomaly
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

function buildFeatureCatalog(featureDefinitions, featureStats = {}) {
  return Object.entries(featureDefinitions ?? {}).reduce((catalog, [featureName, definition]) => {
    const stats = featureStats?.[featureName] ?? {};
    catalog[featureName] = {
      ...definition,
      domain: {
        ...(definition?.domain ?? {}),
        ...(stats ?? {}),
      },
    };
    return catalog;
  }, {});
}

function buildControlDefinitions(displayMetadata, projectionSeed, projectionContext = null) {
  const controlIds = (displayMetadata.knob_controls ?? []).map((id) =>
    id === "population" ? "grace_groundwater_anomaly" : id,
  );

  return controlIds.map((id) => {
    if (id === "grace_groundwater_anomaly" && !displayMetadata.controls?.[id]) {
      return buildGraceControlDefinition(displayMetadata, projectionSeed, projectionContext);
    }

    return {
      id,
      ...displayMetadata.controls[id],
    };
  });
}

function buildInitialControlValues(controlDefinitions, projectionSeed, projectionContext = null) {
  return controlDefinitions.reduce((accumulator, definition) => {
    const observedValue = projectionContext?.latestObservedControlValues?.[definition.id];
    const seedValue = projectionSeed.control_seed_values?.[definition.id];
    accumulator[definition.id] = observedValue ?? seedValue ?? definition.knob.default;
    return accumulator;
  }, {});
}

export function computePopulationChange(absolutePop, previousPop) {
  if (!Number.isFinite(absolutePop) || !Number.isFinite(previousPop) || previousPop === 0) {
    return 0;
  }

  return (absolutePop - previousPop) / previousPop;
}

function buildProjectionContextFromSeed(displayMetadata, projectionSeed) {
  const projectionPolicy = displayMetadata.projection_policy ?? {};
  const featureRowLastObserved = projectionSeed.feature_row_last_observed ?? {};
  const derivedFeatures = projectionPolicy.derived_features ?? [];
  const fixedRawFeatures = projectionPolicy.fixed_raw_features ?? [];
  const targetLagFeatures = Object.keys(featureRowLastObserved).filter((name) =>
    name.startsWith("usdm_sustainability_"),
  );
  const populationSeedAbsolute = projectionSeed.latest_observed_control_values?.population ?? null;

  return {
    contextSource: "projection_seed",
    sourceLastObservedMonth: projectionSeed.source_last_observed_month,
    projectionStartMonth: projectionSeed.projection_start_month,
    featureRowLastObserved,
    latestObservedControlValues: projectionSeed.latest_observed_control_values ?? {},
    populationSeedAbsolute,
    fixedModelInputValues: pickValues(featureRowLastObserved, fixedRawFeatures),
    derivedFeatureValues: pickValues(featureRowLastObserved, derivedFeatures),
    derivedFeatureOverrides: {},
    targetLagValues: pickValues(featureRowLastObserved, targetLagFeatures),
    visualSupportValues: pickValues(featureRowLastObserved, [
      "grace_groundwater_anomaly",
      "grace_available",
      "powell_pool_elevation",
    ]),
    targetHistory: normalizeHistoryRows(projectionSeed.target_history_last6),
    exogenousHistory: normalizeExogenousHistory(projectionSeed.exogenous_history),
  };
}

function buildProjectionContextFromHistoricalBaseline({
  displayMetadata,
  historicalSeries,
  historicalFeatureVectors,
  modelFeatureNames,
  baselineYearMonth = DEFAULT_PROJECTION_BASELINE_MONTH,
}) {
  const baselineRow = (historicalFeatureVectors ?? []).find((row) =>
    row?.yearMonth === baselineYearMonth && row.isModelComplete,
  );

  if (!baselineRow) {
    return null;
  }

  const baselineDate = baselineRow.date instanceof Date
    ? baselineRow.date
    : parseYearMonth(baselineRow.yearMonth);
  if (!(baselineDate instanceof Date) || Number.isNaN(baselineDate.getTime())) {
    return null;
  }

  const projectionPolicy = displayMetadata.projection_policy ?? {};
  const derivedFeatures = projectionPolicy.derived_features ?? [];
  const fixedRawFeatures = projectionPolicy.fixed_raw_features ?? [];
  const featureRowLastObserved = (modelFeatureNames ?? []).reduce((row, featureName) => {
    const value = coerceFiniteNumber(baselineRow[featureName]);
    if (value != null) {
      row[featureName] = value;
    }
    return row;
  }, {});
  const targetLagFeatures = Object.keys(featureRowLastObserved).filter((name) =>
    name.startsWith("usdm_sustainability_"),
  );
  const latestObservedControlValues = pickObservedControlValues(baselineRow);
  const targetHistory = (historicalSeries ?? [])
    .filter((row) =>
      row?.date instanceof Date
      && !Number.isNaN(row.date.getTime())
      && row.date <= baselineDate
      && Number.isFinite(row.score),
    )
    .slice(-6)
    .map((row) => buildHistoryRow(row.yearMonth, row.score))
    .filter(Boolean);

  return {
    contextSource: "historical_baseline",
    sourceLastObservedMonth: baselineYearMonth,
    projectionStartMonth: formatYearMonth(addUtcMonths(baselineDate, 1)),
    featureRowLastObserved,
    latestObservedControlValues,
    populationSeedAbsolute: latestObservedControlValues.population,
    fixedModelInputValues: pickValues(featureRowLastObserved, fixedRawFeatures),
    derivedFeatureValues: pickValues(featureRowLastObserved, derivedFeatures),
    derivedFeatureOverrides: {},
    targetLagValues: pickValues(featureRowLastObserved, targetLagFeatures),
    visualSupportValues: pickValues(featureRowLastObserved, [
      "grace_groundwater_anomaly",
      "grace_available",
      "powell_pool_elevation",
    ]),
    targetHistory,
    exogenousHistory: buildHistoricalExogenousHistory({
      historicalFeatureVectors,
      baselineDate,
      modelFeatureNames,
    }),
  };
}

function buildProjectionContext({
  displayMetadata,
  projectionSeed,
  historicalSeries,
  historicalFeatureVectors,
  modelFeatureNames,
}) {
  return buildProjectionContextFromHistoricalBaseline({
    displayMetadata,
    historicalSeries,
    historicalFeatureVectors,
    modelFeatureNames,
  }) ?? buildProjectionContextFromSeed(displayMetadata, projectionSeed);
}

function buildCurrentPrediction(projectionSeed, historicalSeries, projectionContext = null) {
  if (projectionContext?.contextSource === "historical_baseline") {
    return null;
  }

  const baselinePoint = (historicalSeries ?? []).find((row) =>
    row?.yearMonth === projectionContext?.sourceLastObservedMonth,
  );
  const lastHistoricalPoint = historicalSeries.at(-1) ?? null;

  return (
    projectionSeed.latest_predicted_score
    ?? projectionSeed.latest_observed_score
    ?? baselinePoint?.predictedScore
    ?? baselinePoint?.score
    ?? lastHistoricalPoint?.predictedScore
    ?? lastHistoricalPoint?.score
    ?? null
  );
}

function getHistorySourceType(baseFeature, controlByOutputFeature, fixedRawFeaturesSet, populationOutputFeature) {
  if (baseFeature === TARGET_FEATURE_NAME) {
    return "target_history";
  }

  if (baseFeature === TEMPERATURE_ANOMALY_FEATURE) {
    return "temperature_anomaly_history";
  }

  if (baseFeature === populationOutputFeature) {
    return "population_change_history";
  }

  if (controlByOutputFeature.has(baseFeature)) {
    return "scenario_exogenous_history";
  }

  if (fixedRawFeaturesSet.has(baseFeature)) {
    return "fixed_exogenous_history";
  }

  return "seed_feature_history";
}

function ensureHistoryRequirement(historyRequirements, baseFeature, historySource) {
  if (!historyRequirements[baseFeature]) {
    historyRequirements[baseFeature] = {
      baseFeature,
      historySource,
      lagDepths: [],
      rollingWindows: [],
      requiredHistoryLength: 0,
      derivedFromFeature: null,
    };
  }

  return historyRequirements[baseFeature];
}

export function buildProjectionConnections({
  modelFeatureNames,
  controlDefinitions,
  displayMetadata,
}) {
  const projectionPolicy = displayMetadata?.projection_policy ?? {};
  const featureNames = modelFeatureNames?.length
    ? [...modelFeatureNames]
    : Object.keys(displayMetadata?.features ?? {});
  const calendarFeatures = projectionPolicy.calendar_features ?? [];
  const fixedRawFeaturesSet = new Set(projectionPolicy.fixed_raw_features ?? []);
  const controlByOutputFeature = new Map();
  const controlConnections = {};
  const featurePolicies = {};
  const historyRequirements = {};
  const connectionWarnings = [];
  const populationOutputFeature = projectionPolicy.population_mapping?.output_feature ?? "AZPOP_pct_change";

  controlDefinitions.forEach((definition) => {
    if (definition.model_mapping?.output_feature) {
      controlByOutputFeature.set(definition.model_mapping.output_feature, definition.id);
    }
  });

  featureNames.forEach((featureName) => {
    const lagMatch = LAG_FEATURE_PATTERN.exec(featureName);
    const rollMatch = ROLL_FEATURE_PATTERN.exec(featureName);

    if (lagMatch) {
      const baseFeature = lagMatch[1];
      const lagDepth = Number(lagMatch[2]);
      const historySource = getHistorySourceType(
        baseFeature,
        controlByOutputFeature,
        fixedRawFeaturesSet,
        populationOutputFeature,
      );
      const requirement = ensureHistoryRequirement(historyRequirements, baseFeature, historySource);
      requirement.requiredHistoryLength = Math.max(requirement.requiredHistoryLength, lagDepth);
      addUnique(requirement.lagDepths, lagDepth);
      if (baseFeature === TEMPERATURE_ANOMALY_FEATURE) {
        requirement.derivedFromFeature = "temperature_2m_c";
      }

      featurePolicies[featureName] = {
        featureName,
        type: "lag",
        baseFeature,
        lagDepth,
        historySource,
        controlId: controlByOutputFeature.get(baseFeature) ?? null,
      };
      return;
    }

    if (rollMatch) {
      const baseFeature = rollMatch[1];
      const rollingWindow = Number(rollMatch[2]);
      const historySource = getHistorySourceType(
        baseFeature,
        controlByOutputFeature,
        fixedRawFeaturesSet,
        populationOutputFeature,
      );
      const requirement = ensureHistoryRequirement(historyRequirements, baseFeature, historySource);
      requirement.requiredHistoryLength = Math.max(requirement.requiredHistoryLength, rollingWindow);
      addUnique(requirement.rollingWindows, rollingWindow);
      if (baseFeature === TEMPERATURE_ANOMALY_FEATURE) {
        requirement.derivedFromFeature = "temperature_2m_c";
      }

      featurePolicies[featureName] = {
        featureName,
        type: "roll",
        baseFeature,
        rollingWindow,
        historySource,
        controlId: controlByOutputFeature.get(baseFeature) ?? null,
      };
      return;
    }

    if (calendarFeatures.includes(featureName)) {
      featurePolicies[featureName] = {
        featureName,
        type: "calendar",
        baseFeature: featureName,
      };
      return;
    }

    if (featureName === TEMPERATURE_ANOMALY_FEATURE) {
      featurePolicies[featureName] = {
        featureName,
        type: "temperature_anomaly",
        baseFeature: featureName,
        sourceFeature: "temperature_2m_c",
      };
      return;
    }

    if (featureName === populationOutputFeature) {
      featurePolicies[featureName] = {
        featureName,
        type: "population_change",
        baseFeature: featureName,
      };
      return;
    }

    if (controlByOutputFeature.has(featureName)) {
      featurePolicies[featureName] = {
        featureName,
        type: "scenario_direct",
        baseFeature: featureName,
        controlId: controlByOutputFeature.get(featureName),
      };
      return;
    }

    if (fixedRawFeaturesSet.has(featureName)) {
      featurePolicies[featureName] = {
        featureName,
        type: "fixed_raw",
        baseFeature: featureName,
      };
      return;
    }

    featurePolicies[featureName] = {
      featureName,
      type: "seed_passthrough",
      baseFeature: featureName,
    };
  });

  controlDefinitions.forEach((definition) => {
    const outputFeature = definition.model_mapping?.output_feature;
    if (!outputFeature) {
      return;
    }

    const affectedPolicies = Object.values(featurePolicies).filter((policy) =>
      policy.baseFeature === outputFeature || policy.sourceFeature === outputFeature,
    );

    controlConnections[definition.id] = {
      controlId: definition.id,
      label: definition.label,
      mappingType: definition.model_mapping?.type ?? "unknown",
      outputFeature,
      directFeatureName: outputFeature,
      historyRequirement: historyRequirements[outputFeature] ?? null,
      lagFeatureNames: affectedPolicies
        .filter((policy) => policy.type === "lag" && policy.baseFeature === outputFeature)
        .map((policy) => policy.featureName),
      rollingFeatureNames: affectedPolicies
        .filter((policy) => policy.type === "roll" && policy.baseFeature === outputFeature)
        .map((policy) => policy.featureName),
      derivedFeatureNames: affectedPolicies
        .filter((policy) => policy.type === "temperature_anomaly")
        .map((policy) => policy.featureName),
      derivedLagFeatureNames: affectedPolicies
        .filter((policy) => policy.type === "lag" && policy.baseFeature === TEMPERATURE_ANOMALY_FEATURE)
        .map((policy) => policy.featureName),
      derivedRollingFeatureNames: affectedPolicies
        .filter((policy) => policy.type === "roll" && policy.baseFeature === TEMPERATURE_ANOMALY_FEATURE)
        .map((policy) => policy.featureName),
    };

    if (!featureNames.includes(outputFeature)) {
      connectionWarnings.push(
        `Control "${definition.id}" maps to "${outputFeature}", which is not present in feature_names.json.`,
      );
    }
  });

  return {
    targetFeature: TARGET_FEATURE_NAME,
    modelFeatureNames: featureNames,
    calendarFeatures,
    controlConnections,
    featurePolicies,
    historyRequirements,
    fixedRawFeatures: [...fixedRawFeaturesSet],
    connectionWarnings,
  };
}

function buildFeatureSource(strategy, extra = {}) {
  return {
    strategy,
    ...extra,
  };
}

function resolveHistoryRows(baseFeature, projectionPreparationContext) {
  const {
    targetHistory,
    exogenousHistory,
    displayMetadata,
  } = projectionPreparationContext;

  if (baseFeature === TARGET_FEATURE_NAME) {
    return targetHistory ?? [];
  }

  if (baseFeature === TEMPERATURE_ANOMALY_FEATURE) {
    const temperatureRows = exogenousHistory?.temperature_2m_c?.rows ?? [];
    return buildTemperatureAnomalyHistory(temperatureRows, displayMetadata);
  }

  if (baseFeature === "AZPOP_pct_change") {
    return exogenousHistory?.population_change?.rows ?? [];
  }

  return exogenousHistory?.[baseFeature]?.rows ?? [];
}

function buildProjectionPendingFeatureRow({
  projectionConnections,
  projectionContext,
  targetHistory,
  exogenousHistory,
  nextProjectionMonth,
  controlDefinitions,
  controlValues,
  displayMetadata,
  featureCatalog,
}) {
  const featureRow = {
    ...(projectionContext.featureRowLastObserved ?? {}),
  };
  const featureSources = {};
  const warnings = [];
  const projectedCalendarValues = buildCalendarFeatureValues(nextProjectionMonth);

  projectionConnections.modelFeatureNames.forEach((featureName) => {
    featureSources[featureName] = buildFeatureSource("seed_feature_row");
  });

  projectionConnections.fixedRawFeatures.forEach((featureName) => {
    if (Object.prototype.hasOwnProperty.call(projectionContext.fixedModelInputValues, featureName)) {
      featureRow[featureName] = projectionContext.fixedModelInputValues[featureName];
      featureSources[featureName] = buildFeatureSource("fixed_policy_seed", {
        featureName,
      });
    }
  });

  Object.entries(projectionContext.derivedFeatureOverrides ?? {}).forEach(([featureName, value]) => {
    if (!Number.isFinite(value)) {
      return;
    }

    featureRow[featureName] = value;
    featureSources[featureName] = buildFeatureSource("derived_override", {
      featureName,
    });
  });

  controlDefinitions.forEach((definition) => {
    const outputFeature = definition.model_mapping?.output_feature;
    if (!outputFeature) return;

    const controlValue = controlValues?.[definition.id];
    if (controlValue == null) return;

    if (definition.model_mapping.type === "direct_feature") {
      featureRow[outputFeature] = controlValue;
      featureSources[outputFeature] = buildFeatureSource("scenario_control", {
        controlId: definition.id,
        outputFeature,
      });
      return;
    }

    if (definition.model_mapping.type === "pct_change_from_absolute_series") {
      const previousPopulation =
        getLatestHistoryValue(exogenousHistory?.population?.rows)
        ?? projectionContext.latestObservedControlValues?.population
        ?? projectionContext.populationSeedAbsolute;
      const percentChange = computePopulationChange(
        controlValue,
        previousPopulation,
      );
      featureRow[outputFeature] = percentChange;
      featureSources[outputFeature] = buildFeatureSource("scenario_population_change", {
        controlId: definition.id,
        outputFeature,
      });
    }
  });

  if (Number.isFinite(projectedCalendarValues.month_sin)) {
    featureRow.month_sin = projectedCalendarValues.month_sin;
    featureSources.month_sin = buildFeatureSource("projection_calendar", {
      projectionMonth: nextProjectionMonth,
    });
  }

  if (Number.isFinite(projectedCalendarValues.month_cos)) {
    featureRow.month_cos = projectedCalendarValues.month_cos;
    featureSources.month_cos = buildFeatureSource("projection_calendar", {
      projectionMonth: nextProjectionMonth,
    });
  }

  if (featureRow.temperature_2m_c != null) {
    const temperatureBaseline = getTemperatureBaseline(displayMetadata, nextProjectionMonth);
    if (Number.isFinite(temperatureBaseline)) {
      featureRow.temperature_2m_c_anomaly = featureRow.temperature_2m_c - temperatureBaseline;
      featureSources.temperature_2m_c_anomaly = buildFeatureSource("derived_temperature_anomaly", {
        projectionMonth: nextProjectionMonth,
        sourceFeature: "temperature_2m_c",
      });
    }
  }

  Object.values(projectionConnections.featurePolicies).forEach((policy) => {
    if (policy.type !== "lag" && policy.type !== "roll") {
      return;
    }

    const historyRows = resolveHistoryRows(policy.baseFeature, {
      targetHistory,
      exogenousHistory,
      displayMetadata,
    });
    const historyValue = policy.type === "lag"
      ? getLagValue(historyRows, policy.lagDepth)
      : getRollingAverage(historyRows, policy.rollingWindow);

    if (Number.isFinite(historyValue)) {
      featureRow[policy.featureName] = historyValue;
      featureSources[policy.featureName] = buildFeatureSource(
        policy.type === "lag" ? "history_lag" : "history_roll",
        {
          baseFeature: policy.baseFeature,
          historySource: policy.historySource,
          lagDepth: policy.lagDepth ?? null,
          rollingWindow: policy.rollingWindow ?? null,
        },
      );
      return;
    }

    warnings.push(
      `Falling back to the seed feature row for "${policy.featureName}" because ${policy.baseFeature} history was unavailable.`,
    );
    featureSources[policy.featureName] = buildFeatureSource("seed_feature_row_fallback", {
      baseFeature: policy.baseFeature,
      historySource: policy.historySource,
    });
  });

  projectionConnections.modelFeatureNames.forEach((featureName) => {
    if (featureRow[featureName] != null) {
      return;
    }

    const median = featureCatalog?.[featureName]?.domain?.median;
    if (Number.isFinite(median)) {
      featureRow[featureName] = median;
      featureSources[featureName] = buildFeatureSource("feature_catalog_median_fallback", {
        featureName,
      });
      return;
    }

    if (!Object.prototype.hasOwnProperty.call(featureRow, featureName)) {
      featureRow[featureName] = 0;
      featureSources[featureName] = buildFeatureSource("zero_fallback", {
        featureName,
      });
      warnings.push(`Feature "${featureName}" had no seed value or metadata median; using 0.`);
    }
  });

  return {
    featureRow,
    featureSources,
    warnings,
  };
}

export function buildProjectionPreparation({
  projectionConnections,
  projectionContext,
  controlDefinitions,
  controlValues,
  currentPrediction,
  displayMetadata,
  featureCatalog,
  seedState = null,
  status = "ready",
}) {
  if (!projectionConnections || !projectionContext) {
    return null;
  }

  const stepIndex = Number.isInteger(seedState?.stepIndex) ? seedState.stepIndex : 0;
  const nextProjectionMonth = seedState?.nextProjectionMonth ?? projectionContext.projectionStartMonth;
  const targetHistory = cloneHistoryRows(seedState?.targetHistory ?? projectionContext.targetHistory);
  const exogenousHistory = cloneExogenousHistory(
    seedState?.exogenousHistory ?? projectionContext.exogenousHistory,
  );
  const latestScenarioScore = Number.isFinite(seedState?.latestScenarioScore)
    ? seedState.latestScenarioScore
    : currentPrediction;

  const pendingProjection = nextProjectionMonth
    ? buildProjectionPendingFeatureRow({
        projectionConnections,
        projectionContext,
        targetHistory,
        exogenousHistory,
        nextProjectionMonth,
        controlDefinitions,
        controlValues,
        displayMetadata,
        featureCatalog,
      })
    : {
        featureRow: null,
        featureSources: {},
        warnings: [],
      };

  return {
    status,
    stepIndex,
    nextProjectionMonth,
    latestScenarioScore,
    scenarioControls: { ...(controlValues ?? {}) },
    targetHistory,
    exogenousHistory,
    historyRequirements: projectionConnections.historyRequirements,
    pendingFeatureRow: pendingProjection.featureRow,
    pendingFeatureSources: pendingProjection.featureSources,
    connectionWarnings: [
      ...(projectionConnections.connectionWarnings ?? []),
      ...(pendingProjection.warnings ?? []),
    ],
  };
}

export function buildProjectionPoint({
  yearMonth,
  score,
  monthIndex,
  featureRow = null,
  featureSources = null,
  scenarioControls = null,
}) {
  const numericScore = Number(score);
  const date = parseYearMonth(yearMonth);

  if (!yearMonth || !(date instanceof Date) || Number.isNaN(date.getTime()) || !Number.isFinite(numericScore)) {
    return null;
  }

  return {
    yearMonth,
    date,
    predictedScore: numericScore,
    actualScore: null,
    score: numericScore,
    residual: null,
    source: "projection",
    isProjectionAnchor: false,
    monthIndex,
    featureRow: featureRow ? { ...featureRow } : null,
    featureSources: featureSources
      ? Object.fromEntries(
          Object.entries(featureSources).map(([key, value]) => [key, { ...value }]),
        )
      : null,
    scenarioControls: scenarioControls ? { ...scenarioControls } : null,
  };
}

export function buildProjectionSeries({
  historicalSeries,
  projectionPoints = [],
  projectionAnchorYearMonth = null,
}) {
  const projectionSeries = [];
  const projectionAnchorPoint = projectionAnchorYearMonth
    ? historicalSeries.find((point) => point?.yearMonth === projectionAnchorYearMonth) ?? null
    : null;
  const lastHistoricalPoint = projectionAnchorPoint ?? historicalSeries.at(-1) ?? null;

  if (lastHistoricalPoint?.date && Number.isFinite(lastHistoricalPoint.score)) {
    projectionSeries.push({
      yearMonth: lastHistoricalPoint.yearMonth,
      date: lastHistoricalPoint.date,
      predictedScore: lastHistoricalPoint.score,
      actualScore: lastHistoricalPoint.actualScore ?? lastHistoricalPoint.score,
      score: lastHistoricalPoint.score,
      residual: null,
      source: "projection_anchor",
      isProjectionAnchor: true,
      monthIndex: 0,
    });
  }

  projectionPoints.forEach((point, index) => {
    const date = point?.date instanceof Date ? point.date : parseYearMonth(point?.yearMonth);
    if (!(date instanceof Date) || Number.isNaN(date.getTime()) || !Number.isFinite(point?.score)) {
      return;
    }

    projectionSeries.push({
      ...point,
      date,
      source: "projection",
      isProjectionAnchor: false,
      monthIndex: point.monthIndex ?? index + 1,
    });
  });

  return projectionSeries;
}

export function advanceProjectionPreparation({
  projectionPreparation,
  projectionConnections,
  projectionContext,
  controlDefinitions,
  controlValues,
  predictedScore,
  displayMetadata,
  featureCatalog,
}) {
  if (!projectionPreparation || !projectionConnections || !projectionContext) {
    return null;
  }

  const currentProjectionMonth = projectionPreparation.nextProjectionMonth;
  const currentStepIndex = Number.isInteger(projectionPreparation.stepIndex)
    ? projectionPreparation.stepIndex
    : 0;
  const nextTargetHistory = trimHistoryRows(
    [
      ...cloneHistoryRows(projectionPreparation.targetHistory),
      buildHistoryRow(currentProjectionMonth, predictedScore),
    ].filter(Boolean),
    projectionPreparation.historyRequirements?.[TARGET_FEATURE_NAME]?.requiredHistoryLength ?? 6,
  );

  let nextExogenousHistory = cloneExogenousHistory(projectionPreparation.exogenousHistory);

  const populationDefinition = controlDefinitions.find(
    (definition) => definition.model_mapping?.type === "pct_change_from_absolute_series",
  );

  if (populationDefinition) {
    const populationValue = Number(controlValues?.[populationDefinition.id]);
    if (Number.isFinite(populationValue)) {
      nextExogenousHistory = appendHistoryValue(
        nextExogenousHistory,
        "population",
        currentProjectionMonth,
        populationValue,
        nextExogenousHistory.population?.depth ?? 2,
      );
    }
  }

  Object.entries(projectionPreparation.historyRequirements ?? {}).forEach(([baseFeature, requirement]) => {
    if (baseFeature === TARGET_FEATURE_NAME || baseFeature === TEMPERATURE_ANOMALY_FEATURE) {
      return;
    }

    const pendingValue = Number(projectionPreparation.pendingFeatureRow?.[baseFeature]);
    const fallbackValue =
      getLatestHistoryValue(nextExogenousHistory[getHistoryStorageFeature(baseFeature)]?.rows)
      ?? Number(projectionContext.featureRowLastObserved?.[baseFeature]);
    const nextValue = Number.isFinite(pendingValue) ? pendingValue : fallbackValue;

    if (!Number.isFinite(nextValue)) {
      return;
    }

    nextExogenousHistory = appendHistoryValue(
      nextExogenousHistory,
      baseFeature,
      currentProjectionMonth,
      nextValue,
      requirement.requiredHistoryLength,
    );
  });

  const nextProjectionDate = addUtcMonths(parseYearMonth(currentProjectionMonth), 1);
  const nextProjectionMonth = formatYearMonth(nextProjectionDate);

  return buildProjectionPreparation({
    projectionConnections,
    projectionContext,
    controlDefinitions,
    controlValues,
    currentPrediction: predictedScore,
    displayMetadata,
    featureCatalog,
    seedState: {
      stepIndex: currentStepIndex + 1,
      nextProjectionMonth,
      latestScenarioScore: predictedScore,
      targetHistory: nextTargetHistory,
      exogenousHistory: nextExogenousHistory,
    },
  });
}

export function buildHydratedState(
  displayMetadata,
  projectionSeed,
  historicalSeriesRows,
  historicalFeatureVectorRows,
  modelFeatureNames = [],
  featureStats = {},
) {
  const featureCatalog = buildFeatureCatalog(displayMetadata.features ?? {}, featureStats);
  const historicalSeries = normalizeHistoricalSeries(historicalSeriesRows);
  const historicalFeatureVectors = normalizeHistoricalFeatureVectors(historicalFeatureVectorRows);
  const projectionHorizonMonths =
    displayMetadata.projection_policy?.horizon_months_default
    ?? DEFAULT_PROJECTION_HORIZON_MONTHS;
  const projectionContext = buildProjectionContext({
    displayMetadata,
    projectionSeed,
    historicalSeries,
    historicalFeatureVectors,
    modelFeatureNames,
  });
  const controlDefinitions = buildControlDefinitions(
    displayMetadata,
    projectionSeed,
    projectionContext,
  );
  const currentPrediction = buildCurrentPrediction(
    projectionSeed,
    historicalSeries,
    projectionContext,
  );
  const projectionConnections = buildProjectionConnections({
    modelFeatureNames,
    controlDefinitions,
    displayMetadata,
  });
  const controlValues = buildInitialControlValues(
    controlDefinitions,
    projectionSeed,
    projectionContext,
  );

  return {
    displayMetadata,
    featureCatalog,
    modelFeatureNames,
    controlDefinitions,
    controlValues,
    historicalSeries,
    historicalFeatureVectors,
    projectionSeries: buildProjectionSeries({
      historicalSeries,
      projectionPoints: [],
      projectionAnchorYearMonth: projectionContext?.sourceLastObservedMonth ?? null,
    }),
    projectionHorizonMonths,
    projectionConnections,
    projectionPreparation: buildProjectionPreparation({
      projectionConnections,
      projectionContext,
      controlDefinitions,
      controlValues,
      currentPrediction,
      displayMetadata,
      featureCatalog,
    }),
    projectionContext,
    animationStatus: "idle",
    currentPrediction,
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

export function buildMedianFeatureRow(featureNames, featureCatalog) {
  return featureNames.reduce((row, featureName) => {
    const median = featureCatalog?.[featureName]?.domain?.median;
    if (!Number.isFinite(median)) {
      throw new Error(`Missing median metadata for ${featureName}`);
    }
    row[featureName] = median;
    return row;
  }, {});
}

export function createOrderedFloat32Array(featureNames, featureRow) {
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

export function createScaledFloat32Array(featureNames, featureRow, featureCatalog = {}) {
  const orderedValues = new Float32Array(featureNames.length);

  featureNames.forEach((featureName, index) => {
    if (!Object.prototype.hasOwnProperty.call(featureRow, featureName)) {
      throw new Error(`Feature row is missing required feature "${featureName}"`);
    }

    const rawValue = Number(featureRow[featureName]);
    assertFiniteNumber(rawValue, `Feature "${featureName}" is not finite.`);

    const domain = featureCatalog?.[featureName]?.domain ?? {};
    const mean = Number(domain.mean);
    const std = Number(domain.std);
    const scaledValue = Number.isFinite(mean) && Number.isFinite(std) && std > 0
      ? (rawValue - mean) / std
      : rawValue;

    assertFiniteNumber(scaledValue, `Scaled feature "${featureName}" is not finite.`);
    orderedValues[index] = scaledValue;
  });

  return orderedValues;
}

export function extractScalarScore(outputValue) {
  if (!outputValue?.data?.length) {
    throw new Error("Model output tensor is empty.");
  }

  const score = Number(outputValue.data[0]);
  assertFiniteNumber(score, "Model output score is not finite.");
  return clipScore(score);
}

export async function fetchJson(basePath, fileName) {
  const response = await fetch(`${basePath}/${fileName}`);
  if (!response.ok) {
    throw new Error(`Failed to load ${fileName}`);
  }
  return response.json();
}

export async function fetchCsv(basePath, fileName) {
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
