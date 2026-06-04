import * as d3 from "d3";
import { getAquiferColor } from "./cross-section-waterline.js";
import { formatNumber } from "../runtime-utils.js";

const CHART_WIDTH = 1040;
const CHART_HEIGHT = 460;
const MARGINS = {
  top: 18,
  right: 28,
  bottom: 46,
  left: 66,
};
const OVERVIEW_RAIL = {
  bottomInset: 10,
  height: 28,
  gap: 8,
};
const TYPOGRAPHY = {
  axisSize: 16,
  axisWeight: 700,
  sectionLabelSize: 18,
  sectionLabelWeight: 800,
  overviewLabelSize: 14,
  overviewLabelWeight: 800,
};
const Y_DOMAIN = [0, 100];
const COLORS = {
  grid: "rgba(111, 95, 79, 0.16)",
  axis: "#6f5f4f",
  historical: "#70879a",
  historicalPoint: "rgba(112, 135, 154, 0.7)",
  projection: "#d68c2d",
  projectionFill: "rgba(214, 140, 45, 0.12)",
  projectionPoint: "#d68c2d",
  brushFill: "rgba(13, 108, 116, 0.12)",
  brushStroke: "rgba(13, 108, 116, 0.4)",
  activeRing: "#0d6c74",
};
const SMALL_TICK_FORMAT = d3.utcFormat("%b %Y");
const YEAR_TICK_FORMAT = d3.utcFormat("%Y");

function isDrawablePoint(point) {
  return (
    point?.date instanceof Date
    && !Number.isNaN(point.date.getTime())
    && Number.isFinite(point.score)
  );
}

function getHistoricalSeries(state) {
  return (state.historicalSeries ?? []).filter(isDrawablePoint);
}

function getProjectionSeries(state) {
  return (state.projectionSeries ?? []).filter(isDrawablePoint);
}

function getPointKey(point) {
  return point ? `${point.source}:${point.yearMonth}:${point.monthIndex ?? 0}` : "";
}

function pointMatches(left, right) {
  return getPointKey(left) !== "" && getPointKey(left) === getPointKey(right);
}

function resolveSelectedPoint(state) {
  const selectedPoint = state.selectedPoint;
  if (!selectedPoint) {
    return null;
  }

  const interactionPoints = [
    ...getHistoricalSeries(state),
    ...getProjectionSeries(state).filter((point) => !point.isProjectionAnchor),
  ];

  return interactionPoints.find((point) => pointMatches(point, selectedPoint)) ?? null;
}

function getProjectionSnapshot(points) {
  return points
    .filter((point) => !point.isProjectionAnchor)
    .map((point) => `${getPointKey(point)}:${Number(point.score).toFixed(2)}`);
}

function getAppendedProjectionSegment(previousSnapshot, projectionSeries) {
  const projectionFuture = projectionSeries.filter((point) => !point.isProjectionAnchor);
  const currentSnapshot = getProjectionSnapshot(projectionSeries);

  if (projectionFuture.length !== previousSnapshot.length + 1) {
    return null;
  }

  const isPrefixMatch = previousSnapshot.every((entry, index) => entry === currentSnapshot[index]);
  if (!isPrefixMatch) {
    return null;
  }

  const newPoint = projectionFuture.at(-1) ?? null;
  const previousPoint = projectionSeries.at(-2) ?? null;
  if (!isDrawablePoint(newPoint) || !isDrawablePoint(previousPoint)) {
    return null;
  }

  return {
    newPoint,
    previousPoint,
  };
}

function getMonthSpan(domain) {
  if (!Array.isArray(domain) || domain.length !== 2) return 0;

  const [start, end] = domain;
  if (!(start instanceof Date) || !(end instanceof Date)) return 0;

  return (
    (end.getUTCFullYear() - start.getUTCFullYear()) * 12
    + (end.getUTCMonth() - start.getUTCMonth())
  );
}

function createXAxis(scale, domain) {
  const monthSpan = getMonthSpan(domain);

  if (monthSpan <= 18) {
    return d3.axisBottom(scale)
      .ticks(d3.utcMonth.every(2))
      .tickFormat(SMALL_TICK_FORMAT);
  }

  if (monthSpan <= 48) {
    return d3.axisBottom(scale)
      .ticks(d3.utcMonth.every(6))
      .tickFormat(SMALL_TICK_FORMAT);
  }

  return d3.axisBottom(scale)
    .ticks(d3.utcYear.every(2))
    .tickFormat(YEAR_TICK_FORMAT);
}

function createLineGenerator(xScale, yScale) {
  return d3.line()
    .defined(isDrawablePoint)
    .x((point) => xScale(point.date))
    .y((point) => yScale(point.score));
}


function describePoint(point) {
  const dateLabel = SMALL_TICK_FORMAT(point.date);
  const scoreLabel = formatNumber(point.score, 1);

  if (point.source === "projection") {
    return `Projected month · ${dateLabel} · score ${scoreLabel}`;
  }

  if (point.source === "projection_anchor") {
    return `Projection seed · ${dateLabel} · score ${scoreLabel}`;
  }

  return `Historical point · ${dateLabel} · score ${scoreLabel}`;
}

function getDefaultStatusMessage() {
  return "Click a point to inspect the knob-driven inputs and related model features for that month.";
}

function clampBrushDomain(brushDomain, fullDomain) {
  if (!brushDomain) return null;
  if (!Array.isArray(brushDomain) || brushDomain.length !== 2) return null;

  const [fullStart, fullEnd] = fullDomain;
  const [brushStart, brushEnd] = brushDomain;

  if (
    !(fullStart instanceof Date)
    || !(fullEnd instanceof Date)
    || !(brushStart instanceof Date)
    || !(brushEnd instanceof Date)
  ) {
    return null;
  }

  const clampedStart = new Date(Math.max(fullStart.getTime(), brushStart.getTime()));
  const clampedEnd = new Date(Math.min(fullEnd.getTime(), brushEnd.getTime()));

  if (clampedStart >= clampedEnd) {
    return null;
  }

  return [clampedStart, clampedEnd];
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function coerceFiniteNumber(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
}

function getControlDisplayLabel(definition) {
  if (definition?.id === "grace_groundwater_anomaly") {
    return "Groundwater Level";
  }

  return definition?.label ?? "Control";
}

function isTemperatureControl(definition) {
  return definition?.id === "temperature_2m_c";
}

function toTemperatureFahrenheit(value) {
  if (!Number.isFinite(value)) return value;
  return (value * 9) / 5 + 32;
}

function formatControlValueWithUnit(value, definition) {
  if (!Number.isFinite(value)) {
    return "Unavailable";
  }

  if (isTemperatureControl(definition)) {
    return `${formatNumber(toTemperatureFahrenheit(value), definition?.decimals ?? 1)} F`;
  }

  const unitSuffix = definition?.unit ? ` ${definition.unit}` : "";
  return `${formatNumber(value, definition?.decimals ?? 0)}${unitSuffix}`;
}

function formatFeatureName(featureName) {
  return featureName
    ?.replaceAll("_", " ")
    ?.replace(/\b\w/g, (character) => character.toUpperCase()) ?? "Feature";
}

function getFeatureDefinition(state, featureName) {
  return state.featureCatalog?.[featureName] ?? state.displayMetadata?.features?.[featureName] ?? null;
}

function getFeatureLabel(state, featureName) {
  return getFeatureDefinition(state, featureName)?.label ?? formatFeatureName(featureName);
}

function formatFeatureValueWithUnit(state, featureName, value) {
  if (!Number.isFinite(value)) {
    return "Unavailable";
  }

  const featureDefinition = getFeatureDefinition(state, featureName);
  const unitSuffix = featureDefinition?.unit ? ` ${featureDefinition.unit}` : "";
  return `${formatNumber(value, featureDefinition?.decimals ?? 2)}${unitSuffix}`;
}

function getHistoricalFeatureVectorRow(state, point) {
  return (state.historicalFeatureVectors ?? []).find((row) => row?.yearMonth === point?.yearMonth) ?? null;
}

function getHistoricalControlValue(definition, historicalRow) {
  if (!historicalRow) {
    return null;
  }

  const mappingType = definition?.model_mapping?.type;
  const inputFeature = definition?.model_mapping?.input_feature;
  const outputFeature = definition?.model_mapping?.output_feature;

  if (mappingType === "pct_change_from_absolute_series") {
    return coerceFiniteNumber(historicalRow?.[inputFeature]);
  }

  return (
    coerceFiniteNumber(historicalRow?.[inputFeature])
    ?? coerceFiniteNumber(historicalRow?.[outputFeature])
    ?? coerceFiniteNumber(historicalRow?.[definition?.id])
  );
}

function getProjectionControlValue(definition, point) {
  return (
    coerceFiniteNumber(point?.scenarioControls?.[definition?.id])
    ?? coerceFiniteNumber(point?.featureRow?.[definition?.model_mapping?.input_feature])
    ?? coerceFiniteNumber(point?.featureRow?.[definition?.model_mapping?.output_feature])
  );
}

function getControlValueForPoint(definition, point, historicalRow) {
  return point?.source === "projection"
    ? getProjectionControlValue(definition, point)
    : getHistoricalControlValue(definition, historicalRow);
}

function getControlConnection(state, definition) {
  return state.projectionConnections?.controlConnections?.[definition?.id] ?? null;
}

function getAffectedFeatureNames(state, definition) {
  const connection = getControlConnection(state, definition);
  const orderedFeatures = [];

  if (
    definition?.model_mapping?.type !== "direct_feature"
    && connection?.directFeatureName
  ) {
    orderedFeatures.push(connection.directFeatureName);
  }

  [
    ...(connection?.derivedFeatureNames ?? []),
    ...(connection?.lagFeatureNames ?? []),
    ...(connection?.rollingFeatureNames ?? []),
    ...(connection?.derivedLagFeatureNames ?? []),
    ...(connection?.derivedRollingFeatureNames ?? []),
  ].forEach((featureName) => {
    if (featureName && !orderedFeatures.includes(featureName)) {
      orderedFeatures.push(featureName);
    }
  });

  return orderedFeatures;
}

function buildAffectedFeatureEntries({
  state,
  definition,
  point,
  featureRow,
}) {
  return getAffectedFeatureNames(state, definition)
    .map((featureName) => {
      const value = coerceFiniteNumber(featureRow?.[featureName]);
      if (!Number.isFinite(value)) {
        return null;
      }

      return {
        featureName,
        label: getFeatureLabel(state, featureName),
        value,
      };
    })
    .filter(Boolean);
}

function populatePointLegend(legend, state, point) {
  legend.replaceChildren();

  if (!state.bundleReady || !point) {
    legend.hidden = true;
    return false;
  }

  const historicalRow = point.source === "projection" ? null : getHistoricalFeatureVectorRow(state, point);
  const featureRow = point.source === "projection" ? point.featureRow ?? null : historicalRow;
  const header = document.createElement("div");
  header.className = "time-series-point-legend__header";

  const title = document.createElement("h3");
  title.className = "time-series-point-legend__title";
  const titleDate = document.createElement("span");
  titleDate.textContent = `${SMALL_TICK_FORMAT(point.date)} · `;

  const titleScore = document.createElement("span");
  titleScore.className = "time-series-point-legend__score";
  titleScore.style.color = getAquiferColor(point.score);
  titleScore.textContent = `Sustainability Score ${formatNumber(point.score, 1)}`;

  title.append(titleDate, titleScore);
  header.append(title);

  const controls = document.createElement("div");
  controls.className = "time-series-point-legend__controls";

  (state.controlDefinitions ?? []).forEach((definition) => {
    const controlValue = getControlValueForPoint(definition, point, historicalRow);
    const featureEntries = buildAffectedFeatureEntries({
      state,
      definition,
      point,
      featureRow,
    });

    if (controlValue == null && featureEntries.length === 0) {
      return;
    }

    const card = document.createElement("article");
    card.className = "time-series-point-legend__control";

    const cardHeader = document.createElement("div");
    cardHeader.className = "time-series-point-legend__control-header";

    const label = document.createElement("strong");
    label.textContent = getControlDisplayLabel(definition);

    const value = document.createElement("span");
    value.className = "time-series-point-legend__control-value";
    value.textContent = formatControlValueWithUnit(controlValue, definition);

    cardHeader.append(label, value);
    card.appendChild(cardHeader);

    if (featureEntries.length) {
      const featureList = document.createElement("div");
      featureList.className = "time-series-point-legend__feature-list";

      featureEntries.forEach((entry) => {
        const feature = document.createElement("div");
        feature.className = "time-series-point-legend__feature";

        const copyRow = document.createElement("div");
        copyRow.className = "time-series-point-legend__feature-copy";

        const featureLabel = document.createElement("span");
        featureLabel.className = "time-series-point-legend__feature-label";
        featureLabel.textContent = entry.label;

        const featureValue = document.createElement("span");
        featureValue.className = "time-series-point-legend__feature-value";
        featureValue.textContent = formatFeatureValueWithUnit(state, entry.featureName, entry.value);

        copyRow.append(featureLabel, featureValue);
        feature.appendChild(copyRow);
        featureList.appendChild(feature);
      });

      card.appendChild(featureList);
    }

    controls.appendChild(card);
  });

  legend.append(header, controls);
  legend.hidden = false;
  return true;
}

function renderPointLegend({
  legend,
  chartWrap,
  state,
  point,
  pointX,
  pointY,
}) {
  const inView = (
    Number.isFinite(pointX)
    && Number.isFinite(pointY)
    && pointX >= MARGINS.left
    && pointX <= CHART_WIDTH - MARGINS.right
    && pointY >= MARGINS.top
    && pointY <= CHART_HEIGHT - MARGINS.bottom
  );

  if (!point || !inView || !populatePointLegend(legend, state, point)) {
    legend.hidden = true;
    legend.replaceChildren();
    return;
  }

  const wrapRect = chartWrap.getBoundingClientRect();
  if (wrapRect.width <= 0 || wrapRect.height <= 0) {
    legend.hidden = true;
    return;
  }

  const pointXPx = (pointX / CHART_WIDTH) * wrapRect.width;
  const pointYPx = (pointY / CHART_HEIGHT) * wrapRect.height;
  const overlayPadding = 12;
  const gap = 18;

  legend.hidden = false;
  legend.dataset.side = "right";
  legend.style.left = `${overlayPadding}px`;
  legend.style.top = `${overlayPadding}px`;
  legend.style.visibility = "hidden";

  const legendWidth = legend.offsetWidth;
  const legendHeight = legend.offsetHeight;
  const canFitRight = pointXPx + gap + legendWidth <= wrapRect.width - overlayPadding;
  const canFitLeft = pointXPx - gap - legendWidth >= overlayPadding;
  const side = canFitRight || !canFitLeft ? "right" : "left";
  const left = side === "right"
    ? Math.min(wrapRect.width - legendWidth - overlayPadding, pointXPx + gap)
    : Math.max(overlayPadding, pointXPx - legendWidth - gap);
  const top = clampNumber(
    pointYPx - (legendHeight / 2),
    overlayPadding,
    wrapRect.height - legendHeight - overlayPadding,
  );
  const arrowY = clampNumber(pointYPx - top, 18, Math.max(18, legendHeight - 18));

  legend.dataset.side = side;
  legend.style.left = `${left}px`;
  legend.style.top = `${top}px`;
  legend.style.setProperty("--legend-arrow-y", `${arrowY}px`);
  legend.style.visibility = "visible";
}

export function createTimeSeriesPanel(runtime) {
  const panel = document.createElement("section");
  panel.className = "panel viz-card time-series-panel";

  const shell = document.createElement("div");
  shell.className = "time-series-shell";

  const chartWrap = document.createElement("div");
  chartWrap.className = "time-series-chart";
  chartWrap.hidden = true;
  chartWrap.innerHTML = `
    <div class="time-series-chart-legend" aria-hidden="true">
      <span class="time-series-legend__item">
        <span class="time-series-legend__swatch" style="--swatch:${COLORS.historical}"></span>
        Historical
      </span>
      <span class="time-series-legend__item">
        <span class="time-series-legend__swatch" style="--swatch:${COLORS.projection}"></span>
        Projection
      </span>
    </div>
  `;

  const svg = d3.select(chartWrap)
    .append("svg")
    .attr("class", "time-series-svg")
    .attr("viewBox", `0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`)
    .attr("preserveAspectRatio", "xMidYMid meet");

  const pointLegend = document.createElement("aside");
  pointLegend.className = "time-series-point-legend";
  pointLegend.hidden = true;
  chartWrap.appendChild(pointLegend);

  const resetButton = document.createElement("button");
  resetButton.type = "button";
  resetButton.className = "time-series-reset";
  resetButton.textContent = "Reset zoom";
  resetButton.disabled = true;

  const status = document.createElement("p");
  status.className = "time-series-status";
  status.hidden = true;

  const footer = document.createElement("div");
  footer.className = "time-series-footer";

  const unlimitedToggle = document.createElement("button");
  unlimitedToggle.type = "button";
  unlimitedToggle.className = "time-series-unlimited-toggle";
  unlimitedToggle.textContent = "Unlimited forecast: off";

  footer.append(resetButton, status, unlimitedToggle);
  shell.append(chartWrap, footer);
  panel.append(shell);

  let brushDomain = null;
  let lastProjectionSnapshot = [];
  let unlimitedProjection = false;

  unlimitedToggle.addEventListener("click", () => {
    unlimitedProjection = !unlimitedProjection;
    unlimitedToggle.textContent = `Unlimited forecast: ${unlimitedProjection ? "on" : "off"}`;
    runtime.setProjectionHorizon(unlimitedProjection ? null : 24);
  });

  svg.on("dblclick", (event) => {
    if (!brushDomain) return;

    event.preventDefault();
    event.stopPropagation();
    brushDomain = null;
    render(runtime.getState());
  });

  resetButton.addEventListener("click", () => {
    brushDomain = null;
    render(runtime.getState());
  });

  pointLegend.addEventListener("click", (event) => {
    event.stopPropagation();
  });

  chartWrap.addEventListener("click", () => {
    if (!runtime.getState().selectedPoint) return;
    runtime.setSelectedPoint(null);
  });

  if (typeof ResizeObserver !== "undefined") {
    const resizeObserver = new ResizeObserver(() => {
      if (runtime.getState().selectedPoint) {
        render(runtime.getState());
      }
    });
    resizeObserver.observe(chartWrap);
  }

  function render(state) {
    const activePoint = resolveSelectedPoint(state);

    if (!state.bundleReady) {
      chartWrap.hidden = true;
      status.hidden = true;
      status.textContent = "";
      pointLegend.hidden = true;
      pointLegend.replaceChildren();
      lastProjectionSnapshot = [];
      return;
    }

    const historicalSeries = getHistoricalSeries(state);
    const projectionSeries = getProjectionSeries(state);
    const projectionFuture = projectionSeries.filter((point) => !point.isProjectionAnchor);
    const lastHistoricalPoint = historicalSeries.at(-1) ?? null;
    const allDomainPoints = [...historicalSeries, ...projectionFuture];

    if (!allDomainPoints.length) {
      chartWrap.hidden = true;
      status.hidden = true;
      status.textContent = "";
      pointLegend.hidden = true;
      pointLegend.replaceChildren();
      lastProjectionSnapshot = [];
      return;
    }

    const fullDomain = d3.extent(allDomainPoints, (point) => point.date);
    const hasValidDomain = fullDomain[0] instanceof Date && fullDomain[1] instanceof Date;

    if (!hasValidDomain) {
      chartWrap.hidden = true;
      status.hidden = true;
      status.textContent = "";
      pointLegend.hidden = true;
      pointLegend.replaceChildren();
      lastProjectionSnapshot = [];
      return;
    }

    brushDomain = clampBrushDomain(brushDomain, fullDomain);

    chartWrap.hidden = false;
    status.hidden = false;
    resetButton.disabled = !brushDomain;

    const xDomain = brushDomain ?? fullDomain;
    const fullXScale = d3.scaleUtc()
      .domain(fullDomain)
      .range([MARGINS.left, CHART_WIDTH - MARGINS.right]);
    const xScale = d3.scaleUtc()
      .domain(xDomain)
      .range([MARGINS.left, CHART_WIDTH - MARGINS.right]);
    const plotBottom = CHART_HEIGHT - MARGINS.bottom;
    const plotHeight = CHART_HEIGHT - MARGINS.top - MARGINS.bottom;
    const yScale = d3.scaleLinear()
      .domain(Y_DOMAIN)
      .range([plotBottom, MARGINS.top]);
    const lineGenerator = createLineGenerator(xScale, yScale);

    const projectionStartDate = projectionFuture[0]?.date ?? null;
    const projectionSnapshot = getProjectionSnapshot(projectionSeries);
    const appendedProjectionSegment = getAppendedProjectionSegment(
      lastProjectionSnapshot,
      projectionSeries,
    );

    svg.selectAll("*").remove();

    const defs = svg.append("defs");
    const clipId = `time-series-clip-${Math.random().toString(36).slice(2, 8)}`;
    defs.append("clipPath")
      .attr("id", clipId)
      .append("rect")
      .attr("x", MARGINS.left)
      .attr("y", MARGINS.top)
      .attr("width", CHART_WIDTH - MARGINS.left - MARGINS.right)
      .attr("height", plotHeight);

    svg.append("rect")
      .attr("x", MARGINS.left)
      .attr("y", MARGINS.top)
      .attr("width", CHART_WIDTH - MARGINS.left - MARGINS.right)
      .attr("height", plotHeight)
      .attr("rx", 16)
      .attr("fill", "rgba(255, 255, 255, 0.48)")
      .attr("stroke", "rgba(79, 61, 42, 0.08)");

    if (projectionStartDate && projectionStartDate <= xDomain[1]) {
      const projectionZoneStart = Math.max(MARGINS.left, xScale(projectionStartDate));
      const projectionZoneWidth = Math.max(0, CHART_WIDTH - MARGINS.right - projectionZoneStart);

      svg.append("rect")
        .attr("x", projectionZoneStart)
        .attr("y", MARGINS.top)
        .attr("width", projectionZoneWidth)
        .attr("height", plotHeight)
        .attr("fill", COLORS.projectionFill);
    }

    const plot = svg.append("g").attr("clip-path", `url(#${clipId})`);

    plot.selectAll(".time-series-grid")
      .data(yScale.ticks(5))
      .join("line")
      .attr("class", "time-series-grid")
      .attr("x1", MARGINS.left)
      .attr("x2", CHART_WIDTH - MARGINS.right)
      .attr("y1", (tick) => yScale(tick))
      .attr("y2", (tick) => yScale(tick))
      .attr("stroke", COLORS.grid)
      .attr("stroke-width", 1.1);

    svg.append("g")
      .attr("class", "time-series-axis")
      .attr("transform", `translate(0, ${plotBottom})`)
      .call(createXAxis(xScale, xDomain))
      .call((group) => {
        group.selectAll("text")
          .attr("fill", COLORS.axis)
          .attr("font-size", TYPOGRAPHY.axisSize)
          .attr("font-weight", TYPOGRAPHY.axisWeight);
        group.selectAll("line")
          .attr("stroke", COLORS.grid)
          .attr("stroke-width", 1.1);
        group.select("path")
          .attr("stroke", COLORS.grid)
          .attr("stroke-width", 1.1);
      });

    svg.append("g")
      .attr("class", "time-series-axis")
      .attr("transform", `translate(${MARGINS.left}, 0)`)
      .call(d3.axisLeft(yScale).ticks(5).tickSizeOuter(0))
      .call((group) => {
        group.selectAll("text")
          .attr("fill", COLORS.axis)
          .attr("font-size", TYPOGRAPHY.axisSize)
          .attr("font-weight", TYPOGRAPHY.axisWeight);
        group.selectAll("line")
          .attr("stroke", COLORS.grid)
          .attr("stroke-width", 1.1);
        group.select("path")
          .attr("stroke", COLORS.grid)
          .attr("stroke-width", 1.1);
      });

    const historicalPath = plot.append("path")
      .datum(historicalSeries)
      .attr("fill", "none")
      .attr("stroke", COLORS.historical)
      .attr("stroke-width", 2.8)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("d", lineGenerator);

    const historicalDots = plot.selectAll(".time-series-historical-point")
      .data(historicalSeries)
      .join("circle")
      .attr("class", "time-series-historical-point")
      .attr("cx", (point) => xScale(point.date))
      .attr("cy", (point) => yScale(point.score))
      .attr("r", 4.5)
      .attr("fill", COLORS.historicalPoint);

    const staticProjectionSeries = appendedProjectionSegment
      ? projectionSeries.slice(0, -1)
      : projectionSeries;

    const projectionPath = plot.append("path")
      .datum(staticProjectionSeries)
      .attr("fill", "none")
      .attr("stroke", COLORS.projection)
      .attr("stroke-width", 3.2)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("d", lineGenerator);

    let animatedProjectionSegmentPath = null;
    if (appendedProjectionSegment) {
      animatedProjectionSegmentPath = plot.append("path")
        .datum([appendedProjectionSegment.previousPoint, appendedProjectionSegment.newPoint])
        .attr("fill", "none")
        .attr("stroke", COLORS.projection)
        .attr("stroke-width", 3.2)
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("d", lineGenerator);
    }

    const projectionDots = plot.selectAll(".time-series-projection-point")
      .data(projectionFuture)
      .join("circle")
      .attr("class", "time-series-projection-point")
      .attr("cx", (point) => xScale(point.date))
      .attr("cy", (point) => yScale(point.score))
      .attr("fill", COLORS.projectionPoint);

    if (appendedProjectionSegment && animatedProjectionSegmentPath) {
      const pathNode = animatedProjectionSegmentPath.node();

      if (pathNode) {
        const totalLength = pathNode.getTotalLength();

        animatedProjectionSegmentPath
          .attr("stroke-dasharray", `${totalLength} ${totalLength}`)
          .attr("stroke-dashoffset", totalLength)
          .transition()
          .duration(900)
          .ease(d3.easeLinear)
          .attr("stroke-dashoffset", 0);
      }

      projectionDots
        .attr("r", (point) => pointMatches(point, appendedProjectionSegment.newPoint) ? 0 : 6)
        .attr("opacity", (point) => pointMatches(point, appendedProjectionSegment.newPoint) ? 0.2 : 0.95);

      projectionDots
        .filter((point) => pointMatches(point, appendedProjectionSegment.newPoint))
        .transition()
        .duration(220)
        .attr("r", pointMatches(appendedProjectionSegment.newPoint, activePoint) ? 10.5 : 6)
        .attr("opacity", 0.95);
    } else {
      projectionPath
        .attr("stroke-dasharray", null)
        .attr("stroke-dashoffset", null);

      projectionDots
        .attr("r", 6)
        .attr("opacity", 0.95);
    }

    const interactionPoints = [...historicalSeries, ...projectionFuture];

    if (activePoint) {
      const isProjection = activePoint.source === "projection";
      const dots = isProjection ? projectionDots : historicalDots;
      dots.filter((d) => pointMatches(d, activePoint)).attr("r", isProjection ? 10.5 : 9.5);
    }

    plot.selectAll(".time-series-hit")
      .data(interactionPoints)
      .join("circle")
      .attr("class", "time-series-hit")
      .attr("cx", (point) => xScale(point.date))
      .attr("cy", (point) => yScale(point.score))
      .attr("r", 15)
      .attr("fill", "transparent")
      .style("cursor", "pointer")
      .on("mouseenter", (_, point) => {
        const isProjection = point.source === "projection";
        const dots = isProjection ? projectionDots : historicalDots;
        dots.filter((d) => pointMatches(d, point)).attr("r", isProjection ? 10.5 : 9.5);
        status.hidden = false;
        status.textContent = describePoint(point);
      })
      .on("mouseleave", (_, point) => {
        if (!pointMatches(activePoint, point)) {
          const isProjection = point.source === "projection";
          const dots = isProjection ? projectionDots : historicalDots;
          dots.filter((d) => pointMatches(d, point)).attr("r", isProjection ? 6 : 4.5);
        }
        if (activePoint) {
          status.textContent = describePoint(activePoint);
        } else {
          status.hidden = false;
          status.textContent = getDefaultStatusMessage();
        }
      })
      .on("click", (event, point) => {
        event.stopPropagation();
        runtime.setSelectedPoint(pointMatches(activePoint, point) ? null : point);
      });

    if (activePoint && isDrawablePoint(activePoint)) {
      plot.append("circle")
        .attr("cx", xScale(activePoint.date))
        .attr("cy", yScale(activePoint.score))
        .attr("r", 14)
        .attr("fill", "none")
        .attr("stroke", COLORS.activeRing)
        .attr("stroke-width", 2.2);
    }

    const brushBottom = CHART_HEIGHT - OVERVIEW_RAIL.bottomInset;
    const brushTop = brushBottom - OVERVIEW_RAIL.height;
    const brushYScale = d3.scaleLinear()
      .domain(Y_DOMAIN)
      .range([brushBottom - 3, brushTop + 3]);
    const overviewLineGenerator = createLineGenerator(fullXScale, brushYScale);

    svg.append("rect")
      .attr("x", MARGINS.left)
      .attr("y", brushTop)
      .attr("width", CHART_WIDTH - MARGINS.left - MARGINS.right)
      .attr("height", brushBottom - brushTop)
      .attr("rx", 12)
      .attr("fill", "rgba(13, 108, 116, 0.06)")
      .attr("stroke", "rgba(13, 108, 116, 0.14)");

    svg.append("path")
      .datum(historicalSeries)
      .attr("fill", "none")
      .attr("stroke", "rgba(112, 135, 154, 0.55)")
      .attr("stroke-width", 1.5)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("d", overviewLineGenerator);

    svg.append("path")
      .datum(projectionSeries)
      .attr("fill", "none")
      .attr("stroke", "rgba(214, 140, 45, 0.8)")
      .attr("stroke-width", 1.6)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("d", overviewLineGenerator);

    svg.append("text")
      .attr("x", CHART_WIDTH - MARGINS.right)
      .attr("y", brushTop - OVERVIEW_RAIL.gap)
      .attr("text-anchor", "end")
      .attr("fill", COLORS.axis)
      .attr("font-size", TYPOGRAPHY.overviewLabelSize)
      .attr("font-weight", TYPOGRAPHY.overviewLabelWeight)

    const brush = d3.brushX()
      .extent([
        [MARGINS.left, brushTop],
        [CHART_WIDTH - MARGINS.right, brushBottom],
      ])
      .on("end", (event) => {
        if (!event.sourceEvent) return;
        if (!event.selection) return;

        const [x0, x1] = event.selection;
        if (Math.abs(x1 - x0) < 16) return;

        brushDomain = [fullXScale.invert(x0), fullXScale.invert(x1)];
        render(state);
      });

    const brushGroup = svg.append("g")
      .attr("class", "time-series-brush")
      .call(brush);

    if (brushDomain) {
      brushGroup.call(brush.move, brushDomain.map((date) => fullXScale(date)));
    }

    if (activePoint) {
      status.textContent = describePoint(activePoint);
    } else {
      status.textContent = getDefaultStatusMessage();
    }
    if (activePoint && isDrawablePoint(activePoint)) {
      renderPointLegend({
        legend: pointLegend,
        chartWrap,
        state,
        point: activePoint,
        pointX: xScale(activePoint.date),
        pointY: yScale(activePoint.score),
      });
    } else {
      pointLegend.hidden = true;
      pointLegend.replaceChildren();
    }
    lastProjectionSnapshot = projectionSnapshot;
  }

  runtime.subscribe(render);
  render(runtime.getState());
  return panel;
}
