import * as d3 from "d3";
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

export function createTimeSeriesPanel(runtime) {
  const panel = document.createElement("section");
  panel.className = "panel viz-card time-series-panel";
  panel.style.gridTemplateRows = "minmax(0, 1fr)";

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
  let activePoint = null;
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

  function render(state) {
    if (!state.bundleReady) {
      chartWrap.hidden = true;
      status.hidden = true;
      status.textContent = "";
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
      lastProjectionSnapshot = [];
      return;
    }

    const fullDomain = d3.extent(allDomainPoints, (point) => point.date);
    const hasValidDomain = fullDomain[0] instanceof Date && fullDomain[1] instanceof Date;

    if (!hasValidDomain) {
      chartWrap.hidden = true;
      status.hidden = true;
      status.textContent = "";
      lastProjectionSnapshot = [];
      return;
    }

    brushDomain = clampBrushDomain(brushDomain, fullDomain);

    chartWrap.hidden = false;
    status.hidden = !activePoint;
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

    if (activePoint) {
      const isProjection = activePoint.source === "projection";
      const dots = isProjection ? projectionDots : historicalDots;
      dots.filter((d) => pointMatches(d, activePoint)).attr("r", isProjection ? 10.5 : 9.5);
    }

    const interactionPoints = [...historicalSeries, ...projectionFuture];

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
          status.hidden = true;
          status.textContent = "";
        }
      })
      .on("click", (event, point) => {
        event.stopPropagation();
        activePoint = pointMatches(activePoint, point) ? null : point;
        render(state);
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
      status.hidden = false;
      status.textContent = describePoint(activePoint);
    } else {
      status.hidden = true;
      status.textContent = "";
    }
    lastProjectionSnapshot = projectionSnapshot;

    if (
      activePoint
      && !interactionPoints.some((point) => pointMatches(point, activePoint))
    ) {
      activePoint = lastHistoricalPoint;
      if (activePoint) {
        status.hidden = false;
        status.textContent = describePoint(activePoint);
      } else {
        status.hidden = true;
        status.textContent = "";
      }
    }
  }

  runtime.subscribe(render);
  render(runtime.getState());
  return panel;
}
