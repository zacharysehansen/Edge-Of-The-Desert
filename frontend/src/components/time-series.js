import { formatDateRange, formatNumber } from "../runtime.js";

export function createTimeSeriesPanel(runtime) {
  const panel = document.createElement("section");
  panel.className = "panel viz-card";

  const header = document.createElement("div");
  header.className = "panel-header";
  header.innerHTML = `
    <div>
      <h2>Time Series</h2>
      <p class="panel-meta">Step 3 summary of the historical series and projection seed.</p>
    </div>
  `;

  const body = document.createElement("div");
  body.className = "viz-placeholder viz-placeholder--summary";

  panel.append(header, body);

  function render(state) {
    if (!state.bundleReady) {
      body.textContent = "Waiting for historical_sustainability.csv and historical_feature_vectors.csv.";
      return;
    }

    const historicalStart = state.historicalSeries[0]?.yearMonth;
    const historicalEnd = state.historicalSeries.at(-1)?.yearMonth;
    const modelCompleteCount = state.historicalFeatureVectors.filter(
      (row) => row.isModelComplete,
    ).length;

    body.innerHTML = `
      <div class="summary-grid">
        <div class="summary-grid__item">
          <strong>${formatNumber(state.historicalSeries.length, 0)}</strong>
          <span>historical score rows loaded</span>
        </div>
        <div class="summary-grid__item">
          <strong>${formatNumber(modelCompleteCount, 0)}</strong>
          <span>model-complete feature rows loaded</span>
        </div>
      </div>
      <p class="summary-copy">
        Historical contract window:
        <strong>${formatDateRange(historicalStart, historicalEnd)}</strong>.
        The first projected month is <strong>${state.projectionContext.projectionStartMonth}</strong>.
      </p>
    `;
  }

  runtime.subscribe(render);
  render(runtime.getState());
  return panel;
}
