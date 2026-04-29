import "./styles/app.css";
import { createControlPanel } from "./components/control-panel.js";
import { createCrossSectionPanel } from "./components/cross-section.js";
import { createTimeSeriesPanel } from "./components/time-series.js";
import { createRuntime, formatDateRange } from "./runtime.js";

const appRoot = document.querySelector("#app");
const runtime = createRuntime();

const shell = document.createElement("div");
shell.className = "app-shell";

const header = document.createElement("header");
header.className = "app-header";

const headerText = document.createElement("div");
headerText.innerHTML = `
  <p class="eyebrow">Southwest Water Sustainability</p>
  <h1>Edge Of The Desert</h1>
  <p class="subhead">
    Adjust the scenario controls to explore predicted water sustainability conditions.
  </p>
`;

const statusChip = document.createElement("div");
statusChip.className = "status-chip";
statusChip.textContent = "Loading browser bundle";

header.append(headerText, statusChip);

const layout = document.createElement("div");
layout.className = "app-layout";

const vizPanel = document.createElement("main");
vizPanel.className = "viz-panel";
vizPanel.append(
  createCrossSectionPanel(runtime),
  createTimeSeriesPanel(runtime),
);

layout.append(createControlPanel(runtime), vizPanel);
shell.append(header, layout);
appRoot.append(shell);

function updateHeader(state) {
  if (state.dataStatus === "loading") {
    statusChip.textContent = "Loading browser bundle";
    return;
  }

  if (state.dataStatus === "error") {
    statusChip.textContent = "Bundle load failed";
    return;
  }

  if (!state.bundleReady || !state.displayMetadata) {
    statusChip.textContent = "Waiting for bundle";
    return;
  }

  if (state.modelStatus === "loading") {
    statusChip.textContent = "Initializing ONNX session";
    return;
  }

  if (state.modelStatus === "error") {
    statusChip.textContent = "ONNX init failed";
    return;
  }

  const summaryParts = [
    `${state.displayMetadata.selected_feature_count} features`,
    formatDateRange(
      state.displayMetadata.timeline.model_complete_start,
      state.displayMetadata.timeline.model_complete_end,
    ),
  ];

  if (state.modelReady && state.modelInfo) {
    summaryParts.push(`smoke ${state.modelInfo.smokePrediction.toFixed(2)}`);
  }

  statusChip.textContent = summaryParts.join(" · ");
}

runtime.subscribe(updateHeader);
updateHeader(runtime.getState());

void runtime.load();

window.edgeOfTheDesert = {
  runtime,
};
