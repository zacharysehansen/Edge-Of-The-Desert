import "./styles/app.css";
import { createControlPanel } from "./components/control-panel.js";
import { createCrossSectionPanel } from "./components/cross-section.js";
import { createTimeSeriesPanel } from "./components/time-series.js";
import { createRuntime } from "./runtime.js";
import { formatDateRange } from "./runtime-utils.js";

const appRoot = document.querySelector("#app");
const runtime = createRuntime();

const shell = document.createElement("div");
shell.className = "app-shell";

const layout = document.createElement("div");
layout.className = "app-layout";

const vizPanel = document.createElement("main");
vizPanel.className = "viz-panel";
const controlPanel = createControlPanel(runtime);

vizPanel.append(
  createCrossSectionPanel(runtime),
  createTimeSeriesPanel(runtime),
);

layout.append(vizPanel, controlPanel);
shell.append(layout);
appRoot.append(shell);

function armAmbientAudioOnFirstGesture() {
  let triggered = false;

  async function handleFirstGesture() {
    if (triggered) return;
    triggered = true;

    try {
      await runtime.enableAmbientAudio();
    } catch (error) {
      console.warn("Ambient audio failed to initialize.", error);
    }
  }

  window.addEventListener("pointerdown", handleFirstGesture, { once: true, passive: true });
  window.addEventListener("keydown", handleFirstGesture, { once: true });
}

armAmbientAudioOnFirstGesture();

function updateHeader(state) {
  if (state.dataStatus === "loading") {
    return;
  }

  if (state.dataStatus === "error") {
    return;
  }

  if (!state.bundleReady || !state.displayMetadata) {
    return;
  }

  if (state.modelStatus === "loading") {
    return;
  }

  if (state.modelStatus === "error") {
    return;
  }

}

runtime.subscribe(updateHeader);
updateHeader(runtime.getState());

void runtime.load();

window.edgeOfTheDesert = {
  runtime,
};
