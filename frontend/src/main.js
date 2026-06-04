import "./styles/app.css";
import { createControlPanel } from "./components/control-panel.js";
import { createCrossSectionPanel } from "./components/cross-section.js";
import { createTimeSeriesPanel } from "./components/time-series.js";
import { createRuntime } from "./runtime.js";

const appRoot = document.querySelector("#app");
const runtime = createRuntime();

const shell = document.createElement("div");
shell.className = "app-shell";

const header = document.createElement("header");
header.className = "app-header";
header.innerHTML = `
  <div class="app-header__content">
    <p class="eyebrow">Arizona Water Sustainability Explorer</p>
    <h1>Edge Of The Desert</h1>
  </div>
`;

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
shell.append(header, layout);
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

void runtime.load();

window.edgeOfTheDesert = {
  runtime,
};
