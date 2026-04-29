import * as d3 from "d3";
import sceneUrl from "../../../Aquafer_cross_section.svg?url";
import {
  ATMOSPHERE_SCENE_STYLES,
  createAtmosphereController,
} from "./cross-section-atmosphere.js";
import {
  createIrrigationController,
  IRRIGATION_SCENE_STYLES,
} from "./cross-section-irrigation.js";
import {
  createPipeController,
  PIPE_SCENE_STYLES,
} from "./cross-section-pipe.js";
import {
  createSnowController,
  SNOW_SCENE_STYLES,
} from "./cross-section-snow.js";
import {
  createTemperatureController,
  TEMPERATURE_SCENE_STYLES,
} from "./cross-section-temperature.js";
import {
  createWaterlineController,
  getAquiferColor,
  WATERLINE_SCENE_STYLES,
} from "./cross-section-waterline.js";

const SVG_BOUNDS = {
  width: 1540,
  height: 774,
};
const FALLBACK_BOUNDS = {
  svg: { x: 0, y: 0, width: 1540, height: 774 },
  sky: { x: 0, y: 0, width: 1540, height: 283 },
  soil: { x: 1.4, y: 278.6, width: 1536.8, height: 260.7 },
  secondGroundLayer: { x: 1.4, y: 278.6, width: 1536.8, height: 260.7 },
  aquiferWater: { x: 1.3, y: 585.8, width: 1536.8, height: 133.9 },
  bedrock: { x: 1.5, y: 656.7, width: 1536.6, height: 117 },
  waterInPipe: { x: 1131.2, y: 445.9, width: 14.7, height: 89.4 },
  waterArrows: { x: 338.4, y: 635.3, width: 748.1, height: 34.2 },
};

let sceneMarkupPromise;

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function selectFirst(svg, selectors) {
  for (const selector of selectors) {
    const selection = svg.select(selector);
    if (!selection.empty()) {
      return selection;
    }
  }

  return d3.select(null);
}

function getBounds(selection, fallback) {
  const node = selection?.node?.() ?? selection;
  if (!node) return fallback;

  try {
    const box = node.getBBox();
    if (Number.isFinite(box.width) && Number.isFinite(box.height) && (box.width || box.height)) {
      return {
        x: box.x,
        y: box.y,
        width: box.width,
        height: box.height,
      };
    }
  } catch {
    // Fall back to static bounds when getBBox is unavailable.
  }

  return fallback;
}

function readNodeFill(node, fallback) {
  const sourceFill = node.style.fill || node.getAttribute("fill") || getComputedStyle(node).fill;
  const color = d3.color(sourceFill);
  if (!color || color.opacity === 0) return fallback;
  return color.formatHex();
}

function createScoreMoodController({ svg, sceneContentGroup }) {
  const vegetationGroup = selectFirst(svg, ["#vegetation", "#Green"]);

  if (!vegetationGroup.empty()) {
    vegetationGroup.classed("scene-score-vegetation", true);
  }

  const vegetationPieces = vegetationGroup.empty()
    ? []
    : vegetationGroup.selectAll("path, polygon, rect, circle, ellipse").nodes().map((node) => ({
        selection: d3.select(node),
        originalFill: readNodeFill(node, "#5a7f28"),
      }));

  function update({ safeScore }) {
    const scoreNorm = clamp01(safeScore / 100);
    const dryFactor = clamp01((0.5 - scoreNorm) / 0.5);
    const healthyFactor = clamp01((scoreNorm - 0.5) / 0.5);

    const sceneBrightness = 1 + healthyFactor * 0.08 + dryFactor * 0.24;
    const sceneSaturation = 1 + healthyFactor * 0.18 - dryFactor * 0.32;
    const sceneContrast = 1 + healthyFactor * 0.07 - dryFactor * 0.16;
    const sceneBlur = dryFactor * 0.75;

    sceneContentGroup.style(
      "filter",
      `brightness(${sceneBrightness}) saturate(${sceneSaturation}) contrast(${sceneContrast}) blur(${sceneBlur}px)`,
    );

    vegetationPieces.forEach(({ selection, originalFill }) => {
      const tonedFill = scoreNorm < 0.5
        ? d3.interpolateHcl("#8e7241", originalFill)(scoreNorm / 0.5)
        : d3.interpolateHcl(originalFill, "#4f9826")(healthyFactor * 0.85);

      selection
        .style("fill", tonedFill)
        .attr("fill", tonedFill);
    });

    if (!vegetationGroup.empty()) {
      vegetationGroup.style(
        "filter",
        `saturate(${0.62 + scoreNorm * 0.92}) brightness(${0.78 + scoreNorm * 0.36})`,
      );
    }
  }

  return { update };
}

async function loadSceneMarkup() {
  if (!sceneMarkupPromise) {
    sceneMarkupPromise = fetch(sceneUrl).then(async (response) => {
      if (!response.ok) {
        throw new Error(`Failed to load aquifer SVG (${response.status})`);
      }
      return response.text();
    });
  }

  return sceneMarkupPromise;
}

async function buildScene(container) {
  container.innerHTML = "";

  const style = document.createElement("style");
  style.textContent = `
    .cross-section-svg [data-layer="scene-content"] {
      transition: filter 700ms ease;
    }
    .cross-section-svg .scene-score-vegetation,
    .cross-section-svg .scene-score-vegetation path,
    .cross-section-svg .scene-score-vegetation polygon,
    .cross-section-svg .scene-score-vegetation rect,
    .cross-section-svg .scene-score-vegetation circle,
    .cross-section-svg .scene-score-vegetation ellipse {
      transition: fill 700ms ease, filter 700ms ease;
    }
    .cross-section-svg .scene-chip rect,
    .cross-section-svg .scene-chip text {
      transition: opacity 700ms ease, fill 700ms ease, stroke 700ms ease, transform 700ms ease;
    }
${ATMOSPHERE_SCENE_STYLES}
${IRRIGATION_SCENE_STYLES}
${PIPE_SCENE_STYLES}
${SNOW_SCENE_STYLES}
${TEMPERATURE_SCENE_STYLES}
${WATERLINE_SCENE_STYLES}
  `;
  container.appendChild(style);

  const parser = new DOMParser();
  const markup = await loadSceneMarkup();
  const documentNode = parser.parseFromString(markup, "image/svg+xml");
  const svgNode = documentNode.documentElement;
  svgNode.setAttribute("class", "cross-section-svg");
  svgNode.setAttribute("width", "100%");
  svgNode.setAttribute("height", "auto");
  container.appendChild(svgNode);

  const svg = d3.select(svgNode);
  const defs = svg.select("defs").empty()
    ? svg.insert("defs", ":first-child")
    : svg.select("defs");
  const sceneContentGroup = svg.append("g").attr("data-layer", "scene-content");
  const sceneContentNode = sceneContentGroup.node();

  Array.from(svgNode.children)
    .filter((node) => node !== defs.node() && node !== sceneContentNode)
    .forEach((node) => sceneContentNode.appendChild(node));

  const aquiferWaterGroup = selectFirst(svg, ["#aquifer_water", "#ground_level_4"]);
  const secondGroundLayerGroup = selectFirst(svg, ["#ground_layer_2"]);
  const soilGroup = selectFirst(svg, ["#ground_layer_1"]);
  const bedrockGroup = selectFirst(svg, ["#bedrock"]);
  const waterInPipeGroup = selectFirst(svg, ["#water_in_pipe", "#Water_in_pipe"]);
  const waterArrowsGroup = selectFirst(svg, ["#water_arrows"]);

  const bounds = {
    svg: FALLBACK_BOUNDS.svg,
    sky: getBounds(selectFirst(svg, ["#sky", "#sky_and_clouds"]), FALLBACK_BOUNDS.sky),
    soil: getBounds(soilGroup, FALLBACK_BOUNDS.soil),
    secondGroundLayer: getBounds(secondGroundLayerGroup, FALLBACK_BOUNDS.secondGroundLayer),
    aquiferWater: getBounds(aquiferWaterGroup, FALLBACK_BOUNDS.aquiferWater),
    bedrock: getBounds(bedrockGroup, FALLBACK_BOUNDS.bedrock),
    waterInPipe: getBounds(waterInPipeGroup, FALLBACK_BOUNDS.waterInPipe),
    waterArrows: getBounds(waterArrowsGroup, FALLBACK_BOUNDS.waterArrows),
  };

  const irrigationControl = createIrrigationController({ svg });
  const pipeControl = createPipeController({
    defs,
    bounds,
    waterInPipeGroup,
    waterArrowsGroup,
  });
  const snowCover = createSnowController({ svg });
  const overlayGroup = svg.append("g").attr("data-layer", "dynamic-overlays");

  const atmosphere = createAtmosphereController({
    svg,
    bounds,
    svgWidth: SVG_BOUNDS.width,
  });
  const temperatureControl = createTemperatureController({
    svg,
  });
  const scoreMood = createScoreMoodController({
    svg,
    sceneContentGroup,
  });

  const waterline = createWaterlineController({
    defs,
    overlayGroup,
    aquiferWaterGroup,
    bounds,
    svgWidth: SVG_BOUNDS.width,
  });

  const chipGroup = overlayGroup.append("g").attr("class", "scene-chip");
  const chipBackground = chipGroup
    .append("rect")
    .attr("x", SVG_BOUNDS.width - 196)
    .attr("y", 20)
    .attr("width", 168)
    .attr("height", 52)
    .attr("rx", 12)
    .attr("fill", getAquiferColor(50))
    .attr("opacity", 0.9);

  const chipText = chipGroup
    .append("text")
    .attr("x", SVG_BOUNDS.width - 112)
    .attr("y", 52)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .attr("font-size", "24px")
    .attr("font-family", "Georgia, serif")
    .attr("fill", "#ffffff")
    .text("Score: --");

  function update(inputs) {
    const {
      score,
      grace,
      powell,
      snow,
      precipitation,
      temperature,
      irrigation,
      publicSupply,
      domains,
    } = inputs;

    const { irrigationNorm } = irrigationControl.update({ irrigation, domains });
    pipeControl.update({ irrigationNorm, publicSupply, domains });

    const { safeScore, waterColor } = waterline.update({ score, grace, powell, domains });
    atmosphere.update({ precipitation, domains });
    snowCover.update({ snow, domains });
    temperatureControl.update({ temperature, precipitation, domains });

    chipBackground.attr("fill", waterColor);
    chipText.text(`Score: ${safeScore.toFixed(0)}`);
    scoreMood.update({ safeScore });
  }

  return { update };
}

function getControlDomain(state, id, fallback) {
  const definition = (state.controlDefinitions ?? []).find((control) => control.id === id);
  if (!definition?.knob) return fallback;
  return [definition.knob.min, definition.knob.max];
}

function getCurrentValue(state, id, fallback = null) {
  const value = state.controlValues?.[id];
  if (value != null) return value;
  const definition = (state.controlDefinitions ?? []).find((control) => control.id === id);
  return definition?.knob?.default ?? fallback;
}

function getInputs(state) {
  const grace =
    getCurrentValue(
      state,
      "grace_groundwater_anomaly",
      state.projectionContext?.visualSupportValues?.grace_groundwater_anomaly
      ?? state.projectionContext?.featureRowLastObserved?.grace_groundwater_anomaly
      ?? 0,
    );

  return {
    score: state.currentPrediction ?? state.modelInfo?.smokePrediction ?? null,
    grace,
    powell:
      getCurrentValue(state, "powell_pool_elevation")
      ?? state.projectionContext?.visualSupportValues?.powell_pool_elevation
      ?? 3580,
    snow: getCurrentValue(state, "snow_water_equivalent_in", 0),
    precipitation: getCurrentValue(state, "precipitation_mm_day", 0),
    temperature: getCurrentValue(state, "temperature_2m_c", 0),
    irrigation: getCurrentValue(state, "irrigation_total_withdrawal_mgd", 0),
    publicSupply: getCurrentValue(state, "public_supply_groundwater_mgd", 0),
    domains: {
      grace: getControlDomain(state, "grace_groundwater_anomaly", [
        state.featureCatalog?.grace_groundwater_anomaly?.domain?.p5 ?? -100,
        state.featureCatalog?.grace_groundwater_anomaly?.domain?.p95 ?? 100,
      ]),
      powell: getControlDomain(state, "powell_pool_elevation", [3490, 3700]),
      snow: getControlDomain(state, "snow_water_equivalent_in", [0, 40]),
      precipitation: getControlDomain(state, "precipitation_mm_day", [0, 5]),
      temperature: getControlDomain(state, "temperature_2m_c", [0, 35]),
      irrigation: getControlDomain(state, "irrigation_total_withdrawal_mgd", [0, 1000]),
      publicSupply: getControlDomain(state, "public_supply_groundwater_mgd", [0, 500]),
    },
  };
}

export function createCrossSectionPanel(store) {
  const panel = document.createElement("section");
  panel.className = "panel viz-card";

  const header = document.createElement("div");
  header.className = "panel-header";
  header.innerHTML = `
    <div>
      <h2>Aquifer Cross-Section</h2>
      <p class="panel-meta">Authored SVG scene driven by live weather, storage, and demand signals.</p>
    </div>
  `;

  const sceneContainer = document.createElement("div");
  sceneContainer.className = "cross-section-container";

  const statusLine = document.createElement("p");
  statusLine.className = "cross-section-status";

  panel.append(header, sceneContainer, statusLine);

  let scene = null;
  let scenePromise = null;
  let sceneError = null;

  function updateStatus(state) {
    const scoreText = Number.isFinite(state.currentPrediction)
      ? `Current score ${Math.round(state.currentPrediction)}.`
      : "Current score pending.";
    statusLine.textContent =
        `${scoreText} Powell and GRACE raise and lower the water table, snow shifts from dusty brown to bright white with SNOTEL conditions, precipitation turns the sky dark and stormy, and pipe height reflects combined irrigation and public-supply groundwater withdrawals.`;
  }

  async function ensureScene() {
    if (scene || scenePromise) return scenePromise;
    sceneContainer.innerHTML = `<p class="cross-section-status">Loading illustrated scene.</p>`;
    scenePromise = buildScene(sceneContainer)
      .then((builtScene) => {
        scene = builtScene;
        sceneError = null;
        scenePromise = null;
        render(store.getState());
      })
      .catch((error) => {
        sceneError = error;
        scenePromise = null;
        render(store.getState());
      });

    return scenePromise;
  }

  function render(state) {
    if (!state.bundleReady) {
      sceneContainer.innerHTML = `<p class="cross-section-status">Waiting for visual-support values.</p>`;
      statusLine.textContent = "";
      return;
    }

    if (state.modelStatus === "error") {
      sceneContainer.innerHTML = `<p class="cross-section-status cross-section-status--error">Model error: ${state.modelError ?? "unknown"}</p>`;
      statusLine.textContent = "";
      return;
    }

    if (sceneError) {
      sceneContainer.innerHTML = `<p class="cross-section-status cross-section-status--error">Scene load failed: ${sceneError.message}</p>`;
      statusLine.textContent = "";
      return;
    }

    if (!scene) {
      void ensureScene();
      return;
    }

    scene.update(getInputs(state));
    updateStatus(state);
  }

  store.subscribe(render);
  render(store.getState());

  return panel;
}
