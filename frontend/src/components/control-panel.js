import {
  formatControlValue,
  formatNumber,
} from "../runtime.js";

function renderLoadingState(stack, message) {
  stack.innerHTML = `
    <section class="control-card control-card--status">
      <h3>Data Contract</h3>
      <p>${message}</p>
    </section>
  `;
}

function buildSliderCard(definition, currentValue, onInput, onReset) {
  const { knob } = definition;
  const card = document.createElement("section");
  card.className = "control-card";

  const isPopulation = definition.model_mapping.type === "pct_change_from_absolute_series";
  const mappingNote = isPopulation
    ? `Maps to <code>${definition.model_mapping.output_feature}</code> via percent change.`
    : `Controls <code>${definition.model_mapping.output_feature}</code> directly.`;

  card.innerHTML = `
    <div class="control-card__topline">
      <h3>${definition.label}</h3>
      <span class="control-value js-live-value" data-id="${definition.id}">
        ${formatControlValue(currentValue, definition)} ${definition.unit}
      </span>
    </div>
    <input
      class="control-slider"
      type="range"
      data-id="${definition.id}"
      min="${knob.min}"
      max="${knob.max}"
      step="${knob.step ?? (knob.max - knob.min) / 200}"
      value="${currentValue}"
    />
    <div class="control-card__footer">
      <span class="control-note">p5 ${formatControlValue(knob.min, definition)} &ndash; p95 ${formatControlValue(knob.max, definition)} ${definition.unit}</span>
      <button class="control-reset js-reset" data-id="${definition.id}" data-default="${knob.default}">Reset</button>
    </div>
    <p class="control-note">${mappingNote}</p>
  `;

  card.querySelector(".control-slider").addEventListener("input", (e) => {
    onInput(definition.id, parseFloat(e.target.value), definition);
  });

  card.querySelector(".js-reset").addEventListener("click", () => {
    onReset(definition.id, knob.default, definition);
  });

  return card;
}

export function createControlPanel(runtime) {
  const panel = document.createElement("aside");
  panel.className = "panel control-panel";

  const header = document.createElement("div");
  header.className = "panel-header";
  header.innerHTML = `
    <div>
      <h2>Controls</h2>
      <p class="panel-meta"></p>
    </div>
  `;

  const stack = document.createElement("div");
  stack.className = "control-stack";


  panel.append(header, stack);

  let cardsBuilt = false;

  function handleKnobChange(id, rawValue, definition) {
    const liveSpan = stack.querySelector(`.js-live-value[data-id="${id}"]`);
    if (liveSpan) {
      liveSpan.textContent = `${formatControlValue(rawValue, definition)} ${definition.unit}`;
    }
    const slider = stack.querySelector(`.control-slider[data-id="${id}"]`);
    if (slider) slider.value = rawValue;

    runtime.setControlValue(id, rawValue);
  }

  function buildCards(state) {
    stack.innerHTML = "";

    state.controlDefinitions.forEach((definition) => {
      const currentValue = state.controlValues[definition.id] ?? definition.knob.default;
      const card = buildSliderCard(definition, currentValue, handleKnobChange, handleKnobChange);
      stack.appendChild(card);
    });

    cardsBuilt = true;
  }

  function render(state) {
    if (state.dataStatus === "loading") {
      renderLoadingState(stack, "Loading bundle...");
      cardsBuilt = false;
      return;
    }

    if (state.dataStatus === "error") {
      renderLoadingState(stack, `Bundle failed to load: ${state.bundleError ?? "unknown error"}`);
      cardsBuilt = false;
      return;
    }

    if (!state.bundleReady || state.controlDefinitions.length === 0) {
      renderLoadingState(stack, "Waiting for control metadata...");
      cardsBuilt = false;
      return;
    }

    if (!cardsBuilt) {
      buildCards(state);
    }

  }

  runtime.subscribe(render);
  render(runtime.getState());
  return panel;
}
