import "./knob.js";

import {
  formatControlValue,
  formatNumber,
} from "../runtime-utils.js";

function isTemperatureControl(definition) {
  return definition?.id === "temperature_2m_c";
}

function getDisplayLabel(definition) {
  if (definition?.id === "grace_groundwater_anomaly") {
    return "Groundwater Level";
  }

  return definition?.label ?? "Control";
}

function toTemperatureFahrenheit(value) {
  if (!Number.isFinite(value)) return value;
  return (value * 9) / 5 + 32;
}

function formatValueWithUnit(value, definition) {
  if (isTemperatureControl(definition)) {
    return `${formatNumber(toTemperatureFahrenheit(value), definition?.decimals ?? 1)} F`;
  }

  const unitSuffix = definition?.unit ? ` ${definition.unit}` : "";
  return `${formatControlValue(value, definition)}${unitSuffix}`;
}

function renderLoadingState(stack, message) {
  stack.innerHTML = `
    <section class="control-card control-card--status">
      <h3>Data Contract</h3>
      <p>${message}</p>
    </section>
  `;
}

function buildKnobCard(definition, currentValue, onInput, onReset) {
  const { knob } = definition;
  const card = document.createElement("section");
  const displayLabel = getDisplayLabel(definition);
  const step = Number.isFinite(knob.step) && knob.step > 0
    ? knob.step
    : Math.max((knob.max - knob.min) / 200, 0.1);

  card.className = "control-card";
  card.dataset.controlId = definition.id;
  card.innerHTML = `
    <div class="control-card__topline">
      <h3>${displayLabel}</h3>
    </div>
    <div class="control-card__shell">
      <div class="control-range-limit control-range-limit--min">
        <span class="control-range-value">${formatValueWithUnit(knob.min, definition)}</span>
      </div>
      <div class="control-card__dial">
        <div class="js-knob-mount"></div>
        <span class="control-value js-live-value" data-id="${definition.id}">
          ${formatValueWithUnit(currentValue, definition)}
        </span>
        <button
          class="control-reset js-reset"
          type="button"
          data-id="${definition.id}"
          aria-label="Reset ${displayLabel}"
        >
          Reset
        </button>
      </div>
      <div class="control-range-limit control-range-limit--max">
        <span class="control-range-value">${formatValueWithUnit(knob.max, definition)}</span>
      </div>
    </div>
  `;

  const knobElement = document.createElement("my-knob");
  knobElement.className = "control-knob js-control-knob";
  knobElement.dataset.id = definition.id;
  knobElement.setAttribute("label", displayLabel);
  knobElement.setAttribute("min", String(knob.min));
  knobElement.setAttribute("max", String(knob.max));
  knobElement.setAttribute("step", String(step));
  knobElement.setAttribute("value", String(currentValue));
  card.querySelector(".js-knob-mount").appendChild(knobElement);

  knobElement.addEventListener("input", (event) => {
    const nextValue = Number(event.currentTarget?.value);
    if (!Number.isFinite(nextValue)) return;
    onInput(definition.id, nextValue, definition);
  });

  card.querySelector(".js-reset").addEventListener("click", () => {
    onReset(definition.id, knob.default, definition);
  });

  return card;
}

export function createControlPanel(runtime) {
  const panel = document.createElement("aside");
  panel.className = "panel control-panel";

  const stack = document.createElement("div");
  stack.className = "control-stack";

  panel.append(stack);

  let cardsBuilt = false;
  let renderedControlSignature = "";

  function syncControlCardValue(id, rawValue, definition) {
    if (!Number.isFinite(rawValue)) return;

    const liveSpan = stack.querySelector(`.js-live-value[data-id="${id}"]`);
    if (liveSpan) {
      liveSpan.textContent = formatValueWithUnit(rawValue, definition);
    }

    const knobElement = stack.querySelector(`.js-control-knob[data-id="${id}"]`);
    if (knobElement) {
      knobElement.value = rawValue;
    }
  }

  function handleKnobChange(id, rawValue, definition) {
    syncControlCardValue(id, rawValue, definition);
    runtime.setControlValue(id, rawValue);
  }

  function buildCards(state) {
    stack.innerHTML = "";

    state.controlDefinitions.forEach((definition) => {
      const currentValue = state.controlValues[definition.id] ?? definition.knob.default;
      const card = buildKnobCard(definition, currentValue, handleKnobChange, handleKnobChange);
      stack.appendChild(card);
    });

    cardsBuilt = true;
    renderedControlSignature = state.controlDefinitions.map((definition) => definition.id).join("|");
  }

  function render(state) {
    if (state.dataStatus === "loading") {
      renderLoadingState(stack, "Loading bundle...");
      cardsBuilt = false;
      renderedControlSignature = "";
      return;
    }

    if (state.dataStatus === "error") {
      renderLoadingState(stack, `Bundle failed to load: ${state.bundleError ?? "unknown error"}`);
      cardsBuilt = false;
      renderedControlSignature = "";
      return;
    }

    if (!state.bundleReady || state.controlDefinitions.length === 0) {
      renderLoadingState(stack, "Waiting for control metadata...");
      cardsBuilt = false;
      renderedControlSignature = "";
      return;
    }

    const controlSignature = state.controlDefinitions.map((definition) => definition.id).join("|");
    if (!cardsBuilt || renderedControlSignature !== controlSignature) {
      buildCards(state);
    }

    state.controlDefinitions.forEach((definition) => {
      const nextValue = state.controlValues[definition.id] ?? definition.knob.default;
      syncControlCardValue(definition.id, nextValue, definition);
    });
  }

  runtime.subscribe(render);
  render(runtime.getState());
  return panel;
}
