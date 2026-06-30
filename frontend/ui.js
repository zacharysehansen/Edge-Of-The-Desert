import {
  MONTHS, SCENARIO_DURATION_OPTIONS, SLIDER_DEFS, SLIDER_STATS, OUTPUT_DEFS, OUTPUT_STATS, TOP_INPUTS,
  state,
} from './state.js';
import { loadModels, runAll } from './models.js';


let debounceTimer = null;


function scheduleInference() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(async () => {
    try {
      const results = await runAll(state.sliders, state.month, state.scenarioDurationMonths);
      if (results) updateAllOutputs();
    } catch (err) {
      console.error('[EotD] inference error:', err);
    }
  }, 80);
}


function buildPanelHeading(panelId, title) {
  const panel = document.getElementById(panelId);
  const heading = document.createElement('h2');
  heading.textContent = title;
  if (title === 'The Human Factor'){
    heading.classList.add('human-title'); 
  }
  else{
    heading.classList.add('climate-title'); 
  }

  panel.appendChild(heading);
}



function buildSliderPanel(panelId, keys) {
  const panel = document.getElementById(panelId);

  for (const key of keys) {
    const def   = SLIDER_DEFS[key];
    const stats = SLIDER_STATS[key];
    const wrap  = document.createElement('div');
    wrap.className = 'slider-group';

    const labelRow = document.createElement('div');
    labelRow.className = 'slider-label-row';

    const label = document.createElement('label');
    label.htmlFor = `slider-${key}`;
    label.textContent = def.label;

    const valueDisplay = document.createElement('span');
    valueDisplay.className = 'slider-value';
    valueDisplay.id = `val-${key}`;
    valueDisplay.textContent = formatSliderValue(key, stats.default);

    labelRow.appendChild(label);
    labelRow.appendChild(valueDisplay);

    const input = document.createElement('input');
    input.type = 'range';
    input.id = `slider-${key}`;
    input.min  = stats.min;
    input.max  = stats.max;
    input.step = computeStep(stats.min, stats.max);
    input.value = stats.default;

    input.addEventListener('input', () => {
      state.sliders[key] = parseFloat(input.value);
      document.getElementById(`val-${key}`).textContent = formatSliderValue(key, state.sliders[key]);
      scheduleInference();
    });

    const unitLabel = document.createElement('span');
    unitLabel.className = 'slider-unit';
    unitLabel.textContent = def.unit;

    wrap.appendChild(labelRow);
    wrap.appendChild(input);
    wrap.appendChild(unitLabel);
    panel.appendChild(wrap);
  }
}


function buildMonthDropdown() {
  const panel = document.getElementById('panel-climate');

  const wrap = document.createElement('div');
  wrap.className = 'slider-group';

  const labelRow = document.createElement('div');
  labelRow.className = 'slider-label-row';

  const label = document.createElement('label');
  label.htmlFor = 'month-select';
  label.textContent = 'Month';

  labelRow.appendChild(label);
  wrap.appendChild(labelRow);

  const select = document.createElement('select');
  select.id = 'month-select';

  MONTHS.forEach((name, i) => {
    const opt = document.createElement('option');
    opt.value = i + 1;
    opt.textContent = name;
    if (i + 1 === 7) opt.selected = true;
    select.appendChild(opt);
  });

  select.addEventListener('change', () => {
    state.month = parseInt(select.value, 10);
    scheduleInference();
  });

  wrap.appendChild(select);
  panel.appendChild(wrap);
}


function buildDurationDropdown() {
  const panel = document.getElementById('panel-climate');

  const wrap = document.createElement('div');
  wrap.className = 'slider-group';

  const labelRow = document.createElement('div');
  labelRow.className = 'slider-label-row';

  const label = document.createElement('label');
  label.htmlFor = 'duration-select';
  label.textContent = 'Scenario duration';

  labelRow.appendChild(label);
  wrap.appendChild(labelRow);

  const select = document.createElement('select');
  select.id = 'duration-select';

  SCENARIO_DURATION_OPTIONS.forEach(({ value, label: text }) => {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = text;
    if (value === state.scenarioDurationMonths) opt.selected = true;
    select.appendChild(opt);
  });

  select.addEventListener('change', () => {
    state.scenarioDurationMonths = parseInt(select.value, 10);
    scheduleInference();
  });

  wrap.appendChild(select);
  panel.appendChild(wrap);
}


function buildOutputPanel() {
  const panel = document.getElementById('panel-outputs');

  const heading = document.createElement('h2');
  heading.textContent = 'The Water Factor';
  heading.classList.add('water-title');
  panel.appendChild(heading);

  for (const key of Object.keys(OUTPUT_DEFS)) {
    const def  = OUTPUT_DEFS[key];
    const block = document.createElement('div');
    block.className = 'output-block';
    block.id = `output-${key}`;

    const header = document.createElement('div');
    header.className = 'output-header';

    const name = document.createElement('span');
    name.className = 'output-name';
    name.textContent = def.label;

    const unit = document.createElement('span');
    unit.className = 'output-unit';
    unit.textContent = def.unit;

    header.appendChild(name);
    header.appendChild(unit);

    const barWrap = document.createElement('div');
    barWrap.className = 'bar-wrap';

    const barFill = document.createElement('div');
    barFill.className = 'bar-fill';
    barFill.id = `bar-${key}`;

    const barMarker = document.createElement('div');
    barMarker.className = 'bar-baseline-marker';
    barMarker.id = `marker-${key}`;
    // Position is set later, in updateBaselineMarkers(), once calibrateBaselines()
    // has actually run the model with default sliders. Hidden until then so it
    // never shows a position based on the old precomputed stat.
    barMarker.style.visibility = 'hidden';

    barWrap.appendChild(barFill);
    barWrap.appendChild(barMarker);

    const meta = document.createElement('div');
    meta.className = 'output-meta';

    const scoreEl = document.createElement('span');
    scoreEl.className = 'output-score';
    scoreEl.id = `score-${key}`;
    scoreEl.textContent = 'score: —';

    const deltaEl = document.createElement('span');
    deltaEl.className = 'output-delta';
    deltaEl.id = `delta-${key}`;
    deltaEl.textContent = 'delta: —';

    const rawEl = document.createElement('span');
    rawEl.className = 'output-raw';
    rawEl.id = `raw-${key}`;
    rawEl.textContent = '';

    meta.appendChild(scoreEl);
    meta.appendChild(deltaEl);
    meta.appendChild(rawEl);

    const responds = document.createElement('p');
    responds.className = 'output-responds';
    responds.textContent = `responds mainly to: ${TOP_INPUTS[key]}`;

    const loading = document.createElement('div');
    loading.className = 'output-loading';
    loading.id = `loading-${key}`;
    loading.textContent = 'loading model...';

    block.appendChild(header);
    block.appendChild(barWrap);
    block.appendChild(meta);
    block.appendChild(responds);
    block.appendChild(loading);
    panel.appendChild(block);
  }

  const caveats = document.createElement('ul');
  caveats.className = 'caveats';

  panel.appendChild(caveats);
}


function buildResetButton() {
  const btn = document.getElementById('reset-btn');
  btn.textContent = 'Reset to baseline';
  btn.addEventListener('click', () => {
    for (const key of Object.keys(SLIDER_STATS)) {
      const stats = SLIDER_STATS[key];
      state.sliders[key] = stats.default;
      const input = document.getElementById(`slider-${key}`);
      if (input) input.value = stats.default;
      const valEl = document.getElementById(`val-${key}`);
      if (valEl) valEl.textContent = formatSliderValue(key, stats.default);
    }
    state.scenarioDurationMonths = 12;
    const durationSelect = document.getElementById('duration-select');
    if (durationSelect) durationSelect.value = state.scenarioDurationMonths;
    scheduleInference();
  });
}


function updateBaselineMarkers() {
  for (const key of Object.keys(OUTPUT_DEFS)) {
    const marker = document.getElementById(`marker-${key}`);
    if (!marker) continue;
    const { min, max, baseline } = OUTPUT_STATS[key];
    const baselinePct = Math.max(0, Math.min(100, ((baseline - min) / (max - min)) * 100));
    marker.style.left = `${baselinePct.toFixed(1)}%`;
    marker.style.visibility = 'visible';
  }
}


function updateAllOutputs() {
  for (const [key, out] of Object.entries(state.outputs)) {
    const def = OUTPUT_DEFS[key];
    const loadingEl = document.getElementById(`loading-${key}`);
    const barEl     = document.getElementById(`bar-${key}`);
    const scoreEl   = document.getElementById(`score-${key}`);
    const deltaEl   = document.getElementById(`delta-${key}`);
    const rawEl     = document.getElementById(`raw-${key}`);

    if (out.error) {
      if (loadingEl) loadingEl.textContent = 'model unavailable';
      continue;
    }

    if (out.loading || out.score === null) continue;

    if (loadingEl) loadingEl.style.display = 'none';
    if (barEl) barEl.style.width = `${out.score.toFixed(1)}%`;
    if (scoreEl) scoreEl.textContent = `score: ${Math.round(out.score)} / 100`;
    if (rawEl && out.rawValue !== null) {
      rawEl.textContent = `(${formatRaw(key, out.rawValue)} ${def.unit})`;
    }

    if (deltaEl && out.delta !== null) {
      const sign = out.delta > 0 ? '+' : '';
      deltaEl.textContent = `delta: ${sign}${out.delta} from baseline`;

      const isGood = out.delta === 0
        ? 'neutral'
        : def.higherIsBetter
          ? (out.delta > 0 ? 'positive' : 'negative')
          : (out.delta < 0 ? 'positive' : 'negative');

      deltaEl.className = `output-delta delta-${isGood}`;
    }
  }
}


function onModelReady(key, success) {
  const loadingEl = document.getElementById(`loading-${key}`);
  if (!success) {
    if (loadingEl) loadingEl.textContent = 'model unavailable';
    state.outputs[key] = { score: null, rawValue: null, delta: null, loading: false, error: true };
  } else {
    if (loadingEl) loadingEl.textContent = 'ready';
    state.outputs[key].loading = false;
  }

  const allDone = Object.keys(OUTPUT_DEFS).every(
    k => !state.outputs[k].loading
  );
  if (allDone) scheduleInference();
}


function formatSliderValue(key, value) {
  if (key === 'population') return Math.round(value).toLocaleString();
  if (key === 'irrigation_total_withdrawal_mgd') return Math.round(value).toLocaleString();
  if (key === 'public_supply_groundwater_mgd') return Math.round(value).toLocaleString();
  if (key === 'impervious_pct') return value.toFixed(2);
  if (key === 'precipitation_mm_day') return value.toFixed(2);
  if (key === 'temperature_2m_c') return value.toFixed(1);
  return Math.round(value).toString();
}


function formatRaw(key, value) {
  if (key === 'ndvi') return value.toFixed(3);
  if (key === 'grace') return value.toFixed(1);
  if (key === 'surface_water') return Math.round(value);
  if (key === 'groundwater') return Math.round(value);
  if (key === 'wildfire') return value.toFixed(3);
  if (key === 'wildlife') return value.toFixed(3);
  return value.toFixed(2);
}


function computeStep(min, max) {
  const range = max - min;
  if (range > 500000) return 10000;
  if (range > 1000)   return 10;
  if (range > 100)    return 1;
  if (range > 10)     return 0.1;
  return 0.01;
}


async function init() {
  const humanKeys    = Object.keys(SLIDER_DEFS).filter(k => SLIDER_DEFS[k].panel === 'human');
  const climateKeys  = Object.keys(SLIDER_DEFS).filter(k => SLIDER_DEFS[k].panel === 'climate');

  // Human panel: heading then sliders
  buildPanelHeading('panel-human', 'The Human Factor');
  buildSliderPanel('panel-human', humanKeys);

  // Climate panel: heading first, then dropdowns, then sliders
  buildPanelHeading('panel-climate', 'The Climate Factor');
  buildMonthDropdown();
  buildDurationDropdown();
  buildSliderPanel('panel-climate', climateKeys);

  buildOutputPanel();
  buildResetButton();

  await loadModels(onModelReady);
  updateBaselineMarkers();
}


export { init };