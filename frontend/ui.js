import {
  MONTHS, SCENARIO_DURATION_OPTIONS, SLIDER_DEFS, SLIDER_POLICY, OUTPUT_DEFS, OUTPUT_STATS,
  TOP_INPUTS, TIER_BASIS, state, sliderRaw,
} from './state.js';
import { LEVER_INFO, CLIMATE_ONLY_OUTPUTS } from './structural.js';
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
    const def    = SLIDER_DEFS[key];
    const policy = SLIDER_POLICY[key];
    const wrap   = document.createElement('div');
    wrap.className = 'slider-group';

    const labelRow = document.createElement('div');
    labelRow.className = 'slider-label-row';

    const label = document.createElement('label');
    label.htmlFor = `slider-${key}`;
    label.textContent = def.label;

    const valueDisplay = document.createElement('span');
    valueDisplay.className = 'slider-value';
    valueDisplay.id = `val-${key}`;

    labelRow.appendChild(label);
    labelRow.appendChild(valueDisplay);

    // The slider carries a POLICY DELTA (PHASE3_PLAN.md D5), not a raw value:
    // "+40% of annual withdrawal", not "4,815 MGD", which only ever meant "June".
    const input = document.createElement('input');
    input.type = 'range';
    input.id = `slider-${key}`;
    input.min  = policy.min;
    input.max  = policy.max;
    input.step = policy.step;
    input.value = state.sliders[key];

    input.addEventListener('input', () => {
      state.sliders[key] = parseFloat(input.value);
      refreshSliderReadout(key);
      scheduleInference();
    });

    const unitLabel = document.createElement('span');
    unitLabel.className = 'slider-unit';
    unitLabel.textContent = policy.unit;

    wrap.appendChild(labelRow);
    wrap.appendChild(input);
    wrap.appendChild(unitLabel);
    panel.appendChild(wrap);
  }
}


// The readout shows the policy delta AND the raw value it reconstructs to for the
// selected month, because the second one is what the models actually see and it
// moves when the month dropdown moves.
function refreshSliderReadout(key) {
  const el = document.getElementById(`val-${key}`);
  if (!el) return;
  const delta = state.sliders[key];
  const raw = sliderRaw(key, delta, state.month);
  el.textContent = delta === 0
    ? `normal (${formatRawSlider(key, raw)})`
    : `${formatPolicyDelta(key, delta)} → ${formatRawSlider(key, raw)}`;
}


function refreshAllSliderReadouts() {
  for (const key of Object.keys(SLIDER_POLICY)) refreshSliderReadout(key);
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
    // Every raw value is delta + that month's climatology, so all of them move.
    refreshAllSliderReadouts();
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

    const info = TOP_INPUTS[key];

    const responds = document.createElement('p');
    responds.className = 'output-responds';
    responds.textContent = `responds mainly to: ${info.summary}`;

    // Layer 4: where this number came from. Filled by updateProvenance() on every
    // inference pass; the static text is only what shows before the first run.
    const provenance = document.createElement('div');
    provenance.className = 'provenance';
    provenance.id = `prov-${key}`;
    provenance.textContent = CLIMATE_ONLY_OUTPUTS.includes(key)
      ? 'climate only — no structural path to this output'
      : '';

    const loading = document.createElement('div');
    loading.className = 'output-loading';
    loading.id = `loading-${key}`;
    loading.textContent = 'loading model...';

    block.appendChild(header);
    block.appendChild(barWrap);
    block.appendChild(meta);
    block.appendChild(responds);
    block.appendChild(provenance);
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
    for (const key of Object.keys(SLIDER_POLICY)) {
      state.sliders[key] = SLIDER_POLICY[key].default;
      const input = document.getElementById(`slider-${key}`);
      if (input) input.value = state.sliders[key];
      refreshSliderReadout(key);
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


function signed(points) {
  const rounded = Math.abs(points) < 0.05 ? 0 : points;
  return `${rounded > 0 ? '+' : rounded < 0 ? '\u2212' : ''}${Math.abs(rounded).toFixed(1)}`;
}


function provenanceRow(className, label, points, title, badge) {
  const row = document.createElement('div');
  row.className = `prov-row ${className}`;
  if (title) row.title = title;

  const name = document.createElement('span');
  name.className = 'prov-label';
  name.textContent = label;
  row.appendChild(name);

  if (badge) row.appendChild(badge);

  const value = document.createElement('span');
  value.className = 'prov-points';
  value.textContent = signed(points);
  row.appendChild(value);

  return row;
}


// Layer 4 (PHASE3_PLAN.md §4). Each card says how much of its movement came from the
// learned climate model and how much from the structural levers, and for the human
// part, which lever and on what basis. This is not a disclaimer — it answers *why*
// the number moved, which the black box never could.
function updateProvenance(key, provenance) {
  const host = document.getElementById(`prov-${key}`);
  if (!host) return;
  host.textContent = '';

  if (!provenance) return;

  host.appendChild(provenanceRow(
    'prov-climate', 'climate (learned model)', provenance.climatePoints,
    TIER_BASIS.climate.detail,
  ));

  if (CLIMATE_ONLY_OUTPUTS.includes(key)) {
    const note = document.createElement('p');
    note.className = 'prov-none';
    note.textContent = 'no structural path to this output';
    note.title = 'Ignition is not the limiting factor for large-fire extent in the Southwest, which is what this target measures. No defensible coefficient from a human lever to it could be sourced, so none is invented.';
    host.appendChild(note);
    return;
  }

  if (provenance.levers.length === 0) {
    const idle = document.createElement('div');
    idle.className = 'prov-row prov-idle';
    idle.textContent = 'human levers at normal';
    host.appendChild(idle);
    return;
  }

  host.appendChild(provenanceRow(
    'prov-human', 'human levers (structural)', provenance.humanPoints,
    'Supplied by Layer 2, not by the learned model. Signs and magnitudes come from water balance, land-cover arithmetic and published policy.',
  ));

  const list = document.createElement('div');
  list.className = 'prov-levers';
  for (const lever of provenance.levers) {
    const info = LEVER_INFO[lever.id] ?? {};
    // `via` disambiguates levers that share a slider and an output: population
    // reaches streamflow twice, through effluent and through pumping, in opposite
    // directions. Without it the card lists "Population" twice with no explanation.
    const label = lever.kind === 'transfer'
      ? `via ${OUTPUT_DEFS[lever.from]?.label ?? lever.from}`
      : info.via
        ? `${SLIDER_DEFS[lever.slider]?.label ?? lever.slider} (${info.via})`
        : SLIDER_DEFS[lever.slider]?.label ?? lever.slider;

    const tier = TIER_BASIS[lever.tier] ?? TIER_BASIS['structural-only'];

    const badge = document.createElement('span');
    badge.className = `prov-badge prov-badge-${lever.tier}`;
    badge.textContent = tier.label;
    badge.title = tier.detail;

    list.appendChild(provenanceRow(
      `prov-lever prov-tier-${lever.tier}`,
      label,
      lever.points,
      [info.mechanism, info.evidence && `Evidence: ${info.evidence}`, tier.detail]
        .filter(Boolean).join('\n\n'),
      badge,
    ));
  }
  host.appendChild(list);
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

    updateProvenance(key, out.provenance);

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


function formatPolicyDelta(key, delta) {
  const policy = SLIDER_POLICY[key];
  const sign = delta > 0 ? '+' : '';
  if (policy.mode === 'scale') return `${sign}${Math.round(delta)}%`;
  if (key === 'population')    return `${sign}${Math.round(delta).toLocaleString()}`;
  if (key === 'mead_pool_elevation') return `${sign}${Math.round(delta)} ft`;
  return `${sign}${delta.toFixed(2)}`;
}


// The reconstructed raw value, in the units the models were trained on.
function formatRawSlider(key, raw) {
  const unit = SLIDER_DEFS[key].unit;
  if (key === 'population')      return `${Math.round(raw).toLocaleString()}`;
  if (key === 'impervious_pct')  return `${raw.toFixed(2)} ${unit}`;
  if (key === 'precipitation_mm_day') return `${raw.toFixed(2)} ${unit}`;
  if (key === 'temperature_2m_c')     return `${raw.toFixed(1)} ${unit}`;
  if (key === 'nclimdiv_pdsi')        return raw.toFixed(2);
  return `${Math.round(raw).toLocaleString()} ${unit}`;
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
  refreshAllSliderReadouts();

  await loadModels(onModelReady);
  updateBaselineMarkers();
}


export { init };