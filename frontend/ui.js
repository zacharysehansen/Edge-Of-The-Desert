import {
  MONTHS, SCENARIO_DURATION_OPTIONS, SLIDER_DEFS, SLIDER_POLICY, OUTPUT_DEFS, OUTPUT_STATS,
  TOP_INPUTS, TIER_BASIS, state, sliderRaw,
} from './state.js';
import { LEVER_INFO, CLIMATE_ONLY_OUTPUTS, PARAM_BANDS } from './structural.js';
import { loadModels, runAll } from './models.js';


// Readable names for the banded constants, for the tooltip that explains a range.
// Only the wording lives here — every number comes from PARAM_BANDS, i.e. from
// structural_params.json itself. A renamed constant falls back to its raw name, which
// is ugly but never wrong.
const PARAM_LABELS = {
  storage_af_per_ft: 'aquifer storage coefficient',
  specific_yield: 'specific yield',
  alluvial_fraction: 'alluvial fraction of the region',
  ndvi_impervious: 'NDVI of paved surface',
  ndvi_irrigated_crop: 'NDVI of irrigated cropland',
  region_share_of_az_reduction: "region's share of the Arizona CAP cut",
  groundwater_substitution_fraction: 'share of lost CAP water replaced by pumping',
  effluent_return_fraction: 'effluent returned to stream',
  runoff_coefficient_impervious: 'runoff coefficient, paved',
  runoff_coefficient_natural: 'runoff coefficient, desert',
  stream_capture_fraction: 'stream capture fraction',
  transfer_surface_water_to_wildlife: 'streamflow \u2192 bird abundance edge',
};


// "storage_af_per_ft" means nothing on a card; "aquifer storage coefficient
// (184,023-289,336 AF/ft, MEASURED)" is the sentence that makes a wide range
// legible. The status matters as much as the numbers: a range from an UNTESTED
// assumption and a range from a t = +1.78 measurement are different claims.
function describeParam(name) {
  const spec = PARAM_BANDS[name];
  const label = PARAM_LABELS[name] ?? name;
  if (!spec?.band) return label;
  const fmt = (v) => (Math.abs(v) >= 1000
    ? Math.round(v).toLocaleString('en-US')
    : Number(v.toPrecision(3)).toString());
  return `${label} (${fmt(spec.band[0])}\u2013${fmt(spec.band[1])}, ${spec.status})`;
}


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

    // Which way is up. Three of the six outputs are named for a quantity that rises
    // when things get WORSE -- "Groundwater Depth vs Normal" goes up as the water
    // table falls, and "Wildfire Risk Index" up as more burns -- so a rising bar reads
    // as good news unless the card says otherwise. The delta colour already encodes
    // it (higherIsBetter), but colour alone is not a label, and it is not available to
    // anyone reading the number rather than the hue.
    if (def.rising) {
      const dir = document.createElement('div');
      dir.className = 'output-direction';
      dir.textContent = `bar rises \u2192 ${def.rising}`;
      header.appendChild(dir);
    }

    const stockEl = document.createElement('div');
    stockEl.className = 'output-stock';
    stockEl.id = `stock-${key}`;
    stockEl.textContent = '—';

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

    // An output drawn on a physical scale wider than its history (groundwater, §35)
    // marks where the history sits, so the month-to-month wobble stays visible.
    const stats = OUTPUT_STATS[key];
    if (stats.natural_min !== undefined) {
      const pct = v => ((v - stats.min) / (stats.max - stats.min)) * 100;
      const band = document.createElement('div');
      band.className = 'bar-natural-range';
      band.style.left = `${pct(stats.natural_min).toFixed(1)}%`;
      band.style.width = `${(pct(stats.natural_max) - pct(stats.natural_min)).toFixed(1)}%`;
      band.title = 'Normal range: 90% of past months fall inside this band';
      barWrap.appendChild(band);
    }

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
    block.appendChild(stockEl);
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


// A range is only worth printing when it is wider than the precision the number is
// shown to. Below that it is visual noise that implies a spurious distinction.
function bandText(band) {
  if (!band) return null;
  const [lo, hi] = band;
  if (hi - lo < 0.1) return null;
  // Dropped when the band is an artefact of rounding rather than of uncertainty.
  if (signed(lo) === signed(hi)) return null;
  return `${signed(lo)} to ${signed(hi)}`;
}


function provenanceRow(className, label, points, title, badge, band) {
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

  // The range sits after the point estimate rather than replacing it: the point
  // estimate is what the rest of the card's arithmetic uses, and a card that showed
  // only a range could not be reconciled with the bar above it.
  const range = bandText(band);
  if (range) {
    const span = document.createElement('span');
    span.className = 'prov-band';
    span.textContent = range;
    row.appendChild(span);
  }

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

  // The learned models predict THIS MONTH's change from last month (§4). Layer 3
  // integrates the human levers over the scenario because a storage balance
  // accumulates and its physics is known; it does not integrate the learned climate
  // term (§13, §32: the models carry no dynamics, and a multiplier cannot fix a sign).
  // So this row is the model's one-step change in the selected month under the
  // scenario's climate, NOT the effect of the whole duration -- and the card says so,
  // rather than letting the duration selector imply it (PHASE3_PLAN.md §31).
  const isResidual = !CLIMATE_ONLY_OUTPUTS.includes(key) && key !== 'wildlife';
  host.appendChild(provenanceRow(
    'prov-climate',
    isResidual ? 'climate (learned model) \u2014 this month\u2019s change' : 'climate (learned model)',
    provenance.climatePoints,
    TIER_BASIS.climate.detail + (isResidual
      ? '\n\nThis is the learned model\u2019s change in the selected month under the scenario\u2019s climate, not the accumulated effect over the scenario duration. The duration selector integrates the human levers below (a storage balance accumulates); the learned climate term is not integrated, because the models carry no dynamics of their own and a fitted multiplier cannot correct a sign (PHASE3_PLAN.md \u00a731\u2013\u00a732).'
      : ''),
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
    'Supplied by Layer 2, not by the learned model. Signs and magnitudes come from water balance, land-cover arithmetic and published policy.'
    + (provenance.humanBand ? '\n\nThe range spans the declared bands on the parameters this output depends on. It is parameter uncertainty in Layer 2 only — the learned climate term above carries its own error, which is not in this range.' : ''),
    null,
    provenance.humanBand,
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

    // Naming the parameters that drive the width is the difference between "this
    // number is uncertain" and "this number is uncertain BECAUSE storage_af_per_ft is
    // only known to a factor of 1.6" — the second is actionable, the first is a shrug.
    const driverNote = lever.drivers?.length
      ? `Range driven by:\n  \u2022 ${lever.drivers.map(describeParam).join('\n  \u2022 ')}`
      : null;

    list.appendChild(provenanceRow(
      `prov-lever prov-tier-${lever.tier}`,
      label,
      lever.points,
      [info.mechanism, info.evidence && `Evidence: ${info.evidence}`, tier.detail, driverNote]
        .filter(Boolean).join('\n\n'),
      badge,
      lever.pointsBand,
    ));

    // Some levers are honest and useless at the same time. Urbanization moves the
    // eight-county mean NDVI by about a point, because the slider's whole range is ~2%
    // of the region's area and a regional mean is built to average that away — while on
    // the land that actually got paved, NDVI falls 16%. Reporting only the regional
    // number reads as "urbanization does nothing", which is the opposite of what was
    // measured. Where a lever declares a local effect, the card states both scales.
    if (info.local_effect) {
      const note = document.createElement('div');
      note.className = 'prov-local';
      note.textContent = info.local_effect.headline;
      note.title = info.local_effect.detail;
      list.appendChild(note);
    }
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
      rawEl.textContent = `anomaly: ${formatRaw(key, out.rawValue)} ${def.unit}`;
    }

    const stockEl = document.getElementById(`stock-${key}`);
    if (stockEl && out.rawValue !== null) {
      const stockText = formatStock(key, out.rawValue);
      stockEl.textContent = stockText ?? '—';
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


function formatStock(key, rawValue) {
  const stock = OUTPUT_STATS[key]?.stock;
  if (!stock) return null;
  if (key === 'surface_water') {
    const logBaseline = stock.stock_log_baseline ?? Math.log(stock.stock_reference);
    const cfs = Math.exp(logBaseline + rawValue);
    return `${Math.round(cfs).toLocaleString()} ${stock.stock_unit}`;
  }
  if (key === 'groundwater') {
    const depth = stock.stock_reference + rawValue;
    return `${depth.toFixed(1)} ${stock.stock_unit}`;
  }
  if (key === 'ndvi') return `${rawValue.toFixed(3)} ${stock.stock_unit}`;
  if (key === 'wildfire') return `${rawValue.toFixed(3)} ${stock.stock_unit}`;
  if (key === 'grace') return `${rawValue.toFixed(4)} ${stock.stock_unit}`;
  if (key === 'wildlife') return `${rawValue.toFixed(3)} ${stock.stock_unit}`;
  return null;
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