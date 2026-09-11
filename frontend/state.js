import computedStats from './computed_stats.json' with { type: 'json' };
import topInputs from './top_inputs.json' with { type: 'json' };

const MONTHS = [
  "January","February","March","April","May","June",
  "July","August","September","October","November","December"
];

const SCENARIO_DURATION_OPTIONS = [
  { value: 1,  label: "1 month" },
  { value: 3,  label: "3 months" },
  { value: 6,  label: "6 months" },
  { value: 12, label: "1 year" },
  { value: 24, label: "2 years" },
  { value: 36, label: "3 years" },
];

const SLIDER_DEFS = {
  population:               { label: "Population",              unit: "people",  panel: "human"   },
  irrigation_total_withdrawal_mgd: { label: "Irrigation Withdrawal",    unit: "MGD",     panel: "human"   },
  public_supply_groundwater_mgd:   { label: "Public Supply Groundwater", unit: "MGD",     panel: "human"   },
  impervious_pct:           { label: "Urbanization",            unit: "%",       panel: "human"   },
  mead_pool_elevation:      { label: "Lake Mead Level",         unit: "ft",      panel: "human"   },
  precipitation_mm_day:     { label: "Precipitation",           unit: "mm/day",  panel: "climate" },
  temperature_2m_c:         { label: "Temperature",             unit: "°C",      panel: "climate" },
  // Drought is exposed as PDSI, not USDM DSCI. PDSI is signed (negative = dry, positive =
  // wet), runs back to 1895, and is the exact input the two strongest models use (surface
  // water and wildlife). The USDM DSCI that NDVI/GRACE/wildfire want is *derived* from it
  // in models.js — those two indices only correlate -0.66, so that derivation is
  // approximate, and it is the price of having one honest drought control instead of two
  // that can contradict each other.
  nclimdiv_pdsi:            { label: "Drought (PDSI)",          unit: "",        panel: "climate" },
};

// Loaded dynamically from computed_stats.json. The raw p5/p95/p50 are no longer a
// control surface — SLIDER_POLICY below is — but they stay in the file because
// scripts/phase3/slider_sensitivity.py reads them for the historical sweep in
// PHASE3_PLAN.md §1.
const SLIDER_STATS = computedStats.SLIDER_STATS;

// ── Policy reparameterization (PHASE3_PLAN.md D5) ────────────────────────────
//
// The sliders no longer carry raw model inputs. A raw p5..p95 range is not a
// policy axis: irrigation is 95.2% a repeating month-of-year template, so its raw
// maximum means *June*, and population is 0.4% within-year, so its raw minimum
// means *2002*. Dragging either asked a question month_sin/month_cos already
// answers, which is part of why the learned models found nothing there.
//
// Each slider now carries a DELTA from a deseasonalized baseline, in units a
// person can defend (people added, % of annual withdrawal, points of impervious
// cover, feet of reservoir elevation). The raw value the models were trained on
// is reconstructed by adding the month-of-year climatology back, so the month
// dropdown supplies seasonality and the slider supplies policy.
//
// usdm_dsci deliberately has no policy block: it is not a control, models.js
// derives it from the PDSI slider.
const SLIDER_POLICY = Object.fromEntries(
  Object.entries(SLIDER_STATS)
    .filter(([, stats]) => stats.policy)
    .map(([key, stats]) => [key, stats.policy]),
);

// Month arithmetic wraps: a lag of 3 from January is October, and it carries
// October's climatology, not January's.
function wrapMonth(monthNum) {
  return ((Math.round(monthNum) - 1) % 12 + 12) % 12 + 1;
}

// raw(month, d) = baseline * (1 + d/100) * seasonal[month]   (mode "scale")
//               = baseline +  d          + seasonal[month]   (mode "offset")
function sliderRaw(key, delta, month) {
  const policy = SLIDER_POLICY[key];
  if (!policy) throw new Error(`No policy block for slider "${key}"`);
  const seasonal = policy.seasonal[wrapMonth(month) - 1];
  const d = Number.isFinite(delta) ? delta : 0;
  return policy.mode === 'scale'
    ? policy.baseline * (1 + d / 100) * seasonal
    : policy.baseline + d + seasonal;
}

// The climatological normal for that month — what the model sees at delta 0.
function sliderNormal(key, month) {
  return sliderRaw(key, 0, month);
}

const ZERO_DELTAS = Object.fromEntries(Object.keys(SLIDER_POLICY).map(k => [k, 0]));

const HUMAN_SLIDERS = Object.keys(SLIDER_DEFS).filter(k => SLIDER_DEFS[k].panel === 'human');

// The learned layer is a CLIMATE model and PHASE3_PLAN.md §4 writes it as one:
//
//     output = ML_climate(climate, season, lag1) + Σ_j β_j · (lever_j − baseline_j)
//
// so the human levers are held at their climatological normal when the ONNX models
// run, and Layer 2 owns the entire human response. Feeding the human deltas into the
// learned models as well did two bad things: it double-counted the human response over
// eight lever/output paths (§11.3 — irrigation → groundwater is the only one it named,
// and it is the fifth largest), and it imported D2's month-dependent wrong signs, which
// the §7 acceptance gate then caught overwhelming the structural term for urbanization
// → NDVI and Mead → GRACE.
//
// HUMAN_SLIDERS is derived from `panel: 'human'` rather than hardcoded, so a new human
// lever is covered automatically — but a human lever placed in the climate panel for
// layout reasons would silently start double-counting. That is what
// `slider_sensitivity.py --mode no-double-count` gates: it asserts Layer 1's output is
// bit-identical across every human lever's full policy range, in all 12 months.
//
// Nothing measurable is lost. §10 established the panel cannot identify these
// coefficients: of 21 lever/target pairs, 12 had no effect at all and only 3 had a
// stable sign.
function climateOnly(deltas) {
  const result = { ...deltas };
  for (const key of HUMAN_SLIDERS) result[key] = 0;
  return result;
}

const OUTPUT_DEFS = {
  // `feedsInto` records a link that is actually wired, and it had drifted. GRACE -> NDVI
  // is real: the NDVI model carries the whole grace_groundwater_anomaly block. NDVI ->
  // wildlife was NOT — runPipeline passed ndvi into the wildlife model, but wildlife's
  // 12 features do not include it, so createInputFloat32Array silently dropped it. The
  // transfer was also estimated on the panel and came back null (t = -0.19 / +0.17,
  // p ~ 0.85, sign flips), so the claim is removed rather than implemented.
  //
  // The GRACE unit is metres, not centimetres: the series is lwe_thickness straight from
  // the source .nc4 with no conversion, and its sd of 0.0479 is 4.8 cm, not 0.05 mm.
  grace:        { label: "GRACE Groundwater Anomaly", unit: "m",     higherIsBetter: true,  feedsInto: "ndvi"     },
  ndvi:         { label: "NDVI Vegetation Health",    unit: "NDVI",  higherIsBetter: true,  feedsInto: null       },
  // Both of these are per-station anomaly indices, not levels. A regional mean of raw
  // well depths / raw discharge is a mean over whichever stations reported that month,
  // so it moves with the roster as much as with the water; the models are trained on the
  // centered version instead. Groundwater is still "depth to water", so positive = deeper
  // = less water = worse. Surface water is a log ratio, so 0 = normal flow and +0.7 ≈ 2x.
  groundwater:  { label: "Groundwater Depth vs Normal", unit: "ft",       higherIsBetter: false, feedsInto: null       },
  // Structural, not learned: the wildlife model has no discharge feature either, but
  // there is a real riparian transfer edge in Layer 2 carrying it (structural.js).
  surface_water:{ label: "Streamflow vs Normal",        unit: "log ratio", higherIsBetter: true,  feedsInto: "wildlife" },
  wildfire:     { label: "Wildfire Risk Index",       unit: "",      higherIsBetter: false, feedsInto: null       },
  // Per-route log-abundance anomaly: 0 = an average year, + = more birds than normal.
  wildlife:     { label: "Bird Abundance vs Normal", unit: "log ratio", higherIsBetter: true,  feedsInto: null       },
};

// Loaded dynamically from computed_stats.json (p5, p95, p50 as baseline)
const OUTPUT_STATS = computedStats.OUTPUT_STATS;

// Generated from the deployed models' own importance sidecars by
// scripts/phase3/top_inputs.py (PHASE3_PLAN.md D4). It used to be a hand-written
// table here, and it had drifted into claiming wildfire responds to population and
// NDVI to public supply — neither model has ever carried those features. Deriving
// it from model/*_feature_importance.json means it cannot drift again.
//
// Each entry also carries `human_feature_count`: how many of that model's inputs
// are human levers at all. Four of the six are zero, which is the D1 defect, and
// the card says so rather than implying a reach the model does not have.
const TOP_INPUTS = topInputs;

// What each provenance tier actually claims. PHASE3_PLAN.md §11.1: "two independent
// methods agree on the direction" and "one method, and the other found nothing" are
// different epistemic claims, and distinguishing them is the whole point of this
// architecture over the black box it replaces.
const TIER_BASIS = {
  climate: {
    label: 'climate',
    detail: 'Learned model (Phase 2), trained on observed climate. Human levers are held at their climatological normal here — the structural layer answers those.',
  },
  corroborated: {
    label: 'corroborated',
    detail: 'Water balance or published policy, AND an independent regression on the observed panel agrees on the sign after climate and trend controls.',
  },
  'structural-only': {
    label: 'structural',
    detail: 'Water balance, land-cover arithmetic or published policy. The panel measured no independent effect, so the magnitude rests on the cited parameters alone.',
  },
};

const state = {
  month: 7,
  scenarioDurationMonths: 12,
  // Policy DELTAS, not raw values. 0 everywhere = the climatological normal for
  // the selected month in the reference year.
  sliders: Object.fromEntries(
    Object.keys(SLIDER_POLICY).map(k => [k, SLIDER_POLICY[k].default])
  ),
  outputs: Object.fromEntries(
    Object.keys(OUTPUT_DEFS).map(k => [k, { score: null, rawValue: null, delta: null, provenance: null, loading: true, error: false }])
  ),
  // Flat map of every feature name to its current value.
  // Built fresh by runAll() before each inference pass.
  featureCatalog: {},
};

function getMonthEncoding(monthNum) {
  const angle = (2 * Math.PI * (monthNum - 1)) / 12;
  return { month_sin: Math.sin(angle), month_cos: Math.cos(angle) };
}

function normalizeOutput(modelKey, rawValue) {
  const { min, max } = OUTPUT_STATS[modelKey];
  return Math.max(0, Math.min(100, ((rawValue - min) / (max - min)) * 100));
}

function computeDelta(modelKey, currentScore) {
  const { min, max, baseline } = OUTPUT_STATS[modelKey];
  const baselineScore = Math.max(0, Math.min(100, ((baseline - min) / (max - min)) * 100));
  return Math.round(currentScore - baselineScore);
}

export {
  MONTHS,
  SCENARIO_DURATION_OPTIONS,
  SLIDER_DEFS,
  SLIDER_POLICY,
  ZERO_DELTAS,
  HUMAN_SLIDERS,
  climateOnly,
  OUTPUT_DEFS,
  OUTPUT_STATS,
  TOP_INPUTS,
  TIER_BASIS,
  state,
  wrapMonth,
  sliderRaw,
  sliderNormal,
  getMonthEncoding,
  normalizeOutput,
  computeDelta,
};
