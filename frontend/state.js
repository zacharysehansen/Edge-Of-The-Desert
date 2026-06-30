import computedStats from './computed_stats.json' with { type: 'json' };

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
  impervious_pct:           { label: "Urbanization (Impervious %)", unit: "%",     panel: "human"   },
  mead_pool_elevation:      { label: "Lake Mead Level",         unit: "ft",      panel: "human"   },
  precipitation_mm_day:     { label: "Precipitation",           unit: "mm/day",  panel: "climate" },
  temperature_2m_c:         { label: "Temperature",             unit: "°C",      panel: "climate" },
  usdm_dsci:                { label: "Drought Index (DSCI)",    unit: "",        panel: "climate" },
};

// Loaded dynamically from computed_stats.json (p5, p95, p50)
const SLIDER_STATS = computedStats.SLIDER_STATS;

const OUTPUT_DEFS = {
  grace:        { label: "GRACE Groundwater Anomaly", unit: "cm",    higherIsBetter: true,  feedsInto: "ndvi"     },
  ndvi:         { label: "NDVI Vegetation Health",    unit: "NDVI",  higherIsBetter: true,  feedsInto: "wildlife" },
  groundwater:  { label: "Groundwater Well Depth",    unit: "ft",    higherIsBetter: false, feedsInto: null       },
  surface_water:{ label: "Surface Water Discharge",   unit: "cfs",   higherIsBetter: true,  feedsInto: null       },
  wildfire:     { label: "Wildfire Risk Index",       unit: "",      higherIsBetter: false, feedsInto: null       },
  wildlife:     { label: "Wildlife Abundance",        unit: "index", higherIsBetter: true,  feedsInto: null       },
};

// Loaded dynamically from computed_stats.json (p5, p95, p50 as baseline)
const OUTPUT_STATS = computedStats.OUTPUT_STATS;

const TOP_INPUTS = {
  grace:         "temperature, precipitation anomaly, seasonal position",
  ndvi:          "temperature (lag), public supply groundwater, precipitation",
  groundwater:   "precipitation, temperature, drought index",
  surface_water: "precipitation anomaly, temperature anomaly (lag), precipitation",
  wildfire:      "population, urbanization (lag), temperature (lag)",
  wildlife:      "NDVI (annual mean), drought index, public supply groundwater",
};

const state = {
  month: 7,
  scenarioDurationMonths: 12,
  sliders: Object.fromEntries(
    Object.keys(SLIDER_STATS).map(k => [k, SLIDER_STATS[k].default])
  ),
  outputs: Object.fromEntries(
    Object.keys(OUTPUT_DEFS).map(k => [k, { score: null, rawValue: null, delta: null, loading: true, error: false }])
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
  SLIDER_STATS,
  OUTPUT_DEFS,
  OUTPUT_STATS,
  TOP_INPUTS,
  state,
  getMonthEncoding,
  normalizeOutput,
  computeDelta,
};