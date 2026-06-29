const MONTHS = [
  "January","February","March","April","May","June",
  "July","August","September","October","November","December"
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

const SLIDER_STATS = {
  population:                      { min: 1800000, max: 2600000,  default: 2100000  },
  irrigation_total_withdrawal_mgd: { min: 400,     max: 1800,     default: 950      },
  public_supply_groundwater_mgd:   { min: 80,      max: 320,      default: 175      },
  impervious_pct:                  { min: 2.5,     max: 6.5,      default: 4.2      },
  mead_pool_elevation:             { min: 1050,    max: 1220,     default: 1130     },
  precipitation_mm_day:            { min: 0.3,     max: 5.2,      default: 1.4      },
  temperature_2m_c:                { min: 14,      max: 32,       default: 22       },
  usdm_dsci:                       { min: 0,       max: 400,      default: 150      },
};

const OUTPUT_DEFS = {
  grace:        { label: "GRACE Groundwater Anomaly", unit: "cm",    higherIsBetter: true,  feedsInto: "ndvi"     },
  ndvi:         { label: "NDVI Vegetation Health",    unit: "NDVI",  higherIsBetter: true,  feedsInto: "wildlife" },
  groundwater:  { label: "Groundwater Well Depth",    unit: "ft",    higherIsBetter: false, feedsInto: null       },
  surface_water:{ label: "Surface Water Discharge",   unit: "cfs",   higherIsBetter: true,  feedsInto: null       },
  wildfire:     { label: "Wildfire Risk Index",       unit: "",      higherIsBetter: false, feedsInto: null       },
  wildlife:     { label: "Wildlife Abundance",        unit: "index", higherIsBetter: true,  feedsInto: null       },
};

const OUTPUT_STATS = {
  grace:         { min: -8,   max: 4,    baseline: -1.5  },
  ndvi:          { min: 0.10, max: 0.45, baseline: 0.25  },
  groundwater:   { min: 50,   max: 250,  baseline: 140   },
  surface_water: { min: 5,    max: 900,  baseline: 60    },
  wildfire:      { min: 0,    max: 1,    baseline: 0.18  },
  wildlife:      { min: 800,  max: 2200, baseline: 1400  },
};

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
  sliders: Object.fromEntries(
    Object.keys(SLIDER_STATS).map(k => [k, SLIDER_STATS[k].default])
  ),
  outputs: Object.fromEntries(
    Object.keys(OUTPUT_DEFS).map(k => [k, { score: null, rawValue: null, delta: null, loading: true, error: false }])
  ),
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