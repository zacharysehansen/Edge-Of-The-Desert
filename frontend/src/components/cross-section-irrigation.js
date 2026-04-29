export const IRRIGATION_SCENE_STYLES = "";

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function normalize(value, domain, fallback = 0.5) {
  if (!Number.isFinite(value)) return fallback;
  if (!Array.isArray(domain) || domain.length !== 2) return fallback;

  const [minValue, maxValue] = domain;
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || minValue === maxValue) {
    return fallback;
  }

  return clamp01((value - minValue) / (maxValue - minValue));
}

export function createIrrigationController() {
  function update({ irrigation, domains }) {
    const irrigationNorm = normalize(irrigation, domains.irrigation);

    return { irrigationNorm };
  }

  return { update };
}
