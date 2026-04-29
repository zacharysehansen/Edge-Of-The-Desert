import * as d3 from "d3";

export const TEMPERATURE_SCENE_STYLES = `
  .cross-section-svg .scene-sky,
  .cross-section-svg .scene-mountain path {
    transition: fill 700ms ease;
  }
`;

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

function selectFirst(svg, selectors) {
  for (const selector of selectors) {
    const selection = svg.select(selector);
    if (!selection.empty()) {
      return selection;
    }
  }

  return d3.select(null);
}

export function createTemperatureController({ svg }) {
  const skyPath = selectFirst(svg, ["#sky path"]);
  if (!skyPath.empty()) {
    skyPath.classed("scene-sky", true);
  }

  const mountainGroup = selectFirst(svg, ["#mountain", "#Mountain", "#mountains", "#Mountains"]);
  if (!mountainGroup.empty()) {
    mountainGroup.classed("scene-mountain", true);
  }

  const mountainPieces = mountainGroup.empty()
    ? []
    : mountainGroup.selectAll("path").nodes().map((node) => ({
        selection: d3.select(node),
        originalFill:
          d3.color(node.style.fill || node.getAttribute("fill") || "#8a7560")?.formatHex()
          ?? "#8a7560",
      }));

  function update({ temperature, precipitation, domains }) {
    const tempNorm = normalize(temperature, domains.temperature);
    const precipNorm = normalize(precipitation, domains.precipitation, 0);
    const drySkyFactor = clamp01((0.22 - precipNorm) / 0.22);
    const skyHeatTint = tempNorm * drySkyFactor * 0.7;
    const mountainHeatTint = tempNorm * 0.45;

    if (!skyPath.empty() && drySkyFactor > 0) {
      skyPath.style("fill", d3.interpolateRgb("#cbe7f5", "#b7c96f")(skyHeatTint));
    }

    mountainPieces.forEach(({ selection, originalFill }) => {
      const heatedColor = d3.interpolateRgb(originalFill, "#c47c2b")(mountainHeatTint);
      selection
        .style("fill", heatedColor)
        .attr("fill", heatedColor);
    });
  }

  return { update };
}
