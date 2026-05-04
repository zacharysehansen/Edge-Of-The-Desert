import * as d3 from "d3";

const AQUIFER_COLOR = d3
  .scaleLinear()
  .domain([0, 33, 66, 100])
  .range(["#9f7b59", "#b18b65", "#6fa4cb", "#1971aa"])
  .interpolate(d3.interpolateHcl);

export const WATERLINE_SCENE_STYLES = `
  .cross-section-svg .scene-water-line,
  .cross-section-svg .scene-water-label,
  .cross-section-svg .scene-dry-zone {
    transition: opacity 700ms ease, fill 700ms ease, stroke 700ms ease, transform 700ms ease;
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

function waterTableY(
  grace,
  powell,
  graceDomain,
  powellDomain,
  aquiferBounds,
  secondGroundLayerBounds,
  bedrockBounds,
) {
  const graceNorm = normalize(grace, graceDomain);
  const powellNorm = normalize(powell, powellDomain);
  const blended = Math.min(Math.pow(0.6 * graceNorm + 0.4 * powellNorm, 1.8), 0.6);

  const originalTop = aquiferBounds.y + aquiferBounds.height * 0.18;  const top =
    secondGroundLayerBounds?.y != null
      ? Math.min(originalTop, secondGroundLayerBounds.y + secondGroundLayerBounds.height * 0.01)
      : originalTop;
  const bottom = Math.min(
    aquiferBounds.y + aquiferBounds.height * 0.98,
    bedrockBounds.y + 14,
  );

  return bottom - blended * (bottom - top);
}

export function getAquiferColor(score) {
  const safeScore = Number.isFinite(score) ? Math.max(0, Math.min(100, score)) : 50;
  return AQUIFER_COLOR(safeScore);
}

export function createWaterlineController({
  defs,
  overlayGroup,
  aquiferWaterGroup,
  bounds,
  svgWidth,
}) {
  const uid = Math.random().toString(36).slice(2, 8);
  const aquiferClipId = `aquifer-water-clip-${uid}`;

  const aquiferClipRect = defs
    .append("clipPath")
    .attr("id", aquiferClipId)
    .append("rect")
    .attr("x", bounds.aquiferWater.x)
    .attr("y", bounds.aquiferWater.y)
    .attr("width", bounds.aquiferWater.width)
    .attr("height", bounds.aquiferWater.height);

  if (!aquiferWaterGroup.empty()) {
    aquiferWaterGroup.attr("clip-path", `url(#${aquiferClipId})`);
  }

  const defaultWaterColor = getAquiferColor(50);

  const waterLine = overlayGroup
    .append("line")
    .attr("class", "scene-water-line")
    .attr("x1", 0)
    .attr("y1", bounds.aquiferWater.y)
    .attr("x2", svgWidth)
    .attr("y2", bounds.aquiferWater.y)
    .attr("stroke", defaultWaterColor)
    .attr("stroke-width", 5)
    .attr("stroke-dasharray", "28 16")
    .attr("opacity", 0.72);

  const waterLineLabel = overlayGroup
    .append("text")
    .attr("class", "scene-water-label")
    .attr("x", svgWidth - 30)
    .attr("y", bounds.aquiferWater.y + 8)
    .attr("text-anchor", "end")
    .attr("font-size", "24px")
    .attr("font-family", "Georgia, serif")
    .attr("fill", defaultWaterColor)
    .attr("opacity", 0.88)
    .text("Water Line");

  function update({ score, grace, powell, domains }) {
    const safeScore = Number.isFinite(score) ? Math.max(0, Math.min(100, score)) : 50;
    const waterColor = AQUIFER_COLOR(safeScore);
    const wtY = waterTableY(
      grace,
      powell,
      domains.grace,
      domains.powell,
      bounds.aquiferWater,
      bounds.secondGroundLayer,
      bounds.bedrock,
    );
    const dryHeight = Math.max(0, wtY - bounds.aquiferWater.y);
    const waterHeight = Math.max(0, bounds.aquiferWater.y + bounds.aquiferWater.height - wtY);

    aquiferClipRect
      .attr("y", wtY)
      .attr("height", waterHeight);

    waterLine
      .attr("y1", wtY)
      .attr("y2", wtY)
      .attr("stroke", waterColor);

    waterLineLabel
      .attr("y", Math.max(bounds.secondGroundLayer.y + 24, wtY - 12))
      .attr("fill", waterColor);

    return { safeScore, waterColor, waterTableY: wtY };
  }

  return { update };
}
