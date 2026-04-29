import * as d3 from "d3";

export const SNOW_SCENE_STYLES = `
  .cross-section-svg .scene-snow path {
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

function selectFirst(svg, selectors) {
  for (const selector of selectors) {
    const selection = svg.select(selector);
    if (!selection.empty()) {
      return selection;
    }
  }

  return d3.select(null);
}

export function createSnowController({ svg }) {
  const snowGroup = selectFirst(svg, ["#snow", "#Snow"]);

  if (!snowGroup.empty()) {
    snowGroup.classed("scene-snow", true);
  }

  const snowPieces = snowGroup.empty()
  ? []
  : snowGroup.selectAll("path").nodes().map((node) => ({
      selection: d3.select(node),
      originalFill:
        d3.color(node.style.fill || node.getAttribute("fill") || "#f7f7f7")?.formatHex()
        ?? "#f7f7f7",
    }));

  console.log("sample originalFill:", snowPieces[0]?.originalFill);


  function update({ snow, domains }) {
    const snowNorm = normalize(snow, domains.snow);
    console.log("snowNorm:", snowNorm);
    const snowMeltTone = "#BAA29A";
    

    snowPieces.forEach(({ selection, originalFill }) => {
      selection.style("fill", d3.interpolateHcl(snowMeltTone, originalFill)(snowNorm));
    });

    if (!snowGroup.empty()) {
      snowGroup.attr("opacity", 1);
    }
  }

  return { update };
}