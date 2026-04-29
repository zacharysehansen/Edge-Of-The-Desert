export const PIPE_SCENE_STYLES = `
  .cross-section-svg .scene-water-arrows {
    animation-name: svgArrowPoint;
    animation-timing-function: ease-in-out;
    animation-iteration-count: infinite;
    transition: opacity 450ms ease;
  }
  .cross-section-svg .scene-water-arrow {
    transition: opacity 450ms ease;
  }
  @keyframes svgArrowPoint {
    0% { transform: translateX(0px); }
    18% { transform: translateX(16px); }
    100% { transform: translateX(0px); }
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

function getArrowLocalBounds(node) {
  const pathNode = node.querySelector("path");
  const targetNode = pathNode ?? node;

  try {
    const box = targetNode.getBBox();
    if (Number.isFinite(box.width) && Number.isFinite(box.height) && (box.width || box.height)) {
      return box;
    }
  } catch {
    // Ignore bbox errors and fall back below.
  }

  return { x: 0, y: 0, width: 1, height: 1 };
}

function buildScaledArrowTransform(originalTransform, anchorX, anchorY, scale) {
  const scaleTransform = `translate(${anchorX} ${anchorY}) scale(${scale}) translate(${-anchorX} ${-anchorY})`;
  return originalTransform ? `${originalTransform} ${scaleTransform}` : scaleTransform;
}

export function createPipeController({
  defs,
  bounds,
  waterInPipeGroup,
  waterArrowsGroup,
}) {
  let lastArrowDuration = null;
  let arrowResumeTimer = null;

  const uid = Math.random().toString(36).slice(2, 8);
  const pipeClipId = `pipe-water-clip-${uid}`;
  const pipeClipRect = defs
    .append("clipPath")
    .attr("id", pipeClipId)
    .append("rect")
    .attr("x", bounds.waterInPipe.x)
    .attr("y", bounds.waterInPipe.y)
    .attr("width", bounds.waterInPipe.width)
    .attr("height", bounds.waterInPipe.height);
  const waterArrowItems = waterArrowsGroup.empty()
    ? []
    : waterArrowsGroup.selectAll("g").nodes().map((node) => {
        const bounds = getArrowLocalBounds(node);
        return {
          selection: waterArrowsGroup.select(function selectNode() {
            return node;
          }),
          originalTransform: node.getAttribute("transform") ?? "",
          anchorX: bounds.x + bounds.width / 2,
          anchorY: bounds.y + bounds.height,
        };
      });

  if (!waterInPipeGroup.empty()) {
    waterInPipeGroup.attr("clip-path", `url(#${pipeClipId})`);
  }

  if (!waterArrowsGroup.empty()) {
    waterArrowsGroup
      .classed("scene-water-arrows", true)
      .style("animation-duration", "2.2s");
  }

  if (waterArrowItems.length) {
    waterArrowItems
      .forEach(({ selection, originalTransform }) => {
        selection
          .classed("scene-water-arrow", true)
          .attr("transform", originalTransform);
      });
  }

  function update({ irrigationNorm, publicSupply, domains }) {
    const publicSupplyNorm = normalize(publicSupply, domains.publicSupply);
    const demandNorm = clamp01(irrigationNorm * 0.6 + publicSupplyNorm * 0.4);
    const pipeLevel = bounds.waterInPipe.height * (0.18 + demandNorm * 0.82);

    pipeClipRect
      .attr("y", bounds.waterInPipe.y + bounds.waterInPipe.height - pipeLevel)
      .attr("height", pipeLevel);

    if (!waterInPipeGroup.empty()) {
      waterInPipeGroup
        .attr("opacity", 0.34 + demandNorm * 0.66)
        .style(
          "filter",
          `saturate(${0.8 + demandNorm * 0.65}) brightness(${0.9 + demandNorm * 0.16})`,
        );
    }

    if (!waterArrowsGroup.empty()) {
      waterArrowsGroup.attr("opacity", 0.12 + demandNorm * 0.88);
    }

    if (waterArrowItems.length) {
      const duration = Math.round((2.2 - demandNorm * 1.6) * 10) / 10;
      const scale = 1 + demandNorm * 1.5;

      waterArrowItems.forEach(({ selection, originalTransform, anchorX, anchorY }) => {
        selection.attr(
          "transform",
          buildScaledArrowTransform(originalTransform, anchorX, anchorY, scale),
        );
      });

      clearTimeout(arrowResumeTimer);
      waterArrowsGroup.style("animation-play-state", "paused");
      arrowResumeTimer = setTimeout(() => {
        if (duration !== lastArrowDuration) {
          lastArrowDuration = duration;
          waterArrowsGroup.style("animation-duration", `${duration}s`);
        }
        waterArrowsGroup.style("animation-play-state", "running");
      }, 450);
    }
  }

  return { update };
}
