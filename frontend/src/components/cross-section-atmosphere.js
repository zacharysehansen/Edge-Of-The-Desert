import * as d3 from "d3";

const CLOUD_SELECTORS = ["#cloud", "#cloud1", "#cloud2"];
const MOUNTAIN_SELECTORS = ["#mountain", "#Mountain", "#mountains", "#Mountains"];
const SNOW_SELECTORS = ["#snow", "#Snow"];
const RAIN_DROP_COUNT = 90;
const RAIN_THRESHOLD = 0.55;
const MOUNTAIN_SHADE_THRESHOLD = 0.7;

export const ATMOSPHERE_SCENE_STYLES = `
  @keyframes svgCloudDrift {
    from {
      transform: translate(var(--cloud-start-x, 0px), var(--cloud-offset-y, 0px))
        scale(var(--cloud-scale, 1));
    }
    to {
      transform: translate(var(--cloud-end-x, 2400px), var(--cloud-offset-y, 0px))
        scale(var(--cloud-scale, 1));
    }
  }
  @keyframes svgRainFall {
    from { transform: translateY(var(--rain-start-y, 0px)); }
    to   { transform: translateY(var(--rain-travel, 520px)); }
  }
  .cross-section-svg .scene-cloud {
    animation-name: svgCloudDrift;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    transform-box: fill-box;
    transform-origin: center;
    will-change: transform, opacity;
  }
  .cross-section-svg .scene-cloud path,
  .cross-section-svg .scene-sky {
    transition: opacity 700ms ease, fill 700ms ease, stroke 700ms ease, transform 700ms ease;
  }
  .cross-section-svg .scene-mountain-storm-shade {
    pointer-events: none;
    transition: opacity 700ms ease;
  }
  .cross-section-svg .scene-raindrop {
    animation-name: svgRainFall;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    transform-box: fill-box;
    will-change: transform;
  }
`;

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function easeOutPower(value, power = 2) {
  return 1 - (1 - clamp01(value)) ** power;
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

function stripIds(node) {
  if (!node?.removeAttribute) return;

  node.removeAttribute("id");
  node.removeAttribute("serif:id");

  Array.from(node.children ?? []).forEach(stripIds);
}

function getDefs(svg) {
  return svg.select("defs").empty()
    ? svg.insert("defs", ":first-child")
    : svg.select("defs");
}

function getCloudRoot(svg, baseClouds) {
  const root = selectFirst(svg, ["#sky_and_clouds"]);
  if (!root.empty()) return root;

  const fallbackParent = baseClouds[0]?.node()?.parentNode ?? null;
  return fallbackParent ? d3.select(fallbackParent) : d3.select(null);
}

function createCloudInstances(svg, svgWidth) {
  const baseClouds = CLOUD_SELECTORS
    .map((selector) => svg.select(selector))
    .filter((selection) => !selection.empty());

  if (!baseClouds.length) {
    return [];
  }

  const cloudRoot = getCloudRoot(svg, baseClouds);
  if (cloudRoot.empty()) {
    return [];
  }

  const repeatCount = 3;
  const slotSpacing = svgWidth * 0.28;
  const startX = -svgWidth * 0.9;
  const travelDistance = svgWidth + slotSpacing * 4;
  const instances = [];
  let slot = 0;

  baseClouds.forEach((baseCloud, baseIndex) => {
    for (let repeatIndex = 0; repeatIndex < repeatCount; repeatIndex += 1) {
      let selection = baseCloud;

      if (repeatIndex > 0) {
        const cloneNode = baseCloud.node().cloneNode(true);
        stripIds(cloneNode);
        cloudRoot.node().appendChild(cloneNode);
        selection = d3.select(cloneNode);
      }

      const slotOffset = startX + slotSpacing * slot;
      const baseYOffset = (baseIndex - 1) * 14 + repeatIndex * 8;
      const scaleFactor = 0.9 + repeatIndex * 0.07 + baseIndex * 0.03;
      const opacityFactor = 0.8 + ((baseIndex + repeatIndex) % 3) * 0.08;
      const duration = 66 + baseIndex * 7 + repeatIndex * 5;

      selection
        .classed("scene-cloud", true)
        .style("--cloud-start-x", `${slotOffset}px`)
        .style("--cloud-end-x", `${slotOffset + travelDistance}px`)
        .style("--cloud-offset-y", `${baseYOffset}px`)
        .style("--cloud-scale", `${scaleFactor}`)
        .style("animation-duration", `${duration}s`);

      instances.push({
        selection,
        baseYOffset,
        scaleFactor,
        opacityFactor,
        stormLift: 2 + repeatIndex * 1.2,
      });

      slot += 1;
    }
  });

  return instances;
}

function insertAfter(targetNode, newNode) {
  const parentNode = targetNode?.parentNode;
  if (!parentNode) return null;

  if (targetNode.nextSibling) {
    parentNode.insertBefore(newNode, targetNode.nextSibling);
  } else {
    parentNode.appendChild(newNode);
  }

  return newNode;
}

function createMountainShadeLayer(svg, defs, svgWidth, svgHeight) {
  const mountainGroup = selectFirst(svg, MOUNTAIN_SELECTORS);
  const snowGroup = selectFirst(svg, SNOW_SELECTORS);
  const targetSelections = [mountainGroup, snowGroup].filter((selection) => !selection.empty());

  if (!targetSelections.length) {
    return null;
  }

  const uid = Math.random().toString(36).slice(2, 8);
  const clipId = `mountain-storm-shade-${uid}`;
  const clipPath = defs.append("clipPath").attr("id", clipId);

  targetSelections.forEach((selection) => {
    const cloneNode = selection.node().cloneNode(true);
    stripIds(cloneNode);
    clipPath.node().appendChild(cloneNode);
  });

  const anchorNode = targetSelections[targetSelections.length - 1].node();
  const shadeNode = svg.node().ownerDocument.createElementNS("http://www.w3.org/2000/svg", "g");
  insertAfter(anchorNode, shadeNode);

  const shadeGroup = d3.select(shadeNode)
    .attr("class", "scene-mountain-storm-shade")
    .attr("clip-path", `url(#${clipId})`)
    .attr("opacity", 0);

  shadeGroup.append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", svgWidth)
    .attr("height", svgHeight)
    .attr("fill", "#12171f");

  return shadeGroup;
}

function createRainLayer(svg, defs, svgWidth, svgHeight) {
  const clipHeight = svgHeight / 3;
  const uid = Math.random().toString(36).slice(2, 8);
  const clipId = `rain-clip-region-${uid}`;

  defs.append("clipPath")
    .attr("id", clipId)
    .append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", svgWidth)
    .attr("height", clipHeight);

  const rainGroup = svg.append("g")
    .attr("class", "scene-rain")
    .attr("opacity", 0)
    .attr("clip-path", `url(#${clipId})`);

  const travelDistance = clipHeight * 1.2;

  for (let i = 0; i < RAIN_DROP_COUNT; i++) {
    const x = Math.random() * svgWidth;
    const startY = Math.random() * clipHeight;
    const dropLength = 18 + Math.random() * 16;
    const angle = 0.18;
    const duration = 0.55 + Math.random() * 0.35;
    const delay = -(Math.random() * duration * 3);

    rainGroup.append("line")
      .attr("x1", x)
      .attr("y1", startY)
      .attr("x2", x - dropLength * angle)
      .attr("y2", startY + dropLength)
      .attr("stroke", "#a8cfe0")
      .attr("stroke-width", 1.8)
      .attr("stroke-linecap", "round")
      .classed("scene-raindrop", true)
      .style("--rain-start-y", `${startY}px`)
      .style("--rain-travel", `${travelDistance}px`)
      .style("animation-duration", `${duration}s`)
      .style("animation-delay", `${delay}s`);
  }

  return rainGroup;
}

export function createAtmosphereController({ svg, bounds, svgWidth }) {
  const svgHeight = bounds?.svg?.height ?? bounds?.height ?? 500;
  const defs = getDefs(svg);

  const skyPath = selectFirst(svg, ["#sky path"]);
  if (!skyPath.empty()) {
    skyPath.classed("scene-sky", true);
  }

  const cloudInstances = createCloudInstances(svg, svgWidth);
  const mountainShadeGroup = createMountainShadeLayer(svg, defs, svgWidth, svgHeight);
  const rainGroup = createRainLayer(svg, defs, svgWidth, svgHeight);

  function update({ precipitation, domains }) {
    const precipNorm = normalize(precipitation, domains.precipitation);
    const storminess = easeOutPower(precipNorm, 1.9);
    const skyColor = d3.interpolateRgb("#cbe7f5", "#415563")(precipNorm);
    const cloudColor = d3.interpolateRgb("#fbfcfd", "#262A32")(storminess);
    const cloudOpacity = 0.36 + storminess * 0.54;
    const cloudScale = 1 + storminess * 0.42;

    if (!skyPath.empty()) {
      skyPath.style("fill", skyColor);
    }

    cloudInstances.forEach((cloud) => {
      cloud.selection
        .attr("opacity", clamp01(cloudOpacity * cloud.opacityFactor))
        .style("--cloud-offset-y", `${cloud.baseYOffset + precipNorm * cloud.stormLift}px`)
        .style("--cloud-scale", `${cloudScale * cloud.scaleFactor}`);

      cloud.selection
        .selectAll("path")
        .style("fill", cloudColor)
        .attr("fill", cloudColor);
    });

    const rainProgress = storminess > RAIN_THRESHOLD
      ? clamp01((storminess - RAIN_THRESHOLD) / (1 - RAIN_THRESHOLD))
      : 0;
    const mountainShadeProgress = storminess > MOUNTAIN_SHADE_THRESHOLD
      ? easeOutPower(
          (storminess - MOUNTAIN_SHADE_THRESHOLD) / (1 - MOUNTAIN_SHADE_THRESHOLD),
          1.4,
        )
      : 0;

    rainGroup.attr("opacity", rainProgress * 0.72);

    if (mountainShadeGroup) {
      mountainShadeGroup.attr("opacity", mountainShadeProgress * 0.18);
    }

    return { precipNorm, storminess };
  }

  return { update };
}
