const SVG_NS = "http://www.w3.org/2000/svg";
const VIEWBOX_SIZE = 100;

function createSvgElement(tagName, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tagName);
  Object.entries(attributes).forEach(([key, value]) => {
    element.setAttribute(key, String(value));
  });
  return element;
}

class Knob extends HTMLElement {
  static options = {
    scale: {
      linear: "linear",
      log: "log",
    },
  };

  constructor(opts = {}) {
    super();
    this._opts = opts;
    this._value = null;
    this._draggingPointerId = null;
    this._lastClientY = 0;
    this._listenersAttached = false;
    this._surface = this._buildSvg();

    this._handlePointerDown = this._handlePointerDown.bind(this);
    this._handlePointerMove = this._handlePointerMove.bind(this);
    this._handlePointerUp = this._handlePointerUp.bind(this);
    this._handleWheel = this._handleWheel.bind(this);
    this._handleKeyDown = this._handleKeyDown.bind(this);
  }

  connectedCallback() {
    const options = this._opts;
    const computedStyle = getComputedStyle(this);

    this._min = this._readNumberOption(options.min, "min", 0);
    this._max = this._readNumberOption(options.max, "max", 100);
    this._step = this._readNumberOption(options.step, "step", 1);
    this._scale = options.scale ?? this.getAttribute("scale") ?? "linear";
    this._size = this._readNumberOption(
      options.size,
      "size",
      this._readStyleNumber(computedStyle, "--knob-size", 84),
    );

    this._progressColor = this._readColorOption(
      options.progresscolor,
      "progresscolor",
      computedStyle,
      "--knob-progress-color",
      "#0d6c74",
    );
    this._trackColor = this._readColorOption(
      options.trackcolor,
      "trackcolor",
      computedStyle,
      "--knob-track-color",
      "rgba(180,170,158,0.3)",
    );
    this._knobColor = this._readColorOption(
      options.knobcolor,
      "knobcolor",
      computedStyle,
      "--knob-fill-color",
      "#3d3b38",
    );
    this._pointerColor = this._readColorOption(
      options.pointercolor,
      "pointercolor",
      computedStyle,
      "--knob-pointer-color",
      "rgba(255,252,245,0.9)",
    );

    if (this._value === null) {
      const attributeValue = this.getAttribute("value");
      this._value = attributeValue !== null ? parseFloat(attributeValue) : this._min;
    }

    if (!Number.isFinite(this._step) || this._step <= 0) {
      this._step = 1;
    }

    if (!Number.isFinite(this._min)) this._min = 0;
    if (!Number.isFinite(this._max)) this._max = 100;
    if (this._max < this._min) {
      [this._min, this._max] = [this._max, this._min];
    }

    this._value = this._clamp(this._value);

    this.style.display = "inline-flex";
    this.style.flexDirection = "column";
    this.style.alignItems = "center";
    this.style.justifyContent = "center";
    this.style.cursor = "ns-resize";
    this.style.userSelect = "none";
    this.style.touchAction = "none";
    this.style.outline = "none";

    this.setAttribute("role", "slider");
    this.setAttribute("aria-orientation", "vertical");
    if (!this.hasAttribute("tabindex")) {
      this.setAttribute("tabindex", "0");
    }
    if (this.hasAttribute("label") && !this.hasAttribute("aria-label")) {
      this.setAttribute("aria-label", this.getAttribute("label"));
    }

    this._resizeSurface();

    if (this.firstChild !== this._surface) {
      this.replaceChildren(this._surface);
    }

    if (!this._listenersAttached) {
      this._attachListeners();
      this._listenersAttached = true;
    }

    this._draw();
    this._updateAria();
  }

  get value() {
    return this._value;
  }

  set value(nextValue) {
    const clampedValue = this._clamp(parseFloat(nextValue));
    if (!Number.isFinite(clampedValue)) return;

    this._value = clampedValue;
    this._draw();
    this._updateAria();
  }

  _buildSvg() {
    const svg = createSvgElement("svg", {
      viewBox: `0 0 ${VIEWBOX_SIZE} ${VIEWBOX_SIZE}`,
      "aria-hidden": "true",
    });
    const group = createSvgElement("g");

    this._trackPath = createSvgElement("path", {
      fill: "none",
      "stroke-linecap": "round",
      "vector-effect": "non-scaling-stroke",
    });
    this._progressPath = createSvgElement("path", {
      fill: "none",
      "stroke-linecap": "round",
      "vector-effect": "non-scaling-stroke",
    });
    this._knobCircle = createSvgElement("circle");
    this._pointerLine = createSvgElement("line", {
      "stroke-linecap": "round",
      "vector-effect": "non-scaling-stroke",
    });

    group.append(
      this._trackPath,
      this._progressPath,
      this._knobCircle,
      this._pointerLine,
    );
    svg.appendChild(group);
    return svg;
  }

  _readNumberOption(optionValue, attributeName, fallbackValue) {
    if (Number.isFinite(optionValue)) {
      return optionValue;
    }

    const attributeValue = this.getAttribute(attributeName);
    if (attributeValue != null) {
      const parsed = parseFloat(attributeValue);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }

    return fallbackValue;
  }

  _readStyleNumber(computedStyle, propertyName, fallbackValue) {
    const rawValue = computedStyle.getPropertyValue(propertyName).trim();
    const parsedValue = parseFloat(rawValue);
    return Number.isFinite(parsedValue) ? parsedValue : fallbackValue;
  }

  _readColorOption(optionValue, attributeName, computedStyle, propertyName, fallbackValue) {
    if (optionValue) {
      return optionValue;
    }

    const attributeValue = this.getAttribute(attributeName);
    if (attributeValue) {
      return attributeValue;
    }

    const styleValue = computedStyle.getPropertyValue(propertyName).trim();
    return styleValue || fallbackValue;
  }

  _clamp(nextValue) {
    const fallbackValue = Number.isFinite(this._value) ? this._value : this._min;
    if (!Number.isFinite(nextValue)) return fallbackValue;

    const span = this._max - this._min;
    if (span <= 0) return this._min;

    const snappedValue = Math.round((nextValue - this._min) / this._step) * this._step + this._min;
    return Math.min(this._max, Math.max(this._min, parseFloat(snappedValue.toPrecision(10))));
  }

  _norm() {
    const span = this._max - this._min;
    if (span <= 0) return 0;

    if (
      this._scale === "log"
      && this._min > 0
      && this._max > 0
      && this._value > 0
    ) {
      return Math.log(this._value / this._min) / Math.log(this._max / this._min);
    }

    return (this._value - this._min) / span;
  }

  _resizeSurface() {
    const size = Math.max(56, Math.round(this._size));
    this._surface.style.width = `${size}px`;
    this._surface.style.height = `${size}px`;
  }

  _polarToCartesian(radius, angleRadians) {
    const center = VIEWBOX_SIZE / 2;
    return {
      x: center + Math.cos(angleRadians) * radius,
      y: center + Math.sin(angleRadians) * radius,
    };
  }

  _describeArc(radius, startAngle, endAngle) {
    const angleDelta = endAngle - startAngle;
    if (angleDelta <= 0) {
      return "";
    }

    const start = this._polarToCartesian(radius, startAngle);
    const end = this._polarToCartesian(radius, endAngle);
    const largeArcFlag = angleDelta > Math.PI ? 1 : 0;

    return [
      "M", start.x, start.y,
      "A", radius, radius, 0, largeArcFlag, 1, end.x, end.y,
    ].join(" ");
  }

  _draw() {
    const center = VIEWBOX_SIZE / 2;
    const trackRadius = VIEWBOX_SIZE * 0.42;
    const knobRadius = VIEWBOX_SIZE * 0.33;
    const lineWidth = VIEWBOX_SIZE * 0.07;
    const startAngle = Math.PI * 0.75;
    const endAngle = Math.PI * 2.25;
    const valueAngle = startAngle + this._norm() * (endAngle - startAngle);
    const pointerEnd = this._polarToCartesian(knobRadius * 0.62, valueAngle);

    this._trackPath.setAttribute("d", this._describeArc(trackRadius, startAngle, endAngle));
    this._trackPath.setAttribute("stroke", this._trackColor);
    this._trackPath.setAttribute("stroke-width", String(lineWidth));

    const progressPath = valueAngle > startAngle
      ? this._describeArc(trackRadius, startAngle, valueAngle)
      : "";
    this._progressPath.setAttribute("d", progressPath);
    this._progressPath.setAttribute("stroke", this._progressColor);
    this._progressPath.setAttribute("stroke-width", String(lineWidth));

    this._knobCircle.setAttribute("cx", String(center));
    this._knobCircle.setAttribute("cy", String(center));
    this._knobCircle.setAttribute("r", String(knobRadius));
    this._knobCircle.setAttribute("fill", this._knobColor);

    this._pointerLine.setAttribute("x1", String(center));
    this._pointerLine.setAttribute("y1", String(center));
    this._pointerLine.setAttribute("x2", String(pointerEnd.x));
    this._pointerLine.setAttribute("y2", String(pointerEnd.y));
    this._pointerLine.setAttribute("stroke", this._pointerColor);
    this._pointerLine.setAttribute("stroke-width", String(VIEWBOX_SIZE * 0.05));
  }

  _updateAria() {
    this.setAttribute("aria-valuemin", String(this._min));
    this.setAttribute("aria-valuemax", String(this._max));
    this.setAttribute("aria-valuenow", String(this._value));
  }

  _emitInput() {
    this._updateAria();
    this.dispatchEvent(new Event("input", { bubbles: true }));
  }

  _applyIncrement(delta) {
    const nextValue = this._clamp(this._value + delta);
    if (nextValue === this._value) return;

    this._value = nextValue;
    this._draw();
    this._emitInput();
  }

  _handlePointerDown(event) {
    this._draggingPointerId = event.pointerId;
    this._lastClientY = event.clientY;
    this._surface.setPointerCapture(event.pointerId);
    this.focus();
    event.preventDefault();
  }

  _handlePointerMove(event) {
    if (event.pointerId !== this._draggingPointerId) return;

    const deltaY = this._lastClientY - event.clientY;
    this._lastClientY = event.clientY;
    const range = Math.max(this._max - this._min, this._step);
    this._applyIncrement((deltaY / 160) * range);
  }

  _handlePointerUp(event) {
    if (event.pointerId !== this._draggingPointerId) return;

    if (this._surface.hasPointerCapture(event.pointerId)) {
      this._surface.releasePointerCapture(event.pointerId);
    }
    this._draggingPointerId = null;
  }

  _handleWheel(event) {
    event.preventDefault();
    const direction = event.deltaY < 0 ? 1 : -1;
    this._applyIncrement(direction * this._step);
  }

  _handleKeyDown(event) {
    const largeStep = this._step * 10;

    switch (event.key) {
      case "ArrowUp":
      case "ArrowRight":
        event.preventDefault();
        this._applyIncrement(this._step);
        break;
      case "ArrowDown":
      case "ArrowLeft":
        event.preventDefault();
        this._applyIncrement(-this._step);
        break;
      case "PageUp":
        event.preventDefault();
        this._applyIncrement(largeStep);
        break;
      case "PageDown":
        event.preventDefault();
        this._applyIncrement(-largeStep);
        break;
      case "Home":
        event.preventDefault();
        this.value = this._min;
        this._emitInput();
        break;
      case "End":
        event.preventDefault();
        this.value = this._max;
        this._emitInput();
        break;
      default:
        break;
    }
  }

  _attachListeners() {
    this._surface.addEventListener("pointerdown", this._handlePointerDown);
    this._surface.addEventListener("pointermove", this._handlePointerMove);
    this._surface.addEventListener("pointerup", this._handlePointerUp);
    this._surface.addEventListener("pointercancel", this._handlePointerUp);
    this._surface.addEventListener("wheel", this._handleWheel, { passive: false });
    this.addEventListener("keydown", this._handleKeyDown);
  }
}

if (!window.customElements.get("my-knob")) {
  window.customElements.define("my-knob", Knob);
}
