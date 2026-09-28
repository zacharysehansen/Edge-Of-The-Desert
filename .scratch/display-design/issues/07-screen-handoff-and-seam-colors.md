# 07 — What the screen and painters need

**What to build:** The two outputs that make the seam disappear. First, the screen area: the imagery (and DEM) cropped exactly to the screen's display rectangle, with its geographic reference and pixel dimensions, so the display software draws the valley aligned with the printed terrain. Second, the painters' color sheet: colors sampled from the same imagery at points along the seam and the lower slopes, each given as hex and CIELAB values with a marker showing where on the ring it applies. It's printed as a swatch sheet and also available as data.

This covers the painting side of the agreed calibration. The other side (tuning the screen's edge pixels to the paint under installation lighting) and the fixed 2–3 cm base-imagery border on screen belong to the display software; record both as requirements in the handoff.

**Blocked by:** 02 — Source data, 04 — Screen cutout and lip.

**Status:** done — `python -m terrain handoff <layout>` writes `outputs/terrain/<layout>/handoff/` (option B: screen image 3840 x 2160 px at 9.22 m/px, lip hides 10 px per side; 40 seam colours and 29 lower-slope colours; runs in about 10 s)

**Notes:** the screen image is one image pixel per panel pixel. The panel's resolution is a config value (`screen.resolution_px`, placeholder 4K) and must have the layout's aspect, or the run stops, because square pixels would otherwise stretch the map. Colours are averaged in linear light over a 4 mm window. CIELAB is given under D65, the white point sRGB is defined against; a colorimeter set to D50 will read slightly different numbers. The lower-slope row (100 mm out) is sampled only where terrain is printed. The fixed band (25 mm) and feather (20 mm) widths are placeholders from the design doc, handed over in pixels.

- [x] Screen-area image matches the display area's aspect ratio and position exactly, with its bounds recorded in UTM 12N and lat/lon
- [x] Seam colors are sampled at a configurable spacing along all four sides of the seam, averaged over a small window to ignore single-pixel noise
- [x] Swatch sheet (PDF or printable image) shows each color, its hex and Lab values, and a numbered location on a small map of the ring
- [x] A short handoff note lists the display-software requirements: fixed border band, feathering, edge-pixel calibration to the paint
