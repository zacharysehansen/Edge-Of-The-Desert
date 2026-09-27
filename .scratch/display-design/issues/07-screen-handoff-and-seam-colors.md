# 07 — What the screen and painters need

**What to build:** The two outputs that make the seam disappear. First, the screen area: the imagery (and DEM) cropped exactly to the screen's display rectangle, with its geographic reference and pixel dimensions, so the display software draws the valley aligned with the printed terrain. Second, the painters' color sheet: colors sampled from the same imagery at points along the seam and the lower slopes, each given as hex and CIELAB values with a marker showing where on the ring it applies. It's printed as a swatch sheet and also available as data.

This covers the painting side of the agreed calibration. The other side (tuning the screen's edge pixels to the paint under installation lighting) and the fixed 2–3 cm base-imagery border on screen belong to the display software; record both as requirements in the handoff.

**Blocked by:** 02 — Source data, 04 — Screen cutout and lip.

**Status:** ready-for-agent

- [ ] Screen-area image matches the display area's aspect ratio and position exactly, with its bounds recorded in UTM 12N and lat/lon
- [ ] Seam colors are sampled at a configurable spacing along all four sides of the seam, averaged over a small window to ignore single-pixel noise
- [ ] Swatch sheet (PDF or printable image) shows each color, its hex and Lab values, and a numbered location on a small map of the ring
- [ ] A short handoff note lists the display-software requirements: fixed border band, feathering, edge-pixel calibration to the paint
