# 06 — Tiles, sections and plywood templates

**What to build:** The printable package. The finished ring (cutout, lip and trim) is split into print tiles sized to the printer bed. The tiles are grouped into 3–4 transport sections whose boundaries follow ridgelines or washes, so the section seams hide in the landscape. Each tile is a watertight STL named by section and grid position, and neighbouring tiles line up exactly. Each section gets a plywood cutting template at 1:1 scale (SVG or DXF) with tile outlines marked for gluing. The run reports the tile count and an estimated print time at the configured layer height.

**Blocked by:** 04 — Screen cutout and lip, 05 — Valley-facing trim and outer rim.

**Status:** ready-for-agent

- [ ] Tile size is set from a configured printer bed size (default 220 mm)
- [ ] Every tile is watertight and fits the bed; tiles too small to be worth printing are merged into a neighbour
- [ ] Shared edges between neighbouring tiles match exactly (checked automatically)
- [ ] Section boundaries follow terrain lows or ridgelines, and each section fits through a standard doorway (configurable maximum dimension)
- [ ] One plywood template per section, 1:1, with tile outlines and labels
- [ ] A summary reports the tile count, the count per section and the estimated total print hours
