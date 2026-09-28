# 06 — Tiles, sections and plywood templates

**What to build:** The printable package. The finished ring (cutout, lip and trim) is split into print tiles sized to the printer bed. The tiles are grouped into 3–4 transport sections whose boundaries follow ridgelines or washes, so the section seams hide in the landscape. Each tile is a watertight STL named by section and grid position, and neighbouring tiles line up exactly. Each section gets a plywood cutting template at 1:1 scale (SVG or DXF) with tile outlines marked for gluing. The run reports the tile count and an estimated print time at the configured layer height.

**Blocked by:** 04 — Screen cutout and lip, 05 — Valley-facing trim and outer rim.

**Status:** done — `python -m terrain tiles <layout>` (option B at 0.5 mm: 4 sections, 58 tiles, 1.1 GB of STL, all watertight and within a 210 mm tile and the 250 mm build height; rough estimate 586 printer-hours; tallest tile 194.6 mm, longest single print about 35 h). `--plan-only` gives the sections, templates and overview in seconds.

**Notes:** section cuts are cheapest-cost paths from the screen edge to the rim that prefer washes (from the ticket 05 drainage analysis); on flat ground (option B's south strip) they run straight. Pieces of another section smaller than 10% of a square join that square's main tile; pieces under 1% alone in a square are left off (156 mm² on option B). Templates are SVG at true size; a CNC shop may want DXF.

- [x] Tile size is set from a configured printer bed size (default 220 mm)
- [x] Every tile is watertight and fits the bed; tiles too small to be worth printing are merged into a neighbour
- [x] Shared edges between neighbouring tiles match exactly (checked automatically)
- [x] Section boundaries follow terrain lows or ridgelines, and each section fits through a standard doorway (configurable maximum dimension)
- [x] One plywood template per section, 1:1, with tile outlines and labels
- [x] A summary reports the tile count, the count per section and the estimated total print hours
