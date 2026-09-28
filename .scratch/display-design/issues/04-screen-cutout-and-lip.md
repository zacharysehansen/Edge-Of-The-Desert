# 04 — Screen cutout and lip

**What to build:** The terrain ring around the screen. Given the layout from 03, the pipeline removes the screen rectangle from the terrain and builds the printed lip along its inner edge: it overhangs the screen's bezel to hide it, and its lowest surface sits at the height of the glass. The terrain rises from the lip into the real ground slope. Bezel width, glass height and outer screen dimensions are config values with 55" placeholders, so the lip can be regenerated once the team names the actual screen.

**Blocked by:** 01 — Tracer test tile, 03 — Layout config.

**Status:** done — `python -m terrain ring <layout>` writes a 2 mm viewing mesh of the whole ring and the lip test piece (option B: ring 2198 x 1407 x 194.6 mm; test piece 200 x 200 x 13.3 mm, fits a 220 mm bed)

**Design note (added during implementation):** measuring heights from the table's lowest point left the ground at the seam 24–35 mm above the glass, because the valley floor falls about 250 m across the screen's area. Near the screen, heights are now measured from the smoothed ground elevation at the nearest point of the seam, changing gradually to the table-wide reference over `datum_blend_mm` (200 mm). The terrain meets the glass all round the screen, and the 15 mm blend only smooths local bumps.

- [x] Bezel width, glass height above the frame, and outer screen dimensions are config values
- [x] The ring mesh has an opening exactly matching the screen's display area, and a lip overhanging the bezel by a configurable margin
- [x] The lip's inner edge sits at glass height, checked automatically within 0.2 mm
- [x] Terrain near the lip blends from glass height into real exaggerated elevation without a step
- [x] A test-print config produces one short corner section of lip that fits a single bed, for a physical fit test against the screen
