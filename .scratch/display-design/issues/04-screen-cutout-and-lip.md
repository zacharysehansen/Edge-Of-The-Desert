# 04 — Screen cutout and lip

**What to build:** The terrain ring around the screen. Given the layout from 03, the pipeline removes the screen rectangle from the terrain and builds the printed lip along its inner edge: it overhangs the screen's bezel to hide it, and its lowest surface sits at the height of the glass. The terrain rises from the lip into the real ground slope. Bezel width, glass height and outer screen dimensions are config values with 55" placeholders, so the lip can be regenerated once the team names the actual screen.

**Blocked by:** 01 — Tracer test tile, 03 — Layout config.

**Status:** ready-for-agent

- [ ] Bezel width, glass height above the frame, and outer screen dimensions are config values
- [ ] The ring mesh has an opening exactly matching the screen's display area, and a lip overhanging the bezel by a configurable margin
- [ ] The lip's inner edge sits at glass height, checked automatically within 0.2 mm
- [ ] Terrain near the lip blends from glass height into real exaggerated elevation without a step
- [ ] A test-print config produces one short corner section of lip that fits a single bed, for a physical fit test against the screen
