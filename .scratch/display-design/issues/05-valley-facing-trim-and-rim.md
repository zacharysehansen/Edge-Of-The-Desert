# 05 — Valley-facing trim and outer rim

**What to build:** The outer edge of the diorama. Each range is kept only on its valley-facing slope, up to and slightly past its crest, then cut with a clean vertical wall at the table's outer rim. Which sides get trimmed, and how far past the crest the cut sits, are per-side config values. A side that ends up against a wall can then be cut right at the crest, saving print time. Sides with no range (such as the south edge if the Santa Ritas are out) get the configured edge treatment instead.

**Blocked by:** 01 — Tracer test tile, 03 — Layout config.

**Status:** done — crest line from a priority-flood drainage divide: keep what drains out through the screen's outlet (the Santa Cruz, leaving north-west), plus 1 km, smoothed. Printed area: option A 1.58 → 1.31 m² (17% less), option B 2.27 → 1.38 m² (39% less).

**Notes:** the outer wall is vertical and follows the smoothed outline in steps of the mesh resolution (0.5 mm), to be sanded. Where two sides with different rules meet (option B's south strip and the Rincon crest), the edge runs along the screen's diagonal; keep/cut polygons can reshape it.

- [x] Per-side config: trim to crest plus a margin, or a flat valley strip, or no printed terrain
- [x] The outer wall is vertical and flat, meeting the base cleanly
- [x] The crest line used for the trim is derived from the DEM, not hand-drawn, and can be overridden in the config
- [x] The preview from 03 can overlay the trim line, so the cut can be checked by eye
- [x] The printed area reported before and after trimming shows the saving
