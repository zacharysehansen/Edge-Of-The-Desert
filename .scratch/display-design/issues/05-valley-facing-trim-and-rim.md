# 05 — Valley-facing trim and outer rim

**What to build:** The outer edge of the diorama. Each range is kept only on its valley-facing slope, up to and slightly past its crest, then cut with a clean vertical wall at the table's outer rim. Which sides get trimmed, and how far past the crest the cut sits, are per-side config values. A side that ends up against a wall can then be cut right at the crest, saving print time. Sides with no range (such as the south edge if the Santa Ritas are out) get the configured edge treatment instead.

**Blocked by:** 01 — Tracer test tile, 03 — Layout config.

**Status:** ready-for-agent

- [ ] Per-side config: trim to crest plus a margin, or a flat valley strip, or no printed terrain
- [ ] The outer wall is vertical and flat, meeting the base cleanly
- [ ] The crest line used for the trim is derived from the DEM, not hand-drawn, and can be overridden in the config
- [ ] The preview from 03 can overlay the trim line, so the cut can be checked by eye
- [ ] The printed area reported before and after trimming shows the saving
