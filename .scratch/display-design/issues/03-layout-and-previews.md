# 03 — Layout config and questionnaire previews

**What to build:** The layout layer of the pipeline. A config describes the table: the geographic extent, the screen's display size (55" placeholder), its orientation (north-south or east-west) and its position over the basin. The pipeline derives the scale from those values, making the valley floor fill the screen, and from that works out the table's overall footprint. It renders a top-down preview image (hillshade and imagery) with the screen rectangle, the table outline, the range names and the scale/footprint numbers drawn on.

The first real outputs are one preview per main questionnaire option: north-south with the Santa Ritas, and east-west without them with the Catalinas at the head. These get attached to the team questionnaire so the professors can answer the extent and orientation questions from pictures.

**Blocked by:** 01 — Tracer test tile (reuses its reprojection and scaling), 02 — Source data.

**Status:** ready-for-agent

- [ ] Layout config: extent, screen size, orientation and screen position; scale and table footprint are derived, not entered
- [ ] Preview image shows hillshade or imagery, the screen rectangle, the table outline, range labels, scale and footprint in metres
- [ ] Two preview configs exist, one per questionnaire option, and render without edits
- [ ] The crest-to-crest distances quoted in the questionnaire (about 80 km N–S, 52 km E–W) are checked against the DEM, and the questionnaire is corrected if they're off
- [ ] Previews are referenced from the questionnaire
