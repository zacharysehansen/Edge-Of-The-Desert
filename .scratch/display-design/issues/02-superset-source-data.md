# 02 — Source data for the largest possible area

**What to build:** Fetch and assemble every piece of source data the terrain could need, over a superset box covering all questionnaire options: the Santa Catalinas, Rincons, Tucson Mountains and Santa Ritas, with a margin on every side. The result is one merged DEM (1/3 arc-second 3DEP) and one merged aerial imagery mosaic (NAIP, the most recent Arizona year) for that box, both in UTM 12N. Whichever extent the team picks later is then just a crop of data already on disk.

The sandbox can't reach the download servers with curl, so this ships as a script the user runs from their own terminal. It must resume cleanly if interrupted, and skip files already downloaded.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] The superset box is defined once, in the shared config, and covers all four ranges plus a margin of at least 5 km
- [ ] One user-run script downloads all DEM tiles and imagery for the box, skipping files already present
- [ ] DEM tiles are merged into one raster and imagery into one mosaic, both reprojected to UTM 12N
- [ ] A verification step reports coverage (no nodata holes inside the box), resolution and total size on disk
- [ ] Downloads land under the gitignored data directory; nothing large is committed
- [ ] Data sources and licences are recorded next to the data, like the existing data credits
