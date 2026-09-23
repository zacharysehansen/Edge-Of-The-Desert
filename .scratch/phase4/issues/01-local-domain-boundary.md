# 01 — Local domain boundary — resolve the HUC8 mask

**What to build:** A Python module analogous to `scripts/phase1/region.py` that defines the Tucson local tier. It clips the four HUC8 watersheds (15050301 Upper Santa Cruz, 15050302 Rillito, 15050303 Lower Santa Cruz, 15050304 Brawley Wash) to Pima County using `data/raw/wbd/WBD_15_HU2_GDB.gdb` and the TIGER county shapefile. Exposes the local boundary as a GeoDataFrame, a bounding box, and the same `filter_points()` / `clip_raster()` helpers that `region.py` provides. Writes a verification output confirming the domain is ~9,124 km² / ~3,523 mi² / ~8.1% of the eight-county region.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Module loads the WBD GDB already on disk and selects the four HUC8 codes
- [ ] Clips to Pima County (FIPS 04019) using the TIGER county file
- [ ] Exposes `LOCAL_BOUNDARY`, `LOCAL_BBOX`, `filter_points()`, `clip_raster()` with the same interface as `region.py`
- [ ] Writes or prints a verification summary: total area in km² and mi², percentage of the eight-county region
- [ ] Area is within 5% of the 9,124 km² target stated in PHASE4.md §2
