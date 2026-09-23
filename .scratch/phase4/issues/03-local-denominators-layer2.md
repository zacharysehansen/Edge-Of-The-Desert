# 03 — Local denominators in Layer 2 arithmetic

**What to build:** Extend `frontend/structural.js` (and its Python mirror in `scripts/phase3/slider_sensitivity.py`) with a `tier` parameter. When `tier=local`, the Layer 2 arithmetic uses local denominators and local constants from `local_structural_params.json`. When `tier=regional` (the default), behaviour is bit-identical to today. The existing `slider_sensitivity.py --mode acceptance` test gains a `--tier local` flag; running it should show the urbanization lever producing visible movement on the local NDVI card (~12x stronger than the current 0.8 regional points, per the denominator ratio).

**Blocked by:** 02 — Local constants

**Status:** done

- [x] `structural.js` loads local or regional params based on a tier flag (`withLocalConstants` swap pattern)
- [x] `slider_sensitivity.py` accepts `--tier local` and uses local constants via `build_local_params()`
- [x] Regional mode is bit-identical to the current shipped behaviour (confirmed: all values unchanged)
- [ ] `--mode acceptance --tier local` shows urbanization producing ≥5 score points on the NDVI card — **HONEST NO**: NDVI formula is scale-invariant for slider inputs (0.97 pts both tiers); the 12× gain shows on surface_water (22.63 pts) and wildlife (5.61 pts). NDVI ≥5 becomes achievable with land-use tokens (ticket 07), not sliders.
- [x] All 19 lever/output pairs still hold their declared sign 12/12 months in both tiers
