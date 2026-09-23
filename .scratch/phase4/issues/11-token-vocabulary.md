# 11 — Token vocabulary — engine contract for `place(land_use, parcel)`

**What to build:** The engine accepts `place(token_type, parcel_id)` as its primary input contract, where `token_type` is one of the six tokens from LAND_USE_TOKENS.md: `subdivision`, `dense_infill`, `farm`, `retire_farm`, `recharge_basin`, `riparian_restoration`. Each token placement is converted to the appropriate set of flow deltas using the token's physical definition: a subdivision adds pumping, paving, and population; a recharge basin adds inflow to the aquifer stock; etc. The conversion uses the open geometry parameters (`parcels_per_domain`, `acres_per_token`, `token_intensity`) which are carried as explicit parameters until the installation team answers them. `impervious_pct` becomes derived from the token set, not a parallel slider. The slider interface remains as a fallback for headless testing.

**Blocked by:** 03 — Local denominators, 08 — Stock-and-flow state model

**Status:** ready-for-agent

- [ ] Engine exposes `place(token_type, parcel_id)` and `remove(token_type, parcel_id)`
- [ ] All six token types are implemented with their physical conversions to flow deltas
- [ ] Every token has its opposing partner: subdivision↔dense_infill, farm↔retire_farm, recharge_basin↔any pumping
- [ ] Geometry parameters (`parcels_per_domain`, `acres_per_token`) are configurable, not hard-coded
- [ ] `impervious_pct` is computed from the current token set, not independently controlled
- [ ] Slider fallback remains functional for headless testing
- [ ] Token placement inside the local domain affects the local tier; outside does not affect local
