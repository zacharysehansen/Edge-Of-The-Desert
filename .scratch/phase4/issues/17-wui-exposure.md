# 17 — WUI exposure geometry

**What to build:** A geometric readout that computes wildland-urban interface (WUI) exposure: acres and structures within 1 km of burnable fuel at the current token layout. This uses NLCD land-cover data (already in `data/raw/NLCD/`) to identify burnable fuel and the token set to identify development. Pure geometry — no fire-probability claim. Always responds to token placement. Ships regardless of the fire-ignition measurement outcome (ticket 18). This is the "how much of what you just built sits inside the fire-prone edge" sentence from PHASE4.md §10.

**Blocked by:** 03 — Local denominators (needs the local mask and NLCD clipping)

**Status:** ready-for-agent

- [ ] Burnable fuel is identified from NLCD land-cover classes (forest, shrub, grassland)
- [ ] A 1 km buffer around burnable fuel defines the WUI edge
- [ ] Placing a development token (subdivision, dense_infill) within the WUI reports exposed acres
- [ ] The readout updates whenever the token set changes
- [ ] Output is pure geometry: acres exposed and (if structure data is available) structure count
- [ ] No fire-probability coefficient is used — this is area-at-risk, not risk modeling
