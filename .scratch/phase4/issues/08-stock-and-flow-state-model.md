# 08 — Stock-and-flow state model — aquifer and Mead as stocks

**What to build:** The engine gains real stock variables: aquifer storage (acre-feet), Mead pool storage, paved acres, population, and irrigated acres. Controls set flows: pumping rate, recharge rate, paving rate, population growth rate. The engine steps annually; Layer 1 is evaluated once per simulated year for that year's climate forcing, never iterated month-to-month. The existing `reversion_per_month` restoring forces (grace 0.0386, groundwater 0.0157, ndvi 0.2221, surface water 0.3511) are incorporated as the physical damping on each stock. The aquifer stock is bounded by the statutory 1,000 ft limit from A.A.C. R12-15-716(B)(2). This is the "spine" from PHASE4.md §4.2 course A.

**Blocked by:** 06 — Engine skeleton

**Status:** ready-for-agent

- [ ] Aquifer storage is a real state variable in acre-feet, with flows (pumping out, recharge in) and the measured damping rate
- [ ] Mead pool storage is a state variable with DCP shortage tiers driving flow reductions
- [ ] Paved acres, population, and irrigated acres are state variables with rates set by controls
- [ ] The engine steps annually; Layer 1 evaluates each year's climate once, not iteratively
- [ ] The aquifer stock respects the 1,000 ft statutory bound (Tucson AMA)
- [ ] Each stock's restoring force uses the measured `reversion_per_month` values
- [ ] A 100-year default scenario runs without divergence
