# 16 — Evidence horizon and `beyond_evidence` flag

**What to build:** Each output carries a declared evidence horizon: the length of the training record that supports it (NDVI ~24 years, streamflow ~46 years, wildfire ~40 years, nClimDiv ~131 years; the aquifer stock is a mass balance and is valid furthest). Uncertainty bands widen with a stated rule past the evidence horizon (e.g. linear widening proportional to years past the horizon). Past 3× an output's evidence horizon, the engine sets `beyond_evidence: true` in the output contract. The frontend and API surface this flag so the installation team can make it visible or audible.

**Blocked by:** 13 — Session state, 15 — Climate futures

**Status:** ready-for-agent

- [ ] Each output has a declared `evidence_horizon_years` in the engine's output metadata
- [ ] `band_low` and `band_high` widen past the evidence horizon with a stated, documented rule
- [ ] `beyond_evidence` is set to `true` in the output contract when simulated time exceeds 3× the evidence horizon
- [ ] The widening rule is documented (formula, not just code)
- [ ] The frontend surfaces the `beyond_evidence` flag on output cards
- [ ] A test verifies that bands at 2× the evidence horizon are wider than at 1×, and at 3× the flag flips
