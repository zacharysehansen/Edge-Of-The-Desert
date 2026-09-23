# 15 — Climate futures and BAU reference

**What to build:** The engine accepts a climate-future selector with three modes: `hold` (repeat the current climate indefinitely), `observed_trend` (temperature as a mean shift from the nClimDiv 1895–2026 fitted trend; precipitation as widening variance on a flat mean), and `hot_dry` (an aggressive warming scenario). Variability is stochastic, block-resampled from the observed record and warm-shifted, with a fixed seed per session for reproducibility. Each output card shows deviation from business-as-usual, sourced from the OEO population projection and CRSS depletion schedules parsed in ticket 14. The engine emits one seeded trace as the primary channel and an ensemble band alongside it, exposed in the API. The visual layer decides whether to render the band.

**Blocked by:** 10 — Equilibrium emulator, 14 — Parse boundary conditions

**Status:** ready-for-agent

- [ ] Engine accepts a `climate_future` parameter: `hold`, `observed_trend`, `hot_dry`
- [ ] `observed_trend` applies a temperature mean shift from the nClimDiv trend and widens precipitation variance
- [ ] `hot_dry` applies an aggressive warming scenario with reduced precipitation
- [ ] Variability is block-resampled from the observed record with a fixed seed per session
- [ ] BAU reference trajectory is sourced from OEO (population) and CRSS (Mead/CAP)
- [ ] Each output includes a `bau_value` field alongside the scenario value
- [ ] Ensemble band (e.g. 10th–90th percentile over N seeds) is computed and exposed in the API
- [ ] The same seed produces the same trace across runs
