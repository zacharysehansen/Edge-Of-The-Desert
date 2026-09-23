# 12 — Perceptual gate — the Phase 4 ship gate

**What to build:** A test that asserts: a single token placed anywhere in the domain moves at least two outputs by ≥2% of that output's declared display range within 5 simulated years. Every token type must pass. This replaces the Phase 3 §7 criterion ("≥5 score points across the lever's full range at 12 months") as the ship gate. The old criterion is kept as a reported diagnostic but is no longer a gate. Runs as a pytest test against the engine.

**Blocked by:** 11 — Token vocabulary, 09 — Conservation and boundedness gates

**Status:** ready-for-agent

- [ ] For each of the six token types, place one token at every parcel and run a 5-year simulation
- [ ] Assert that at least two outputs move by ≥2% of their declared `display_max - display_min`
- [ ] Every token type passes the gate
- [ ] The test reports which outputs moved and by how much for each token type
- [ ] The old §7 criterion is still computed and reported but does not gate
- [ ] Test runs as pytest and exits nonzero on failure
