# 06 — Engine skeleton — Python package with the provenance schema

**What to build:** A new `engine/` Python package that is the single source of truth for inference. Exposes a stateless core function `simulate(scenario, horizon, seed) -> trajectory` where each output at each timestep carries the output contract from PHASE4.md §8: `value`, `unit`, `display_min`, `display_max`, `band_low`, `band_high`, `provenance` (measured | modelled | declared | extrapolated), and `beyond_evidence` (bool). The engine wraps `onnxruntime` for the six shipped ONNX models, plus the Layer 2 structural arithmetic and Layer 3 mean-reversion integration currently split across `structural.js` and `slider_sensitivity.py`. A `python -m engine --dump` CLI confirms parity with the shipped frontend on a fixed scenario set.

**Blocked by:** 05 — Stock units

**Status:** ready-for-agent

- [ ] `engine/` is a proper Python package with `__init__.py` and importable modules
- [ ] `simulate()` accepts a scenario dict (slider deltas + month + duration) and returns a trajectory
- [ ] Each output at each timestep is a dict matching the §8 contract (value, unit, display_min, display_max, band_low, band_high, provenance, beyond_evidence)
- [ ] `provenance` is mandatory and set correctly for each output
- [ ] ONNX inference matches the shipped `frontend/models.js` results on a fixed scenario set
- [ ] Layer 2 and Layer 3 arithmetic match the shipped `structural.js` results
- [ ] `python -m engine --dump` prints the same fixed-scenario output as `node frontend/catalog.js --dump`
