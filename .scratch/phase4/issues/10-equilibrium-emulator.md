# 10 — Equilibrium-response emulator — climate term for the long horizon

**What to build:** Sweep the six shipped ONNX models over a grid of climate states offline (temperature × precipitation × drought, at representative seasonal points). Fit a smooth response surface for each model. The engine's stock-and-flow rollout uses this surface as "what a year of this climate looks like" instead of iterating the residual models month-to-month (which diverges — PHASE4.md §4.1). The engine relaxes toward the emulated equilibrium at the measured `reversion_per_month` rate. This is course C from §4.2. The boundedness gate from ticket 09 must still pass with the emulator in place.

**Blocked by:** 08 — Stock-and-flow state model

**Status:** ready-for-agent

- [ ] A sweep script evaluates each ONNX model over a climate grid and saves the results
- [ ] A smooth surface (polynomial, spline, or GP) is fitted per model to the sweep results
- [ ] The engine uses the emulated surface for annual climate forcing instead of iterating residual models
- [ ] Relaxation toward equilibrium uses the measured `reversion_per_month` for each output
- [ ] Boundedness gate (ticket 09) passes with the emulator active
- [ ] The emulator's R² against the ONNX models on the sweep grid is reported and >0.9
