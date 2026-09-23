# 09 — Conservation and boundedness gates

**What to build:** Two hard build gates that run as automated tests against the engine. (a) **Conservation**: water in = water out + Δstorage, to a stated tolerance (e.g. 0.1 AF), at every timestep, under the nested budget. A stock model that does not conserve mass is wrong in a way no regression ever could be. (b) **Boundedness**: no output diverges over a 300-year run under any admissible scenario. This turns PHASE3_PLAN.md §32's blow-up into a permanent test. Both gates run as `pytest` tests and exit nonzero on failure, suitable as CI build gates.

**Blocked by:** 08 — Stock-and-flow state model

**Status:** ready-for-agent

- [ ] Conservation test: for every timestep in a suite of scenarios, assert `inflow - outflow - Δstorage < tolerance`
- [ ] Tolerance is stated and justified (e.g. floating-point rounding, not physics)
- [ ] Boundedness test: a 300-year run under max-stress scenarios produces no output outside its declared display range
- [ ] Both tests run as pytest and exit nonzero on failure
- [ ] At least three scenario variants tested: business-as-usual, max-development, max-conservation
