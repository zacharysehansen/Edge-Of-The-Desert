# 04 — Nested budget accounting

**What to build:** When a lever acts inside the local domain, the regional figure is computed as `local + rest-of-region` rather than independently. This ensures that pumping booked locally is not also booked regionally (the double-counting problem from PHASE3_PLAN.md §11.3). A new test mode `slider_sensitivity.py --mode no-double-count-nested` asserts water conservation: total regional pumping equals local pumping plus rest-of-region pumping, to machine precision, for every lever and every month.

**Blocked by:** 03 — Local denominators in Layer 2

**Status:** ready-for-agent

- [ ] Regional Layer 2 output is computed as `local_share + rest_of_region_share`, not independently
- [ ] Local pumping is subtracted from the regional total before computing rest-of-region
- [ ] `--mode no-double-count-nested` test passes: total = local + rest for all levers, all months
- [ ] The existing `--mode no-double-count` (learned outputs unchanged by human levers) still passes
- [ ] No change to any shipped model or regional constant
