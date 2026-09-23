# 05 — Output units — stocks and levels replace anomalies

**What to build:** Convert the four anomaly-based outputs (groundwater depth anomaly, GRACE storage anomaly, surface-water discharge log anomaly, wildlife BBS anomaly) to physical stock units: feet below land surface, acre-feet in storage, cubic feet per second, and abundance index. Update `scripts/phase3/generate_stats.py` to write `display_min` and `display_max` in stock units alongside the existing anomaly statistics. Update the frontend output cards to show the stock-unit values as the primary display, with the anomaly available as a derived view (tooltip or toggle). This is the output-unit half of milestone M2.

**Blocked by:** None — can start immediately.

**Status:** done

- [x] Each anomaly output has a documented conversion to stock units (baseline + anomaly × scale)
- [x] `generate_stats.py` writes `display_min`, `display_max`, and `unit` in stock units for each output
- [x] Frontend cards show stock-unit values as the primary number (e.g. "116.9 ft below land surface")
- [x] Anomaly view is still accessible as secondary line (score/delta/anomaly row below the stock value)
- [x] The conversions are sourced: baseline depth from the well record, storage from the calibration, discharge from the streamflow record
