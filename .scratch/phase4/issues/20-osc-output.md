# 20 — OSC output for sonification

**What to build:** The engine emits every output over Open Sound Control (OSC) at each tick, alongside the existing HTTP/WebSocket channels. Each OSC message carries the full output contract: value, unit, display range, provenance, and beyond_evidence flag. This is the lingua franca for the sonification team (TouchDesigner, Max/MSP, etc.) and lets them map values to sound with a declared full-swing range, tuned once rather than re-scaling every session. The OSC endpoint address and port are configurable.

**Blocked by:** 13 — Session state

**Status:** ready-for-agent

- [ ] The engine sends OSC messages at each tick for every output
- [ ] Each message includes: value, unit, display_min, display_max, band_low, band_high, provenance, beyond_evidence
- [ ] OSC address pattern is configurable (e.g. `/eotd/groundwater`, `/eotd/grace`)
- [ ] Target host and port are configurable via engine config
- [ ] A `python-osc` (or equivalent) dependency is added to `requirements.txt`
- [ ] A test verifies that OSC messages are emitted with the correct schema on each tick
