# 07 — Retire JS reimplementation — frontend calls the engine API

**What to build:** Replace the frontend's in-browser reimplementation of feature reconstruction and structural arithmetic with HTTP calls to the Python engine from ticket 06. Retire `frontend/catalog.js`, `frontend/structural.js`, and `scripts/phase3/check_catalog_parity.py` — the parity check existed only to guard a duplication that this ticket deletes. The frontend becomes a thin rendering layer: it sends slider state to the engine and receives the output contract. Add a minimal HTTP server (FastAPI/uvicorn or similar) that wraps the engine's `simulate()`. Update `check_frontend.mjs` to test the frontend against the API. Record the retirement reason in PROBLEMS.md as PHASE4.md §8 instructs.

**Blocked by:** 06 — Engine skeleton

**Status:** ready-for-agent

- [ ] `frontend/catalog.js` and `frontend/structural.js` are removed from the repo
- [ ] `scripts/phase3/check_catalog_parity.py` is removed and the reason recorded in PROBLEMS.md
- [ ] A minimal HTTP server wraps `engine.simulate()` and serves JSON
- [ ] The frontend fetches inference results from the local API instead of running ONNX in-browser
- [ ] `check_frontend.mjs` still passes, now testing against the API
- [ ] The frontend still loads and renders correctly in a browser
- [ ] ONNX models are still loaded by the engine, not duplicated
