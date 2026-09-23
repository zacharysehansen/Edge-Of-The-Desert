# 19 — Offline vendoring — no CDN dependency

**What to build:** Vendor `onnxruntime-web` and any other CDN-loaded dependency locally so the app boots from a cold, disconnected machine. The importmap in `frontend/index.html` currently points to `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/` — replace it with a local path. Add a `package.json` (or a vendored copy) that pins the version. If the frontend still runs ONNX in-browser (i.e. ticket 07 is not yet merged), vendor the WASM files too. A test confirms no external network fetch is required to load the app.

**Blocked by:** 07 — Retire JS reimplementation

**Status:** ready-for-agent

- [ ] `onnxruntime-web` is vendored into the repo (or, if ticket 07 retired browser ONNX, only the engine's Python `onnxruntime` is needed)
- [ ] The importmap in `index.html` points to local files, not a CDN
- [ ] No external network fetch is required to boot the app
- [ ] A test starts a local server on a blocked network interface and confirms the app loads
- [ ] The vendored version is pinned and documented
