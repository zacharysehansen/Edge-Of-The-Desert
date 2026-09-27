# 01 — Tracer: one printable test tile from a config

**What to build:** The thinnest end-to-end path from elevation data to a printable file. A small config names a patch of the Santa Catalina front (a box in lat/lon), a horizontal scale, a vertical exaggeration and a base thickness. One command fetches the 1/3 arc-second 3DEP DEM covering that patch, reprojects it to UTM 12N, crops it, applies scale and exaggeration, and writes a single watertight STL with a flat base. This tile doubles as the physical test print for exaggeration, layer height and paint, which the terrain design doc called for before committing to the full build.

The reprojection, scaling and meshing written here are the core the later tickets reuse, so keep them as functions that take the config, not a one-off script.

**Blocked by:** None — can start immediately.

**Status:** done — real tile built: 200.0 x 199.0 mm, 66.7 mm tall (5 mm base + 61.7 mm relief), elevation 782–1523 m, 323,188 triangles, watertight

- [x] A config file holds patch bounds, scale (e.g. 1:30,000), vertical exaggeration (default 2.5x) and base thickness; nothing is hard-coded
- [x] One command produces the STL from the config
- [x] The STL is watertight and manifold (checked automatically by edge pairing and signed volume in numpy; trimesh wasn't installable offline)
- [x] Its footprint matches the configured scale within 1 mm, and its relief matches real elevation range × exaggeration ÷ scale within 1 mm
- [x] The tile fits a 220 mm printer bed at the default settings (200 x 199 mm on the real DEM)
- [x] If the DEM fetch can't run in the sandbox (curl is blocked here), the command prints the exact download for the user to run and picks up the file from disk
- [x] New dependencies are added to requirements.txt
