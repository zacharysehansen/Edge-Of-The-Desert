# Data sources

Source data for the terrain diorama. Downloaded files live under `data/` and are not committed; `python -m terrain sources config/sources.toml` fetches them again.

| Product | Source | Resolution used | Licence |
|---|---|---|---|
| Elevation | USGS 3D Elevation Program (3DEP), 1/3 arc-second DEM, tiles n32w111, n32w112, n33w111, n33w112 | about 10 m, resampled to 10 m in UTM 12N | Public domain (U.S. Government work) |
| Aerial imagery | USDA National Agriculture Imagery Program (NAIP), most recent Arizona year, via Microsoft Planetary Computer | 0.6 m native, averaged to 5 m in UTM 12N | Public domain (U.S. Government work) |

Suggested credit: "Elevation data: U.S. Geological Survey, 3D Elevation Program. Imagery: USDA Farm Production and Conservation Business Center, National Agriculture Imagery Program."
