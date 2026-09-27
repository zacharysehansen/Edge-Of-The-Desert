# Terrain Design Document: Tucson Basin Model

## Purpose

A 3D printed terrain model of Tucson and the surrounding mountain ranges, built from USGS elevation data, for a multimedia art installation about the water table.

## Coverage Area

The model should include the Tucson basin and the four ranges that surround it:

| Direction | Range |
|---|---|
| North | Santa Catalina Mountains |
| East | Rincon Mountains |
| South | Santa Rita Mountains |
| West | Tucson Mountains |

The area is roughly 70 km across.

## Data Source

- **Source:** USGS National Map Downloader (apps.nationalmap.gov/downloader)
- **Product:** Elevation Products (3DEP)
- **Resolution:** 1/3 arc-second DEM (about 10 m)
- **Reason:** Higher resolution data (such as 1 m lidar) creates very large files and adds detail too small to print at this scale.

## Model Specifications

| Setting | Value | Notes |
|---|---|---|
| Model width | About 1 m (to be confirmed) | Depends on installation space |
| Vertical exaggeration | 2x to 3x | Without it, relief is only about 3 cm at 1 m wide |
| Base thickness | To be decided | Thicker base makes tiles sturdier and easier to join |
| Tile size | 20 to 25 cm | Match to printer bed size |
| Layer height | Thin (0.1 to 0.15 mm) | Keeps slopes smooth |

### Why Vertical Exaggeration Is Needed

Mount Lemmon rises about 2 km above the valley floor, but the area is about 70 km wide. At true scale on a 1 m model, the mountains would only stand about 3 cm tall and would look flat. A 2x to 3x exaggeration makes the ranges clearly readable.

## Workflow

### 1. Download Elevation Data

1. Open the USGS National Map Downloader.
2. Draw a box around the coverage area.
3. Select Elevation Products (3DEP), 1/3 arc-second DEM.
4. Download all tiles that cover the area.

### 2. Prepare Data in QGIS

1. Install QGIS (free).
2. Load the DEM tiles.
3. Merge the tiles into one raster if there is more than one.
4. Crop the raster to the exact coverage area.

### 3. Convert to 3D Model

1. Install the DEMto3D plugin in QGIS.
2. Set the model's physical size.
3. Set the base thickness.
4. Set the vertical exaggeration.
5. Use the plugin's tiling option to split the model into printable tiles.
6. Export each tile as an STL file.

### 4. Print

1. Slice each STL file in Cura or PrusaSlicer.
2. Use a thin layer height.
3. Print each tile.

### 5. Assemble and Finish

1. Glue the tiles together in a grid.
2. Sand the seams and surfaces.
3. Prime the model.
4. Paint.
5. If the installation involves real water, seal the model with epoxy or a waterproof primer, since standard PLA prints are slightly porous.

## Quick Test Option

Before committing to the full build, make a small test print using **TouchTerrain** (free web app from Iowa State). It lets you pick an area on a map and download printable STL tiles directly, without using QGIS. This is useful for testing exaggeration and layer height settings.

## Possible Help

- Makerspaces or the University of Arizona architecture program for fabrication help
- Local model railroad clubs (SASME, Tucson Garden Railway Society, HOBOE, Tucson N-Trak) for painting and scenery finishing

## Open Questions

- Final model dimensions
- Exact vertical exaggeration value
- Whether the model will be in contact with water
- Whether water table data or other layers will be shown on or around the model
- Paint style (realistic, abstract, or data-driven color)
