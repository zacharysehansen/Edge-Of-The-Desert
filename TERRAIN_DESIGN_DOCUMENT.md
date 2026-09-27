# Terrain Design Document: Tucson Basin Diorama

## Purpose

A 3D-printed terrain diorama of the Tucson basin, built from USGS elevation data, for the land-planning table of The Edge of the Desert installation. The diorama surrounds a flat touchscreen and gives the screen a physical landscape to sit in.

## Concept

The table is a flat touchscreen framed by printed mountains.

- **The screen shows the valley floor.** This is where visitors place land-use pieces and see the effects on water, vegetation and wildlife. The land-use pieces go on the screen only, never on the printed terrain.
- **The diorama shows the ranges around it.** The printed terrain rises from the screen's edge into the surrounding mountains.
- **The map continues across the seam.** The screen and the diorama show the same geography at the same scale, so they read as one surface.
- **The table is viewed from all sides.** Visitors walk around a flat table.

## Coverage Area

The Tucson basin and the ranges that surround it:

| Direction | Range |
|---|---|
| North | Santa Catalina Mountains |
| East | Rincon Mountains |
| South | Santa Rita Mountains (inclusion undecided) |
| West | Tucson Mountains |

How much of the basin to show, whether the Santa Ritas are included, and which way the screen lies are all open. See [Open Questions](#open-questions).

Only the valley-facing slope of each range is printed, up to and slightly past its crest. Visitors standing at the table look inward, so the far side of a range is never seen. Each range ends in a clean vertical wall at the table's outer rim. A side of the table that stands against a wall can be cut right at the crest, which saves print time.

## The Seam Between Screen and Terrain

The seam is the hardest part of the design. It is handled at three levels.

1. **Shape.** A printed lip along the diorama's inner edge overhangs the screen's bezel and hides it. The lip's lowest surface sits at the height of the glass, and the terrain rises from it into the real ground slope.
2. **Colour.** The diorama is hand-painted, and colours come from real aerial imagery. The pipeline samples the imagery along the seam and produces exact colour targets (hex and CIELAB) for the painters.
3. **Light.** The screen emits light and the paint only reflects it, so paint alone can never match the screen in a dark room. The diorama gets directed lighting, and at the installation the screen's edge pixels are calibrated to match the painted terrain under the real lighting. Changing screen colours in software is far easier than repainting.

The display software also keeps a fixed band 2–3 cm wide along the screen's edge that always shows the base imagery and blends gradually into the interactive map. Land-use changes then never alter the colours at the seam.

## Data Sources

| Product | Source | Resolution |
|---|---|---|
| Elevation | USGS 3D Elevation Program (3DEP), 1/3 arc-second DEM | about 10 m |
| Aerial imagery | USDA NAIP, most recent Arizona year | 0.6 m, averaged to 5 m |

Higher-resolution elevation data (such as 1 m lidar) produces very large files and adds detail too small to print at this scale. The screen shows about 10 m per pixel at this scale, so imagery much finer than 5 m adds nothing visible. Credits and licences are listed in [DATA_SOURCES.md](DATA_SOURCES.md).

## Model Specifications

| Setting | Value | Notes |
|---|---|---|
| Screen | 55" touchscreen (placeholder) | Sets the scale: the valley floor fills the screen |
| Scale | About 1:30,000 to 1:41,000 | Derived from the screen size and the chosen extent |
| Table size | About 1.7 × 1.2 m to 2.1 × 1.4 m | Derived from the scale and the extent |
| Vertical exaggeration | 2.5x (test print will confirm) | Without it, relief is only a few centimetres |
| Base thickness | 5 mm | Tiles are glued to plywood, so the base can stay thin |
| Tile size | Set by printer bed (220–250 mm placeholder) | |
| Layer height | 0.1 to 0.15 mm | Keeps slopes smooth |
| Material | PLA, painted | No water touches the terrain, so no sealing is needed |

### Why Vertical Exaggeration Is Needed

Mount Lemmon rises about 2 km above the valley floor. At 1:30,000 that is under 7 cm at true scale, which makes the ranges look flat next to a 1.2 m screen. A 2x to 3x exaggeration makes them read clearly. The test tile uses 2.5x.

## Structure

- A wooden table frame holds the screen and carries its weight and ventilation.
- The printed tiles are glued onto 3–4 plywood boards, one per section. The sections bolt onto the frame, so the table comes apart for moving.
- Section boundaries follow ridgelines and washes, so the joints are hard to see.
- The pipeline produces a 1:1 cutting template for each plywood board, with the tile outlines marked for gluing.

## Workflow

The whole build is produced by one scripted pipeline (the `terrain` package). Every size and setting lives in a config file, so the full print package can be regenerated whenever an answer from the team changes a number.

1. **Fetch the source data.** `python -m terrain sources config/sources.toml` downloads the elevation data and imagery for the largest area the diorama could cover. Any extent the team picks is a crop of this data.
2. **Lay out the table.** Set the extent, the screen size and orientation, and the table outline. The pipeline derives the scale and renders a top-down preview.
3. **Build the ring.** The pipeline cuts the screen opening, forms the lip, trims each range to its valley-facing slope, and adds the vertical outer rim.
4. **Tile it.** The ring is split into bed-sized tiles grouped into transport sections, each exported as a watertight STL, plus the plywood templates, the tile count and an estimated print time.
5. **Hand off the seam.** The pipeline exports the screen area (aligned imagery for the display software) and the painters' colour sheet.
6. **Print.** Slice each STL in Cura or PrusaSlicer at a thin layer height and print it.
7. **Assemble and finish.** Glue the tiles to their plywood boards, sand the seams, prime and paint to the colour sheet. At the installation, calibrate the screen's edge pixels to the paint under the real lighting.

The work is tracked as tickets in `.scratch/display-design/issues/`.

## Test Print

Before the full build, print one test tile to check vertical exaggeration, layer height and paint:

```
python -m terrain fetch config/test_tile.toml
python -m terrain tile config/test_tile.toml
```

The current test tile covers about 6 × 6 km of the Santa Catalina front around the mouth of Sabino Canyon. It prints at 200 × 199 mm and 66.7 mm tall.

TouchTerrain (a free web app from Iowa State) is still handy for a quick look at other areas without running the pipeline.

## Possible Help

- Makerspaces or the University of Arizona architecture program, for fabrication help
- Local model railroad clubs (SASME, Tucson Garden Railway Society, HOBOE, Tucson N-Trak), for painting and scenery finishing

## Open Questions

These are with the team in [the questionnaire](to-questionnaire-terrain-diorama.md):

- The touchscreen model and size
- How much of the basin to show, and whether the Santa Ritas are included
- Which way the screen lies (north-south or east-west)
- The south edge, if the Santa Ritas are out
- Floor space, and whether one side of the table faces a wall
- Room lighting, and whether spotlights can be added
- Available printers and their bed sizes
- Deadline and budget

## Settled

- **Water contact:** none. The terrain stays dry and water is shown by the screen and the groundwater tank.
- **Paint style:** realistic, hand-painted from aerial imagery, matched to the screen at the seam.
