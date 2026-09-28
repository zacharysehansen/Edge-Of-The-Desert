"""Command line: `python -m terrain fetch|tile|sources|preview <config.toml>`."""

import argparse
import sys

from terrain.checks import watertight_problems
from terrain.config import TileConfig, load_config
from terrain.dem import fetch_patch, load_heightfield
from terrain.mesh import heightfield_to_mesh
from terrain.stl import write_stl
from terrain import sources
from terrain.layout import load_layout


def fetch(cfg: TileConfig) -> None:
    print(f"Fetching DEM patch for {cfg.name} from {cfg.dem_url}")
    try:
        fetch_patch(cfg.dem_url, cfg.dem_path, cfg.west, cfg.south, cfg.east, cfg.north)
    except Exception as e:
        sys.exit(f"Download failed: {e}\n"
                 "If this machine has no network access, run the same command from a terminal that does.")
    print(f"Saved {cfg.dem_path}")


def tile(cfg: TileConfig, config_path: str) -> None:
    if not cfg.dem_path.exists():
        sys.exit(f"No DEM at {cfg.dem_path}. Download it first with:\n"
                 f"    python -m terrain fetch {config_path}")
    hf = load_heightfield(cfg.dem_path, cfg.west, cfg.south, cfg.east, cfg.north, cfg.ground_spacing_m)
    mesh = heightfield_to_mesh(hf.elevation_m, cfg.mesh_resolution_mm, cfg.scale,
                               cfg.vertical_exaggeration, cfg.base_thickness_mm)
    problems = watertight_problems(mesh)
    if problems:
        sys.exit("Mesh failed checks:\n  " + "\n  ".join(problems))
    write_stl(mesh, cfg.stl_path, header=f"{cfg.name} 1:{cfg.scale:g} x{cfg.vertical_exaggeration:g}")

    size = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    elev = hf.elevation_m
    print(f"Wrote {cfg.stl_path}  ({len(mesh.faces):,} triangles, watertight)")
    print(f"  ground:    {hf.width_m / 1000:.2f} x {hf.height_m / 1000:.2f} km, "
          f"elevation {elev.min():.0f}-{elev.max():.0f} m")
    print(f"  footprint: {size[0]:.1f} x {size[1]:.1f} mm at 1:{cfg.scale:g}")
    print(f"  height:    {size[2]:.1f} mm ({cfg.base_thickness_mm:g} mm base + "
          f"{size[2] - cfg.base_thickness_mm:.1f} mm relief at {cfg.vertical_exaggeration:g}x)")
    fits = size[0] <= cfg.bed_mm and size[1] <= cfg.bed_mm
    print(f"  bed:       {'fits' if fits else 'DOES NOT FIT'} a {cfg.bed_mm:g} mm bed")


def fetch_sources(config_path: str, only: str | None) -> None:
    cfg = sources.load_sources_config(config_path)
    try:
        if only in (None, "dem"):
            print("DEM tiles:")
            sources.merge_dem(cfg, sources.fetch_dem_tiles(cfg))
            print(f"  merged -> {cfg.dem_path}")
        if only in (None, "imagery"):
            print("Imagery:")
            sources.merge_imagery(cfg, sources.fetch_imagery_items(cfg))
            print(f"  merged -> {cfg.imagery_path}")
    except (RuntimeError, OSError) as e:  # GDAL errors, and network errors (URLError is an OSError)
        sys.exit(f"Download failed: {e}\n"
                 "Anything already saved is kept; re-run the same command to continue.")
    print("Checking coverage:")
    try:
        lines = sources.verify(cfg)
    except ValueError as e:
        sys.exit(f"  {e}")
    for line in lines:
        print(f"  {line}")


PREVIEW_DIR = "docs/previews"


def preview(config_path: str, sources_path: str) -> None:
    from pathlib import Path

    from terrain.preview import render_preview

    layout = load_layout(config_path)
    src = sources.load_sources_config(sources_path)
    if not src.dem_path.exists():
        sys.exit(f"No merged DEM at {src.dem_path}. Run: python -m terrain sources {sources_path}")
    problems = layout.problems()
    if problems:
        sys.exit("Layout problem:\n  " + "\n  ".join(problems))
    imagery = src.imagery_path if src.imagery_path.exists() else None
    out = Path(PREVIEW_DIR) / f"{layout.name}.png"
    render_preview(layout, src.dem_path, imagery, out)
    print(f"{layout.title}")
    for line in layout.summary():
        print(f"  {line}")
    print(f"  preview -> {out}" + ("" if imagery else " (shaded relief only: imagery not merged yet)"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m terrain")
    parser.add_argument("command", choices=["fetch", "tile", "sources", "preview"])
    parser.add_argument("config")
    parser.add_argument("--only", choices=["dem", "imagery"],
                        help="sources: fetch just one product")
    parser.add_argument("--sources", default="config/sources.toml",
                        help="preview: the sources config naming the merged DEM and imagery")
    args = parser.parse_args()
    if args.command == "sources":
        fetch_sources(args.config, args.only)
        return
    if args.command == "preview":
        preview(args.config, args.sources)
        return
    cfg = load_config(args.config)
    if args.command == "fetch":
        fetch(cfg)
    else:
        tile(cfg, args.config)


if __name__ == "__main__":
    main()
