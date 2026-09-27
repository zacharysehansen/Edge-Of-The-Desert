"""Tile config: everything the pipeline needs, loaded from one TOML file."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TileConfig:
    name: str
    west: float
    east: float
    south: float
    north: float
    scale: float
    vertical_exaggeration: float
    base_thickness_mm: float
    mesh_resolution_mm: float
    bed_mm: float
    dem_url: str
    dem_path: Path
    output_dir: Path

    @property
    def ground_spacing_m(self) -> float:
        """Distance on the ground between neighbouring height samples."""
        return self.mesh_resolution_mm * self.scale / 1000.0

    @property
    def stl_path(self) -> Path:
        return self.output_dir / f"{self.name}.stl"


def load_config(path: str | Path) -> TileConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    patch, prn, src, out = raw["patch"], raw["print"], raw["source"], raw["output"]
    if not (patch["west"] < patch["east"] and patch["south"] < patch["north"]):
        raise ValueError("patch bounds must satisfy west < east and south < north")
    return TileConfig(
        name=patch["name"],
        west=patch["west"],
        east=patch["east"],
        south=patch["south"],
        north=patch["north"],
        scale=prn["scale"],
        vertical_exaggeration=prn["vertical_exaggeration"],
        base_thickness_mm=prn["base_thickness_mm"],
        mesh_resolution_mm=prn["mesh_resolution_mm"],
        bed_mm=prn["bed_mm"],
        dem_url=src["dem_url"],
        dem_path=Path(src["dem_path"]),
        output_dir=Path(out["dir"]),
    )
