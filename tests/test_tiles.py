"""Ticket 06: sections, tiles and plywood templates, on the synthetic basin (no network)."""

import re
from dataclasses import replace

import numpy as np
import pytest

from terrain.layout import Trim
from terrain.package import write_plywood_templates, write_tiles
from terrain.ring import Ring
from terrain.tiles import Sections, TilePlan, TileSettings
from test_ring import BUILD, LAYOUT
from test_trim import basin_dem  # noqa: F401  (fixture)

SETTINGS = TileSettings(tile_mm=210.0, max_height_mm=250.0, min_tile_share=0.10, section_count=4,
                        door_mm=800.0, max_length_mm=1600.0, wall_mm=1.2, infill=0.15, flow_mm3_per_s=8.0)


@pytest.fixture(scope="module")
def plan(basin_dem):  # noqa: F811
    ring = Ring(replace(LAYOUT, trim=Trim(margin_m=1000.0)), replace(BUILD, mesh_resolution_mm=4.0), basin_dem)
    sections = Sections(ring, SETTINGS)
    return TilePlan(ring, sections, SETTINGS)


def test_sections_fit_through_the_door(plan):
    s = plan.sections
    assert s.count >= SETTINGS.section_count
    for k in range(1, s.count + 1):
        x0, y0, x1, y1 = s._bbox_mm(s.labels, k)
        short, long_ = sorted((x1 - x0, y1 - y0))
        assert short <= SETTINGS.door_mm + s.step and long_ <= SETTINGS.max_length_mm + s.step


def test_every_printed_point_belongs_to_exactly_one_tile(plan):
    r = plan.ring
    xs = np.arange(1.0, r.table[2], 3.0)
    ys = np.arange(1.0, r.table[3], 3.0)
    owner = plan.owner_at(xs, ys)
    covered = np.zeros(owner.shape, int)
    for t in plan.tiles:
        inside = ((xs >= t.square[0]) & (xs < t.square[2]))[None, :] & ((ys >= t.square[1]) & (ys < t.square[3]))[:, None]
        covered += inside & (owner == t.section)
    assert covered.max() == 1
    assert ((owner > 0) == (covered == 1)).all()


def test_neighbouring_tiles_share_exact_edges(plan):
    by_square = {t.square: t for t in plan.tiles}
    pairs = [(a, by_square[(a.square[2], a.square[1], a.square[2] + plan.size, a.square[3])])
             for a in plan.tiles if (a.square[2], a.square[1], a.square[2] + plan.size, a.square[3]) in by_square]
    assert pairs
    for a, b in pairs[:4]:
        ma, mb = plan.mesh(a), plan.mesh(b)
        line = a.square[2]
        used_a = ma.vertices[np.unique(ma.faces)]
        on_a = {tuple(v) for v in np.round(used_a[np.isclose(used_a[:, 0], line)][:, 1:], 9)}
        on_b = {tuple(v) for v in np.round(mb.vertices[np.isclose(mb.vertices[:, 0], line)][:, 1:], 9)}
        assert on_a <= on_b  # every vertex A puts on the shared line, B has too, at the same height


def test_tiles_are_printable_and_listed(plan, tmp_path):
    rows = write_tiles(plan, tmp_path, None, log=lambda *_: None)  # raises if any tile fails its checks
    assert len(rows) == len(plan.tiles)
    assert all(r["width_mm"] <= SETTINGS.tile_mm and r["depth_mm"] <= SETTINGS.tile_mm for r in rows)
    assert all(r["print_hours"] > 0 for r in rows)
    assert len(list((tmp_path / "tiles").glob("*.stl"))) == len(plan.tiles)


def test_plywood_templates_are_one_to_one(plan, tmp_path):
    paths = write_plywood_templates(plan, tmp_path, LAYOUT.scale)
    assert len(paths) == len({t.section for t in plan.tiles})
    for p in paths:
        head = p.read_text()[:2000]
        width_pt = float(re.search(r'width="([\d.]+)pt"', head).group(1))
        k = int(p.stem[1:])
        # The section as printed (after slivers join their square's main tile), on the template's 4 mm grid.
        xs = np.arange(2.0, plan.ring.table[2], 4.0)
        ys = np.arange(2.0, plan.ring.table[3], 4.0)
        _, cols = np.nonzero(plan.owner_at(xs, ys) == k)
        width = xs[cols].max() - xs[cols].min()
        # The drawing is that width plus two 25 mm margins, at true size (72 pt per inch).
        assert width_pt * 25.4 / 72 == pytest.approx(width + 50, abs=1.0)
