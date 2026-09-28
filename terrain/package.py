"""Ticket 06 outputs: tile STLs, plywood templates, an overview map and a tile list."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from terrain.checks import watertight_problems
from terrain.stl import write_stl
from terrain.tiles import TilePlan

TEMPLATE_RES_MM = 1.0
SECTION_COLOURS = plt.cm.Set2.colors


def _mesh_size(mesh):
    used = mesh.vertices[np.unique(mesh.faces)]
    return used.min(axis=0), used.max(axis=0)


def write_tiles(plan: TilePlan, out_dir: Path, resolution_mm: float | None, log=print) -> list[dict]:
    """Write every tile's STL after checking it; returns one row per tile for the tile list."""
    rows = []
    bed = plan.settings.tile_mm
    for i, tile in enumerate(plan.tiles, 1):
        mesh = plan.mesh(tile, resolution_mm)
        problems = watertight_problems(mesh)
        lo, hi = _mesh_size(mesh)
        size = hi - lo
        if size[0] > bed + 1e-6 or size[1] > bed + 1e-6:
            problems.append(f"{size[0]:.1f} x {size[1]:.1f} mm does not fit the {bed:g} mm tile")
        if hi[2] > plan.settings.max_height_mm:
            problems.append(f"{hi[2]:.1f} mm tall, over the {plan.settings.max_height_mm:g} mm build height")
        if problems:
            raise ValueError(f"tile {tile.name}: " + "; ".join(problems))
        path = out_dir / "tiles" / f"{tile.name}.stl"
        write_stl(mesh, path, header=f"{tile.name} section {tile.section}")
        hours = plan.print_hours(mesh)
        rows.append({"tile": tile.name, "section": tile.section,
                     "x_mm": round(tile.square[0], 1), "y_mm": round(tile.square[1], 1),
                     "width_mm": round(size[0], 1), "depth_mm": round(size[1], 1), "height_mm": round(hi[2], 1),
                     "triangles": len(mesh.faces), "print_hours": round(hours, 1)})
        log(f"  [{i}/{len(plan.tiles)}] {tile.name}: {size[0]:.0f} x {size[1]:.0f} x {hi[2]:.0f} mm, "
            f"about {hours:.0f} h")
    return rows


def write_tile_list(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _grid(x0, y0, x1, y1, res):
    return np.arange(x0 + res / 2, x1, res), np.arange(y0 + res / 2, y1, res)


def write_overview(plan: TilePlan, path: Path, title: str) -> None:
    """Top-down map of the sections and tiles, for planning the build."""
    r = plan.ring
    xs, ys = _grid(0, 0, r.table[2], r.table[3], 4.0)
    owner = plan.owner_at(xs, ys)
    fig, ax = plt.subplots(figsize=(12, 12 * r.table[3] / r.table[2] + 1))
    colours = np.ones((*owner.shape, 3))
    for k in range(1, owner.max() + 1):
        colours[owner == k] = SECTION_COLOURS[(k - 1) % len(SECTION_COLOURS)][:3]
    ax.imshow(colours, origin="lower", extent=[0, r.table[2], 0, r.table[3]])
    for tile in plan.tiles:
        x0, y0, x1, y1 = tile.square
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="black", lw=0.5, alpha=0.5))
    for tile in plan.tiles:  # label each tile at the middle of its own piece
        tx, ty = _grid(*tile.square, 4.0)
        mine = plan.owner_at(tx, ty) == tile.section
        if mine.any():
            rows, cols = np.nonzero(mine)
            ax.text(tx[cols].mean(), ty[rows].mean(), tile.name, ha="center", va="center", fontsize=6)
    for k in range(1, owner.max() + 1):
        rows, cols = np.nonzero(owner == k)
        ax.text(xs[cols].mean(), ys[rows].mean(), f"S{k}", ha="center", va="center", fontsize=18,
                weight="bold", alpha=0.6)
    o = r.opening
    ax.add_patch(plt.Rectangle((o[0], o[1]), o[2] - o[0], o[3] - o[1], fill=False, ec="#c90", lw=1.5))
    ax.set_xlim(0, r.table[2])
    ax.set_ylim(0, r.table[3])
    ax.set_aspect("equal")
    ax.set_xlabel("mm")
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_plywood_templates(plan: TilePlan, out_dir: Path, scale: float) -> list[Path]:
    """One 1:1 SVG per section: plywood cut line, tile outlines and labels, and a scale bar.

    The plywood's inner edge is the screen's outer edge; the tiles overhang it onto
    the bezel and glass, so tile outlines reach past the plywood there.
    """
    r = plan.ring
    so = r.screen_outer
    paths = []
    margin = 25.0
    for k in sorted({t.section for t in plan.tiles}):
        tiles = [t for t in plan.tiles if t.section == k]
        # Section extent from its tiles' pieces.
        xs, ys = _grid(0, 0, r.table[2], r.table[3], 4.0)
        rows, cols = np.nonzero(plan.owner_at(xs, ys) == k)
        x0, x1 = xs[cols].min() - margin, xs[cols].max() + margin
        y0, y1 = ys[rows].min() - margin, ys[rows].max() + margin + 30  # room for the heading
        gx, gy = _grid(x0, y0, x1, y1, TEMPLATE_RES_MM)
        owner = plan.owner_at(gx, gy)
        in_screen = ((gx > so[0]) & (gx < so[2]))[None, :] & ((gy > so[1]) & (gy < so[3]))[:, None]
        board = (owner == k) & ~in_screen

        w_mm, h_mm = x1 - x0, y1 - y0
        fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.axis("off")
        ax.contour(gx, gy, board.astype(float), levels=[0.5], colors="black", linewidths=1.2)
        for t in tiles:
            inside = ((gx >= t.square[0]) & (gx < t.square[2]))[None, :] & ((gy >= t.square[1]) & (gy < t.square[3]))[:, None]
            piece = (owner == k) & inside
            if piece.any():
                ax.contour(gx, gy, piece.astype(float), levels=[0.5], colors="#1f5fbf", linewidths=0.6,
                           linestyles="dashed")
                rr, cc = np.nonzero(piece)
                ax.text(gx[cc].mean(), gy[rr].mean(), t.name, ha="center", va="center", fontsize=14, color="#1f5fbf")
        ax.text(x0 + 10, y1 - 12, f"Section S{k} plywood, 1:1. Solid line: cut. Dashed: tile outlines "
                f"(they overhang the screen edge). Model scale 1:{scale:,.0f}.", fontsize=12, va="top")
        ax.plot([x0 + 10, x0 + 110], [y1 - 28, y1 - 28], color="black", lw=2)
        ax.text(x0 + 115, y1 - 28, "100 mm: check before cutting", fontsize=10, va="center")
        path = out_dir / "plywood" / f"S{k}.svg"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="svg")
        plt.close(fig)
        paths.append(path)
    return paths
