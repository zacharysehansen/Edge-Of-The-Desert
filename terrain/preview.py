"""Top-down preview of a layout: shaded relief (or imagery), screen, table outline, labels."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from osgeo import gdal

from terrain.dem import UTM_12N
from terrain.layout import Layout, to_utm

gdal.UseExceptions()
PREVIEW_RES_M = 40.0
MARGIN_M = 4000.0  # show a little of what lies outside the table

# Desert hypsometric tint for the relief-only preview: pale valley floor to dark mountain tops.
DESERT = LinearSegmentedColormap.from_list(
    "desert", ["#e9dcc0", "#d4bf94", "#b39570", "#8a7a5c", "#6f7a5a", "#dcdcd6"])

RANGES = [  # label positions (lon, lat), placed on each range's valley-facing side
    ("Santa Catalina Mts", -110.80, 32.40),
    ("Rincon Mts", -110.60, 32.17),
    ("Santa Rita Mts", -110.85, 31.74),
    ("Tucson Mts", -111.10, 32.27),
    ("Tucson", -110.97, 32.22),
]


def _read_grid(path: Path, bounds, res: float, bands: list[int]) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    ds = gdal.Warp("", str(path), format="MEM", dstSRS=UTM_12N, outputBounds=bounds,
                   xRes=res, yRes=res, resampleAlg="average")
    data = np.stack([ds.GetRasterBand(b).ReadAsArray() for b in bands]).astype(float)
    ds = None
    return data


def render_preview(layout: Layout, dem_path: Path, imagery_path: Path | None, out_png: Path) -> None:
    tx0, ty0, tx1, ty1 = layout.table_utm
    bounds = (tx0 - MARGIN_M, ty0 - MARGIN_M, tx1 + MARGIN_M, ty1 + MARGIN_M)
    elev = _read_grid(dem_path, bounds, PREVIEW_RES_M, [1])[0]
    shade = LightSource(azdeg=315, altdeg=40)
    rgb = _read_grid(imagery_path, bounds, PREVIEW_RES_M, [1, 2, 3]) if imagery_path else None
    if rgb is not None:
        base = np.clip(np.moveaxis(rgb, 0, -1) / 255.0, 0, 1)
        image = shade.shade_rgb(base, elev, vert_exag=2.0, dx=PREVIEW_RES_M, dy=PREVIEW_RES_M, blend_mode="soft")
    else:
        image = shade.shade(elev, cmap=DESERT, vert_exag=2.0, dx=PREVIEW_RES_M, dy=PREVIEW_RES_M,
                            blend_mode="soft", vmin=650, vmax=2900)

    # Axes in km from the table's south-west corner.
    def km(x, y):
        return (x - tx0) / 1000, (y - ty0) / 1000

    left, bottom = km(bounds[0], bounds[1])
    right, top = km(bounds[2], bounds[3])
    fig, ax = plt.subplots(figsize=(9, 9 * (top - bottom) / (right - left) + 1.2))
    ax.imshow(image, extent=[left, right, bottom, top], interpolation="bilinear")

    # Dim everything outside the table.
    tw, th = (tx1 - tx0) / 1000, (ty1 - ty0) / 1000
    for rect in [(left, bottom, -left, top - bottom), (tw, bottom, right - tw, top - bottom),
                 (0, bottom, tw, -bottom), (0, th, tw, top - th)]:
        ax.add_patch(Rectangle(rect[:2], rect[2], rect[3], color="black", alpha=0.45, lw=0))
    ax.add_patch(Rectangle((0, 0), tw, th, fill=False, ec="white", lw=2.5, label="table edge"))
    sx0, sy0, sx1, sy1 = layout.screen_utm
    (a, b), (c, d) = km(sx0, sy0), km(sx1, sy1)
    ax.add_patch(Rectangle((a, b), c - a, d - b, fill=False, ec="#ffcc00", lw=2.5, label="screen"))

    for label, lon, lat in RANGES:
        x, y = km(*to_utm(lon, lat))
        if left < x < right and bottom < y < top:
            ax.text(x, y, label, color="white", fontsize=10, ha="center", va="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0))

    ax.annotate("N", xy=(0.97, 0.95), xytext=(0.97, 0.88), xycoords="axes fraction", color="white",
                ha="center", fontsize=12, weight="bold", arrowprops=dict(arrowstyle="-|>", color="white"))
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xlabel("km")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.set_title(layout.title + "\n" + " · ".join(layout.summary()), fontsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
