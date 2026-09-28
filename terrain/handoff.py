"""Ticket 07: what the screen and the painters need to make the seam disappear.

Screen area: the imagery and DEM cropped exactly to the display rectangle, one
image pixel per panel pixel, with its bounds in UTM 12N and lon/lat, so the display
software draws the valley in line with the printed terrain.

Seam colours: the imagery sampled at regular spacing along the seam (and a row out
on the lower slopes), each averaged over a small window, given as hex and CIELAB
for the painters, with a numbered map of where each colour goes.
"""

import csv
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from osgeo import gdal
from pyproj import Transformer
from rasterio.windows import from_bounds

from terrain.dem import UTM_12N
from terrain.ring import Rect, Ring

gdal.UseExceptions()
MAP_RES_M = 40.0  # imagery resolution for the swatch sheet's location map
SWATCHES_PER_PAGE = (6, 8)  # columns, rows


@dataclass(frozen=True)
class HandoffSettings:
    screen_px: tuple[int, int]  # panel's native pixels, long side first
    spacing_mm: float
    window_mm: float
    lower_slope_mm: float
    fixed_band_mm: float
    feather_mm: float


def load_handoff_settings(path: str | Path) -> HandoffSettings:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    s, d = raw["seam_colors"], raw["display_software"]
    a, b = raw["screen"]["resolution_px"]
    return HandoffSettings(screen_px=(max(a, b), min(a, b)), spacing_mm=s["spacing_mm"],
                           window_mm=s["window_mm"], lower_slope_mm=s["lower_slope_mm"],
                           fixed_band_mm=d["fixed_band_mm"], feather_mm=d["feather_mm"])


# --- colour ------------------------------------------------------------------------

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB values in 0-1 to linear light."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


# sRGB primaries to CIE XYZ, and the D65 white point sRGB is defined against.
SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255, last axis RGB) to CIELAB under D65, 2-degree observer."""
    xyz = srgb_to_linear(np.asarray(rgb, float) / 255) @ SRGB_TO_XYZ.T / D65
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    return np.stack([L, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def window_mean(pixels: np.ndarray) -> np.ndarray:
    """Mean colour of an (..., 3) block of 0-255 sRGB pixels, mixed in linear light like paint under light."""
    lin = srgb_to_linear(pixels.reshape(-1, 3).astype(float) / 255).mean(axis=0)
    return np.round(linear_to_srgb(lin) * 255)


# --- screen area -------------------------------------------------------------------

def screen_pixels(ring: Ring, settings: HandoffSettings) -> tuple[int, int]:
    """(width, height) of the north-up screen image: the panel's pixels, turned to match the orientation."""
    long_px, short_px = settings.screen_px
    w, h = (long_px, short_px) if ring.layout.orientation == "ew" else (short_px, long_px)
    dw, dh = ring.layout.screen_mm
    # The screen's ground rectangle has the display's aspect, so square panel pixels
    # need a panel of that aspect too; otherwise the image would be stretched.
    if abs(w * dh / dw - h) > 1:
        a, b = ring.layout.screen_aspect
        raise ValueError(f"screen resolution {long_px} x {short_px} is not {a}:{b} like the layout's screen")
    return w, h


def _warp(src: Path, dst: Path, bounds, size, alg: str, dtype=gdal.GDT_Unknown) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    gdal.Warp(str(dst), str(src), format="GTiff", dstSRS=UTM_12N, outputBounds=bounds,
              width=size[0], height=size[1], resampleAlg=alg, outputType=dtype,
              creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])


def corners_lonlat(bounds) -> dict:
    """The UTM rectangle's four corners in NAD83 lon/lat (a UTM rectangle is not a lon/lat box)."""
    x0, y0, x1, y1 = bounds
    to_ll = Transformer.from_crs(UTM_12N, "EPSG:4269", always_xy=True)
    return {k: [round(v, 7) for v in to_ll.transform(x, y)]
            for k, (x, y) in {"nw": (x0, y1), "ne": (x1, y1), "se": (x1, y0), "sw": (x0, y0)}.items()}


def write_screen_area(ring: Ring, settings: HandoffSettings, dem_path: Path, imagery_path: Path,
                      out_dir: Path) -> dict:
    """Imagery (GeoTIFF and PNG) and DEM cropped to the display area, plus a JSON of where it sits."""
    lay, build = ring.layout, ring.build
    bounds = lay.screen_utm
    w, h = screen_pixels(ring, settings)
    ground_px = (bounds[2] - bounds[0]) / w
    with rasterio.open(imagery_path) as src:
        alg = "average" if ground_px > src.res[0] else "cubic"
    tif, png, dem = out_dir / "screen_imagery.tif", out_dir / "screen_imagery.png", out_dir / "screen_dem.tif"
    _warp(imagery_path, tif, bounds, (w, h), alg)
    gdal.Translate(str(png), str(tif), format="PNG", bandList=[1, 2, 3])
    _warp(dem_path, dem, bounds, (w, h), "average" if ground_px > 10 else "bilinear", gdal.GDT_Float32)
    (out_dir / "screen_imagery.png.aux.xml").unlink(missing_ok=True)

    mm_px = lay.screen_mm[0] / w
    info = {
        "layout": lay.name,
        "orientation": lay.orientation,
        "image_is_north_up": True,
        "rotate_for_panel": ("none" if lay.orientation == "ew" else
                             "the panel is mounted with its long side north-south; turn the image to "
                             "match how the panel is driven"),
        "pixels": [w, h],
        "crs": UTM_12N,
        "bounds_utm": dict(zip(["west", "south", "east", "north"], [round(v, 2) for v in bounds])),
        "corners_lonlat_nad83": corners_lonlat(bounds),
        "ground_m_per_px": round(ground_px, 4),
        "scale": round(lay.scale, 1),
        "display_mm": [round(v, 2) for v in lay.screen_mm],
        "display_mm_per_px": round(mm_px, 5),
        "hidden_under_lip_px": round(build.lip_overhang_mm / mm_px, 1),
        "fixed_band_px": round(settings.fixed_band_mm / mm_px, 1),
        "feather_px": round(settings.feather_mm / mm_px, 1),
        "files": {"imagery_geotiff": tif.name, "imagery_png": png.name, "dem_geotiff_m": dem.name},
    }
    (out_dir / "screen.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


# --- seam colours ------------------------------------------------------------------

def _outset(r: Rect, d: float) -> Rect:
    return r[0] - d, r[1] - d, r[2] + d, r[3] + d


def perimeter_points(r: Rect, spacing_mm: float) -> list[tuple[str, float, float]]:
    """Evenly spaced points round rectangle r, clockwise from its north-west corner.

    Each side gets its own even spacing no wider than `spacing_mm`; each corner is
    listed once, with the side that starts there.
    """
    x0, y0, x1, y1 = r
    sides = [("north", (x0, y1), (x1, y1)), ("east", (x1, y1), (x1, y0)),
             ("south", (x1, y0), (x0, y0)), ("west", (x0, y0), (x0, y1))]
    pts = []
    for name, (ax, ay), (bx, by) in sides:
        n = max(int(np.ceil(np.hypot(bx - ax, by - ay) / spacing_mm)), 1)
        for t in np.arange(n) / n:
            pts.append((name, ax + t * (bx - ax), ay + t * (by - ay)))
    return pts


@dataclass(frozen=True)
class Swatch:
    number: int
    row: str  # "seam" or "lower slope"
    side: str
    x_mm: float
    y_mm: float
    easting: float
    northing: float
    rgb: tuple[int, int, int]

    @property
    def hex(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)

    @property
    def lab(self) -> tuple[float, float, float]:
        return tuple(round(float(v), 1) for v in srgb_to_lab(np.array(self.rgb)))


def sample_seam_colors(ring: Ring, settings: HandoffSettings, imagery_path: Path) -> list[Swatch]:
    """Colour at each sample point along the seam and the lower-slope row, window-averaged."""
    lay = ring.layout
    tx0, ty0, _, _ = lay.table_utm
    m_per_mm = lay.scale / 1000
    half = settings.window_mm / 2 * m_per_mm
    rows = [("seam", ring.opening)]
    if settings.lower_slope_mm > 0:
        rows.append(("lower slope", _outset(ring.opening, settings.lower_slope_mm)))
    swatches = []
    with rasterio.open(imagery_path) as src:
        for row, rect in rows:
            for side, x, y in perimeter_points(rect, settings.spacing_mm):
                if row != "seam" and not ring.outline.keep(np.array([x]), np.array([y]))[0, 0]:
                    continue  # this point is off the printed terrain
                e, n = tx0 + x * m_per_mm, ty0 + y * m_per_mm
                win = from_bounds(e - half, n - half, e + half, n + half, transform=src.transform)
                block = src.read([1, 2, 3], window=win.round_offsets().round_lengths(), boundless=True)
                rgb = window_mean(np.moveaxis(block, 0, -1))
                swatches.append(Swatch(len(swatches) + 1, row, side, round(x, 1), round(y, 1),
                                       round(e, 1), round(n, 1), tuple(int(v) for v in rgb)))
    return swatches


def write_seam_colors_csv(swatches: list[Swatch], path: Path) -> None:
    to_ll = Transformer.from_crs(UTM_12N, "EPSG:4269", always_xy=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "row", "side", "x_mm", "y_mm", "easting", "northing", "lon", "lat",
                    "hex", "r", "g", "b", "L_d65", "a_d65", "b_d65"])
        for s in swatches:
            lon, lat = to_ll.transform(s.easting, s.northing)
            w.writerow([s.number, s.row, s.side, s.x_mm, s.y_mm, s.easting, s.northing,
                        round(lon, 6), round(lat, 6), s.hex, *s.rgb, *s.lab])


def _map_page(pdf: PdfPages, ring: Ring, swatches: list[Swatch], imagery_path: Path, title: str) -> None:
    """The ring from above on the imagery, with every swatch's number where it applies."""
    lay = ring.layout
    tx0, ty0, tx1, ty1 = lay.table_utm
    ds = gdal.Warp("", str(imagery_path), format="MEM", dstSRS=UTM_12N, outputBounds=(tx0, ty0, tx1, ty1),
                   xRes=MAP_RES_M, yRes=MAP_RES_M, resampleAlg="average")
    img = np.stack([ds.GetRasterBand(b).ReadAsArray() for b in (1, 2, 3)], axis=-1)
    ds = None
    tw, th = ring.table[2], ring.table[3]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.imshow(img, extent=[0, tw, 0, th])
    gx, gy = np.linspace(0, tw, 700), np.linspace(0, th, int(700 * th / tw))
    kept = ring.outline.keep(gx, gy)
    ax.contourf(gx, gy, (~kept).astype(float), levels=[0.5, 1.5], colors=["white"], alpha=0.6)
    ax.contour(gx, gy, kept.astype(float), levels=[0.5], colors=["#ff5a36"], linewidths=1)
    o = ring.opening
    ax.add_patch(Rectangle(o[:2], o[2] - o[0], o[3] - o[1], fill=False, ec="#ffcc00", lw=1.5))
    for s in swatches:
        ax.plot(s.x_mm, s.y_mm, "o", ms=9, mfc=s.hex, mec="black" if s.row == "seam" else "white", mew=1)
        ax.annotate(str(s.number), (s.x_mm, s.y_mm), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=6, weight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8, lw=0))
    ax.set_xlim(0, tw)
    ax.set_ylim(0, th)
    ax.set_xlabel("mm from the table's south-west corner (north is up)")
    ax.set_title(f"{title}\nYellow: screen opening. Red: edge of the printed terrain. "
                 "Black ring: seam colour. White ring: lower slope.", fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _swatch_pages(pdf: PdfPages, swatches: list[Swatch], title: str) -> None:
    cols, rows = SWATCHES_PER_PAGE
    per_page = cols * rows
    for start in range(0, len(swatches), per_page):
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(f"{title}  (page {start // per_page + 1} of {-(-len(swatches) // per_page)})\n"
                     "sRGB hex and CIELAB (D65, 2°). Print colours are approximate: "
                     "match paint to the numbers with a colorimeter.", fontsize=8)
        for i, s in enumerate(swatches[start:start + per_page]):
            r, c = divmod(i, cols)
            ax = fig.add_axes([0.04 + c * 0.155, 0.83 - r * 0.108, 0.14, 0.07])
            ax.add_patch(Rectangle((0, 0), 1, 1, color=s.hex))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            L, a, b = s.lab
            ax.set_xlabel(f"{s.number} · {s.row}, {s.side}\n{s.hex}\nL {L:.1f}  a {a:.1f}  b {b:.1f}",
                          fontsize=6.5, labelpad=2)
        pdf.savefig(fig)
        plt.close(fig)


def write_swatch_sheet(ring: Ring, swatches: list[Swatch], imagery_path: Path, path: Path) -> None:
    title = f"Seam colours: {ring.layout.title}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        _map_page(pdf, ring, swatches, imagery_path, title)
        _swatch_pages(pdf, swatches, title)


# --- handoff note ------------------------------------------------------------------

def write_handoff_note(info: dict, settings: HandoffSettings, n_swatches: int, path: Path) -> None:
    b = info["bounds_utm"]
    w, h = info["pixels"]
    text = f"""# Screen and paint handoff: {info['layout']}

Generated by `python -m terrain handoff`. Regenerate after any change to the layout or `config/build.toml`.

## Screen area

- `{info['files']['imagery_png']}` / `{info['files']['imagery_geotiff']}`: the valley floor the screen shows, {w} x {h} px, north up, one image pixel per panel pixel.
- `{info['files']['dem_geotiff_m']}`: elevation in metres on the same grid.
- Bounds (UTM 12N, {info['crs']}): west {b['west']}, south {b['south']}, east {b['east']}, north {b['north']}. Corners in lon/lat are in `screen.json`.
- {info['ground_m_per_px']:.2f} m of ground per pixel; {info['display_mm_per_px']:.3f} mm of glass per pixel; scale 1:{info['scale']:,.0f}.
- Orientation: {"long side east-west, image as is" if info['orientation'] == "ew" else "long side north-south: " + info['rotate_for_panel']}.

Every pixel must be drawn exactly where this image puts it: no letterboxing, cropping or scaling in the display software, or the map will not line up with the printed terrain.

## Requirements for the display software

1. **Hidden edge.** The printed lip covers the outer {info['hidden_under_lip_px']:.0f} px of each side. Nothing interactive should sit there.
2. **Fixed band.** A band {settings.fixed_band_mm:g} mm wide (about {info['fixed_band_px']:.0f} px) along every edge always shows this base imagery, whatever the land-use state, so the colours at the seam never change.
3. **Feathering.** Inside the band, the interactive map fades in from the base imagery over {settings.feather_mm:g} mm (about {info['feather_px']:.0f} px); no hard edge.
4. **Edge-pixel calibration.** At the installation, under the real lighting, the colours of the fixed band are adjusted until they match the painted terrain next to them. Adjust the screen, not the paint: this needs a per-edge (ideally per-segment) colour correction the installer can tune and save, applied to the band only and faded out across the feather.
5. **Brightness.** The screen gives off light and the paint doesn't, so the screen will usually need to run well below full brightness to sit next to the terrain.

## Painters' colour sheet

- `seam_colors.pdf`: a numbered map of the ring and {n_swatches} swatches, each with hex and CIELAB (D65, 2°) values.
- `seam_colors.csv`: the same colours as data, with table and ground coordinates.
- Colours are the imagery averaged over {settings.window_mm:g} mm of table (mixed in linear light), every {settings.spacing_mm:g} mm or closer along the seam{f", and along a second row {settings.lower_slope_mm:g} mm out on the lower slopes" if settings.lower_slope_mm > 0 else ""}.
- The printed sheet only approximates the colours. Mix paint against the Lab numbers with a colorimeter, and check it next to the running screen. The imagery's hillsides have the sun and shadows of the photo baked in, so on steep lower slopes use the colour as a mid-tone, not the final finish.
"""
    path.write_text(text)


def write_handoff(ring: Ring, settings: HandoffSettings, dem_path: Path, imagery_path: Path,
                  out_dir: Path) -> tuple[dict, list[Swatch]]:
    info = write_screen_area(ring, settings, dem_path, imagery_path, out_dir)
    swatches = sample_seam_colors(ring, settings, imagery_path)
    write_seam_colors_csv(swatches, out_dir / "seam_colors.csv")
    write_swatch_sheet(ring, swatches, imagery_path, out_dir / "seam_colors.pdf")
    write_handoff_note(info, settings, len(swatches), out_dir / "HANDOFF.md")
    return info, swatches
