"""High-resolution satellite image of an AOI from Esri World Imagery tiles.

The mosaic is built from the best available imagery (Maxar, Airbus, ...), so its capture
date varies by place and is not "today". Use Sentinel-2 (sentinel.py) for a dated, recent scene.
"""
import io
import math
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTRIBUTION = "Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
TILE = 256
_HEADERS = {"User-Agent": "GeoAI-Platform/0.2 (geological research tool)"}


# ---------- Web-Mercator tile maths ----------
def lonlat_to_global_px(lon, lat, z):
    n = TILE * 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat = max(min(lat, 85.0511), -85.0511)
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def tile_range(bbox, z):
    lon0, lat0, lon1, lat1 = bbox
    x0, y0 = lonlat_to_global_px(lon0, lat1, z)  # top-left
    x1, y1 = lonlat_to_global_px(lon1, lat0, z)  # bottom-right
    return int(x0 // TILE), int(y0 // TILE), int(x1 // TILE), int(y1 // TILE)


def choose_zoom(bbox, max_tiles=64, max_zoom=18, min_zoom=3):
    for z in range(max_zoom, min_zoom - 1, -1):
        tx0, ty0, tx1, ty1 = tile_range(bbox, z)
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= max_tiles:
            return z
    return min_zoom


def padded_bbox(bbox, pad=0.15):
    lon0, lat0, lon1, lat1 = bbox
    dx, dy = (lon1 - lon0) * pad, (lat1 - lat0) * pad
    return lon0 - dx, lat0 - dy, lon1 + dx, lat1 + dy


# ---------- fetching ----------
def fetch_tile(z, x, y, session=None, url=ESRI_URL, timeout=20):
    s = session or requests
    r = s.get(url.format(z=z, x=x, y=y), headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def satellite_image(bbox, max_tiles=64, max_zoom=18, fetcher=None, workers=8):
    """Stitch tiles covering bbox (lon0, lat0, lon1, lat1) and crop to it exactly.

    Returns (PIL.Image, zoom, failed_tile_count). `fetcher(z, x, y)` can be injected for tests.
    """
    z = choose_zoom(bbox, max_tiles, max_zoom)
    tx0, ty0, tx1, ty1 = tile_range(bbox, z)
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    canvas = Image.new("RGB", (nx * TILE, ny * TILE), (40, 40, 40))
    failed = 0

    if fetcher is None:
        session = requests.Session()
        fetcher = lambda zz, xx, yy: fetch_tile(zz, xx, yy, session)  # noqa: E731

    def job(t):
        xx, yy = t
        try:
            return t, fetcher(z, xx, yy)
        except Exception:
            return t, None

    jobs = [(xx, yy) for yy in range(ty0, ty1 + 1) for xx in range(tx0, tx1 + 1)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for (xx, yy), img in ex.map(job, jobs):
            if img is None:
                failed += 1
                continue
            canvas.paste(img.resize((TILE, TILE)), ((xx - tx0) * TILE, (yy - ty0) * TILE))

    if failed == len(jobs):
        raise ConnectionError("No satellite tiles could be downloaded. Check the internet connection.")

    lon0, lat0, lon1, lat1 = bbox
    px0, py0 = lonlat_to_global_px(lon0, lat1, z)
    px1, py1 = lonlat_to_global_px(lon1, lat0, z)
    ox, oy = tx0 * TILE, ty0 * TILE
    crop = canvas.crop((int(px0 - ox), int(py0 - oy), int(math.ceil(px1 - ox)), int(math.ceil(py1 - oy))))
    return crop, z, failed


# ---------- presentation ----------
def _nice_length(m):
    for v in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
        if v >= m:
            return v
    return 100000


def annotated_figure(img, bbox, aoi, title="Satellite image", attribution=ESRI_ATTRIBUTION):
    """Map figure: image + AOI outline + corner labels + north arrow + scale bar + attribution."""
    lon0, lat0, lon1, lat1 = bbox
    lat_c = (lat0 + lat1) / 2
    ratio = (lat1 - lat0) / max((lon1 - lon0) * math.cos(math.radians(lat_c)), 1e-12)
    fig, ax = plt.subplots(figsize=(9, float(np.clip(9 * ratio, 4, 12))))
    ax.imshow(np.asarray(img), extent=[lon0, lon1, lat0, lat1], interpolation="bilinear")
    ax.set_aspect(1 / math.cos(math.radians(lat_c)))

    xs, ys = list(aoi.lon) + [aoi.lon[0]], list(aoi.lat) + [aoi.lat[0]]
    ax.plot(xs, ys, color="#ffd400", lw=2.2)
    for i, (x, y) in enumerate(zip(aoi.lon, aoi.lat)):
        ax.plot(x, y, "o", color="#ffd400", ms=5, mec="black", mew=0.6)
        ax.annotate(f"C{i + 1}", (x, y), xytext=(5, 5), textcoords="offset points", color="white",
                    fontsize=9, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0))

    # scale bar (~20 % of width)
    m_per_deg = 111_320 * math.cos(math.radians(lat_c))
    length_m = _nice_length((lon1 - lon0) * m_per_deg * 0.2)
    sx, sy = lon0 + (lon1 - lon0) * 0.04, lat0 + (lat1 - lat0) * 0.05
    ax.plot([sx, sx + length_m / m_per_deg], [sy, sy], color="white", lw=4, solid_capstyle="butt")
    ax.plot([sx, sx + length_m / m_per_deg], [sy, sy], color="black", lw=1.2, solid_capstyle="butt")
    ax.text(sx + length_m / m_per_deg / 2, sy + (lat1 - lat0) * 0.015,
            f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m} m",
            color="white", ha="center", va="bottom", fontsize=9, weight="bold")

    # north arrow
    ax.annotate("N", xy=(0.95, 0.94), xytext=(0.95, 0.84), xycoords="axes fraction",
                ha="center", va="center", color="white", fontsize=12, weight="bold",
                arrowprops=dict(facecolor="white", edgecolor="black", width=4, headwidth=12))

    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title)
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.text(0.99, 0.005, attribution, ha="right", va="bottom", fontsize=7, color="#555")
    fig.tight_layout()
    return fig
