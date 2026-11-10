"""Recent Sentinel-2 L2A (surface reflectance) scenes for an AOI.

Source: Microsoft Planetary Computer STAC API (free, no account or key needed).
Sentinel-2 revisits every ~5 days at 10-20 m resolution, so the newest cloud-free
scene is usually only days or weeks old.
"""
import math
from collections import defaultdict

import numpy as np

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
# Bands needed for the composites and gold-alteration indices (+ SCL for cloud masking)
BANDS = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12", "SCL"]
BAND_INFO = {
    "B02": "Blue (490 nm, 10 m)", "B03": "Green (560 nm, 10 m)", "B04": "Red (665 nm, 10 m)",
    "B08": "NIR (842 nm, 10 m)", "B8A": "Narrow NIR (865 nm, 20 m)",
    "B11": "SWIR-1 (1610 nm, 20 m)", "B12": "SWIR-2 (2190 nm, 20 m)", "SCL": "Scene classification",
}
SCL_BAD = (0, 1, 3, 8, 9, 10)  # no data, saturated, cloud shadow, cloud medium/high, cirrus
RESOLUTIONS = [10, 20, 30, 60, 120, 240]


# ---------- search ----------
def search_scenes(polygon_geojson, start, end, max_cloud=30, limit=60):
    """Return scene groups (one per date & orbit; neighbouring tiles are merged), newest first."""
    import planetary_computer
    import pystac_client
    from shapely.geometry import shape

    catalog = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=[COLLECTION], intersects=polygon_geojson,
                            datetime=f"{start}/{end}", query={"eo:cloud_cover": {"lt": max_cloud}},
                            sortby=[{"field": "properties.datetime", "direction": "desc"}],
                            max_items=limit)
    items = [it.to_dict() for it in search.items()]
    return group_items(items, shape(polygon_geojson))


def group_items(items, aoi_polygon):
    from shapely.geometry import shape
    from shapely.ops import unary_union

    groups = defaultdict(list)
    for it in items:
        p = it["properties"]
        groups[(p["datetime"][:10], p.get("sat:relative_orbit"))].append(it)
    out = []
    for (date, orbit), its in groups.items():
        footprint = unary_union([shape(i["geometry"]) for i in its])
        cover = footprint.intersection(aoi_polygon).area / aoi_polygon.area if aoi_polygon.area else 0
        out.append({
            "date": date,
            "orbit": orbit,
            "cloud": float(np.mean([i["properties"].get("eo:cloud_cover", np.nan) for i in its])),
            "coverage": float(min(cover, 1.0)),
            "tiles": ", ".join(sorted({i["properties"].get("s2:mgrs_tile", i["id"]) for i in its})),
            "platform": its[0]["properties"].get("platform", "sentinel-2"),
            "items": its,
        })
    out.sort(key=lambda g: (g["date"], g["coverage"]), reverse=True)
    return out


# ---------- target grid ----------
def auto_resolution(aoi, max_pixels=4_000_000):
    e, n = aoi.to_utm()
    area = (max(e) - min(e)) * (max(n) - min(n))
    for r in RESOLUTIONS:
        if area / r ** 2 <= max_pixels:
            return r
    return RESOLUTIONS[-1]


def target_grid(aoi, res, pad_px=4):
    """Output grid in the AOI's UTM zone: (epsg, rasterio Affine transform, width, height)."""
    from rasterio.transform import from_origin

    e, n = aoi.to_utm()
    left, right = min(e) - pad_px * res, max(e) + pad_px * res
    bottom, top = min(n) - pad_px * res, max(n) + pad_px * res
    w, h = int(math.ceil((right - left) / res)), int(math.ceil((top - bottom) / res))
    return aoi.utm_epsg, from_origin(left, top, res, res), w, h


def polygon_mask(aoi, transform, width, height):
    """True inside the AOI polygon."""
    from rasterio.features import geometry_mask
    from shapely.geometry import Polygon

    e, n = aoi.to_utm()
    return geometry_mask([Polygon(zip(e, n))], out_shape=(height, width), transform=transform, invert=True)


# ---------- loading ----------
def _offset(item):
    """L2A processing baseline >= 04.00 (from Jan 2022) stores reflectance with a -1000 offset."""
    try:
        return -1000.0 if float(item["properties"].get("s2:processing_baseline", "0")) >= 4.0 else 0.0
    except ValueError:
        return 0.0


def load_scene(items, aoi, res=None, bands=BANDS, progress=None, opener=None):
    """Read bands for one scene group onto a common UTM grid clipped to the AOI.

    Returns dict(bands={name: float32 array}, transform, epsg, res, inside, clear, date).
    Reflectance bands are 0-1; SCL keeps its class codes. `opener(href)` is injectable for tests.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    if opener is None:
        import planetary_computer
        opener = lambda href: rasterio.open(planetary_computer.sign(href))  # noqa: E731

    res = res or auto_resolution(aoi)
    epsg, transform, w, h = target_grid(aoi, res)
    out = {}
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4",
               GDAL_HTTP_RETRY_DELAY="1", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff")
    steps, done = len(bands) * len(items), 0
    with rasterio.Env(**env):
        for b in bands:
            arr = np.full((h, w), np.nan, np.float32)
            for it in items:
                href = it["assets"][b]["href"]
                with opener(href) as src, WarpedVRT(
                        src, crs=f"EPSG:{epsg}", transform=transform, width=w, height=h,
                        resampling=Resampling.nearest if b == "SCL" else Resampling.bilinear,
                        src_nodata=0, nodata=0) as vrt:
                    data = vrt.read(1).astype(np.float32)
                valid = data != 0
                if b != "SCL":
                    data = (data + _offset(it)) / 10000.0
                fill = np.isnan(arr) & valid
                arr[fill] = data[fill]
                done += 1
                if progress:
                    progress(done / steps, f"Reading {b} ({BAND_INFO.get(b, b)})")
            out[b] = arr

    inside = polygon_mask(aoi, transform, w, h)
    scl = out.get("SCL")
    clear = inside & ~np.isnan(out["B04"])
    if scl is not None:
        clear &= ~np.isin(np.nan_to_num(scl, nan=0).astype(int), SCL_BAD)
    return {"bands": out, "transform": transform, "epsg": epsg, "res": res,
            "inside": inside, "clear": clear, "date": items[0]["properties"]["datetime"][:10]}
