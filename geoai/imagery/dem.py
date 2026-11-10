"""Copernicus DEM GLO-30 (30 m global elevation) via Microsoft Planetary Computer (free, no key)."""
import json

import numpy as np

COLLECTION = "cop-dem-glo-30"
ATTRIBUTION = "Copernicus DEM GLO-30 (ESA / Airbus), via Microsoft Planetary Computer"


def search_dem(polygon_geojson):
    import planetary_computer
    import pystac_client

    from .sentinel import PC_STAC
    catalog = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    geom = json.loads(polygon_geojson) if isinstance(polygon_geojson, str) else polygon_geojson
    return [it.to_dict() for it in catalog.search(collections=[COLLECTION], intersects=geom).items()]


def load_dem(aoi, epsg, transform, width, height, items=None, opener=None):
    """Elevation (m, float32, NaN = no data) on the given grid, mosaicking all DEM tiles needed."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    if items is None:
        items = search_dem(aoi.geometry_json())
    if not items:
        raise ValueError("No elevation data found for this area.")
    if opener is None:
        import planetary_computer
        opener = lambda href: rasterio.open(planetary_computer.sign(href))  # noqa: E731

    out = np.full((height, width), np.nan, np.float32)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4"):
        for it in items:
            with opener(it["assets"]["data"]["href"]) as src:
                nod = src.nodata if src.nodata is not None else -32767
                with WarpedVRT(src, crs=f"EPSG:{epsg}", transform=transform, width=width, height=height,
                               resampling=Resampling.bilinear, src_nodata=nod, nodata=np.nan,
                               dtype="float32") as vrt:
                    data = vrt.read(1)
            ok = np.isfinite(data) & (data > -1000) & np.isnan(out)
            out[ok] = data[ok]
    if np.isnan(out).all():
        raise ValueError("Elevation tiles were found but contain no data for this area.")
    return out
