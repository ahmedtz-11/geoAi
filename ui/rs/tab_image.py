"""High-resolution satellite image tab (Esri World Imagery)."""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pyproj import Transformer

from geoai import export
from geoai.imagery import basemap
from ui.state import fragment


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def cached_basemap(bbox, max_tiles):
    return basemap.satellite_image(bbox, max_tiles=max_tiles)


@fragment
def render(aoi):
    st.caption("Best available high-resolution mosaic (sub-metre to a few metres). Capture date varies by location.")
    detail = st.select_slider("Detail", ["Standard", "High", "Very high"], "High", key="img_detail")
    max_tiles = {"Standard": 36, "High": 100, "Very high": 196}[detail]
    bbox = basemap.padded_bbox(aoi.bbox, 0.15)
    try:
        with st.spinner("Downloading satellite tiles…"):
            img, zoom, failed = cached_basemap(bbox, max_tiles)
    except Exception as e:
        st.error(f"Could not download the satellite image: {e}")
        return
    if failed:
        st.warning(f"{failed} tile(s) could not be downloaded and show as grey.")
    fig = basemap.annotated_figure(img, bbox, aoi, title="Satellite image of the area of interest")
    st.pyplot(fig)
    to3857 = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = to3857.transform(bbox[0], bbox[1])
    x1, y1 = to3857.transform(bbox[2], bbox[3])
    from rasterio.transform import from_bounds
    tr = from_bounds(x0, y0, x1, y1, img.width, img.height)
    e1, e2, e3 = st.columns(3)
    e1.download_button("Map with labels (PNG)", lambda f=fig: export.fig_to_png(f), "aoi_satellite_map.png",
                       "image/png", on_click="ignore", key="img_png1")
    e2.download_button("Image only (PNG)", lambda i=img: export.image_to_png(i), "aoi_satellite.png", "image/png",
                       on_click="ignore", key="img_png2")
    e3.download_button("GeoTIFF (for QGIS/ArcGIS)",
                       lambda i=img, t=tr: export.rgb_geotiff_bytes(np.asarray(i) / 255.0, t, 3857),
                       "aoi_satellite_3857.tif", "image/tiff", on_click="ignore", key="img_tif")
    plt.close(fig)
    st.caption(f"Zoom level {zoom} · {img.width}×{img.height} px · {basemap.ESRI_ATTRIBUTION}")
