"""DEM / hillshade map: shaded relief with contours, hillshade and slope."""
import numpy as np
import streamlit as st

from geoai import export, terrain
from geoai.imagery import dem as dem_mod

from .common import get_scene, header, need_scene, show_fig


@st.fragment
def render(aoi):
    st.caption("Copernicus DEM GLO-30 elevation. Relief shows ridges of resistant rock (quartz veins, BIF, "
               "granites), valleys eroded along faults and soft units, and the drainage pattern.")
    sc = get_scene(aoi)
    if sc is None:
        need_scene()
        return
    if sc.get("dem") is None:
        st.warning(f"Elevation data is not available for this scene: {sc.get('dem_error')}")
        return
    bands, tr, epsg, inside, clear, aoi_utm = header(sc, aoi)
    dem, res_m = sc["dem"], sc["data"]["res"]
    c1, c2, c3, c4 = st.columns(4)
    az = c1.slider("Sun azimuth (°)", 0, 360, step=15, key="rs_dem_az",
                   help="Direction the light comes from. Features perpendicular to it stand out most.")
    alt = c2.slider("Sun altitude (°)", 5, 85, step=5, key="rs_dem_alt")
    z = c3.slider("Vertical exaggeration", 1.0, 5.0, step=0.5, key="rs_dem_z")
    contours = c4.checkbox("Contours", key="rs_dem_contours")
    hs = terrain.hillshade(dem, res_m, az, alt, z)
    slope = terrain.slope_deg(dem, res_m)
    v = dem[inside & np.isfinite(dem)]
    if v.size == 0:
        st.warning("No elevation values inside the area.")
        return
    step = terrain.contour_interval(dem) if contours else None
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lowest", f"{v.min():,.0f} m")
    k2.metric("Highest", f"{v.max():,.0f} m")
    k3.metric("Relief", f"{v.max() - v.min():,.0f} m")
    k4.metric("Mean slope", f"{np.nanmean(slope[inside]):.1f}°")
    left, right = st.columns([3, 2])
    with left:
        show_fig(export.relief_figure(dem, hs, tr, aoi_utm, "Shaded relief (elevation + hillshade)"
                                      + (f", contours every {step} m" if step else ""), step),
                 "dem_shaded_relief.png", key="dem_png_relief")
    with right:
        show_fig(export.raster_figure(hs, tr, aoi_utm, f"Hillshade (sun {az}°, {alt}° high)", "gray",
                                      cbar_label="illumination", lo=0.5, hi=99.5), "hillshade.png",
                 key="dem_png_hs")
        show_fig(export.raster_figure(slope, tr, aoi_utm, "Slope", "YlOrBr", cbar_label="degrees", lo=0, hi=99),
                 "slope.png", key="dem_png_slope")
    st.download_button("Elevation, hillshade, slope, aspect (GeoTIFF)",
                       lambda: export.geotiff_bytes({"elevation_m": dem, "hillshade": hs, "slope_deg": slope,
                                                     "aspect_deg": terrain.aspect_deg(dem, res_m)}, tr, epsg),
                       "dem_products.tif", "image/tiff", on_click="ignore", key="dem_tif")
    st.caption(f"{dem_mod.ATTRIBUTION}. 30 m source resampled to the {res_m} m analysis grid.")
