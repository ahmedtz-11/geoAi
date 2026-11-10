"""Geological lineament map: automatic extraction, rose diagram, density."""
import numpy as np
import pandas as pd
import streamlit as st

from geoai import export, lineaments

from .common import get_scene, header, hillshades, lineament_result, need_scene, show_fig, ss, std_pca
from geoai import terrain


@st.fragment
def render(aoi):
    st.caption("Lineaments are straight to gently curved linear features (valleys, ridges, tonal boundaries) "
               "that often follow faults, shear zones, fractures, dykes and contacts, which control many gold "
               "deposits. They are traced automatically from shaded relief and/or the imagery.")
    sc = get_scene(aoi)
    if sc is None:
        need_scene()
        return
    bands, tr, epsg, inside, clear, aoi_utm = header(sc, aoi)
    res_m = sc["data"]["res"]
    has_dem = sc.get("dem") is not None
    sources = ["Elevation (hillshade)", "Satellite (PC1)", "Both"] if has_dem else ["Satellite (PC1)"]
    if ss.get("rs_lin_source") not in sources:
        ss["rs_lin_source"] = sources[0]
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    source = c1.radio("Trace from", sources, key="rs_lin_source", horizontal=True)
    pct = c2.slider("Selectivity", 80, 98, key="rs_lin_pct",
                    help="Higher = only the sharpest edges become lineaments (fewer, more reliable).")
    sigma = c3.slider("Smoothing", 1.0, 4.0, step=0.5, key="rs_lin_sigma",
                      help="Higher ignores small, local features.")
    length = c4.slider("Min length (m)", 100, 3000, step=50, key="rs_lin_len")
    gap = c5.slider("Max gap (m)", 10, 500, step=10, key="rs_lin_gap",
                    help="Edge pieces separated by less than this are joined into one lineament.")
    with st.spinner("Extracting lineaments…"):
        lines, edges = lineament_result(sc, source, pct, sigma, length, gap)
    if has_dem and source != "Satellite (PC1)":
        bg = terrain.multi_hillshade(sc["dem"], res_m)
        bg_name = "multi-directional hillshade"
    else:
        bg = std_pca(sc)["pcs"][0]
        bg_name = "PC1"
    total_km = sum(l["length_m"] for l in lines) / 1000
    area_km2 = aoi.area_m2 / 1e6
    starts, lens = lineaments.rose_stats(lines)
    order = np.argsort(lens)[::-1]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lineaments", f"{len(lines)}")
    k2.metric("Total length", f"{total_km:,.1f} km")
    k3.metric("Density", f"{total_km / max(area_km2, 1e-9):,.2f} km/km²")
    k4.metric("Main trend", lineaments.strike_label(starts[order[0]] + 5) if lines else "–")
    if not lines:
        st.warning("No lineaments found. Lower the selectivity or the minimum length.")
        return
    left, right = st.columns([3, 2])
    with left:
        show_fig(export.lines_figure(bg, lines, tr, aoi_utm, f"Lineaments on {bg_name} · {sc['data']['date']}"),
                 "lineament_map.png", key="lin_png_map")
    with right:
        show_fig(export.rose_figure(starts, lens, title="Strike rose (length-weighted)"), "lineament_rose.png",
                 key="lin_png_rose")
        trends = pd.DataFrame([{"Trend": lineaments.strike_label(starts[i] + 5),
                                "Strike range": f"{starts[i]:.0f}–{starts[i] + 10:.0f}°",
                                "Length (km)": round(float(lens[i]), 2),
                                "Share": f"{100 * lens[i] / max(lens.sum(), 1e-9):.0f}%"}
                               for i in order[:4] if lens[i] > 0])
        st.dataframe(trends, hide_index=True, width="stretch")
    radius = st.slider("Density search radius (m)", 100, 3000, step=50, key="rs_lin_radius")
    dens = lineaments.density(lines, inside.shape, res_m, radius)
    d1, d2 = st.columns([3, 2])
    with d1:
        show_fig(export.raster_figure(dens, tr, aoi_utm, "Lineament density", "hot_r", valid=inside,
                                      cbar_label="km per km²", lo=0, hi=99.5), "lineament_density.png",
                 key="lin_png_dens")
    with d2:
        st.markdown("**Reading the results**\n\n"
                    "- The **rose diagram** shows the dominant structural directions. In the Lake Victoria "
                    "goldfields, gold often follows shear zones parallel to the greenstone belt.\n"
                    "- **High density** areas and **intersections** of trends are structurally prepared ground, "
                    "worth checking against alteration maps.\n"
                    "- Roads, field boundaries and power lines can be traced too; check them on the "
                    "high-resolution image.")
        geo = lineaments.to_geojson(lines, tr, epsg)
        rows = pd.DataFrame([{"id": i + 1, "length_m": round(l["length_m"], 1), "azimuth": round(l["azimuth"], 1),
                              "trend": lineaments.strike_label(l["azimuth"])} for i, l in enumerate(lines)])
        st.download_button("Lineaments (GeoJSON for QGIS/ArcGIS)", geo, "lineaments.geojson",
                           "application/geo+json", on_click="ignore", key="lin_geojson")
        st.download_button("Lineament list (CSV)", rows.to_csv(index=False), "lineaments.csv", "text/csv",
                           on_click="ignore", key="lin_csv")
        st.download_button("Density (GeoTIFF)",
                           lambda: export.geotiff_bytes({"lineament_density_km_km2": dens}, tr, epsg),
                           "lineament_density.tif", "image/tiff", on_click="ignore", key="lin_tif")
