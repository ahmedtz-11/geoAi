"""Section 1 - Remote sensing.

Input : CSV / Excel with the 4 corner coordinates of an area (lat/long, DMS or UTM).
Output: AOI map + size, then Step-3 image outputs (one module each in ui/rs/):
        high-resolution image, iron oxide, clay/hydroxyl, band ratios, PCA, lithological
        classification, lineaments, DEM/hillshade, integrated prospectivity.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from geoai import coords, export
from geoai.imagery import basemap
from geoai.io import read_raw_table
from ui.rs import (common, data_panel, tab_dem, tab_image, tab_index, tab_lineaments, tab_lithology, tab_pca,
                   tab_prospectivity, tab_ratios)

ss = st.session_state
common.keep_widget_state()
SAMPLE = pd.DataFrame({"corner": ["C1", "C2", "C3", "C4"],
                       "latitude": ["-2.8500", "-2.8500", "-2.8900", "-2.8900"],
                       "longitude": ["32.1500", "32.1900", "32.1900", "32.1500"]})


# ---------- small UI helpers ----------



def map_zoom(bbox):
    span = max(bbox[2] - bbox[0], (bbox[3] - bbox[1]) * 1.3, 1e-6)
    return float(np.clip(math.log2(360 / span) - 0.8, 2, 17))


def aoi_map(aoi, height=460):
    lon, lat = list(aoi.lon) + [aoi.lon[0]], list(aoi.lat) + [aoi.lat[0]]
    fig = go.Figure()
    fig.add_trace(go.Scattermap(lon=lon, lat=lat, mode="lines", fill="toself",
                                fillcolor="rgba(255,212,0,0.12)", line=dict(color="#ffd400", width=3),
                                hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattermap(lon=list(aoi.lon), lat=list(aoi.lat), mode="markers+text",
                                marker=dict(size=9, color="#ffd400"),
                                text=[f"C{i + 1}" for i in range(len(aoi.lon))], textposition="top right",
                                textfont=dict(color="white", size=13),
                                hovertemplate="%{text}<br>Lat %{lat:.6f}<br>Lon %{lon:.6f}<extra></extra>",
                                showlegend=False))
    cx, cy = aoi.centroid
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        map=dict(style="white-bg", center=dict(lon=cx, lat=cy), zoom=map_zoom(aoi.bbox),
                 layers=[dict(below="traces", sourcetype="raster", source=[basemap.ESRI_URL],
                              sourceattribution=basemap.ESRI_ATTRIBUTION)]))
    return fig











# ---------- page ----------
st.title("1 · Remote sensing")
st.caption("Give the 4 corners of your area. GeoAI maps it, measures it, fetches satellite imagery and "
           "highlights surface alteration linked to gold.")

with st.sidebar:
    st.markdown("**Templates**")
    st.download_button("Corners template (lat/long)", SAMPLE.to_csv(index=False), "aoi_corners_latlong.csv",
                       "text/csv", on_click="ignore")
    st.caption("Columns can be named latitude/longitude, lat/lon, Y/X or northing/easting. "
               "Degrees-minutes-seconds like 2°51'00\"S also work.")

# ---------- step 1: coordinates ----------
st.subheader("Step 1 · Area corners")
mode = st.radio("Coordinates", ["Upload file", "Type them in"], horizontal=True, label_visibility="collapsed")
df = None
if mode == "Upload file":
    up = st.file_uploader("CSV, TXT or Excel with one row per corner", type=["csv", "txt", "xlsx", "xls"])
    if up is not None:
        try:
            df = read_raw_table(up)
        except Exception as e:
            st.error(f"Could not read {up.name}: {e}")
else:
    df = st.data_editor(SAMPLE, num_rows="dynamic", width="stretch", key="manual_corners",
                        column_config={c: st.column_config.TextColumn(c) for c in SAMPLE.columns})

if df is None or df.empty:
    st.info("Upload a corners file, or switch to **Type them in** to try the example area.")
    st.stop()

cols = list(df.columns)
gx, gy = coords.guess_columns(df)
c1, c2, c3 = st.columns([1, 1, 2])
xcol = c1.selectbox("Longitude / Easting column", cols, cols.index(gx) if gx in cols else 0)
ycol = c2.selectbox("Latitude / Northing column", cols, cols.index(gy) if gy in cols else min(1, len(cols) - 1))
xs, ys = coords.parse_column(df[xcol]), coords.parse_column(df[ycol])
geo_guess = coords.looks_geographic(xs.dropna(), ys.dropna())
labels = list(coords.CRS_OPTIONS) + ["Other EPSG code…"]
default = 0 if geo_guess else labels.index("WGS84 / UTM zone 36S (EPSG:32736)")
crs_label = c3.selectbox("Coordinate system", labels, default,
                         help="Auto-detected from the values. Tanzania spans UTM zones 35S-37S; "
                              "older maps often use Arc 1960.")
epsg = coords.CRS_OPTIONS.get(crs_label)
if epsg is None:
    epsg = int(c3.number_input("EPSG code", min_value=1024, max_value=999999, value=32736, step=1))

try:
    aoi = coords.aoi_from_table(df, xcol, ycol, epsg)
except Exception as e:
    st.error(str(e))
    st.stop()
for note in aoi.notes:
    st.warning(note)

# ---------- step 2: AOI summary ----------
st.subheader("Step 2 · Area of interest")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Area", f"{aoi.area_m2 / 1e6:,.3f} km²")
m2.metric("Area", f"{aoi.area_m2 / 1e4:,.1f} ha")
m3.metric("Perimeter", f"{aoi.perimeter_m / 1000:,.2f} km")
cx, cy = aoi.centroid
m4.metric("UTM zone", aoi.utm_name.split("zone")[-1].strip(), help=f"EPSG:{aoi.utm_epsg}")
st.caption(f"Centre: latitude {cy:.6f}, longitude {cx:.6f}")

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(aoi_map(aoi), width="stretch", config={"scrollZoom": True})
with right:
    st.dataframe(aoi.corners_table(), hide_index=True, width="stretch")
    st.caption(f"Corners ordered clockwise. Easting/Northing in {aoi.utm_name} (EPSG:{aoi.utm_epsg}).")
    d1, d2, d3 = st.columns(3)
    d1.download_button("GeoJSON", aoi.geojson(), "aoi.geojson", "application/geo+json", on_click="ignore")
    d2.download_button("KML (Google Earth)", export.aoi_kml(aoi), "aoi.kml", "application/vnd.google-earth.kml+xml", on_click="ignore")
    d3.download_button("Corners CSV", aoi.corners_table().to_csv(index=False), "aoi_corners.csv", "text/csv", on_click="ignore")

if aoi.area_m2 > 5_000e6:
    st.warning("This area is very large (> 5,000 km²). Images will be coarse; consider splitting it.")

# ---------- step 3: image outputs ----------
st.subheader("Step 3 · Satellite image outputs")
data_panel.render(aoi)

TABS = ["High-resolution image", "Iron oxide/ferric iron", "Clay/hydroxyl alteration",
        "Band ratios", "PCA images", "Lithological classification", "Geological lineaments",
        "DEM / hillshade", "⭐ Integrated prospectivity"]
RENDER = [tab_image.render, lambda a: tab_index.render(a, "iron"), lambda a: tab_index.render(a, "clay"),
          tab_ratios.render, tab_pca.render, tab_lithology.render, tab_lineaments.render, tab_dem.render,
          tab_prospectivity.render]
# lazy tabs: only the selected tab runs, so the page stays fast
for tab, render in zip(st.tabs(TABS, key="step3_tab", on_change="rerun"), RENDER):
    if tab.open:
        with tab:
            render(aoi)
