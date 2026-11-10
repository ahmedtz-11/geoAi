"""Satellite data loader shared by all Step-3 maps: Sentinel-2 scene + Copernicus 30 m DEM."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from geoai.imagery import dem as dem_mod
from geoai.imagery import sentinel

from .common import cached_search, get_scene, ss


def _load(aoi, g, res):
    bar = st.progress(0.0, f"Loading Sentinel-2 scene of {g['date']}…")
    data = sentinel.load_scene(g["items"], aoi, res, progress=lambda p, t: bar.progress(min(p, 1.0) * 0.85, t))
    bar.progress(0.9, "Loading elevation (Copernicus DEM 30 m)…")
    dem, err = None, None
    try:
        h, w = data["inside"].shape
        dem = dem_mod.load_dem(aoi, data["epsg"], data["transform"], w, h)
    except Exception as e:  # imagery is still useful without elevation
        err = str(e)
    bar.empty()
    ss.s2_scene = {"key": (aoi.key, g["date"], g["orbit"], res), "data": data, "dem": dem, "dem_error": err,
                   "cloud": g["cloud"], "tiles": g["tiles"]}
    ss.pop("rs_memo_store", None)


def _auto(aoi):
    try:
        with st.spinner("Searching the last 6 months for a scene with under 20% cloud…"):
            found = cached_search(aoi.geometry_json(), str(date.today() - timedelta(days=180)), str(date.today()), 20)
        found = [g for g in found if g["coverage"] > 0.99] or found
        if not found:
            st.warning("No scene with under 20% cloud in the last 6 months. Choose a date range below.")
            return
        _load(aoi, found[0], sentinel.auto_resolution(aoi))
        st.rerun()
    except Exception as e:
        st.error(f"Could not get satellite data: {e}")


def _chooser(aoi):
    c1, c2, c3 = st.columns(3)
    start = c1.date_input("From", date.today() - timedelta(days=120), key="dp_start")
    end = c2.date_input("To", date.today(), key="dp_end")
    max_cloud = c3.slider("Max cloud cover (%)", 0, 100, 20, key="dp_cloud")
    if st.button("🔎 Search scenes", key="dp_search"):
        try:
            with st.spinner("Searching the Sentinel-2 archive…"):
                ss.dp_groups = cached_search(aoi.geometry_json(), str(start), str(end), max_cloud)
            ss.dp_groups_key = aoi.key
        except Exception as e:
            st.error(f"Search failed: {e}")
    groups = ss.get("dp_groups") if ss.get("dp_groups_key") == aoi.key else None
    if groups is None:
        return
    if not groups:
        st.warning("No scenes found. Widen the date range or raise the cloud limit.")
        return
    table = pd.DataFrame([{"Date": g["date"], "Cloud (tile) %": round(g["cloud"], 1),
                           "AOI covered %": round(100 * g["coverage"], 1), "Tiles": g["tiles"],
                           "Satellite": g["platform"]} for g in groups])
    st.dataframe(table, hide_index=True, width="stretch", height=min(35 * len(table) + 38, 250))
    full = [i for i, g in enumerate(groups) if g["coverage"] > 0.99]
    pick = st.selectbox("Scene", range(len(groups)), full[0] if full else 0, key="dp_pick",
                        format_func=lambda i: f"{groups[i]['date']} · cloud {groups[i]['cloud']:.0f}% · "
                                              f"covers {100 * groups[i]['coverage']:.0f}% of AOI")
    auto = sentinel.auto_resolution(aoi)
    res = st.selectbox("Pixel size (m)", [r for r in sentinel.RESOLUTIONS if r >= auto], 0, key="dp_res",
                       help="10 m is the finest Sentinel-2 detail; large areas use bigger pixels to stay fast.")
    if st.button("⬇️ Load this scene", key="dp_load", type="primary"):
        try:
            _load(aoi, groups[pick], res)
            st.rerun()
        except Exception as e:
            st.error(f"Could not load the scene: {e}")


def render(aoi):
    sc = get_scene(aoi)
    with st.container(border=True):
        if sc is None:
            st.markdown("**Satellite data for the maps below:** Sentinel-2 multispectral (10–20 m, revisit ~5 days) "
                        "and Copernicus 30 m elevation. Free sources, no account needed.")
            if st.button("🛰️ Get latest scene", type="primary", key="dp_auto"):
                _auto(aoi)
            with st.expander("…or choose a specific date"):
                _chooser(aoi)
        else:
            d = sc["data"]
            clear_pct = 100 * d["clear"].sum() / max(d["inside"].sum(), 1)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Scene date", d["date"])
            k2.metric("Pixel size", f"{d['res']} m")
            k3.metric("Clear pixels in AOI", f"{clear_pct:.0f}%")
            k4.metric("Elevation (DEM)", "Loaded" if sc["dem"] is not None else "Unavailable")
            if clear_pct < 60:
                st.warning("Much of the area is cloud, shadow or missing. Choose another date for reliable maps.")
            if sc["dem"] is None:
                st.warning(f"Elevation could not be loaded ({sc['dem_error']}). DEM and elevation-based "
                           "lineaments are unavailable; satellite-based lineaments still work.")
            with st.expander("Change scene"):
                _chooser(aoi)
            st.caption(f"Tiles {sc['tiles']} · Contains modified Copernicus Sentinel data and Copernicus DEM, "
                       "via Microsoft Planetary Computer.")
