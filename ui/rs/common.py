"""Shared helpers for the remote-sensing tabs: session state, caching, figure display, tables."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pyproj import Transformer

from geoai import export, indices, lineaments, lithology, pca, terrain
from geoai.imagery import sentinel

ss = st.session_state
VEG_NDVI = 0.4

# Widget defaults. Keys start with "rs_" so their values survive switching tabs (see keep_widget_state).
DEFAULTS = {
    "rs_iron_method": "Band ratio", "rs_iron_veg": True,
    "rs_clay_method": "Band ratio", "rs_clay_veg": True,
    "rs_ratio_preset": "Sabins (R=B11/B12, G=B4/B2, B=B8/B4)",
    "rs_ratio_r": "B11/B12", "rs_ratio_g": "B4/B2", "rs_ratio_b": "B8/B4",
    "rs_pca_rgb": "PC1-PC2-PC3",
    "rs_lith_k": 6,
    "rs_lin_source": "Elevation (hillshade)", "rs_lin_pct": 90, "rs_lin_sigma": 2.0,
    "rs_lin_len": 300, "rs_lin_gap": 60, "rs_lin_radius": 500,
    "rs_dem_az": 315, "rs_dem_alt": 45, "rs_dem_z": 1.0, "rs_dem_contours": True,
    "rs_p_w_clay": 1.0, "rs_p_w_iron": 0.7, "rs_p_w_lden": 0.8, "rs_p_w_lprox": 0.6, "rs_p_w_lith": 0.4,
    "rs_p_method": "Weighted sum", "rs_p_gamma": 0.8, "rs_p_veg": True, "rs_p_top": 5.0, "rs_p_minha": 0.5,
}


def keep_widget_state():
    """Streamlit forgets widget values of tabs that are not shown; re-saving them keeps user choices."""
    for k, v in DEFAULTS.items():
        ss.setdefault(k, v)
    for k in [k for k in ss.keys() if isinstance(k, str) and k.startswith("rs_")]:
        ss[k] = ss[k]


# ---------- scene state ----------
def get_scene(aoi):
    s = ss.get("s2_scene")
    return s if s and s["key"][0] == aoi.key else None


def need_scene():
    st.info("Load satellite data first with **Get latest scene** above (or choose a date).")


def memo(scene_key, name, fn, *params):
    """Per-scene cache for heavy results (PCA, classification, lineaments ...)."""
    store = ss.setdefault("rs_memo_store", {})
    if store.get("_scene") != scene_key:
        store.clear()
        store["_scene"] = scene_key
    k = (name,) + tuple(params)
    if k not in store:
        if len(store) > 30:
            for old in [x for x in store if x != "_scene"][:10]:
                store.pop(old, None)
        store[k] = fn()
    return store[k]


@st.cache_data(show_spinner=False, ttl=1800, max_entries=20)
def cached_search(geometry_json, start, end, max_cloud):
    return sentinel.search_scenes(json.loads(geometry_json), start, end, max_cloud)


# ---------- masks & shared products ----------
def usable_mask(sc, hide_veg):
    clear = sc["data"]["clear"]
    if not hide_veg:
        return clear
    return clear & (indices.compute_index(sc["data"]["bands"], "ndvi") <= VEG_NDVI)


def std_pca(sc):
    d = sc["data"]
    return memo(sc["key"], "pca", lambda: pca.pca(d["bands"], pca.PCA_BANDS, d["clear"]))


def crosta(sc, kind):
    d = sc["data"]
    return memo(sc["key"], "crosta", lambda: pca.crosta(d["bands"], d["clear"], kind), kind)


def index_layer(sc, kind, method):
    """Clay ('clay') or iron ('iron') evidence image by band ratio or Crósta PCA."""
    if method.startswith("Crósta"):
        return crosta(sc, "hydroxyl" if kind == "clay" else "iron")["image"]
    return indices.compute_index(sc["data"]["bands"], "clay_hydroxyl" if kind == "clay" else "iron_oxide")


def lith(sc, k):
    d = sc["data"]
    return memo(sc["key"], "lith", lambda: lithology.classify(d["bands"], d["clear"], k), k)


def hillshades(sc):
    dem, res = sc.get("dem"), sc["data"]["res"]
    if dem is None:
        return None
    return memo(sc["key"], "hs4", lambda: [terrain.hillshade(dem, res, a, 30) for a in (0, 45, 90, 135)])


def lineament_result(sc, source, pct, sigma, length, gap):
    d = sc["data"]
    valid = d["inside"] & np.isfinite(d["bands"]["B04"])

    def run():
        imgs = []
        if source in ("Elevation (hillshade)", "Both") and sc.get("dem") is not None:
            imgs += hillshades(sc)
            valid_ = valid & np.isfinite(sc["dem"])
        else:
            valid_ = valid
        if source in ("Satellite (PC1)", "Both") or not imgs:
            imgs.append(std_pca(sc)["pcs"][0])
        return lineaments.detect(imgs, valid_, d["res"], sigma, pct, length, gap)

    return memo(sc["key"], "lin", run, source, pct, sigma, length, gap)


def current_lineaments(sc):
    """Lineaments with the settings currently chosen on the lineament tab."""
    src = ss.get("rs_lin_source", DEFAULTS["rs_lin_source"])
    if sc.get("dem") is None and src != "Satellite (PC1)":
        src = "Satellite (PC1)"
    return lineament_result(sc, src, ss["rs_lin_pct"], ss["rs_lin_sigma"], ss["rs_lin_len"], ss["rs_lin_gap"])


# ---------- display ----------
def show_fig(fig, filename, label="Download PNG", key=None):
    st.pyplot(fig)
    st.download_button(label, lambda: export.fig_to_png(fig), filename, "image/png", key=key or f"dl_{filename}",
                       on_click="ignore")
    plt.close(fig)  # a closed figure can still be saved when the button is clicked


def tif_button(label, layers_fn, filename, key, transform, epsg):
    st.download_button(label, lambda: export.geotiff_bytes(layers_fn(), transform, epsg), filename, "image/tiff",
                       key=key, on_click="ignore")


def zones_table(zones, epsg):
    if not zones:
        return pd.DataFrame()
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    rows = []
    for i, z in enumerate(zones, 1):
        lon, lat = to_ll.transform(z["easting"], z["northing"])
        rows.append({"Zone": f"T{i}", "Latitude": round(lat, 6), "Longitude": round(lon, 6),
                     "Easting": round(z["easting"], 1), "Northing": round(z["northing"], 1),
                     "Area (ha)": round(z["area_ha"], 2), "Mean score": round(z["mean_score"], 3),
                     "Max score": round(z["max_score"], 3)})
    return pd.DataFrame(rows)


def header(sc, aoi):
    d = sc["data"]
    return d["bands"], d["transform"], d["epsg"], d["inside"], d["clear"], aoi.to_utm(d["epsg"])
