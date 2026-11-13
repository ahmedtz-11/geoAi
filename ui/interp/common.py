"""Shared helpers for the Section 2 interface."""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from geoai import export
from ui import state

ss = st.session_state
PREFIX = "s2w_"  # widget keys whose values must survive switching result tabs


# widget-key prefixes owned by each page (cleared when that page's data are re-submitted)
PAGE_PREFIXES = {
    "magnetic": [f"{PREFIX}mag_"],
    "ip": [f"{PREFIX}ip_"],
    "geology": [f"{PREFIX}fav"],
    "geochem": [f"{PREFIX}gc_"],
    "prediction": [f"{PREFIX}ev_", f"{PREFIX}method", f"{PREFIX}top", f"{PREFIX}minha", f"{PREFIX}rs_ds"],
}


def clear_widgets(*pages):
    for page in pages:
        for p in PAGE_PREFIXES.get(page, []):
            state.forget(p)


def new_run_id():
    ss["s2_run_counter"] = ss.get("s2_run_counter", 0) + 1
    return ss["s2_run_counter"]


def default(key, value):
    return state.initial(key, value)


def show_fig(fig, filename, key, label="Download PNG"):
    st.pyplot(fig)
    st.download_button(label, lambda: export.fig_to_png(fig), filename, "image/png", key=key, on_click="ignore")
    plt.close(fig)


def memo(run, name, fn, *params):
    """Cache heavy results inside the run they belong to (so each page keeps its own)."""
    store = run.setdefault("_memo", {})
    k = (name,) + tuple(params)
    if k not in store:
        if len(store) > 40:
            for old in list(store)[:15]:
                store.pop(old, None)
        store[k] = fn()
    return store[k]


def to_lonlat(epsg, x, y):
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(np.asarray(x, float), np.asarray(y, float))
    return np.asarray(lon), np.asarray(lat)


def dataset_picker(items, label, key):
    if len(items) == 1:
        return 0
    return st.selectbox(label, range(len(items)), format_func=lambda i: items[i]["name"], key=key)


def safe_name(s):
    return "".join(c if c.isalnum() else "_" for c in str(s)).strip("_")[:40] or "layer"
