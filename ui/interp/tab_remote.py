"""Remote sensing results: the uploaded layers as maps."""
import numpy as np
import streamlit as st

from geoai import interpret, plots

from .common import dataset_picker, PREFIX, safe_name, show_fig
from ui.state import fragment


@fragment
def render(run):
    items = run["datasets"]["remote"]
    if not items:
        st.info("No remote sensing data were uploaded. Tip: download GeoTIFFs from the Section 1 tabs "
                "(clay, iron oxide, lineament density, prospectivity) and upload them here.")
        return
    i = dataset_picker(items, "Dataset", f"{PREFIX}rs_ds")
    ds = items[i]
    if ds["kind"] == "image_ref":
        x0, x1, y0, y1 = ds["extent"]
        show_fig(plots.image_figure(ds["image"], [x0, x1, y0, y1], ds["name"]), "remote_image.png", f"rs_png_ref_{i}")
        return
    if ds.get("from_image"):
        st.warning("Values come from the colours of a picture (relative 0–1 scale).")
    names = list(ds["layers"])
    if ds["kind"] == "grid":
        arrays, tr = [ds["layers"][n] for n in names], ds["transform"]
    else:
        tr, shape, _ = run["grid"]
        arrays = [interpret.points_to_grid(ds["x"], ds["y"], ds["layers"][n], tr, shape) for n in names]
    for n, a in zip(names, arrays):
        v = a[np.isfinite(a)]
        c1, c2 = st.columns([3, 1])
        with c1:
            show_fig(plots.grid_figure(a, tr, f"{n} · {ds['name']}", "magma"), f"{safe_name(n)}.png",
                     f"rs_png_{i}_{safe_name(n)}")
        with c2:
            if v.size:
                st.metric("Median", f"{np.median(v):.3g}")
                st.metric("P95", f"{np.percentile(v, 95):.3g}")
                st.metric("Max", f"{v.max():.3g}")
    st.caption("Each layer becomes an evidence layer in the Gold prediction tab (high values favourable by default; "
               "change the direction there if low values are favourable).")
