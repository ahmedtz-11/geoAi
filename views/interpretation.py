"""Section 6 - Integrated interpretation & gold prediction.

Combines what was submitted on the data pages (2 Magnetic, 3 IP, 4 Geology, 5 Geochemistry) with remote sensing
(Section 1 results from this session and/or uploaded GeoTIFFs), then predicts gold targets.
"""
import numpy as np
import streamlit as st

from geoai import indices, interpret
from ui.interp import tab_prediction, tab_remote, tab_summary, upload
from ui.interp.common import new_run_id, ss

PAGES = {"magnetic": ("2 · Magnetic", "views/magnetic.py"),
         "ip": ("3 · IP", "views/ip.py"),
         "geology": ("4 · Geology", "views/geology.py"),
         "geochem": ("5 · Geochemistry", "views/geochemistry.py")}


def section1_spec():
    """Clay / hydroxyl and iron oxide indices from the Section 1 scene loaded in this session."""
    sc = ss.get("s2_scene")
    if not sc:
        return None
    d = sc["data"]
    b = d["bands"]
    layers = {"Section 1 clay / hydroxyl (B11/B12)": np.where(d["clear"], indices.compute_index(b, "clay_hydroxyl"), np.nan),
              "Section 1 iron oxide (B4/B2)": np.where(d["clear"], indices.compute_index(b, "iron_oxide"), np.nan)}
    return {"slot": "remote", "name": f"Section 1 · Sentinel-2 {d['date']}", "kind": "raster",
            "data": {"kind": "raster", "layers": layers, "transform": d["transform"], "epsg": d["epsg"]},
            "bands": list(layers)}


st.title("6 · Integrated interpretation & gold prediction")
st.caption("Brings together everything you have submitted on the other pages and predicts where gold is most "
           "likely, with estimated depth and structural trend for each target.")

runs = ss.get("s2_runs", {})
st.subheader("Step 1 · Data to combine")
cols = st.columns(5)
with cols[0], st.container(border=True):
    st.markdown("**1 · Remote sensing**")
    s1 = section1_spec()
    use_s1 = False
    if s1:
        use_s1 = st.checkbox(f"Use Section 1 scene ({ss['s2_scene']['data']['date']}): clay + iron oxide", True,
                             key="use_section1")
    else:
        st.caption("No Section 1 scene loaded in this session.")
    st.page_link("views/remote_sensing.py", label="Open remote sensing", icon="🛰️")
for col, (slot, (label, path)) in zip(cols[1:], PAGES.items()):
    with col, st.container(border=True):
        st.markdown(f"**{label}**")
        e = runs.get(slot)
        if e:
            st.success(f"{len(e['specs'])} file(s) submitted")
            st.caption(", ".join(s["name"] for s in e["specs"]))
        else:
            st.caption("Nothing submitted yet.")
        st.page_link(path, label="Open page", icon="➡️")

rs_specs = []
if st.toggle("Add other remote sensing layers (GeoTIFF, CSV points or a map picture)", key="s5_add_rs"):
    rs_specs = upload.render_slot("remote")

specs = [s for e in runs.values() for s in e["specs"]] + rs_specs + ([s1] if use_s1 and s1 else [])
if not specs:
    st.info("Submit data on at least one of the pages above (or load a Section 1 scene) to build a prediction.")
    st.stop()

settings = {}
for e in runs.values():
    settings.update(e["settings"])
sig = (tuple(e["sig"] for e in runs.values()), upload.signature(rs_specs, {}),
       ss["s2_scene"]["key"] if use_s1 and s1 else None)
comb = ss.get("s2_combined")
if not comb or comb["sig"] != sig:
    bar = st.progress(0.0, "Combining data…")
    try:
        run = interpret.run(specs, settings, progress=lambda p, t: bar.progress(min(p, 1.0), t))
        run["id"] = new_run_id()
        ss.s2_combined = comb = {"sig": sig, "run": run}
    except Exception as e:
        bar.empty()
        st.error(f"Could not combine the data: {e}")
        st.stop()
    bar.empty()
run = comb["run"]

st.subheader("Step 2 · Results")
TABS = [("Summary", lambda: tab_summary.render(run)),
        ("⭐ Gold prediction", lambda: tab_prediction.render(run, tab_summary.findings)),
        ("Remote sensing layers", lambda: tab_remote.render(run))]
for tab, (_, render) in zip(st.tabs([t for t, _ in TABS], key="s5_tab", on_change="rerun"), TABS):
    if tab.open:
        with tab:
            render()
