"""Shared layout for the data pages (Magnetic / IP, Geology, Geochemistry): upload → submit → results.

Each page keeps its own run in st.session_state["s2_runs"][slot]; the Integrated prediction page
combines the submitted data of all pages.
"""
import time

import streamlit as st

from geoai import interpret

from . import upload
from .common import clear_widgets, new_run_id, ss

PREDICTION_PAGE = "views/interpretation.py"


def data_page(slot, title, intro, render_results, settings_ui=None):
    st.title(title)
    st.caption(intro)

    st.subheader("Step 1 · Upload")
    specs = upload.render_slot(slot)
    settings = settings_ui() if settings_ui else {}

    st.subheader("Step 2 · Submit")
    runs = ss.setdefault("s2_runs", {})
    entry = runs.get(slot)
    sig = upload.signature(specs, settings)
    c1, c2 = st.columns([1, 3])
    go = c1.button("🚀 Submit & interpret", type="primary", disabled=not specs, width="stretch", key=f"go_{slot}")
    c2.markdown(f"{len(specs)} file(s) ready" if specs else "Upload at least one file above.")
    if entry and specs and entry["sig"] != sig:
        st.info("Files or settings changed since the last run. Submit again to update the results.")

    if go:
        bar = st.progress(0.0, "Starting…")
        t0 = time.time()
        try:
            run = interpret.run(specs, settings, progress=lambda p, t: bar.progress(min(p, 1.0), t))
            run["id"] = new_run_id()
            runs[slot] = {"run": run, "sig": sig, "specs": specs, "settings": settings}
            clear_widgets(slot)
            ss.pop("s2_combined", None)  # the integrated prediction must be rebuilt
            bar.empty()
            st.success(f"Interpretation finished in {time.time() - t0:.0f} s.")
            st.rerun()
        except Exception as e:
            bar.empty()
            st.error(f"The interpretation failed: {e}")

    entry = runs.get(slot)
    if not entry:
        st.info("Upload your files, check their settings, then press **Submit & interpret**. "
                "Sample files are in the project's `sample_data` folder.")
        return
    st.subheader("Step 3 · Results")
    c1, c2 = st.columns([3, 1])
    c1.caption("Results of the last submitted files: " + ", ".join(s["name"] for s in entry["specs"]))
    if c2.button("Clear results", key=f"clear_{slot}"):
        runs.pop(slot, None)
        ss.pop("s2_combined", None)
        clear_widgets(slot)
        st.rerun()
    for e in entry["run"]["errors"]:
        st.error(e)
    render_results(entry["run"])
    st.divider()
    st.page_link(PREDICTION_PAGE, label="Next: combine with your other data in **6 · Integrated gold prediction**", icon="➡️")
