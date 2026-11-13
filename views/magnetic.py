"""Section 2 - Magnetic: one survey file -> 12 interpreted maps."""
import time
from datetime import date, datetime

import streamlit as st

from geoai import datasets, geophysics, interpret, magnetics, readers
from ui.interp import tab_magnetic, upload
from ui.interp.common import clear_widgets, new_run_id, ss

st.title("2 · Magnetic")
st.caption("Upload one magnetic survey file (ground or airborne TMI). GeoAI produces the standard interpretation "
           "maps: TMI, RTP/RTE, vertical derivatives, analytic signal, tilt, THDR, lineaments, upward continuation, "
           "SPI and Euler depths, and a magnetic target map.")

st.subheader("Step 1 · Upload the survey")
f = st.file_uploader("Magnetic data: .dat (Geosoft XYZ), .csv, .txt, .xyz, .xls/.xlsx, GeoTIFF grid or a map picture",
                     type=readers.ACCEPT, key="up_magnetic")
spec = None
if f is not None:
    with st.container(border=True):
        spec = upload.configure("geophysics", f, 0, survey=geophysics.SURVEYS[0])

settings = {}
if spec is not None:
    st.markdown("**Earth's magnetic field at the survey** (needed for RTP / RTE)")
    try:
        lon, lat = datasets._centre_lonlat(spec)
        c1, c2 = st.columns([1, 3])
        when = c1.date_input("Survey date", date.today(), key="mag_date")
        inc0, dec0, f0, src = magnetics.field_direction(lon, lat, datetime(when.year, when.month, when.day))
        c2.caption(f"From {src} at {lat:.4f}, {lon:.4f}: inclination **{inc0:.1f}°**, declination **{dec0:.1f}°**"
                   + (f", total field {f0:,.0f} nT" if f0 == f0 else "") + ".")
    except Exception:
        inc0, dec0 = -28.0, 0.5
        st.caption("Could not locate the survey; enter the field direction below.")
    c1, c2, c3 = st.columns(3)
    manual = c1.checkbox("Enter inclination / declination myself", key="mag_manual")
    inc = c2.number_input("Inclination (°)", -90.0, 90.0, float(round(inc0, 1)), disabled=not manual, key="mag_inc") \
        if manual else inc0
    dec = c3.number_input("Declination (°)", -180.0, 180.0, float(round(dec0, 1)), disabled=not manual,
                          key="mag_dec") if manual else dec0
    reduction = st.radio("Reduction", magnetics.REDUCTIONS, horizontal=True, key="mag_reduction",
                         help="RTP is unstable close to the magnetic equator; Auto switches to RTE when |inclination| "
                              "< 25°.")
    with st.expander("Advanced settings"):
        st.caption("Gridding, spike and levelling settings shape the **Filled** image. The **Normal** image always "
                   "uses the original processing: every reading, linear interpolation, no filling between lines.")
        b1, b2 = st.columns(2)
        auto_cell = b1.checkbox("Auto grid cell size", True,
                                help="A quarter of the survey line spacing (the usual 1/3-1/5 rule). Smaller "
                                     "cells than this cannot add real detail across the lines, they only add "
                                     "gaps and stripes.")
        cell = None if auto_cell else b1.number_input("Grid cell size (m)", 0.5, 5000.0, 25.0, step=0.5)
        grid_method = b2.selectbox("Gridding", geophysics.GRID_METHODS,
                                   help="Minimum curvature draws the smoothest surface through the readings and "
                                        "fills between the survey lines. Linear interpolation is simpler and "
                                        "leaves the data pattern more visible.")
        d1, d2 = st.columns(2)
        ds_on = d1.checkbox("Remove spikes", True,
                            help="Drops readings that disagree with their neighbours by more than the tolerance "
                                 "below — a magnetometer swung, dropped or carried past a fence records jumps of "
                                 "thousands of nT over a metre, which no rock produces. The grid has to honour "
                                 "every reading, so one spike becomes a bullseye and flattens the colours "
                                 "everywhere else.")
        despike_tol = d1.slider("Spike tolerance (robust σ)", 3.0, 20.0, 6.0, step=0.5, disabled=not ds_on,
                                help="Lower removes more. Check the count reported above the maps.") if ds_on else 0.0
        microlevel = d2.checkbox("Microlevelling (decorrugation)", False,
                                 help="Removes stripes running along the survey lines, left by diurnal drift and "
                                      "instrument offsets between lines (Minty, 1991). Helps a survey walked over "
                                      "several days; on a short, clean survey it can remove real structure that "
                                      "happens to trend along the lines, so check both.")
        a1, a2, a3, a4 = st.columns(4)
        si_label = a1.selectbox("Euler structural index", list(geophysics.STRUCTURAL_INDEX), 2,
                                help="0 contacts/faults, 1 dykes and veins, 2 pipes, 3 compact bodies.")
        win = a2.slider("Euler window (cells)", 5, 25, 10)
        auto = a3.checkbox("Auto noise smoothing", True,
                           help="Upward-continues the field by about one data spacing before derivatives.")
        smooth = None if auto else a3.number_input("Smoothing height (m)", 0.0, 2000.0, 20.0)
        up_auto = a4.checkbox("Auto upward-continuation height", True)
        up_h = None if up_auto else a4.number_input("Height (m)", 10.0, 5000.0, 200.0)
    settings = {"inc": float(inc), "dec": float(dec), "reduction": reduction, "si": geophysics.STRUCTURAL_INDEX[si_label],
                "euler_window": win, "smooth_m": smooth, "up_height": up_h, "cell": cell,
                "grid_method": grid_method, "despike_tol": despike_tol, "microlevel": microlevel, "normal_view": True}

st.subheader("Step 2 · Submit")
runs = ss.setdefault("s2_runs", {})
entry = runs.get("magnetic")
sig = upload.signature([spec] if spec else [], settings)
c1, c2 = st.columns([1, 3])
go = c1.button("🚀 Submit & interpret", type="primary", disabled=spec is None, width="stretch", key="go_magnetic")
if entry and spec and entry["sig"] != sig:
    st.info("File or settings changed since the last run. Submit again to update the maps.")
if go:
    bar = st.progress(0.0, "Starting…")
    t0 = time.time()
    try:
        run = interpret.run([spec], settings, progress=lambda p, t: bar.progress(min(p, 1.0), t))
        if run["errors"] or not run["datasets"]["geophysics"]:
            raise ValueError("; ".join(run["errors"]) or "no magnetic data found")
        run["id"] = new_run_id()
        runs["magnetic"] = {"run": run, "sig": sig, "specs": [spec], "settings": settings}
        clear_widgets("magnetic")
        ss.pop("s2_combined", None)
        bar.empty()
        st.rerun()
    except Exception as e:
        bar.empty()
        st.error(f"The interpretation failed: {e}")

entry = runs.get("magnetic")
if not entry:
    st.info("Upload a file, check the columns and the Earth's field, then press **Submit & interpret**. "
            "Try `sample_data/magnetics_tmi.dat`.")
    st.stop()
st.subheader("Step 3 · Maps")
c1, c2 = st.columns([3, 1])
c1.caption(f"Results for {entry['specs'][0]['name']}")
if c2.button("Clear results", key="clear_magnetic"):
    runs.pop("magnetic", None)
    ss.pop("s2_combined", None)
    clear_widgets("magnetic")
    st.rerun()
tab_magnetic.render(entry["run"])
st.divider()
st.page_link("views/interpretation.py", label="Next: combine with your other data in **6 · Integrated gold "
                                              "prediction**", icon="➡️")
