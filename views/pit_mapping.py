"""Pit mapping: pit coordinates (+ optional picture and notes) -> mineralisation trend, remaining potential
of each pit, proposed neighbour pits, and a pit register kept for future use."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from geoai import coords, export, pits, readers

REGISTER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pit_register.csv")
ss = st.session_state

st.title("Pit mapping")
st.caption("Upload the coordinates of your pits. GeoAI finds the mineralisation trend, rates how much potential "
           "each pit still has, proposes where to sink the next pits, and keeps every coordinate in a register "
           "for future use.")

st.subheader("Step 1 · Upload")
c1, c2, c3 = st.columns(3)
f_tab = c1.file_uploader("Pit coordinates (Excel, CSV or TXT)", type=["xls", "xlsx", "csv", "txt"], key="pit_table")
f_img = c2.file_uploader("Picture: pit sketch, photo or map (optional)", type=["png", "jpg", "jpeg"], key="pit_img")
f_txt = c3.file_uploader("Field notes (TXT, optional)", type=["txt"], key="pit_notes")
c1.caption("One row per pit. Coordinates are required; depth, gold grade, status and vein strike are used when "
           "the file has them. Try `sample_data/pits_sample.xlsx`.")

spec = None
if f_tab is not None:
    try:
        sheet = None
        if f_tab.name.lower().endswith((".xls", ".xlsx")):
            sheets = readers.excel_sheets(f_tab)
            if len(sheets) > 1:
                sheet = st.selectbox("Sheet", sheets, key="pit_sheet")
        df = readers.read_upload(f_tab, sheet)["df"]
    except Exception as e:
        df = None
        st.error(f"Could not read the file: {e}")
    if df is not None and len(df.columns) >= 2:
        with st.container(border=True):
            st.dataframe(df.head(8), hide_index=True, width="stretch")
            cols = list(df.columns)
            gx, gy = coords.guess_columns(df)
            a, b, c = st.columns(3)
            xcol = a.selectbox("Easting / longitude column", cols, cols.index(gx) if gx in cols else 0, key="pit_x")
            ycol = b.selectbox("Northing / latitude column", cols, cols.index(gy) if gy in cols else 1, key="pit_y")
            xs, ys = coords.parse_column(df[xcol]).dropna(), coords.parse_column(df[ycol]).dropna()
            geo = len(xs) > 0 and len(ys) > 0 and coords.looks_geographic(xs, ys)
            names = list(coords.CRS_OPTIONS)
            crs = c.selectbox("Coordinate system", names, 0 if geo else 2, key=f"pit_crs_{int(geo)}")
            guess = pits.guess_fields(df, used=(xcol, ycol))
            opt = ["(none)"] + [k for k in cols if k not in (xcol, ycol)]
            labels = {"pit": "Pit name / ID", "depth": "Pit depth (m)", "grade": "Gold grade", "status": "Status / remarks",
                      "strike": "Vein strike (°)"}
            fields = {}
            for col, (k, lab) in zip(st.columns(len(labels)), labels.items()):
                pick = col.selectbox(lab, opt, opt.index(guess[k]) if guess.get(k) in opt else 0, key=f"pit_f_{k}")
                fields[k] = None if pick == "(none)" else pick
            spec = {"df": df, "x": xcol, "y": ycol, "epsg": coords.CRS_OPTIONS[crs], "fields": fields}

img = Image.open(f_img).convert("RGB") if f_img is not None else None
notes = f_txt.getvalue().decode("utf-8", errors="ignore").strip() if f_txt is not None else ""
extent = None
if img is not None:
    with st.expander("Picture options"):
        if st.checkbox("This picture is a map: draw it behind the pits", key="pit_img_map"):
            st.caption("Type the coordinates of the picture's edges, in the same coordinate system the pits end up "
                       "in (UTM metres).")
            e1, e2, e3, e4 = st.columns(4)
            extent = (e1.number_input("West", value=0.0, format="%.1f"), e2.number_input("East", value=0.0, format="%.1f"),
                      e3.number_input("South", value=0.0, format="%.1f"), e4.number_input("North", value=0.0, format="%.1f"))
            if not (extent[1] > extent[0] and extent[3] > extent[2]):
                extent = None

st.subheader("Step 2 · Submit")
a, b, c = st.columns([2, 1, 1])
project = a.text_input("Project / site name", "My site", key="pit_project",
                       help="Pits are saved in the register under this name.")
max_t = b.slider("Most new pits to propose", 2, 20, 8, key="pit_max")
outs = c.slider("Step-outs beyond each end", 0, 4, 2, key="pit_outs")
if st.button("Submit & interpret", type="primary", disabled=spec is None, key="pit_go"):
    try:
        p, epsg = pits.build(spec["df"], spec["x"], spec["y"], spec["epsg"], spec["fields"])
        tr = pits.trend(p)
        p = pits.potential(p, tr)
        t = pits.propose(p, tr, epsg, max_t, outs)
        ss["pit_result"] = {"pits": p, "trend": tr, "targets": t, "epsg": epsg, "project": project.strip() or "My site",
                            "img": img, "extent": extent, "notes": notes}
    except Exception as e:
        ss.pop("pit_result", None)
        st.error(f"The interpretation failed: {e}")

res = ss.get("pit_result")
if not res:
    st.info("Upload the pit coordinates, check the columns, then press **Submit & interpret**.")
else:
    p, tr, t, epsg = res["pits"], res["trend"], res["targets"], res["epsg"]
    st.subheader("Step 3 · Interpretation")
    k = st.columns(5)
    k[0].metric("Pits", len(p))
    k[1].metric("Trend", f"{tr['azimuth']:.0f}° {pits.strike_label(tr['azimuth'])}")
    k[2].metric("Trend length", f"{tr['length']:.0f} m")
    k[3].metric("High potential", int((p.potential == "High").sum()))
    k[4].metric("Proposed pits", len(t))
    for line in pits.summary(p, tr, t):
        st.markdown("- " + line)
    st.caption("Screening only: the ratings compare pits with each other using the data in your file. They are not "
               "a resource estimate. Confirm with sampling before spending on a new pit.")

    tabs = st.tabs(["Map", "Pits: what is left", "Proposed pits", "Long section", "Picture & notes", "Pit register"])
    with tabs[0]:
        bg = (res["img"], res["extent"]) if res["img"] is not None and res["extent"] else None
        fig = pits.map_figure(p, tr, t, f"Pit map · {res['project']} · EPSG:{epsg}", bg)
        png = export.fig_to_png(fig)
        st.pyplot(fig)
        plt.close(fig)
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("Map (PNG)", png, "pit_map.png", "image/png", key="pit_dl_png")
        d2.download_button("Google Earth (KML)", pits.to_kml(res["project"], p, t), "pits.kml",
                           "application/vnd.google-earth.kml+xml", key="pit_dl_kml")
        st.caption("Circle size follows gold grade. The yellow band is the mineralised corridor; stars are the "
                   "proposed pits.")
    with tabs[1]:
        show = p[["pit", "potential", "score", "grade", "depth", "status", "why", "x", "y", "lat", "lon"]].rename(
            columns={"pit": "Pit", "potential": "Remaining potential", "score": "Score", "grade": "Grade",
                     "depth": "Depth (m)", "status": "Status", "why": "Why", "x": "Easting", "y": "Northing",
                     "lat": "Latitude", "lon": "Longitude"}).sort_values("Score", ascending=False)
        st.dataframe(show.round({"Easting": 1, "Northing": 1, "Latitude": 6, "Longitude": 6}), hide_index=True,
                     width="stretch")
        st.download_button("Pit ratings (CSV)", show.to_csv(index=False), "pit_ratings.csv", "text/csv", key="pit_dl_r")
        st.caption("**High**: on the trend, good grade here or next door, and room left below the pit. "
                   "**Low**: off the trend, poor grade, or recorded as worked out.")
    with tabs[2]:
        if not len(t):
            st.info("No new pits to propose: the pits are already evenly spaced along the trend and step-outs are "
                    "set to 0.")
        else:
            show = t[["target", "kind", "priority", "score", "x", "y", "lat", "lon", "expected_depth", "nearest_pit",
                      "why"]].rename(columns={"target": "Target", "kind": "Type", "priority": "Priority",
                                              "score": "Score", "x": "Easting", "y": "Northing", "lat": "Latitude",
                                              "lon": "Longitude", "expected_depth": "Expected depth (m)",
                                              "nearest_pit": "Nearest pit", "why": "Why here"})
            st.dataframe(show.round({"Easting": 1, "Northing": 1, "Latitude": 6, "Longitude": 6}), hide_index=True,
                         width="stretch")
            st.download_button("Proposed pits (CSV)", show.to_csv(index=False), "proposed_pits.csv", "text/csv",
                               key="pit_dl_t")
            st.caption("**Infill** pits test gaps between two mineralised pits (lowest risk). **Step-out** pits "
                       "extend the trend beyond the last pit; the further out, the lower the score.")
    with tabs[3]:
        if np.isfinite(p.depth).sum() >= 2:
            fig = pits.section_figure(p, tr, t, f"Long section along the trend · {res['project']}")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Add a pit depth column to see the long section and where the ore gets deeper.")
    with tabs[4]:
        if res["img"] is None and not res["notes"]:
            st.info("No picture or notes were uploaded.")
        if res["img"] is not None:
            st.image(res["img"], caption="Uploaded picture", width="stretch")
        if res["notes"]:
            st.markdown("**Field notes**")
            st.text(res["notes"])
    with tabs[5]:
        st.caption(f"The register keeps pit coordinates between sessions, in `data/pit_register.csv` inside the GeoAI "
                   "folder. Saving a project again updates its pits.")
        if st.button(f"Save these {len(p)} pits and {len(t)} proposed pits to the register", key="pit_save"):
            try:
                pits.save_register(REGISTER, pits.register_rows(res["project"], p, t, epsg))
                st.success(f"Saved under “{res['project']}”.")
            except Exception as e:
                st.error(f"Could not save the register: {e}")

st.divider()
st.subheader("Saved pit register")
reg = pits.load_register(REGISTER)
if not len(reg):
    st.caption("Nothing saved yet. Interpret a pit file and press Save in the **Pit register** tab.")
else:
    projects = sorted(reg.project.dropna().unique())
    pick = st.multiselect("Projects", projects, projects, key="pit_reg_projects")
    view = reg[reg.project.isin(pick)]
    st.dataframe(view, hide_index=True, width="stretch", height=min(38 + 35 * len(view), 420))
    r1, r2, _ = st.columns([1, 1, 3])
    r1.download_button("Register (CSV)", view.to_csv(index=False), "pit_register.csv", "text/csv", key="pit_dl_reg")
    with r2.popover("Remove a project"):
        gone = st.selectbox("Project to remove from the register", projects, key="pit_reg_rm")
        if st.button("Remove", key="pit_reg_rm_go"):
            reg[reg.project != gone].to_csv(REGISTER, index=False)
            st.rerun()
