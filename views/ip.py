"""Section 3 - IP (induced polarisation): one file per survey line -> one pseudosection image per file.

Handles plain tables (station, n-level or depth, value) and instrument exports with electrode positions
(IRIS Prosys / ELREC Pro: Tx1, Tx2, Rx1, Rx2, Rho, M ...), including their information lines, remote
electrodes and data-quality columns.
"""
import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from geoai import coords, export, geophysics, interpret, ipdata, plots, readers
from geoai.coords import parse_column
from ui.interp import upload
from ui.interp.common import clear_widgets, new_run_id, safe_name, ss

NONE = "(none)"
MIN_READINGS = 6


def guess(cols, words, exclude=()):
    """First column matching a word. Short words (x, y, z, n, m) must be the whole column name."""
    for w in words:
        for c in cols:
            name = str(c).strip().lower()
            if c in exclude:
                continue
            if (name == w) if len(w) <= 2 else (w in name):
                return c
    return None


def num(text):
    text = (text or "").strip().replace(",", "")
    return float(text) if text else None


st.title("3 · IP (induced polarisation)")
st.caption("Upload one file per survey line (.csv or .txt). Each line gives one pseudosection image showing where "
           "chargeability (or resistivity) is anomalous and at what depth.")

# ---------- step 1: lines ----------
st.subheader("Step 1 · Lines and files")
n_lines = int(st.number_input("How many survey lines?", 1, 30, 1, key="ip_nlines"))
lines = []
for i in range(n_lines):
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        name = c1.text_input(f"Line {i + 1} name", f"L{i + 1}", key=f"ip_name_{i}")
        f = c2.file_uploader(f"File for line {name}", type=["csv", "txt"], key=f"ip_file_{i}")
        p1, p2, p3 = st.columns(3)
        e0 = p1.text_input("Line start easting (m, optional)", "", key=f"ip_e0_{i}",
                           help="With the start northing and bearing, places a line whose stations are distances "
                                "along the line (e.g. 0, 20, 40 …) on the map.")
        north = p2.text_input("Line start northing (m, optional)", "", key=f"ip_north_{i}",
                              help="Alone (no start easting): the line's northing, for files whose station values "
                                   "are already eastings.")
        bearing = p3.text_input("Line bearing (° from north)", "90", key=f"ip_bear_{i}",
                                help="Direction in which station values increase: 90 = towards east, 0 = north.")
        lines.append({"name": name.strip() or f"L{i + 1}", "file": f, "e0": e0, "north": north, "bearing": bearing})

uploaded = [l for l in lines if l["file"] is not None]
specs, settings, summary = [], {}, []
if uploaded:
    first = upload._read(uploaded[0]["file"].name, uploaded[0]["file"].getvalue(), None)
    df0 = first["df"]
    cols = list(df0.columns)
    electrodes = ipdata.has_electrodes(df0)
    st.markdown("**Columns** (the same layout is used for every line file)")
    st.dataframe(df0.head(4).astype(str), height=150, width="stretch")

    if electrodes:
        ec = ipdata.electrode_columns(df0)
        _, arr0, a0 = ipdata.geometry(df0)
        st.success(f"Electrode positions found ({ec['c1']}, {ec['c2']}, {ec['p1']}, {ec['p2']}): **{arr0}** array, "
                   f"dipole a = **{a0:g} m**. Station positions and pseudo-depths are calculated from the electrodes "
                   "(remote electrodes such as 9999999 are ignored).")
        c1, c2 = st.columns(2)
        others = [c for c in readers.numeric_columns(df0) if c not in (ec["m"], ec["rho"])]
        values = [c for c in (ec["m"], ec["rho"]) if c] + others
        labels = {ec["m"]: f"Chargeability {ec['m']} (mV/V)", ec["rho"]: f"Apparent resistivity {ec['rho']} (ohm·m)"}
        vcol = c1.selectbox("Value to show", values, format_func=lambda c: labels.get(c, str(c)), key="ip_vcol_e")
        arrays = ["Pole-dipole", "Dipole-dipole", "Wenner", "Schlumberger", "Pole-pole"]
        array = c2.selectbox("Array", arrays, arrays.index(arr0) if arr0 in arrays else 0, key="ip_array_e",
                             help="Detected from the Array column; used for the pseudo-depths.")
        is_charge = vcol == ec["m"]
    else:
        c1, c2, c3 = st.columns(3)
        xcol = c1.selectbox("Station / distance column", cols,
                            cols.index(guess(cols, ["station", "east", "dist", "x"]) or cols[0]), key="ip_xcol")
        vguess = guess(cols, ["charg", "mv/v", "msec", "ip", "m"]) or guess(cols, ["resist", "ohm", "rho"])
        vopts = [c for c in cols if c != xcol]
        vcol = c2.selectbox("Value to show", vopts, vopts.index(vguess) if vguess in vopts else 0, key="ip_vcol")
        ycol = c3.selectbox("Northing column in the file (optional)", [NONE] + cols,
                            ([NONE] + cols).index(guess(cols, ["north", "y"]) or NONE), key="ip_ycol")
        ycol = None if ycol == NONE else ycol
        c1, c2, c3, c4 = st.columns(4)
        mode = c1.radio("Depth given as", ["n-level", "Depth column"], key="ip_dmode",
                        index=0 if any(str(c).lower() in ("n", "n_level", "nlevel", "level") for c in cols) else 1)
        if mode == "n-level":
            ncands = [c for c in cols if c not in (xcol, vcol)]
            ng = next((c for c in ncands if str(c).lower() in ("n", "n_level", "nlevel", "level")), ncands[0])
            ncol = c2.selectbox("n-level column", ncands, ncands.index(ng), key="ip_ncol")
            array = c3.selectbox("Array", ["Dipole-dipole", "Pole-dipole", "Wenner / Schlumberger"], key="ip_array")
            a_sp = c4.number_input("Dipole spacing a (m)", 1.0, 1000.0, 50.0, key="ip_a")
            depth_cfg = {"depth_mode": "n-level", "ncol": ncol, "array": array, "a_spacing": a_sp}
        else:
            dcands = [c for c in cols if c not in (xcol, vcol)]
            dcol = c2.selectbox("Depth column (m)", dcands, dcands.index(guess(dcands, ["depth", "z"]) or dcands[0]),
                                key="ip_dcol")
            depth_cfg = {"depth_mode": "Depth column", "depthcol": dcol}
        is_charge = not any(w in str(vcol).lower() for w in ["res", "ohm", "rho"])

    with st.expander("Data-quality filters", expanded=electrodes):
        q1, q2, q3, q4 = st.columns(4)
        drop_neg = q1.checkbox("Remove negative / zero values", True, key="ip_q_neg",
                               help="Negative chargeability usually means coupling or noise.")
        drop_contact = q2.checkbox("Remove bad electrode contacts", True, key="ip_q_contact",
                                   disabled=not electrodes, help="Contact values such as INFINI or ------.")
        max_err = q3.number_input("Max chargeability error (ErrM)", 0.0, 1000.0, 10.0, key="ip_q_err",
                                  disabled=not (electrodes and is_charge),
                                  help="Readings whose error is larger than this are removed.") \
            if electrodes and is_charge else None
        drop_n1 = q4.checkbox("Exclude n = 1", False, key="ip_q_n1",
                              help="The shallowest level, next to the current electrode, is often affected by "
                                   "electrode effects and poor contact.")

    survey = geophysics.SURVEYS[1] if is_charge else geophysics.SURVEYS[2]
    needs_crs = any(l["north"] for l in uploaded) or (not electrodes and ycol)
    epsg = None
    if needs_crs:
        epsg = coords.CRS_OPTIONS[st.selectbox("Coordinate system of the line positions",
                                               [k for k in coords.CRS_OPTIONS if "4326" not in k], 1, key="ip_crs")]

    for l in uploaded:
        try:
            data = upload._read(l["file"].name, l["file"].getvalue(), None)
            df = data["df"].copy()
            if electrodes:
                if not ipdata.has_electrodes(df):
                    raise ValueError("no electrode columns (Tx1, Tx2, Rx1, Rx2) like the first file")
                g, arr, a = ipdata.geometry(df, array)
                df["_station_x"], df["_pseudo_depth"] = g.station_x.to_numpy(), g.pseudo_depth.to_numpy()
                keep, why = ipdata.quality_mask(df, vcol, max_err, drop_neg, drop_contact)
                if drop_n1:
                    bad = keep & (g.n_level.to_numpy() == 1)
                    why["n = 1 excluded"] = int(bad.sum())
                    keep &= ~bad
                cfg = {"xcol": "_station_x", "depth_mode": "Depth column", "depthcol": "_pseudo_depth", "ycol": None}
            else:
                missing = [c for c in [xcol, vcol, depth_cfg.get("ncol"), depth_cfg.get("depthcol"), ycol]
                           if c and c not in df.columns]
                if missing:
                    raise ValueError(f"missing column(s) {', '.join(map(str, missing))}")
                v = parse_column(df[vcol]).to_numpy(float)
                keep = np.isfinite(v) & np.isfinite(parse_column(df[xcol]).to_numpy(float))
                why = {"missing value": int((~keep).sum())}
                if drop_neg:
                    bad = keep & (v <= 0)
                    why["negative or zero"] = int(bad.sum())
                    keep &= ~bad
                if drop_n1 and depth_cfg.get("ncol"):
                    bad = keep & (parse_column(df[depth_cfg["ncol"]]).to_numpy(float) == 1)
                    why["n = 1 excluded"] = int(bad.sum())
                    keep &= ~bad
                cfg = {"xcol": xcol, "ycol": ycol, **depth_cfg}
            df = df[keep].reset_index(drop=True)
            stations = df[cfg["xcol"]].nunique()
            e0, n0, brg = num(l["e0"]), num(l["north"]), num(l["bearing"])
            ok = len(df) >= MIN_READINGS and stations >= 2
            summary.append({"Line": l["name"], "File": l["file"].name, "Readings": int(len(keep)),
                            "Kept": int(len(df)), "Removed": ", ".join(f"{k}: {v}" for k, v in why.items() if v) or "—",
                            "Status": "OK" if ok else f"Too few valid readings (need ≥ {MIN_READINGS} over ≥ 2 stations)"})
            if not ok:
                continue
            spec = {"slot": "geophysics", "name": f"{l['name']} · {l['file'].name}", "kind": "table",
                    "data": {"kind": "table", "df": df}, "survey": survey, "geometry": "section", "vcol": vcol,
                    "linecol": None, "line_name": l["name"], "epsg": epsg, **cfg,
                    "line_northing": None, "line_origin": None}
            if not cfg.get("ycol") and n0 is not None:
                if e0 is not None:
                    spec["line_origin"] = (e0, n0, 90.0 if brg is None else brg)
                else:
                    spec["line_northing"] = n0
            specs.append(spec)
        except ValueError as e:
            summary.append({"Line": l["name"], "File": l["file"].name, "Readings": 0, "Kept": 0, "Removed": "—",
                            "Status": f"Not usable: {e}"})
    st.markdown("**Files**")
    st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
    settings = {"ip_value": vcol, "quality": (drop_neg, drop_contact if electrodes else None, max_err, drop_n1)}

# ---------- step 2: submit ----------
st.subheader("Step 2 · Submit")
runs = ss.setdefault("s2_runs", {})
entry = runs.get("ip")
sig = upload.signature([{k: v for k, v in s.items() if k != "data"} | {"rows": len(s["data"]["df"])} for s in specs],
                       settings)
c1, c2 = st.columns([1, 3])
go = c1.button("🚀 Submit & interpret", type="primary", disabled=not specs, width="stretch", key="go_ip")
c2.markdown(f"{len(specs)} of {n_lines} line file(s) ready")
if entry and specs and entry["sig"] != sig:
    st.info("Files or settings changed since the last run. Submit again to update the images.")
if go:
    bar = st.progress(0.0, "Starting…")
    try:
        run = interpret.run(specs, settings, progress=lambda p, t: bar.progress(min(p, 1.0), t))
        run["id"] = new_run_id()
        runs["ip"] = {"run": run, "sig": sig, "specs": specs, "settings": settings}
        clear_widgets("ip")
        ss.pop("s2_combined", None)
        bar.empty()
        st.rerun()
    except Exception as e:
        bar.empty()
        st.error(f"The interpretation failed: {e}")

entry = runs.get("ip")
if not entry:
    st.info("Set the number of lines, upload one file per line, check the columns, then press **Submit & interpret**. "
            "Try the `sample_data/ip_line_*.csv` files.")
    st.stop()

# ---------- step 3: images ----------
run = entry["run"]
st.subheader("Step 3 · Pseudosections (one image per line)")
c1, c2, c3 = st.columns([2, 2, 1])
pct = c1.slider("Anomaly outline (percentile of all values)", 50, 99, 95, key="ip_pct",
                help="The black line encloses values above this percentile of all lines together.")
stretch = c2.slider("Colour stretch (upper percentile)", 80.0, 100.0, 98.0, 0.5, key="ip_stretch",
                    help="Lower it when a few extreme readings make the rest of the section one colour.")
if c3.button("Clear results", key="clear_ip"):
    runs.pop("ip", None)
    ss.pop("s2_combined", None)
    st.rerun()
for e in run["errors"]:
    st.error(e)
items = list(zip(run["datasets"]["geophysics"], run["results"]["geophysics"]))
if not items:
    st.warning("No line could be processed. Check the column choices and the quality filters above.")
    st.stop()
allv = np.concatenate([ds["v"] for ds, _ in items])
thr = float(np.nanpercentile(allv[np.isfinite(allv)], pct))
images = {}
for ds, r in items:
    for lid, L in r["lines"].items():
        d = L["data"].dropna()
        unit = ds["survey"].split("(")[-1].strip(")")
        param = "Chargeability" if "charge" in ds["survey"] else "Resistivity"
        cmap = "jet" if param == "Chargeability" else "viridis"
        title = f"{param} pseudosection · line {lid}  (black outline ≥ {thr:.1f} {unit})"
        try:
            xi, zi, G = geophysics.section_grid(d.x, d.depth, d.v)
            fig = plots.section_figure(xi, zi, G, (d.x, d.depth), thr, title, cmap, f"{param} ({unit})", stretch)
        except Exception:  # too few or aligned readings to interpolate: plot the readings themselves
            fig = plots.section_points_figure(d.x, d.depth, d.v, thr, title, cmap, f"{param} ({unit})", stretch)
        with st.container(border=True):
            st.pyplot(fig)
            an = geophysics.section_anomalies(d.x, d.depth, d.v, thr)
            c1, c2 = st.columns([3, 1])
            if len(an):
                a = an.iloc[0]
                c1.markdown(f"**Line {lid}**: strongest anomaly from **{a.from_x:,.0f} to {a.to_x:,.0f} m**, about "
                            f"**{a.depth_top:.0f}–{a.depth_bottom:.0f} m** deep, peak {a.peak:.1f} {unit} at "
                            f"{a.peak_x:,.0f} m" + (f" · {len(an) - 1} smaller anomaly(ies)" if len(an) > 1 else "")
                            + f" · {len(d)} readings.")
            else:
                c1.markdown(f"**Line {lid}**: no continuous anomaly above the outline threshold · {len(d)} readings.")
            png = export.fig_to_png(fig, 150)
            images[f"ip_{safe_name(lid)}.png"] = png
            c2.download_button("Download PNG", png, f"ip_{safe_name(lid)}.png", "image/png", on_click="ignore",
                               key=f"ip_png_{safe_name(lid)}")
        plt.close(fig)


def _zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in images.items():
            z.writestr(n, b)
    return buf.getvalue()


st.download_button("Download all images (ZIP)", _zip, "ip_pseudosections.zip", "application/zip", on_click="ignore",
                   key="ip_zip")
st.caption("Depths are approximate pseudo-depths (median depth of investigation for the array). High chargeability "
           "usually means disseminated sulphides (pyrite, arsenopyrite), often linked to gold, but graphite also "
           "responds; compare with resistivity and geology. Invert the data (e.g. Res2DInv) for true depths.")
st.divider()
st.page_link("views/interpretation.py", label="Next: combine with your other data in **6 · Integrated gold "
                                              "prediction**", icon="➡️")
