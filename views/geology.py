"""Section 4 - Geology: structural measurements and / or a QDS geological map sheet.

Both uploads are optional; either on its own is enough. The results are stored where the other data pages
keep theirs (st.session_state["s2_runs"]["geology"]), so 6 - Integrated gold prediction picks them up.
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from geoai import coords, export, interpret, plots, readers, structure
from ui.interp.common import clear_widgets, new_run_id, ss, to_lonlat

TABLES = ["csv", "xls", "xlsx"]
MAPS = ["png", "jpg", "jpeg", "tif", "tiff"]
PREDICTION_PAGE = "views/interpretation.py"
NONE = "(none)"

st.title("4 · Geology")
st.caption("Structural measurements from the field and / or a QDS geological map sheet. GeoAI settles which "
           "way each plane dips, separates the readings into structural sets, works out where those sets "
           "cross (the line an ore shoot usually follows), and keeps everything for the integrated "
           "prediction. Upload either input, or both.")


def _named(raw, name):
    buf = io.BytesIO(raw)
    buf.name = name
    return buf


@st.cache_data(show_spinner=False, max_entries=8)
def _read(raw, name, sheet):
    return readers.read_upload(_named(raw, name), sheet)


# ---------------------------------------------------------------- Step 1
st.subheader("Step 1 · Upload")
u1, u2 = st.columns(2)
f_tab = u1.file_uploader("Structural measurements (CSV or Excel)", type=TABLES, key="gl_table")
u1.caption("One row per reading: position, strike, dip, and the dip direction if you recorded it. A structure "
           "column (shear, fault, vein…) and a station ID are used when present. Try "
           "`sample_data/structure_measurements.csv`.")
f_map = u2.file_uploader("QDS geological map sheet (PNG, JPG or GeoTIFF)", type=MAPS, key="gl_map")
u2.caption("A scanned or exported sheet. A GeoTIFF brings its own coordinates; a PNG or JPG needs its edges "
           "typed in. Used as the background map and kept for the prediction page.")

specs, cfg, xs, ys = [], None, [], []
if f_tab is not None:
    raw = f_tab.getvalue()
    df = None
    try:
        sheet = None
        if f_tab.name.lower().endswith((".xls", ".xlsx")):
            sheets = readers.excel_sheets(_named(raw, f_tab.name))
            full = []
            for sh in sheets:
                try:
                    if len(_read(raw, f_tab.name, sh)["df"].columns) >= 3:
                        full.append(sh)
                except Exception:
                    pass
            opts = full or sheets
            sheet = st.selectbox("Sheet", opts, key="gl_sheet") if len(opts) > 1 else opts[0]
        df = _read(raw, f_tab.name, sheet)["df"]
    except Exception as e:
        st.error(f"Could not read the measurements: {e}")
    if df is not None and len(df.columns) >= 3:
        with st.container(border=True):
            st.markdown(f"**{f_tab.name}** · {len(df):,} rows")
            st.dataframe(df.head(5), hide_index=True, width="stretch")
            cols = list(df.columns)
            opt = [NONE] + cols
            gx, gy = coords.guess_columns(df)
            a, b, c = st.columns(3)
            xcol = a.selectbox("Easting / longitude", cols, cols.index(gx) if gx in cols else 0, key="gl_x")
            ycol = b.selectbox("Northing / latitude", cols, cols.index(gy) if gy in cols else 1, key="gl_y")
            xs = coords.parse_column(df[xcol]).dropna()
            ys = coords.parse_column(df[ycol]).dropna()
            geo = len(xs) > 0 and len(ys) > 0 and coords.looks_geographic(xs, ys)
            names = list(coords.CRS_OPTIONS)
            crs = c.selectbox("Coordinate system", names, 0 if geo else
                              names.index("WGS84 / UTM zone 36S (EPSG:32736)"), key=f"gl_crs_{int(geo)}")

            def pick(label, guesses, key, allow_none=True):
                g = structure.guess(df, guesses)
                lst = opt if allow_none else cols
                return st.selectbox(label, lst, lst.index(g) if g in lst else 0, key=key)

            d, e, g2 = st.columns(3)
            with d:
                scol = pick("Strike column", ["strike", "trend"], "gl_strike", False)
            with e:
                dcol = pick("Dip column", ["dip"], "gl_dip", False)
            with g2:
                ddcol = pick("Dip direction (optional)", structure.DIPDIR_COLS, "gl_dd")
            h, i2 = st.columns(2)
            with h:
                tcol = pick("Structure type (optional)", structure.STRUCT_COLS, "gl_type")
            with i2:
                icol = pick("Station ID (optional)", structure.STOP_COLS, "gl_id")
            if ddcol == NONE:
                st.caption("With no dip direction, the right-hand rule is assumed: the plane dips 90° "
                           "clockwise of the strike. If your strikes were not recorded that way, the dips "
                           "below will face the wrong side.")
            sv = coords.parse_column(df[scol]).to_numpy(float)
            dv = coords.parse_column(df[dcol]).to_numpy(float)
            usable = int((np.isfinite(sv) & np.isfinite(dv)).sum())
            st.caption(f"{usable:,} of {len(df):,} rows have both a strike and a dip.")
            if usable >= 3:
                cfg = {"xcol": xcol, "ycol": ycol, "epsg": coords.CRS_OPTIONS[crs], "strikecol": scol,
                       "dipcol": dcol, "ddcol": None if ddcol == NONE else ddcol,
                       "typecol": None if tcol == NONE else tcol, "idcol": None if icol == NONE else icol}
                specs.append({"slot": "geology", "name": f_tab.name, "kind": "table",
                              "data": {"kind": "table", "df": df}, "xcol": xcol, "ycol": ycol,
                              "epsg": cfg["epsg"], "strikecol": scol, "dipcol": dcol,
                              "strike_is_dipdir": False,
                              # the structure name doubles as the "lithology" the prediction page ranks,
                              # so SHEAR and VEIN are picked up there as gold-favourable hosts
                              "lithcol": cfg["typecol"]})
            else:
                st.error("Fewer than three rows have both a strike and a dip.")

if f_map is not None:
    mraw = f_map.getvalue()
    res = None
    try:
        res = _read(mraw, f_map.name, None)
    except Exception as e:
        st.error(f"Could not read the map sheet: {e}")
    if res is not None:
        with st.container(border=True):
            st.markdown(f"**{f_map.name}** · QDS map sheet")
            if res["kind"] == "raster":
                layers = list(res["layers"])
                a0 = res["layers"][layers[0]]
                st.caption(f"GeoTIFF · {len(layers)} band(s) · {a0.shape[1]}×{a0.shape[0]} cells · "
                           f"EPSG:{res.get('epsg')}")
                if not res.get("epsg"):
                    st.warning("This GeoTIFF has no EPSG code, so it may not line up with the measurements.")
                specs.append({"slot": "geology", "name": f_map.name, "kind": "raster", "data": res})
            elif res["kind"] == "image":
                st.image(res["image"], width=280)
                aoi = ss.get("rs_aoi")
                if len(xs) and len(ys):
                    d0 = (float(np.min(xs)), float(np.max(xs)), float(np.min(ys)), float(np.max(ys)))
                    src = "pre-filled from the area your measurements cover"
                elif aoi:
                    bb = aoi["bounds"]
                    d0 = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                    src = "pre-filled from the Section 1 area"
                else:
                    d0 = (0.0, 0.0, 0.0, 0.0)
                    src = ("needed before the sheet can be used: a QDS sheet's corners are printed on it, or "
                           "take them from the sheet index")
                st.caption(f"A picture carries no coordinates, so give the sheet's edges \u2014 {src}.")
                e1, e2, e3, e4 = st.columns(4)
                x0 = e1.number_input("West", value=d0[0], format="%.2f", key="gl_mx0")
                x1 = e2.number_input("East", value=d0[1], format="%.2f", key="gl_mx1")
                y0 = e3.number_input("South", value=d0[2], format="%.2f", key="gl_my0")
                y1 = e4.number_input("North", value=d0[3], format="%.2f", key="gl_my1")
                clist = list(coords.CRS_OPTIONS)
                looks_ll = coords.looks_geographic(pd.Series([x0, x1]), pd.Series([y0, y1]))
                default_crs = next((i for i, k in enumerate(clist)
                                    if cfg and coords.CRS_OPTIONS[k] == cfg["epsg"]),
                                   0 if looks_ll else clist.index("WGS84 / UTM zone 36S (EPSG:32736)"))
                mcrs = st.selectbox("Coordinate system of the edges", clist, default_crs,
                                    key=f"gl_mcrs_{int(looks_ll)}")
                mepsg = coords.CRS_OPTIONS[mcrs]
                if x1 <= x0 or y1 <= y0:
                    st.warning("Enter the sheet's edges (east above west, north above south).")
                elif mepsg == 4326 and not looks_ll:
                    st.warning("Those edges are far too large for degrees of latitude and longitude. They "
                               "look like UTM metres, so pick the matching UTM zone above.")
                elif mepsg != 4326 and looks_ll:
                    st.warning("Those edges look like degrees of latitude and longitude, not UTM metres. "
                               "Pick WGS84 latitude / longitude above.")
                else:
                    specs.append({"slot": "geology", "name": f_map.name, "kind": "image", "data": res,
                                  "extent": (x0, x1, y0, y1), "epsg": mepsg})
            else:
                st.error("That file is neither a picture nor a GeoTIFF.")

# ---------------------------------------------------------------- Step 2
st.subheader("Step 2 · Submit")
runs = ss.setdefault("s2_runs", {})
entry = runs.get("geology")
sig = hash(repr([sorted((k, repr(v)) for k, v in s.items() if k != "data") for s in specs] + [repr(cfg)]))
c1, c2 = st.columns([1, 3])
go = c1.button("🚀 Submit & interpret", type="primary", disabled=not specs, width="stretch", key="gl_go")
c2.markdown(f"{len(specs)} input(s) ready" if specs else "Upload the measurements, the map sheet, or both.")
if entry and specs and entry["sig"] != sig:
    st.info("Files or settings changed since the last run. Submit again to update the results.")
if go:
    try:
        run = interpret.run(specs, {}, progress=None)
        if not run["datasets"]["geology"]:
            raise ValueError("; ".join(run["errors"]) or "nothing usable in these files")
        run["id"] = new_run_id()
        runs["geology"] = {"run": run, "sig": sig, "specs": specs, "settings": {}, "cfg": cfg}
        clear_widgets("geology")
        ss.pop("s2_combined", None)
        st.rerun()
    except Exception as e:
        st.error(f"The interpretation failed: {e}")

entry = runs.get("geology")
if not entry:
    st.info("Upload your measurements and / or the QDS sheet, then press **Submit & interpret**.")
    st.stop()

# ---------------------------------------------------------------- Step 3
run, cfg = entry["run"], entry.get("cfg")
st.subheader("Step 3 · Overview and interpretation")
c1, c2 = st.columns([3, 1])
c1.caption("Results for " + ", ".join(s["name"] for s in entry["specs"]))
if c2.button("Clear results", key="gl_clear"):
    runs.pop("geology", None)
    ss.pop("s2_combined", None)
    clear_widgets("geology")
    st.rerun()
for e in run["errors"]:
    st.warning(e)

bg = next((d for d in run["datasets"]["geology"] if d["kind"] == "image_ref"), None)
pts = next((d for d in run["datasets"]["geology"] if d["kind"] == "points"), None)
background = (bg["image"], tuple(bg["extent"])) if bg is not None else None

if pts is None or cfg is None:
    st.info("Only the map sheet was uploaded, so there is nothing to interpret from measurements. The sheet "
            "is shown below and is kept as the background for the prediction page.")
    if bg is not None:
        fig = plots.image_figure(bg["image"], list(bg["extent"]), f"QDS sheet · {bg['name']}")
        st.pyplot(fig)
        plt.close(fig)
else:
    src = entry["specs"][0]["data"]["df"]
    keep = np.isfinite(coords.parse_column(src[cfg["strikecol"]]).to_numpy(float)) & \
        np.isfinite(coords.parse_column(src[cfg["dipcol"]]).to_numpy(float))
    sub = src[keep].reset_index(drop=True)

    @st.cache_data(show_spinner=False, max_entries=8)
    def _analyse(key, strike, dip, dipdir, types, x, y, ids, max_sets):
        res = structure.resolve(strike, dip, dipdir)
        ok = res.ok.to_numpy()
        sets, labels = structure.find_sets(res.strike[ok], res.dip[ok], max_sets=max_sets)
        return {"res": res, "sets": sets, "labels": labels, "inters": structure.intersections(sets),
                "spread": structure.spread(x, y, ids),
                "types": structure.classify_types(types) if types is not None else None}

    max_sets = st.slider("Most structural sets to look for", 1, 4, 3, key="gl_sets",
                         help="Sets are found by clustering the poles to the planes. A set needs at least "
                              "three readings, so a small survey may give fewer than you ask for.")
    strike = coords.parse_column(sub[cfg["strikecol"]]).to_numpy(float)
    dip = coords.parse_column(sub[cfg["dipcol"]]).to_numpy(float)
    dipdir = coords.parse_column(sub[cfg["ddcol"]]).to_numpy(float) if cfg["ddcol"] else None
    types = sub[cfg["typecol"]].astype(str).to_numpy() if cfg["typecol"] else None
    ids = sub[cfg["idcol"]].astype(str).to_numpy() if cfg["idcol"] else np.arange(1, len(sub) + 1).astype(str)
    A = _analyse(run["id"], strike, dip, dipdir, types, pts["x"], pts["y"], ids, max_sets)
    res, sets, labels, inters, sp = (A[k] for k in ("res", "sets", "labels", "inters", "spread"))
    ok = res.ok.to_numpy()

    k = st.columns(5)
    k[0].metric("Readings used", f"{int(ok.sum())} of {len(res)}")
    k[1].metric("Stations", sp["stations"])
    k[2].metric("Structural sets", len(sets))
    k[3].metric("Main set", f"{sets[0]['strike']:.0f}°/{sets[0]['dip']:.0f}°" if sets else "—")
    k[4].metric("Shoot plunge", f"{inters[0]['plunge']:.0f}° → {inters[0]['trend']:.0f}°" if inters else "—")

    narrative = structure.narrative(res, A["types"], sets, labels, inters, sp)
    st.markdown("### Interpretation")
    for line in narrative:
        st.markdown("- " + line)
    st.caption("Structural measurements say where gold could be trapped, not whether it is there. Test these "
               "directions against the soil anomalies and the geophysics before committing to a drill hole.")

    tabs = st.tabs(["Map", "Stereonet", "Strike rose", "Sets", "Readings", "QDS sheet"])
    with tabs[0]:
        fig = structure.map_figure(pts["x"], pts["y"], res, labels, A["types"], sets, background, ids,
                                   "Structural measurements" + (" on the QDS sheet" if background else ""))
        st.pyplot(fig)
        png = export.fig_to_png(fig)
        plt.close(fig)
        st.download_button("Map (PNG)", png, "structure_map.png", "image/png", on_click="ignore",
                           key="gl_png_map")
        st.caption("Each long tick runs along the strike of a plane, with a short tick towards the dip, "
                   "coloured by set. A cross marks a reading that was set aside.")
    with tabs[1]:
        fig = structure.stereonet_figure(res, labels, sets, inters)
        st.pyplot(fig)
        png = export.fig_to_png(fig)
        plt.close(fig)
        st.download_button("Stereonet (PNG)", png, "stereonet.png", "image/png", on_click="ignore",
                           key="gl_png_stereo")
        st.caption("Each dot is the pole to one plane; the curve is the mean plane of its set and the square "
                   "its pole. The star is where two sets cross, the line an ore shoot usually follows.")
    with tabs[2]:
        starts = np.arange(0, 180, 10.0)
        h, _ = np.histogram(np.asarray(res.strike[ok]) % 180, bins=np.arange(0, 190, 10))
        fig = export.rose_figure(starts, h.astype(float), title="Strike rose (field measurements)")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Strikes in 10° bins, drawn both ways because a strike has no direction.")
    with tabs[3]:
        if not sets:
            st.info("No set could be separated from these readings.")
        else:
            tbl = pd.DataFrame([{"Set": s["set"], "Readings": s["n"],
                                 "Mean strike": f"{s['strike']:.0f}° ({structure.strike_label(s['strike'])})",
                                 "Mean dip": f"{s['dip']:.0f}° {structure.dip_word(s['dip'])}",
                                 "Dips towards": f"{s['dipdir']:.0f}°",
                                 "Tightness κ": round(s["kappa"], 1) if np.isfinite(s["kappa"]) else "",
                                 "Mean within ±": f"{s['a95']:.0f}°" if np.isfinite(s["a95"]) else "",
                                 "Spread": f"{s['scatter']:.0f}°"} for s in sets])
            st.dataframe(tbl, hide_index=True, width="stretch")
            if inters:
                st.markdown("**Where the sets cross** — the line an ore shoot usually follows")
                st.dataframe(pd.DataFrame([{"Sets": i["sets"], "Plunge": f"{i['plunge']:.0f}°",
                                            "Towards": f"{i['trend']:.0f}° "
                                                       f"({structure.strike_label(i['trend'])})",
                                            "Angle between planes": f"{i['angle']:.0f}°"} for i in inters]),
                             hide_index=True, width="stretch")
            st.download_button("Sets (CSV)", tbl.to_csv(index=False), "structural_sets.csv", "text/csv",
                               on_click="ignore", key="gl_csv_sets")
    with tabs[4]:
        lon, lat = to_lonlat(run["epsg"], pts["x"], pts["y"])
        set_of = np.array([""] * len(res), dtype=object)
        set_of[np.flatnonzero(ok)] = [sets[j]["set"] if j >= 0 and j < len(sets) else "" for j in labels]
        out = pd.DataFrame({"Station": ids, "Easting": np.round(pts["x"], 1),
                            "Northing": np.round(pts["y"], 1), "Latitude": np.round(lat, 6),
                            "Longitude": np.round(lon, 6), "Strike as recorded": res.strike_in.round(0),
                            "Dip": res.dip.round(0), "Strike (right-hand rule)": res.strike.round(0),
                            "Dips towards": res.dipdir.round(0),
                            "Dip direction off by": res.offset.round(0)})
        if A["types"] is not None:
            out["Structure"] = A["types"]
        out["Set"] = set_of
        out["Used"] = np.where(ok, "yes", "set aside")
        st.dataframe(out, hide_index=True, width="stretch", height=360)
        st.download_button("Readings (CSV)", out.to_csv(index=False), "structural_readings.csv", "text/csv",
                           on_click="ignore", key="gl_csv_read")
        if (~ok).any():
            st.caption("The readings marked **set aside** have a dip direction that cannot belong to their "
                       "strike, so one of the two was mis-recorded. They are left out of the sets.")
    with tabs[5]:
        if bg is None:
            st.info("No QDS map sheet was uploaded.")
        else:
            fig = plots.image_figure(bg["image"], list(bg["extent"]), f"QDS sheet · {bg['name']}",
                                     pts=(pts["x"], pts["y"]))
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Your measurement stations are drawn on the sheet, so you can check that the sheet's "
                       "edges were entered correctly.")

    entry["analysis"] = {"sets": sets, "intersections": inters, "spread": sp, "narrative": narrative,
                         "n_used": int(ok.sum()), "n_total": int(len(res)), "sets_of_reading": set_of}

st.divider()
st.page_link(PREDICTION_PAGE, icon="➡️",
             label="Next: combine with your other data in **6 · Integrated gold prediction**")
