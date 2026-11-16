"""Section 5 - Geochemistry: soil samples with a gold assay, read and interpreted for targeting.

The results are stored where the other data pages keep theirs (st.session_state["s2_runs"]["geochem"]),
so 6 - Integrated gold prediction picks them up unchanged.
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from geoai import coords, export, geochem, geophysics, interpret, plots, readers, soilgold
from ui.interp.common import clear_widgets, new_run_id, ss, to_lonlat

ACCEPT = ["csv", "xls", "xlsx"]
PREDICTION_PAGE = "views/interpretation.py"

st.title("5 · Geochemistry")
st.caption("Soil samples with a gold assay. GeoAI works out where background stops, groups the anomalous "
           "samples into zones with their trend and extent, says which are worth following up and which are "
           "probably a single lucky grain, and keeps everything for the integrated prediction.")

# ---------------------------------------------------------------- Step 1
st.subheader("Step 1 · Upload")
f = st.file_uploader("Soil samples: CSV or Excel with the sample position, the sample ID and the gold assay",
                     type=ACCEPT, key="gc_file")
st.caption("One row per sample. Needed: easting / northing (or longitude / latitude) and the gold assay. "
           "A sample ID column is used if there is one. Values like `<0.01` or a negative code are read as "
           "below detection. Try `sample_data/soil_gold.xlsx`.")

def _named(raw, name):
    """read_upload picks the reader from the file name, so the buffer has to carry one."""
    buf = io.BytesIO(raw)
    buf.name = name
    return buf


@st.cache_data(show_spinner=False, max_entries=8)
def _read(raw, name, sheet):
    return readers.read_upload(_named(raw, name), sheet)["df"]


spec = None
if f is not None:
    raw = f.getvalue()
    df = None
    try:
        sheet = None
        if f.name.lower().endswith((".xls", ".xlsx")):
            sheets = readers.excel_sheets(_named(raw, f.name))
            full = []
            for sh in sheets:                         # skip the empty sheets Excel leaves behind
                try:
                    if len(_read(raw, f.name, sh).columns) >= 3:
                        full.append(sh)
                except Exception:
                    pass
            opts = full or sheets
            sheet = st.selectbox("Sheet", opts, key="gc_sheet") if len(opts) > 1 else opts[0]
        df = _read(raw, f.name, sheet)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
    if df is not None:
        if len(df.columns) < 3 or len(df) < 3:
            st.error("The file needs at least three columns (position and assay) and a few rows.")
        else:
            with st.container(border=True):
                st.dataframe(df.head(6), hide_index=True, width="stretch")
                cols = list(df.columns)
                num = readers.numeric_columns(df)
                gx, gy = coords.guess_columns(df)
                a, b, c = st.columns(3)
                xcol = a.selectbox("Easting / longitude", cols, cols.index(gx) if gx in cols else 0, key="gc_x")
                ycol = b.selectbox("Northing / latitude", cols, cols.index(gy) if gy in cols else 1, key="gc_y")
                xs, ys = coords.parse_column(df[xcol]).dropna(), coords.parse_column(df[ycol]).dropna()
                geo = len(xs) > 0 and len(ys) > 0 and coords.looks_geographic(xs, ys)
                names = list(coords.CRS_OPTIONS)
                crs = c.selectbox("Coordinate system", names, 0 if geo else
                                  names.index("WGS84 / UTM zone 36S (EPSG:32736)"), key=f"gc_crs_{int(geo)}")
                rest = [k for k in num if k not in (xcol, ycol)]
                if not rest:
                    st.error("No numeric column is left for the gold assay.")
                else:
                    au_guess = next((k for k in rest if any(w in str(k).lower() for w in
                                                            ("au", "gold", "assay", "grade"))), rest[0])
                    d, e2 = st.columns(2)
                    aucol = d.selectbox("Gold assay column", rest, rest.index(au_guess), key="gc_au")
                    vals, _ = geochem.clean_values(df[aucol])
                    unit = e2.selectbox("Assay unit", list(soilgold.UNITS),
                                        list(soilgold.UNITS).index(soilgold.guess_unit(vals)), key="gc_unit",
                                        help="Guessed from the size of the numbers. Soil gold runs from a few "
                                             "ppb to a few thousand ppb, so 0.05 is ppm and 50 is ppb. "
                                             "Everything below is reported in both.")
                    idopts = ["(row number)"] + [k for k in cols if k not in (xcol, ycol, aucol)]
                    guess = next((k for k in idopts[1:] if any(w in str(k).lower() for w in
                                                               ("sample", "id", "no", "point"))), None)
                    idcol = st.selectbox("Sample ID column", idopts, idopts.index(guess) if guess else 0,
                                         key="gc_id")
                    good = int(np.isfinite(vals).sum())
                    if good:
                        st.caption(f"{good:,} of {len(df):,} rows have a usable gold value, from "
                                   f"{np.nanmin(vals):,.4g} to {np.nanmax(vals):,.4g} {unit.split(' ')[0]}.")
                    if good >= 5:
                        spec = {"slot": "geochem", "name": f.name, "kind": "table",
                                "data": {"kind": "table", "df": df}, "xcol": xcol, "ycol": ycol,
                                "epsg": coords.CRS_OPTIONS[crs], "au": aucol, "pathfinders": [],
                                "idcol": None if idcol == idopts[0] else idcol, "unit": unit}
                    else:
                        st.error("Fewer than five usable gold values in that column.")

# ---------------------------------------------------------------- Step 2
st.subheader("Step 2 · Submit")
runs = ss.setdefault("s2_runs", {})
entry = runs.get("geochem")
sig = hash(repr(sorted((k, repr(v)) for k, v in (spec or {}).items() if k != "data")))
c1, c2 = st.columns([1, 3])
go = c1.button("🚀 Submit & interpret", type="primary", disabled=spec is None, width="stretch", key="gc_go")
if entry and spec and entry["sig"] != sig:
    c2.info("File or settings changed since the last run. Submit again to update the results.")
if go:
    try:
        run = interpret.run([spec], {}, progress=None)
        if run["errors"] or not run["datasets"]["geochem"]:
            raise ValueError("; ".join(run["errors"]) or "no usable samples found")
        run["id"] = new_run_id()
        runs["geochem"] = {"run": run, "sig": sig, "specs": [spec], "settings": {}, "unit": spec["unit"]}
        clear_widgets("geochem")
        ss.pop("s2_combined", None)          # the integrated prediction must be rebuilt
        st.rerun()
    except Exception as e:
        st.error(f"The interpretation failed: {e}")

entry = runs.get("geochem")
if not entry:
    st.info("Upload your samples, check the columns, then press **Submit & interpret**.")
    st.stop()

# ---------------------------------------------------------------- Step 3
run = entry["run"]
ds = run["datasets"]["geochem"][0]
unit_name = entry.get("unit", list(soilgold.UNITS)[0])
u = unit_name.split(" ")[0]
mult = soilgold.UNITS[unit_name]
x, y = ds["x"], ds["y"]
v = ds["table"][ds["au"]].to_numpy(float)

st.subheader("Step 3 · Overview and interpretation")
c1, c2 = st.columns([3, 1])
c1.caption(f"Results for {entry['specs'][0]['name']} · gold column “{ds['au']}” in {u}")
if c2.button("Clear results", key="gc_clear"):
    runs.pop("geochem", None)
    ss.pop("s2_combined", None)
    clear_widgets("geochem")
    st.rerun()


@st.cache_data(show_spinner=False, max_entries=8)
def _analyse(key, x, y, v, mult, method):
    lay = soilgold.layout(x, y)
    pops = soilgold.populations(v)
    limits = soilgold.thresholds(v, method, pops)
    return {"layout": lay, "stats": soilgold.summary(v, mult), "pops": pops, "limits": limits,
            "classes": geochem.classify(v, limits),
            "zones": soilgold.zones(x, y, v, limits[1], lay["spacing"]),
            "floor": soilgold.detection_floor(v)}


method = st.radio("Where background stops", soilgold.METHODS, horizontal=True, key="gc_method",
                  help="The first asks the data: it fits a background population and a mineralised one, and "
                       "puts the line where the mineralised one takes over. The others are the usual fixed "
                       "rules, which need you to already know what background looks like here.")
A = _analyse(run["id"], x, y, v, mult, method)
lay, stats, pops, limits, cls, z = (A[k] for k in ("layout", "stats", "pops", "limits", "classes", "zones"))
if method == soilgold.METHODS[0] and not (pops.get("ok") and pops["n_pop"] == 2 and pops.get("limits")):
    st.warning("No separate mineralised population could be fitted, so the thresholds below fall back to "
               "**mean + σ on log values**.")

k = st.columns(5)
k[0].metric("Samples", f"{stats['n']:,}")
k[1].metric("Median", f"{stats['median']:,.4g} {u}", help=f"{stats['median_ppb']:,.0f} ppb")
k[2].metric("Highest", f"{stats['max']:,.4g} {u}", help=f"{stats['max_ppb']:,.0f} ppb")
k[3].metric("Anomalous samples", int((cls >= 2).sum()))
k[4].metric("Zones to follow up", int((z.samples >= 3).sum()) if len(z) else 0)

narrative = soilgold.narrative(stats, lay, pops, limits, z, x, y, mult, unit_name, A["floor"])
st.markdown("### Interpretation")
for line in narrative:
    st.markdown("- " + line)
st.caption("Soil geochemistry shows where gold reaches the surface, not how much is underneath. Every zone "
           "below needs checking on the ground before it is worth drilling.")

tabs = st.tabs(["Map", "Anomaly zones", "Distribution", "Interpolated", "Statistics", "All samples"])
with tabs[0]:
    fig = soilgold.zone_map_figure(x, y, v, cls, z, u)
    st.pyplot(fig)
    png = export.fig_to_png(fig)
    plt.close(fig)
    st.download_button("Map (PNG)", png, "soil_gold_map.png", "image/png", on_click="ignore", key="gc_png_map")
    st.caption(f"Circle size follows the assay. Thresholds: weak ≥ {limits[0]:,.4g}, anomalous ≥ "
               f"{limits[1]:,.4g}, strongly anomalous ≥ {limits[2]:,.4g} {u}. Dashed rings are the zones.")
with tabs[1]:
    if not len(z):
        st.info("No sample reaches the anomalous threshold, so there is nothing to rank.")
    else:
        lon, lat = to_lonlat(run["epsg"], z.peak_x.to_numpy(), z.peak_y.to_numpy())
        show = pd.DataFrame({
            "Zone": z.zone, "Samples": z.samples, f"Peak ({u})": z.peak.round(4),
            "Peak (ppb)": (z.peak * mult).round(0), f"Mean ({u})": z["mean"].round(4),
            "Peak easting": z.peak_x.round(0), "Peak northing": z.peak_y.round(0),
            "Latitude": np.round(lat, 6), "Longitude": np.round(lon, 6),
            "Length (m)": z.length.round(0),
            "Trend": [f"{a:.0f}° {soilgold.strike_label(a)}" if np.isfinite(a) else "" for a in z.azimuth],
            "Open towards": [", ".join(soilgold.open_edges(r, x, y, lay["spacing"])) for _, r in z.iterrows()],
            "Follow up": np.where(z.samples >= 3, "Yes, infill", "Re-sample first")})
        st.dataframe(show, hide_index=True, width="stretch")
        st.download_button("Zones (CSV)", show.to_csv(index=False), "gold_zones.csv", "text/csv",
                           on_click="ignore", key="gc_csv_zones")
        st.caption("Ranked by peak assay. A zone of one or two samples is not a target yet: one high sample "
                   "beside ordinary ones is as often a single grain in the sieve as it is a discovery.")
with tabs[2]:
    a, b = st.columns(2)
    with a:
        fig = plots.hist_figure(v, limits, f"Gold distribution ({u})", u)
        st.pyplot(fig)
        plt.close(fig)
    with b:
        fig = soilgold.probability_figure(v, pops, limits, u)
        st.pyplot(fig)
        plt.close(fig)
    st.caption("On the right, one population plots as a single straight line of points. A step in the points "
               "where the two fitted lines cross means a mineralised population sits on top of background.")
with tabs[3]:
    try:
        lv = np.log10(np.where(v > 0, v, np.nan))
        g = geophysics.grid_points(x, y, lv)
        fig = plots.grid_figure(g["z"], g["transform"], f"Gold in soil, interpolated (log10 {u})", "YlOrRd",
                                f"log10 {u}", overlay_pts=(x, y))
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Minimum-curvature surface through the log of the assays, blanked away from the samples. "
                   "What happens between samples is a guess: trust the sample points, not the colours "
                   "between them.")
    except Exception as e:
        st.info(f"The sample positions are too close to a line to interpolate: {e}")
with tabs[4]:
    rows = [("Samples", f"{stats['n']:,}"), ("Area covered", f"{lay['area_ha']:,.1f} ha"),
            ("Sample spacing", f"{lay['spacing']:,.0f} m"), ("Duplicate positions", f"{lay['duplicates']}"),
            ("Minimum", f"{stats['min']:,.4g} {u}"), ("25th percentile", f"{stats['p25']:,.4g} {u}"),
            ("Median", f"{stats['median']:,.4g} {u}"), ("Mean", f"{stats['mean']:,.4g} {u}"),
            ("Geometric mean", f"{stats['geomean']:,.4g} {u}"), ("75th percentile", f"{stats['p75']:,.4g} {u}"),
            ("90th percentile", f"{stats['p90']:,.4g} {u}"), ("95th percentile", f"{stats['p95']:,.4g} {u}"),
            ("98th percentile", f"{stats['p98']:,.4g} {u}"), ("Maximum", f"{stats['max']:,.4g} {u}"),
            ("Peak / median contrast", f"{stats['contrast']:,.1f}×"),
            ("Coefficient of variation", f"{stats['cv']:,.2f}"),
            ("Spread of the logs (σ)", f"{stats['log_sd']:,.2f}"),
            ("Weak anomaly threshold", f"{limits[0]:,.4g} {u}"),
            ("Anomalous threshold", f"{limits[1]:,.4g} {u}"),
            ("Strongly anomalous threshold", f"{limits[2]:,.4g} {u}")]
    st.dataframe(pd.DataFrame(rows, columns=["", "Value"]), hide_index=True, width="stretch", height=520)
with tabs[5]:
    zone_of = np.array([""] * len(v), dtype=object)
    for r in z.itertuples():
        zone_of[r.indices] = r.zone
    lon, lat = to_lonlat(run["epsg"], x, y)
    table = soilgold.report_table(ds["ids"], x, y, v, cls, lon, lat, mult, zone_of)
    st.dataframe(table.sort_values("Au", ascending=False), hide_index=True, width="stretch", height=420)
    st.download_button("All samples with classes and zones (CSV)", table.to_csv(index=False),
                       "soil_gold_classified.csv", "text/csv", on_click="ignore", key="gc_csv_all")

# keep the interpretation with the run, so the integrated prediction and its report can use it
entry["analysis"] = {"layout": lay, "stats": stats, "pops": pops, "limits": limits, "zones": z,
                     "floor": A["floor"], "method": method, "unit": unit_name, "narrative": narrative}
st.divider()
st.page_link(PREDICTION_PAGE, icon="➡️",
             label="Next: combine with your other data in **6 · Integrated gold prediction**")
