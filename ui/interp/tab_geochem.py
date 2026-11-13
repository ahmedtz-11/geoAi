"""Geochemistry results: statistics, thresholds, anomaly maps, correlations, gold index."""
import numpy as np
import pandas as pd
import streamlit as st

from geoai import geochem, geophysics, interpret, plots

from .common import PREFIX, dataset_picker, default, memo, show_fig, ss, to_lonlat
from ui.state import fragment


@fragment
def render(run):
    items = run["datasets"]["geochem"]
    if not items:
        st.info("No geochemical data were uploaded.")
        return
    i = dataset_picker(items, "Dataset", f"{PREFIX}gc_ds")
    ds = items[i]
    method = st.radio("Anomaly thresholds", geochem.METHODS, horizontal=True,
                      key=default(f"{PREFIX}gc_method", run["settings"].get("geochem_method", geochem.METHODS[0])),
                      help="Assay data are skewed, so thresholds are set on log values. Percentiles suit small "
                           "or mixed data sets.")
    r = memo(run, "geochem", lambda: interpret.analyze_geochem(ds, method), i, method)
    t = ds["table"]
    cols = list(t.columns)
    stats = r["stats"].copy()
    st.markdown(f"**{len(t):,} samples** · statistics per element (below-detection values set to half the "
                "detection limit)")
    st.dataframe(stats.round(3), width="stretch")
    el = st.selectbox("Element", cols, cols.index(ds["au"]) if ds["au"] in cols else 0,
                      key=default(f"{PREFIX}gc_el_{i}", ds["au"] or cols[0]))
    v = t[el].to_numpy(float)
    lim = r["limits"][el]
    cls = geochem.classify(v, lim)
    a, b = st.columns(2)
    with a:
        show_fig(plots.points_figure(ds["x"], ds["y"], classes=cls, labels=geochem.CLASSES,
                                     colors=geochem.CLASS_COLORS, title=f"{el} anomaly classes",
                                     sizes=[10, 22, 36, 54]), f"{el}_classes.png", f"gc_png_cls_{i}")
    with b:
        show_fig(plots.hist_figure(v, lim, f"{el} distribution", el), f"{el}_histogram.png", f"gc_png_hist_{i}")
        n_an = int((cls >= 2).sum())
        st.metric("Anomalous samples (≥ anomalous threshold)", f"{n_an} of {int(np.isfinite(v).sum())}")
    try:
        lv = np.log10(np.where(v > 0, v, np.nan))
        g = memo(run, "gc_grid", lambda: geophysics.grid_points(ds["x"], ds["y"], lv), i, el)
        c1, c2 = st.columns(2)
        with c1:
            show_fig(plots.grid_figure(g["z"], g["transform"], f"{el} gridded (log10)", "YlOrRd", "log10",
                                       overlay_pts=(ds["x"], ds["y"])), f"{el}_grid.png", f"gc_png_grid_{i}")
        with c2:
            if len(cols) > 1:
                show_fig(plots.corr_figure(r["corr"]), "correlation.png", f"gc_png_corr_{i}")
                st.caption("Elements correlated with Au are its pathfinders here (As, Sb, Bi, Te, W are common in "
                           "orogenic gold).")
    except ValueError as e:
        st.warning(str(e))
    if "index" in r:
        st.markdown("**Multi-element gold index** (Au weight 2 + pathfinders weighted by their correlation with Au)")
        c1, c2 = st.columns([3, 2])
        with c1:
            show_fig(plots.points_figure(ds["x"], ds["y"], values=r["index"], title="Gold index", cmap="inferno",
                                         label="index (z)"), "gold_index.png", f"gc_png_idx_{i}")
        with c2:
            st.dataframe(pd.DataFrame({"Element": list(r["index_weights"]),
                                       "Weight": np.round(list(r["index_weights"].values()), 2)}),
                         hide_index=True, width="stretch")
        au = ds["au"]
        lon, lat = to_lonlat(run["epsg"], ds["x"], ds["y"])
        out = pd.DataFrame({"Sample": ds["ids"], "Easting": np.round(ds["x"], 1), "Northing": np.round(ds["y"], 1),
                            "Latitude": np.round(lat, 6), "Longitude": np.round(lon, 6)})
        for c in cols:
            out[c] = t[c].round(3)
        out["Au class"] = [geochem.CLASSES[c] if c >= 0 else "" for c in r["au_class"]]
        out["Gold index"] = np.round(r["index"], 3)
        top = out[r["au_class"] >= 2].sort_values(au, ascending=False)
        st.markdown(f"**Anomalous gold samples** ({len(top)})")
        st.dataframe(top.head(200), hide_index=True, width="stretch", height=260)
        st.download_button("All samples with classes (CSV)", out.to_csv(index=False), "geochem_classified.csv",
                           "text/csv", on_click="ignore", key=f"gc_csv_{i}")
