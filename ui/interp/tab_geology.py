"""Geology results: lithology map, favourable host rocks, structural measurements."""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from geoai import export, geology, plots

from .common import PREFIX, default, show_fig, ss
from ui.state import fragment


def _background(run):
    for ds in run["datasets"]["geology"]:
        if ds["kind"] == "image_ref":
            return ds
    return None


@fragment
def render(run):
    items = run["datasets"]["geology"]
    if not items:
        st.info("No geological data were uploaded.")
        return
    pts = [d for d in items if d["kind"] == "points"]
    bg = _background(run)
    if pts:
        liths = run["lithologies"]
        default(f"{PREFIX}fav", run["fav_default"])
        st.multiselect("Gold-favourable host rocks (used in the prediction)", liths, key=f"{PREFIX}fav",
                       help="Pre-selected from names like greenstone, BIF, quartz vein, shear, schist, mafic.")
        x = np.concatenate([d["x"] for d in pts])
        y = np.concatenate([d["y"] for d in pts])
        lith = np.concatenate([np.asarray(d["lith"]).astype(str) for d in pts])
        names = sorted(set(lith))
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(len(names))]
        classes = np.array([names.index(l) for l in lith])
        background = None
        if bg is not None:
            from rasterio.transform import from_bounds
            x0, x1, y0, y1 = bg["extent"]
            arr = np.asarray(bg["image"])
            background = (arr, from_bounds(x0, y0, x1, y1, arr.shape[1], arr.shape[0]), None)
        fav = set(ss.get(f"{PREFIX}fav", []))
        labels = [f"{n} ★" if n in fav else n for n in names]
        a, b = st.columns([3, 2])
        with a:
            show_fig(plots.points_figure(x, y, classes=classes, labels=labels, colors=colors,
                                         title="Lithology (★ = favourable host)", background=background),
                     "lithology_map.png", "geo_png_lith")
        with b:
            summ = geology.lith_summary(lith)
            summ["Favourable"] = summ["Lithology"].isin(fav).map({True: "★", False: ""})
            st.dataframe(summ, hide_index=True, width="stretch")
        strikes = np.concatenate([d["strike"] for d in pts if d.get("strike") is not None]) if \
            any(d.get("strike") is not None for d in pts) else np.array([])
        strikes = strikes[np.isfinite(strikes)]
        if strikes.size:
            st.markdown(f"**Structural measurements** ({strikes.size})")
            starts, counts = geology.rose_from_strikes(strikes)
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                show_fig(export.rose_figure(starts, counts, title="Strike rose (field measurements)"),
                         "strike_rose.png", "geo_png_rose")
            dips = np.concatenate([d["dip"] for d in pts if d.get("dip") is not None]) if \
                any(d.get("dip") is not None for d in pts) else None
            if dips is not None:
                s_all = np.concatenate([d["strike"] for d in pts if d.get("strike") is not None])
                ok = np.isfinite(s_all) & np.isfinite(dips)
                if ok.any():
                    with c2:
                        px, py = geology.poles(s_all[ok], dips[ok])
                        show_fig(plots.stereonet_figure(px, py), "stereonet.png", "geo_png_stereo")
            with c3:
                main = starts[np.argmax(counts)] + 5
                from geoai.lineaments import strike_label
                st.metric("Main strike", f"{main:.0f}°")
                st.metric("Trend", strike_label(main))
                if dips is not None and np.isfinite(dips).any():
                    st.metric("Median dip", f"{np.nanmedian(dips):.0f}°")
    elif bg is not None:
        st.caption("Only a map picture was uploaded; it is shown below as a reference.")
    if bg is not None:
        show_fig(plots.image_figure(bg["image"], [bg["extent"][0], bg["extent"][1], bg["extent"][2], bg["extent"][3]],
                                    f"Geological map · {bg['name']}"), "geological_map.png", "geo_png_bg")
