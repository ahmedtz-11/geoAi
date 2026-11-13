"""Magnetic / IP results: enhanced maps, structures and strike, depth estimates, IP sections."""
import numpy as np
import pandas as pd
import streamlit as st

from geoai import export, geophysics, lineaments, plots

from .common import PREFIX, dataset_picker, default, memo, safe_name, show_fig, ss
from ui.state import fragment


def _magnetic(run, ds, r, i):
    g, prods = r["grid"], r["products"]
    view = st.radio("Show", ["Enhanced maps", "Structures & strike", "Depth estimates"], horizontal=True,
                    key=default(f"{PREFIX}mag_view_{i}", "Enhanced maps"))
    names = [k for k in prods]
    if view == "Enhanced maps":
        st.caption(f"Grid cell {g['cell']:.0f} m. Derivative maps use the field smoothed by {r['smooth_m']:.0f} m "
                   "upward continuation to suppress line noise.")
        pick = st.selectbox("Large map", names, names.index("Tilt derivative (°)"),
                            key=default(f"{PREFIX}mag_big_{i}", "Tilt derivative (°)"))
        cmap = {"Tilt derivative (°)": "RdBu_r", "Residual": "RdBu_r", "Analytic signal": "magma",
                "Horizontal gradient": "inferno", "First vertical derivative": "RdBu_r"}.get(pick, "jet")
        c1, c2 = st.columns([3, 2])
        with c1:
            show_fig(plots.grid_figure(prods[pick], g["transform"], f"{pick} · {ds['name']}", cmap),
                     f"{safe_name(pick)}.png", f"mag_png_big_{i}")
        with c2:
            st.markdown(
                "**What each map shows**\n\n"
                "- **Residual**: local anomalies after removing the deep regional field.\n"
                "- **Analytic signal**: peaks over the edges of magnetic bodies (no need to correct for the "
                "Earth's field direction, useful near the equator).\n"
                "- **Tilt derivative**: balances strong and weak anomalies; the zero line follows contacts and "
                "faults.\n"
                "- **Horizontal gradient**: ridges over contacts and faults.\n"
                "- **First vertical derivative**: sharpens shallow sources.\n\n"
                "Gold in greenstone belts often sits on **magnetic lows** along shear zones, where alteration "
                "destroyed magnetite, next to magnetic BIF or mafic units.")
        show_fig(plots.panels([prods[n] for n in names], names, g["transform"],
                              ["jet", "jet", "RdBu_r", "magma", "RdBu_r", "inferno", "RdBu_r"], ncols=4),
                 "magnetic_products.png", f"mag_png_all_{i}")
        st.download_button("All magnetic grids (GeoTIFF)",
                           lambda: export.geotiff_bytes({safe_name(n): prods[n] for n in names}, g["transform"],
                                                        run["epsg"]),
                           f"magnetic_grids_{safe_name(ds['name'])}.tif", "image/tiff", on_click="ignore",
                           key=f"mag_tif_{i}")
    elif view == "Structures & strike":
        st.caption("Faults, contacts and dykes traced automatically on the tilt derivative. Their strike is the "
                   "main structural trend controlling where gold-bearing fluids flowed.")
        c1, c2 = st.columns(2)
        pct = c1.slider("Selectivity", 80, 98, key=default(f"{PREFIX}mag_pct_{i}", 90),
                        help="Higher keeps only the clearest structures.")
        lcells = c2.slider("Min length (cells)", 5, 40, key=default(f"{PREFIX}mag_len_{i}", 15))
        lines = memo(run, "maglin", lambda: lineaments.detect([prods["Tilt derivative (°)"]], g["valid"], g["cell"], 1.5,
                                                         pct, lcells * g["cell"], 2 * g["cell"],
                                                         merge_m=3 * g["cell"])[0], i, pct, lcells)
        starts, lens = lineaments.rose_stats(lines)
        order = np.argsort(lens)[::-1]
        k1, k2, k3 = st.columns(3)
        k1.metric("Structures", len(lines))
        k2.metric("Total length", f"{sum(l['length_m'] for l in lines) / 1000:,.1f} km")
        k3.metric("Main trend", lineaments.strike_label(starts[order[0]] + 5) if lines else "–")
        if lines:
            a, b = st.columns([3, 2])
            with a:
                show_fig(export.lines_figure(prods["Tilt derivative (°)"], lines, g["transform"], None,
                                             f"Magnetic structures on tilt derivative · {ds['name']}", "RdBu_r"),
                         "magnetic_structures.png", f"mag_png_lin_{i}")
            with b:
                show_fig(export.rose_figure(starts, lens, title="Strike of magnetic structures"),
                         "magnetic_rose.png", f"mag_png_rose_{i}")
                st.dataframe(pd.DataFrame([{"Trend": lineaments.strike_label(starts[j] + 5),
                                            "Strike": f"{starts[j]:.0f}–{starts[j] + 10:.0f}°",
                                            "Length (km)": round(float(lens[j]), 2)} for j in order[:4] if lens[j] > 0]),
                             hide_index=True, width="stretch")
                st.download_button("Structures (GeoJSON)", lineaments.to_geojson(lines, g["transform"],
                                                                                 run["epsg"]),
                                   "magnetic_structures.geojson", "application/geo+json", on_click="ignore",
                                   key=f"mag_geojson_{i}")
        else:
            st.warning("No structures found; lower the selectivity.")
    else:
        eu = r["euler"]
        st.caption(f"Euler deconvolution (structural index {run['settings'].get('si', 1.0)}) estimates the depth "
                   "to the top of magnetic sources from the field and its gradients, window by window. The power "
                   "spectrum gives average depths of deep and shallow source groups.")
        if len(eu):
            q = eu.depth.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Euler solutions", f"{len(eu):,}")
            k2.metric("Median depth", f"{q[0.5]:.0f} m")
            k3.metric("Middle 50%", f"{q[0.25]:.0f}–{q[0.75]:.0f} m")
            k4.metric("Shallowest 10%", f"< {q[0.1]:.0f} m")
            a, b = st.columns([3, 2])
            with a:
                show_fig(plots.euler_figure(r["products"]["Residual"], g["transform"], eu,
                                            f"Euler depth solutions on residual field · {ds['name']}"),
                         "euler_solutions.png", f"mag_png_euler_{i}")
            with b:
                show_fig(plots.hist_figure(eu.depth, None, "Depth of Euler solutions", "Depth (m)", log=False),
                         "euler_histogram.png", f"mag_png_eh_{i}")
                st.download_button("Euler solutions (CSV)", eu.round(1).to_csv(index=False), "euler_solutions.csv",
                                   "text/csv", on_click="ignore", key=f"mag_euler_csv_{i}")
        else:
            st.warning("No reliable Euler solutions. Try another structural index or window size (Advanced settings).")
        k, lp, _ = r["spectrum"]
        st.markdown("**Power spectrum**: adjust the fit ranges to the straight parts of the curve")
        c1, c2 = st.columns(2)
        deep = c1.slider("Deep segment (fraction of k)", 0.0, 0.6, step=0.01,
                         key=default(f"{PREFIX}spec_deep_{i}", (0.02, 0.15)))
        shallow = c2.slider("Shallow segment (fraction of k)", 0.05, 1.0, step=0.01,
                            key=default(f"{PREFIX}spec_sh_{i}", (0.15, 0.40)))
        fits = geophysics.spectral_depths(k, lp, deep, shallow)
        a, b = st.columns([3, 2])
        with a:
            show_fig(plots.spectrum_figure(k, lp, fits), "power_spectrum.png", f"mag_png_spec_{i}")
        with b:
            for nm, f in fits.items():
                st.metric(f"{nm.capitalize()} sources", f"≈ {f['depth']:.0f} m")
            st.caption("Steeper slopes = deeper sources. The noisy high-wavenumber tail should not be fitted.")


def _ip_grid(ds, r, i):
    g = r["grid"]
    z = g["z"]
    charge = "charge" in ds["survey"]
    v = z[np.isfinite(z)]
    m, s = float(np.mean(v)), float(np.std(v))
    lim = [m + s, m + 2 * s]
    cls = np.full(z.shape, np.nan)
    cls[np.isfinite(z)] = np.digitize(z[np.isfinite(z)], lim)
    a, b = st.columns(2)
    with a:
        show_fig(plots.grid_figure(z, g["transform"], f"{ds['survey']} · {ds['name']}",
                                   "jet" if charge else "viridis", ds["survey"].split("(")[-1].strip(")")),
                 "ip_map.png", f"ip_png_map_{i}")
    with b:
        show_fig(plots.grid_figure(cls, g["transform"], "Anomaly classes (0 < mean+σ, 1 < mean+2σ, 2 above)",
                                   "YlOrRd" if charge else "PuBu", "class", lo=0, hi=100), "ip_classes.png",
                 f"ip_png_cls_{i}")
    cell = g["cell"]
    st.dataframe(pd.DataFrame({"Class": ["Background", "Elevated (mean+σ)", "Strong (mean+2σ)"],
                               "From": [f"< {lim[0]:.2f}", f"{lim[0]:.2f}", f"{lim[1]:.2f}"],
                               "Area (ha)": [round(float((cls == c).sum() * cell ** 2 / 1e4), 1) for c in range(3)]}),
                 hide_index=True, width="stretch")
    if charge:
        st.caption("High chargeability marks disseminated sulphides (pyrite, arsenopyrite, pyrrhotite), often "
                   "associated with gold, but also graphite. Compare with resistivity and geology.")


def _ip_section(ds, r, i):
    lines = list(r["lines"])
    lid = st.selectbox("Line", lines, key=default(f"{PREFIX}ip_line_{i}", lines[0])) if len(lines) > 1 else lines[0]
    d = r["lines"][lid]["data"]
    pct = st.slider("Anomaly threshold (percentile of all values)", 50, 99,
                    key=default(f"{PREFIX}ip_pct_{i}", 90))
    thr = float(np.nanpercentile(ds["v"], pct))
    xi, zi, G = geophysics.section_grid(d.x, d.depth, d.v)
    show_fig(plots.section_figure(xi, zi, G, (d.x, d.depth), thr,
                                  f"{ds['survey']} pseudosection · line {lid} (black line = {thr:.1f})",
                                  "jet" if "charge" in ds["survey"] else "viridis",
                                  ds["survey"].split("(")[-1].strip(")")), f"ip_section_{safe_name(lid)}.png",
             f"ip_png_sec_{i}_{safe_name(lid)}")
    an = geophysics.section_anomalies(d.x, d.depth, d.v, thr)
    if len(an):
        st.markdown(f"**Anomalies on line {lid}** (depth = approximate pseudo-depth)")
        st.dataframe(an.rename(columns={"from_x": "From (m)", "to_x": "To (m)", "depth_top": "Top (m)",
                                        "depth_bottom": "Bottom (m)", "peak": "Peak value", "peak_x": "Peak at (m)"})
                     .round(1), hide_index=True, width="stretch")
    else:
        st.info("No values above the threshold on this line.")
    if r.get("grid") is not None:
        st.markdown("**Plan map** (maximum value at each station, all depths)")
        _ip_grid(ds, {"grid": r["grid"]}, f"{i}_plan")
    st.caption("Pseudo-depths from n-levels are approximate (about 0.2–0.4 × dipole spacing per level). "
               "Invert the data (e.g. Res2DInv) for true depths.")


@fragment
def render(run):
    items = run["datasets"]["geophysics"]
    if not items:
        st.info("No magnetic or IP data were uploaded.")
        return
    i = dataset_picker(items, "Dataset", f"{PREFIX}gp_ds")
    ds, r = items[i], run["results"]["geophysics"][i]
    if ds.get("from_image"):
        st.warning("These values come from the colours of a picture (relative 0–1 scale, not real units). "
                   "Use the original data for reliable depths.")
    if ds["survey"].startswith("Magnetic"):
        _magnetic(run, ds, r, i)
    elif ds["kind"] == "section":
        _ip_section(ds, r, i)
    else:
        _ip_grid(ds, r, i)
