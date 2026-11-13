"""Magnetic results: one tab per map (TMI, RTP/RTE, 1VD, 2VD, AS, tilt, THDR, lineaments, upward
continuation, SPI depth, Euler 3D, final target map) plus an overview with every map."""
import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from geoai import export, geophysics, interpret, lineaments, plots, prospectivity
from geoai.geophysics import upward

from .common import PREFIX, default, memo, safe_name, show_fig, ss, to_lonlat
from ui.state import fragment

INFO = {
    "TMI": ("turbo", "nT", "Total magnetic intensity as measured (gridded). Anomalies are shifted and asymmetric "
                          "because the Earth's field is inclined."),
    "RTP": ("turbo", "nT", "Reduced to the pole: anomalies moved over their sources and made symmetric, as if "
                          "measured at the magnetic pole. A low-latitude amplitude correction keeps it stable."),
    "RTE": ("turbo", "nT", "Reduced to the equator: used at low magnetic latitudes, where RTP is unstable. "
                          "Magnetic bodies appear as lows elongated E-W; derivative maps use the inverted RTE."),
    "1VD": ("RdBu_r", "nT/m", "First vertical derivative: sharpens shallow sources and narrows anomalies, "
                             "outlining near-surface magnetic units, dykes and BIF."),
    "2VD": ("RdBu_r", "nT/m²", "Second vertical derivative: even sharper; its zero line approximates the edges of "
                              "shallow bodies. Noisy data can make it speckled."),
    "Analytic signal": ("magma", "nT/m", "3-D analytic signal: peaks over the edges of magnetic bodies whatever the "
                                         "field direction, which makes it robust near the equator."),
    "Tilt derivative": ("RdBu_r", "degrees", "Tilt angle: balances strong and weak anomalies. The 0° line (black) "
                                             "follows contacts and faults; positive over sources."),
    "THDR": ("inferno", "nT/m", "Total horizontal derivative: ridges over contacts and faults (steepest lateral "
                                "change of the reduced field)."),
}


VIEWS = ["Normal", "Filled"]
VIEW_KEY = f"{PREFIX}mag_image_type"


def filled():
    return ss.get(VIEW_KEY, VIEWS[0]) == VIEWS[1]


def stretch():
    """Filled: Oasis montaj's equal-area colour method. Normal: the original linear stretch."""
    return plots.STRETCHES[0] if filled() else plots.STRETCHES[1]


def palette(own):
    """Filled: Geosoft rainbow. Normal: each map's original colours."""
    return "geosoft" if filled() else own


def view(arr, r):
    return arr


def _map(run, r, name, arr, i, cmap=None, label=None, contour0=False, pts=None):
    g = r["grid"]
    arr = view(arr, r)
    cm, unit, info = INFO.get(name, ("turbo", "", ""))
    fig = plots.grid_figure(arr, g["transform"], f"{name} · {r['_name']}", palette(cmap or cm), label or unit,
                            overlay_pts=pts, stretch=stretch(), smooth=filled())
    if contour0:
        ax = fig.axes[0]
        h, w = arr.shape
        t = g["transform"]
        xs = t.c + t.a * (np.arange(w) + 0.5)
        ys = t.f + t.e * (np.arange(h) + 0.5)
        ax.contour(xs, ys, np.where(np.isfinite(arr), arr, np.nan), levels=[0], colors="k", linewidths=0.8)
    a, b = st.columns([3, 1])
    with a:
        show_fig(fig, f"{safe_name(name)}.png", f"mag_png_{safe_name(name)}_{i}")
    with b:
        if info:
            st.caption(info)
        v = arr[np.isfinite(arr)]
        if v.size:
            st.metric("Min", f"{v.min():,.4g}")
            st.metric("Max", f"{v.max():,.4g}")
        st.download_button("GeoTIFF", lambda: export.geotiff_bytes({safe_name(name): arr}, g["transform"], run["epsg"]),
                           f"{safe_name(name)}.tif", "image/tiff", on_click="ignore", key=f"mag_tif_{safe_name(name)}_{i}")


def _all_maps_zip(run, r):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, arr in r["maps"].items():
            cm, unit, _ = INFO.get(name.split(" ")[0] if name.startswith("Upward") else name, ("turbo", "", ""))
            if name == "SPI depth":
                cm, unit = "viridis_r", "m"
            fig = plots.grid_figure(view(arr, r), r["grid"]["transform"], f"{name} · {r['_name']}", palette(cm), unit,
                                    stretch=stretch(), smooth=filled())
            z.writestr(f"{safe_name(name)}.png", export.fig_to_png(fig, 150))
            plt.close(fig)
        z.writestr("magnetic_maps.tif", export.geotiff_bytes({safe_name(k): v for k, v in r["maps"].items()},
                                                             r["grid"]["transform"], run["epsg"]))
    return buf.getvalue()


def _lineaments(run, r, i):
    g = r["grid"]
    c1, c2 = st.columns(2)
    pct = c1.slider("Selectivity", 80, 98, key=default(f"{PREFIX}mag_pct_{i}", 90),
                    help="Higher keeps only the clearest structures.")
    lcells = c2.slider("Min length (cells)", 5, 40, key=default(f"{PREFIX}mag_len_{i}", 15))
    tilt = view(r["maps"]["Tilt derivative"], r)
    lines = memo(run, "maglin", lambda: lineaments.detect([tilt], g["valid"], g["cell"], 1.5, pct, lcells * g["cell"],
                                                           2 * g["cell"], merge_m=3 * g["cell"])[0], i, pct, lcells)
    starts, lens = lineaments.rose_stats(lines)
    order = np.argsort(lens)[::-1]
    k1, k2, k3 = st.columns(3)
    k1.metric("Lineaments", len(lines))
    k2.metric("Total length", f"{sum(l['length_m'] for l in lines) / 1000:,.1f} km")
    k3.metric("Main trend", lineaments.strike_label(starts[order[0]] + 5) if lines else "–")
    if not lines:
        st.warning("No lineaments found; lower the selectivity.")
        return
    a, b = st.columns([3, 2])
    with a:
        show_fig(export.lines_figure(tilt, lines, g["transform"], None, f"Magnetic lineaments on tilt derivative · "
                                                                       f"{r['_name']}", "RdBu_r"),
                 "magnetic_lineaments.png", f"mag_png_lin_{i}")
    with b:
        show_fig(export.rose_figure(starts, lens, title="Strike of magnetic lineaments"), "magnetic_rose.png",
                 f"mag_png_rose_{i}")
        st.dataframe(pd.DataFrame([{"Trend": lineaments.strike_label(starts[j] + 5),
                                    "Strike": f"{starts[j]:.0f}–{starts[j] + 10:.0f}°",
                                    "Length (km)": round(float(lens[j]), 2)} for j in order[:4] if lens[j] > 0]),
                     hide_index=True, width="stretch")
        st.download_button("Lineaments (GeoJSON)", lineaments.to_geojson(lines, g["transform"], run["epsg"]),
                           "magnetic_lineaments.geojson", "application/geo+json", on_click="ignore",
                           key=f"mag_geojson_{i}")
    st.caption("Faults, shear zones, dykes and contacts traced automatically on the tilt derivative. Their strike is "
               "the main structural control on where gold-bearing fluids flowed.")


def _upward(run, r, i):
    g = r["grid"]
    h = st.slider("Continuation height (m)", 10, 2000, step=10,
                  key=default(f"{PREFIX}mag_up_{i}", int(round(r["up_height"]))),
                  help="Higher = view from further above: shallow, local anomalies fade, deep regional sources "
                       "remain.")
    arr = memo(run, "up", lambda: np.where(g["valid"], upward(g["zf"], g["cell"], h), np.nan).astype(np.float32), i, h)
    INFO["Upward continuation"] = ("turbo", "nT", f"Field as it would be measured {h} m higher. Use it to separate "
                                                  "deep (regional) from shallow sources; the difference from the TMI "
                                                  "is the residual field.")
    _map(run, r, "Upward continuation", arr, i)


def _spi(run, r, i):
    g = r["grid"]
    pts = r["spi_points"]
    st.caption("Source Parameter Imaging (contact model): depth = 1 / local wavenumber of the tilt angle, computed "
               "over the contacts (where the tilt is near 0°). Coloured points show depths to the top of magnetic "
               "contacts.")
    if not len(pts):
        st.warning("No SPI solutions; the data may be too smooth or too noisy.")
        return
    q = pts.depth.quantile([0.25, 0.5, 0.75])
    k1, k2, k3 = st.columns(3)
    k1.metric("SPI points", f"{len(pts):,}")
    k2.metric("Median depth", f"{q[0.5]:.0f} m")
    k3.metric("Middle 50%", f"{q[0.25]:.0f}–{q[0.75]:.0f} m")
    a, b = st.columns([3, 2])
    with a:
        fig = plots.euler_figure(view(r["maps"]["Tilt derivative"], r), g["transform"], pts,
                                 f"SPI depth to contacts · {r['_name']}")
        show_fig(fig, "spi_depth.png", f"mag_png_spi_{i}")
    with b:
        show_fig(plots.hist_figure(pts.depth, None, "SPI depths", "Depth (m)", log=False), "spi_hist.png",
                 f"mag_png_spih_{i}")
        st.download_button("SPI depth grid (GeoTIFF)",
                           lambda: export.geotiff_bytes({"spi_depth_m": r["maps"]["SPI depth"]}, g["transform"],
                                                        run["epsg"]), "spi_depth.tif", "image/tiff", on_click="ignore",
                           key=f"mag_spi_tif_{i}")
        st.download_button("SPI points (CSV)", pts.round(1).to_csv(index=False), "spi_points.csv", "text/csv",
                           on_click="ignore", key=f"mag_spi_csv_{i}")


def _euler(run, r, i):
    g, eu = r["grid"], r["euler"]
    st.caption(f"3-D Euler deconvolution (structural index {run['settings'].get('si', 1.0)}): depth to the top of "
               "magnetic sources, solved window by window from the field and its gradients.")
    if not len(eu):
        st.warning("No reliable Euler solutions. Try another structural index or window size (Advanced settings).")
        return
    q = eu.depth.quantile([0.1, 0.25, 0.5, 0.75])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Solutions", f"{len(eu):,}")
    k2.metric("Median depth", f"{q[0.5]:.0f} m") 
    k3.metric("Middle 50%", f"{q[0.25]:.0f}–{q[0.75]:.0f} m")
    k4.metric("Shallowest 10%", f"< {q[0.1]:.0f} m")
    fig3 = go.Figure(go.Scatter3d(
        x=eu.x, y=eu.y, z=-eu.depth, mode="markers",
        marker=dict(size=3, color=eu.depth, colorscale="Plasma_r", colorbar=dict(title="Depth (m)"), opacity=0.85),
        hovertemplate="E %{x:.0f}<br>N %{y:.0f}<br>depth %{marker.color:.0f} m<extra></extra>"))
    fig3.update_layout(height=560, margin=dict(l=0, r=0, t=30, b=0), title="Euler solutions in 3-D (drag to rotate)",
                       scene=dict(xaxis_title="Easting (m)", yaxis_title="Northing (m)", zaxis_title="Elevation (m)",
                                  aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.4)))
    st.plotly_chart(fig3, width="stretch", key=f"mag_euler3d_{i}")
    a, b = st.columns([3, 2])
    with a:
        bg = view(r["maps"].get("RTP", r["maps"].get("RTE")), r)
        show_fig(plots.euler_figure(bg, g["transform"], eu, f"Euler solutions (plan) · {r['_name']}"),
                 "euler_solutions.png", f"mag_png_euler_{i}")
    with b:
        show_fig(plots.hist_figure(eu.depth, None, "Euler depths", "Depth (m)", log=False), "euler_hist.png",
                 f"mag_png_eh_{i}")
        st.download_button("Euler solutions (CSV)", eu.round(1).to_csv(index=False), "euler_solutions.csv", "text/csv",
                           on_click="ignore", key=f"mag_euler_csv_{i}")


TARGET_LAYERS = {
    "Structure density": ("High", 1.0, "Concentration of lineaments (faults, contacts)."),
    "Proximity to structures": ("Low", 0.8, "Distance to the nearest lineament (closer is better)."),
    "Magnetic low (alteration)": ("Low", 0.6, "Magnetite destroyed by hydrothermal alteration along shears."),
    "Analytic signal": ("High", 0.3, "Edges of magnetic units such as BIF and mafic rocks."),
}


def _targets(run, ds, r, i):
    g = r["grid"]
    st.caption("Magnetic-only gold targets: ground with dense structures close to faults and contacts, with magnetic "
               "lows from alteration. Combine with IP, geology and geochemistry on the Integrated page.")
    cols = st.columns(4)
    w = {k: cols[j].slider(k, 0.0, 1.0, step=0.1, key=default(f"{PREFIX}mag_w{j}_{i}", d[1]), help=d[2])
         for j, (k, d) in enumerate(TARGET_LAYERS.items())}
    c1, c2 = st.columns(2)
    top = c1.slider("Top % as targets", 1.0, 20.0, step=0.5, key=default(f"{PREFIX}mag_top_{i}", 5.0))
    min_ha = c2.slider("Smallest target (ha)", 0.0, 20.0, step=0.5, key=default(f"{PREFIX}mag_minha_{i}", 1.0))
    lines = r["lineaments"]
    valid = g["valid"]
    red = r["maps"]["RTP"] if "RTP" in r["maps"] else -r["maps"]["RTE"]  # magnetic bodies positive

    def members():
        m = {"Analytic signal": prospectivity.fuzzy_linear(r["maps"]["Analytic signal"], valid),
             "Magnetic low (alteration)": prospectivity.fuzzy_linear(red, valid, invert=True)}
        if lines:
            m["Structure density"] = prospectivity.fuzzy_linear(
                lineaments.density(lines, valid.shape, g["cell"], 8 * g["cell"]), valid, 0, 98)
            m["Proximity to structures"] = prospectivity.fuzzy_distance(
                lineaments.distance(lines, valid.shape, g["cell"]), valid, max(3 * g["cell"], 100))
        return m
    mem = memo(run, "mag_members", members, i)
    used = {k: v for k, v in w.items() if v > 0 and k in mem}
    score, _ = interpret.knowledge_driven({k: mem[k] for k in used}, used)
    if score is None:
        st.info("Give at least one layer a weight above 0.")
        return
    min_px = max(1, int(round(min_ha * 1e4 / g["cell"] ** 2)))
    zones, t = interpret.describe_targets(score, np.isfinite(score), g["transform"], top, min_px,
                                          {k: mem[k] for k in used}, {"geophysics": [r]}, {"geophysics": [ds]})
    if len(t):
        lon, lat = to_lonlat(run["epsg"], t["Easting"], t["Northing"])
        t.insert(1, "Latitude", np.round(lat, 6))
        t.insert(2, "Longitude", np.round(lon, 6))
    a, b = st.columns([3, 2])
    with a:
        show_fig(plots.grid_figure(view(score, r), g["transform"], f"Magnetic target map (pink = targets) · {r['_name']}",
                                   "inferno", "score", targets=zones["targets"], lo=1, hi=99),
                 "magnetic_targets.png", f"mag_png_tgt_{i}")
    with b:
        st.markdown(f"**{len(t)} targets**")
        if len(t):
            st.dataframe(t[["Target", "Latitude", "Longitude", "Area (ha)", "Score", "Est. depth (m)", "Trend"]],
                         hide_index=True, width="stretch", height=min(38 + 35 * len(t), 380))
            st.download_button("Targets (CSV)", t.to_csv(index=False), "magnetic_targets.csv", "text/csv",
                               on_click="ignore", key=f"mag_tgt_csv_{i}")
        st.download_button("Target score (GeoTIFF)",
                           lambda: export.geotiff_bytes({"magnetic_target_score": score,
                                                         "target": zones["targets"].astype(np.float32)},
                                                        g["transform"], run["epsg"]),
                           "magnetic_targets.tif", "image/tiff", on_click="ignore", key=f"mag_tgt_tif_{i}")
    if len(t):
        with st.expander("Target details"):
            st.dataframe(t, hide_index=True, width="stretch")


@fragment
def render(run):
    items = run["datasets"]["geophysics"]
    if not items:
        st.info("No magnetic data.")
        return
    i = 0
    ds = items[i]
    normal = (run["results"].get("geophysics_normal") or [None])[i]
    if normal is None or "maps" not in normal:
        st.caption("This file is already a grid, so there is one image type.")
        ss[VIEW_KEY] = VIEWS[1]
    else:
        st.radio("Image type", VIEWS, horizontal=True, key=default(VIEW_KEY, VIEWS[0]),
                 help="**Normal**: the readings gridded as they are, in the original colours; ground between "
                      "survey lines that no reading reaches stays blank. **Filled**: spikes removed, a "
                      "continuous surface across the whole survey, drawn the way Oasis montaj draws it. "
                      "Applies to every map below and to the downloads.")
    use_filled = filled() or normal is None or "maps" not in normal
    r = run["results"]["geophysics"][i] if use_filled else normal
    # each image type keeps its own cached lineaments, targets and continuation
    run = {**run, "_memo": run.setdefault("_memo_filled" if use_filled else "_memo_normal", {})}
    r["_name"] = ds["name"]
    if "maps" not in r:
        st.error("This dataset could not be processed as magnetics.")
        return
    if ds.get("from_image"):
        st.warning("Values come from the colours of a picture (relative 0–1 scale, not nT). Trends are usable; "
                   "depths are not reliable.")
    red = r["reduction"]
    gm = r["grid"].get("geometry", {})
    layout = (f" · {gm['n_lines']} survey lines {gm['line_spacing']:.1f} m apart, stations {gm['station']:.1f} m"
              if gm.get("is_line_survey") else "")
    dsp = r.get("despiked") or {}
    spikes = (f" · {dsp['dropped']:,} of {dsp['total']:,} readings removed as spikes "
              f"(> {dsp['tol']:.0f}σ, σ = {dsp['sigma']:,.0f} nT)" if dsp.get("dropped") else "")
    levelled = f" · microlevelled, correction {r['microlevelled']:,.0f} nT rms" if r.get("microlevelled") else ""
    st.caption(f"Earth's field used: inclination {r['inc']:.1f}°, declination {r['dec']:.1f}° → **{red}** · grid cell "
               f"{r['grid']['cell']:.1f} m · noise smoothing {r['smooth_m']:.0f} m (derivatives)"
               f"{layout}{spikes}{levelled}.")
    names = ["Overview", "TMI", f"{red}", "1VD", "2VD", "Analytic signal", "Tilt derivative", "THDR", "Lineaments",
             "Upward continuation", "SPI depth", "Euler 3D", "Final target map"]
    tabs = st.tabs(names, key="mag_tabs", on_change="rerun")
    maps = r["maps"]
    for tab, name in zip(tabs, names):
        if not tab.open:
            continue
        with tab:
            if name == "Overview":
                show = [k for k in maps if k != "SPI depth"]
                cms = [palette(INFO.get(k, ("turbo",))[0] if not k.startswith("Upward") else "turbo") for k in show]
                show_fig(plots.panels([view(maps[k], r) for k in show], show, r["grid"]["transform"], cms, ncols=4,
                                      suptitle=f"Magnetic maps · {ds['name']}", stretch=stretch(),
                                      smooth=filled()),
                         "magnetic_overview.png",
                         f"mag_png_over_{i}")
                st.download_button("Download all maps (ZIP of PNGs + multi-band GeoTIFF)", lambda: _all_maps_zip(run, r),
                                   "magnetic_maps.zip", "application/zip", on_click="ignore", key=f"mag_zip_{i}")
            elif name == "TMI":
                pts = (ds["x"], ds["y"]) if ds["kind"] == "points" and len(ds["x"]) < 20000 else None
                _map(run, r, "TMI", maps["TMI"], i, pts=pts)
            elif name in ("RTP", "RTE", "1VD", "2VD", "Analytic signal", "THDR"):
                _map(run, r, name, maps[name], i)
            elif name == "Tilt derivative":
                _map(run, r, name, maps[name], i, contour0=True)
            elif name == "Lineaments":
                _lineaments(run, r, i)
            elif name == "Upward continuation":
                _upward(run, r, i)
            elif name == "SPI depth":
                _spi(run, r, i)
            elif name == "Euler 3D":
                _euler(run, r, i)
            elif name == "Final target map":
                _targets(run, ds, r, i)
