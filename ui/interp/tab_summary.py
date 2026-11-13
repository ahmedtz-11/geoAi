"""Summary: what was analysed, key findings, overview map with the top targets."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pyproj import CRS

from geoai import lineaments

from . import predict
from .common import PREFIX, show_fig, ss, to_lonlat
from ui.state import fragment


def findings(run):
    out = []
    for ds, r in zip(run["datasets"]["geophysics"], run["results"]["geophysics"]):
        if ds["survey"].startswith("Magnetic") and r.get("products"):
            starts, lens = lineaments.rose_stats(r.get("lineaments") or [])
            if lens.sum():
                o = np.argsort(lens)[::-1]
                tr = [f"{lineaments.strike_label(starts[j] + 5)} ({starts[j]:.0f}–{starts[j] + 10:.0f}°)"
                      for j in o[:2] if lens[j] > 0.15 * lens.sum()]
                out.append(f"Magnetics ({ds['name']}): main structural trend{'s' if len(tr) > 1 else ''} "
                           f"{' and '.join(tr)}.")
            eu = r.get("euler")
            if eu is not None and len(eu):
                q = eu.depth.quantile([0.25, 0.5, 0.75])
                out.append(f"Magnetic source depths (Euler, {len(eu)} solutions): median {q[0.5]:.0f} m, "
                           f"middle half {q[0.25]:.0f}–{q[0.75]:.0f} m.")
        elif r.get("lines"):
            for lid, L in r["lines"].items():
                an = L["anomalies"]
                if len(an):
                    a = an.iloc[0]
                    out.append(f"IP line {lid}: strongest anomaly {a.from_x:.0f}–{a.to_x:.0f} m, from about "
                               f"{a.depth_top:.0f} to {a.depth_bottom:.0f} m deep (peak {a.peak:.1f}).")
        elif r.get("limits"):
            out.append(f"{ds['survey']} ({ds['name']}): strong anomalies above {r['limits'][1]:.2f}.")
    for ds, r in zip(run["datasets"]["geochem"], run["results"]["geochem"]):
        au = ds["au"]
        if au and "au_class" in r:
            lim = r["limits"][au]
            n = int((r["au_class"] >= 2).sum())
            mx = np.nanmax(ds["table"][au])
            corr = r["corr"][au].drop(au)
            pf = [c for c, v in corr.items() if v >= 0.3]
            out.append(f"Geochemistry ({ds['name']}): {n} samples anomalous in gold (≥ {lim[1]:.3g}), maximum "
                       f"{mx:.3g}" + (f"; pathfinders correlated with Au: {', '.join(pf)}." if pf else "."))
    pts = [d for d in run["datasets"]["geology"] if d["kind"] == "points"]
    if pts:
        fav = ss.get(f"{PREFIX}fav", run["fav_default"])
        s = np.concatenate([d["strike"] for d in pts if d.get("strike") is not None]) if \
            any(d.get("strike") is not None for d in pts) else np.array([])
        s = s[np.isfinite(s)]
        txt = f"Geology: {sum(len(d['x']) for d in pts)} mapping points; favourable hosts: {', '.join(fav) or 'none chosen'}"
        if s.size:
            h, e = np.histogram(s % 180, bins=np.arange(0, 190, 10))
            txt += f"; measured structures strike mainly {lineaments.strike_label(e[np.argmax(h)] + 5)}"
        out.append(txt + ".")
    return out


@fragment
def render(run):
    data = run["datasets"]
    counts = {k: len(v) for k, v in data.items()}
    labels = {"geophysics": "Magnetic / IP", "geology": "Geology", "remote": "Remote sensing", "geochem": "Geochemistry"}
    cols = st.columns(4)
    for c, (k, lab) in zip(cols, labels.items()):
        c.metric(lab, f"{counts[k]} file{'s' if counts[k] != 1 else ''}" if counts[k] else "—")
    st.caption(f"All data converted to {CRS.from_epsg(run['epsg']).name} (EPSG:{run['epsg']})"
               + (f" · analysis grid {run['grid'][2]:.0f} m" if run["grid"] else ""))
    for e in run["errors"]:
        st.error(e)
    st.markdown("**Key findings**")
    f = findings(run)
    ev, s, pred = predict.current(run)
    t = pred.get("targets") if pred else None
    if t is not None and len(t):
        lon, lat = to_lonlat(run["epsg"], t["Easting"], t["Northing"])
        for j in range(min(3, len(t))):
            row = t.iloc[j]
            f.append(f"Target {row.Target}: {lat[j]:.5f}, {lon[j]:.5f} · score {row.Score:.2f} · {row['Area (ha)']} ha"
                     + (f" · est. depth ~{row['Est. depth (m)']:.0f} m" if pd.notna(row["Est. depth (m)"]) else "")
                     + (f" · trend {row.Trend}" if row.Trend else "") + ".")
    st.markdown("\n".join(f"- {x}" for x in f) if f else "_No findings yet._")
    if not run["grid"]:
        return
    tr, shape, cell = run["grid"]
    fig, ax = plt.subplots(figsize=(8, 7))
    if pred and pred.get("score") is not None:
        ext = [tr.c, tr.c + shape[1] * tr.a, tr.f + shape[0] * tr.e, tr.f]
        im = ax.imshow(pred["score"], extent=ext, cmap="inferno", alpha=0.85)
        plt.colorbar(im, ax=ax, shrink=0.75, label="gold prospectivity")
    styles = {"geophysics": ("#1f77b4", 2, "Geophysics"), "geology": ("#2ca02c", 14, "Geology"),
              "geochem": ("#ff7f0e", 5, "Geochemistry"), "remote": ("#9467bd", 3, "Remote sensing points")}
    for slot, (col, size, lab) in styles.items():
        xs, ys = [], []
        for d in data[slot]:
            if d["kind"] in ("points",) or (d["kind"] == "section" and d.get("y") is not None):
                xs.append(d["x_map"] if d["kind"] == "section" else d["x"])
                ys.append(d["y"])
        if xs:
            ax.scatter(np.concatenate(xs), np.concatenate(ys), s=size, c=col, alpha=0.45, label=lab, linewidths=0)
    if t is not None and len(t):
        ax.scatter(t.Easting, t.Northing, marker="*", s=260, c="#00e5ff", edgecolors="k", label="Targets", zorder=5)
        for _, row in t.head(10).iterrows():
            ax.annotate(row.Target, (row.Easting, row.Northing), xytext=(6, 6), textcoords="offset points",
                        color="white", weight="bold", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, lw=0))
    ax.legend(loc="upper left", bbox_to_anchor=(1.25, 1), frameon=False, fontsize=8)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("Overview: data and gold targets", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    show_fig(fig, "overview_map.png", "sum_png")
    st.caption("Targets use the current settings of the Gold prediction tab.")
