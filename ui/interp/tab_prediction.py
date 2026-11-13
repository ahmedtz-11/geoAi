"""Gold prediction: evidence weights, knowledge-driven and Random Forest prediction, ranked targets."""
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from geoai import export, plots

from . import predict
from .common import PREFIX, default, show_fig, ss, to_lonlat
from ui.state import fragment


def targets_with_ll(run, t):
    if t is None or not len(t):
        return t
    lon, lat = to_lonlat(run["epsg"], t["Easting"], t["Northing"])
    t = t.copy()
    t.insert(1, "Latitude", np.round(lat, 6))
    t.insert(2, "Longitude", np.round(lon, 6))
    return t


def report_html(run, pred, s, findings):
    tr = run["grid"][0]
    fig = plots.grid_figure(pred["score"], tr, "Gold prospectivity", "inferno", "score",
                            targets=pred["zones"]["targets"] if pred.get("zones") else None)
    png = base64.b64encode(export.fig_to_png(fig, 150)).decode()
    t = targets_with_ll(run, pred.get("targets"))
    used = "".join(f"<li>{k}: weight {w:.1f}, {s['favourable'][k].lower()} values favourable</li>"
                   for k, w in s["weights"].items() if w > 0)
    files = "".join(f"<li>{d['name']} ({slot})</li>" for slot, ds in run["datasets"].items() for d in ds)
    return f"""<html><head><meta charset="utf-8"><title>GeoAI gold prediction report</title></head>
<body style="font-family:Arial,sans-serif;max-width:960px;margin:24px auto;color:#222">
<h1>GeoAI · Gold interpretation & prediction</h1><p>Generated {datetime.now():%Y-%m-%d %H:%M} ·
coordinates EPSG:{run['epsg']}</p>
<h2>Data</h2><ul>{files}</ul>
<h2>Key findings</h2><ul>{''.join(f'<li>{f}</li>' for f in findings)}</ul>
<h2>Prediction</h2><p>Method: {s['method']}. Top {s['top']:.1f}% of the area, zones ≥ {s['min_ha']:.1f} ha.</p>
<ul>{used}</ul><img src="data:image/png;base64,{png}" style="width:100%">
<h2>Ranked targets</h2>{t.to_html(index=False) if t is not None and len(t) else '<p>No targets.</p>'}
<p style="color:#777;font-size:12px">Targets rank ground for follow-up (mapping, soil infill, trenching, drilling).
They are not proof of mineralisation. Depths are estimates from Euler deconvolution, IP pseudo-depths or the
power spectrum.</p></body></html>"""


@fragment
def render(run, findings_fn):
    if not run["grid"]:
        st.info("The uploaded data have no map coordinates, so a prediction map can't be made.")
        return
    ev, s, pred = predict.current(run)
    if not ev:
        st.info("No evidence layers could be built from the uploaded data.")
        return
    st.caption("Each evidence layer is scaled 0 (unfavourable) to 1 (favourable). Set how much each one counts "
               "(0 = ignore) and whether high or low values are favourable for gold.")
    with st.container(border=True):
        h = st.columns([4, 2, 2])
        h[0].markdown("**Evidence layer**")
        h[1].markdown("**Weight**")
        h[2].markdown("**Favourable values**")
        for name, e in ev.items():
            c = st.columns([4, 2, 2])
            c[0].markdown(f"{name}  \n<span style='color:#888;font-size:0.8em'>{e['source']} · {e['info']}</span>",
                          unsafe_allow_html=True)
            c[1].slider("w", 0.0, 1.0, step=0.1, key=default(predict.wkey(name, "w"), e["default_weight"]),
                        label_visibility="collapsed")
            c[2].radio("dir", ["High", "Low"], horizontal=True, key=default(predict.wkey(name, "dir"), e["favourable"]),
                       label_visibility="collapsed")
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.radio("Prediction method", predict.METHODS, key=default(f"{PREFIX}method", predict.METHODS[0]),
             help="Knowledge-driven uses your weights. Data-driven learns from where gold-anomalous soil samples "
                  "are, using the other layers. 'Average' combines both.")
    c2.slider("Top % as targets", 1.0, 20.0, step=0.5, key=default(f"{PREFIX}top", 5.0))
    c3.slider("Smallest target (ha)", 0.0, 20.0, step=0.5, key=default(f"{PREFIX}minha", 1.0))
    ev, s, pred = predict.current(run)
    if pred is None or pred.get("score") is None:
        msg = (pred or {}).get("rf", {}) or {}
        st.warning(msg.get("error") or "Give at least one evidence layer a weight above 0.")
        return
    tr = run["grid"][0]
    zones, t = pred["zones"], targets_with_ll(run, pred["targets"])
    k1, k2, k3 = st.columns(3)
    k1.metric("Targets", len(t))
    k2.metric("Target area", f"{zones['targets'].sum() * run['grid'][2] ** 2 / 1e4:,.1f} ha")
    if pred.get("rf") and pred["rf"].get("auc") is not None:
        k3.metric("Random Forest skill (AUC)", f"{pred['rf']['auc']:.2f}",
                  help="Area under the ROC curve from spatial cross-validation: 0.5 = no better than chance, "
                       "0.7-0.8 = useful, > 0.8 = strong.")
    a, b = st.columns([3, 2])
    with a:
        show_fig(plots.grid_figure(pred["score"], tr, "Gold prospectivity (pink = targets)", "inferno", "score",
                                   targets=zones["targets"], lo=1, hi=99), "gold_prospectivity.png", "pred_png_map")
    with b:
        st.markdown("**Ranked targets**")
        if len(t):
            st.dataframe(t[["Target", "Latitude", "Longitude", "Area (ha)", "Score", "Est. depth (m)", "Trend"]],
                         hide_index=True, width="stretch", height=min(38 + 35 * len(t), 380))
        else:
            st.info("No targets; lower the smallest target size.")
    if len(t):
        with st.expander("Target details (depth source, trend source, supporting evidence)", expanded=True):
            st.dataframe(t, hide_index=True, width="stretch")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Targets (CSV)", t.to_csv(index=False) if len(t) else "", "gold_targets.csv", "text/csv",
                       on_click="ignore", key="pred_csv", disabled=not len(t))
    d2.download_button("Prediction + evidence (GeoTIFF)",
                       lambda: export.geotiff_bytes({"gold_prospectivity": pred["score"],
                                                     "target": zones["targets"].astype(np.float32),
                                                     **{f"ev_{i + 1}": pred["members"][k] for i, k in
                                                        enumerate(pred["members"])}}, tr, run["epsg"]),
                       "gold_prediction.tif", "image/tiff", on_click="ignore", key="pred_tif")
    d3.download_button("Report (HTML)", lambda: report_html(run, pred, s, findings_fn(run)), "geoai_gold_report.html",
                       "text/html", on_click="ignore", key="pred_html")
    rf = pred.get("rf")
    if rf and "importances" in rf:
        st.markdown(f"**Random Forest**: trained on {rf['n_pos']} gold-anomalous and {rf['n_neg']} background samples "
                    f"from {pred.get('rf_source')}")
        c1, c2 = st.columns([3, 2])
        with c1:
            show_fig(plots.grid_figure(rf["prob"], tr, "Random Forest probability of a gold anomaly", "viridis",
                                       "probability", lo=0, hi=100), "rf_probability.png", "pred_png_rf")
        with c2:
            show_fig(plots.bar_figure(rf["importances"], "What the model relies on", "importance"),
                     "rf_importance.png", "pred_png_imp")
            if rf["auc"] is not None and rf["auc"] < 0.65:
                st.warning("Weak skill: the evidence layers do not explain where gold samples are anomalous.")
    elif rf and "error" in rf:
        st.info(f"Random Forest not available: {rf['error']}")
    used = [k for k, w in s["weights"].items() if w > 0]
    if used:
        st.markdown("**Evidence layers used** (0 = unfavourable, 1 = favourable)")
        show_fig(plots.panels([pred["members"][k] for k in used], used, tr, "viridis", ncols=4, lo=0, hi=100),
                 "evidence_layers.png", "pred_png_ev")
