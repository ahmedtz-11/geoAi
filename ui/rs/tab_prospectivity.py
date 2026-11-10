"""Integrated mineral prospectivity map (fuzzy overlay of alteration, structure and lithology)."""
import numpy as np
import streamlit as st

from geoai import export, lineaments, prospectivity

from .common import (current_lineaments, get_scene, header, index_layer, lith, need_scene, show_fig, ss,
                     usable_mask, zones_table)

LAYERS = {
    "clay": ("Clay / hydroxyl alteration", "rs_p_w_clay", "Purples"),
    "iron": ("Iron oxide (gossan)", "rs_p_w_iron", "YlOrRd"),
    "lden": ("Lineament density", "rs_p_w_lden", "hot_r"),
    "lprox": ("Proximity to lineaments", "rs_p_w_lprox", "Reds"),
    "lith": ("Favourable spectral units", "rs_p_w_lith", "Greens"),
}


@st.fragment
def render(aoi):
    st.caption("Combines the evidence for gold into one map. Each layer is scaled from 0 (unfavourable) to 1 "
               "(most favourable), weighted, and summed. The brightest ground has alteration, structure and "
               "favourable rock together: the best places to start field checks.")
    sc = get_scene(aoi)
    if sc is None:
        need_scene()
        return
    bands, tr, epsg, inside, clear, aoi_utm = header(sc, aoi)
    res_m = sc["data"]["res"]

    st.markdown("**Evidence weights** (0 = ignore)")
    cols = st.columns(5)
    w = {k: cols[i].slider(name, 0.0, 1.0, step=0.1, key=key) for i, (k, (name, key, _)) in enumerate(LAYERS.items())}
    o1, o2, o3, o4 = st.columns(4)
    method = o1.radio("Combine by", ["Weighted sum", "Fuzzy gamma"], key="rs_p_method",
                      help="Weighted sum averages the evidence. Fuzzy gamma rewards places where several layers "
                           "are high at once.")
    gamma = o2.slider("Gamma", 0.5, 0.95, step=0.05, key="rs_p_gamma", disabled=method != "Fuzzy gamma")
    top = o3.slider("Top % as targets", 1.0, 20.0, step=0.5, key="rs_p_top")
    min_ha = o4.slider("Smallest target (ha)", 0.0, 5.0, step=0.1, key="rs_p_minha")
    hide_veg = st.checkbox("Exclude dense vegetation (alteration can't be seen there)", key="rs_p_veg")
    valid = usable_mask(sc, hide_veg) if (w["clay"] or w["iron"]) else inside & np.isfinite(bands["B04"])
    valid = valid & inside

    with st.spinner("Building evidence layers…"):
        layers = {}
        if w["clay"]:
            layers["clay"] = prospectivity.fuzzy_linear(index_layer(sc, "clay", ss["rs_clay_method"]), valid)
        if w["iron"]:
            layers["iron"] = prospectivity.fuzzy_linear(index_layer(sc, "iron", ss["rs_iron_method"]), valid)
        if w["lden"] or w["lprox"]:
            lines, _ = current_lineaments(sc)
            if lines:
                if w["lden"]:
                    layers["lden"] = prospectivity.fuzzy_linear(
                        lineaments.density(lines, inside.shape, res_m, ss["rs_lin_radius"]), valid, 0, 98)
                if w["lprox"]:
                    layers["lprox"] = prospectivity.fuzzy_distance(
                        lineaments.distance(lines, inside.shape, res_m), valid, half_m=max(2 * res_m, 150))
            else:
                st.warning("No lineaments with the current settings; structure layers skipped.")
        if w["lith"]:
            lres = lith(sc, ss["rs_lith_k"])
            units = list(lres["means"].index)
            default = [u for u, h in zip(units, lres["means"]["hint"]) if "alteration" in h or "Iron" in h]
            fav = st.multiselect("Favourable spectral units (from the lithological classification)", units,
                                 default=[u for u in ss.get("rs_p_units", default) if u in units],
                                 format_func=lambda u: f"{u}: {lres['means'].loc[u, 'hint']}")
            ss["rs_p_units"] = fav
            idx = [units.index(u) for u in fav]
            layers["lith"] = np.where(valid, np.isin(lres["labels"], idx).astype(np.float32), np.nan)
    if not layers:
        st.info("Give at least one evidence layer a weight above 0.")
        return
    score = prospectivity.combine(layers, w, valid, method, gamma)
    min_px = max(1, int(round(min_ha * 1e4 / res_m ** 2)))
    z = prospectivity.targets(score, valid, top, min_px, tr)
    zt = zones_table(z["zones"], epsg)

    k1, k2, k3 = st.columns(3)
    k1.metric("Target zones", f"{len(zt)}")
    k2.metric("Target area", f"{z['targets'].sum() * res_m ** 2 / 1e4:,.1f} ha")
    k3.metric("Area evaluated", f"{100 * np.isfinite(score).sum() / max(inside.sum(), 1):.0f}% of AOI")
    left, right = st.columns([3, 2])
    with left:
        show_fig(export.raster_figure(score, tr, aoi_utm, f"Integrated gold prospectivity · {sc['data']['date']}",
                                      "inferno", overlay=z["targets"], cbar_label="prospectivity (0–1)", lo=1, hi=99),
                 "prospectivity_map.png", key="p_png_map")
    with right:
        st.markdown("**Ranked targets** (pink on the map)")
        if len(zt):
            st.dataframe(zt.drop(columns=["Easting", "Northing"]), hide_index=True, width="stretch",
                         height=min(35 * len(zt) + 38, 360))
            st.download_button("Targets (CSV)", zt.to_csv(index=False), "prospectivity_targets.csv", "text/csv",
                               on_click="ignore", key="p_csv")
        else:
            st.info("No target zones; lower the smallest target size.")
        st.download_button("Prospectivity + evidence (GeoTIFF)",
                           lambda: export.geotiff_bytes({"prospectivity": score,
                                                         "target": z["targets"].astype(np.float32),
                                                         **{f"evidence_{k}": a for k, a in layers.items()}}, tr, epsg),
                           "prospectivity.tif", "image/tiff", on_click="ignore", key="p_tif")
    st.markdown("**Evidence layers used** (0 = unfavourable, 1 = favourable)")
    keys = list(layers)
    show_fig(export.panels_figure([layers[k] for k in keys], [f"{LAYERS[k][0]} (w={w[k]:.1f})" for k in keys],
                                  [LAYERS[k][2] for k in keys], tr, aoi_utm=aoi_utm, ncols=min(len(keys), 5),
                                  lo=0, hi=100), "prospectivity_evidence.png", key="p_png_ev")
    st.caption("Alteration layers follow the method chosen on their own tabs, lineaments use the settings on the "
               "lineament tab, and spectral units come from the lithological classification. This map ranks "
               "ground for follow-up; confirm targets with soil geochemistry, geophysics (Section 2) and mapping.")
