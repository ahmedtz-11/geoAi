"""Iron oxide / ferric iron map and clay / hydroxyl alteration map (same layout, different index)."""
import numpy as np
import pandas as pd
import streamlit as st

from geoai import export, indices

from .common import crosta, get_scene, header, index_layer, need_scene, show_fig, ss, usable_mask
from ui.state import fragment

CFG = {
    "iron": {
        "name": "Iron oxide", "ratio": "B4/B2", "cmap": "YlOrRd", "colors": indices.IRON_COLORS,
        "class_title": "Ferric iron anomaly classes", "file": "iron_oxide",
        "caption": "Iron Oxide Index = Red / Blue (Sentinel-2 B4/B2). High values mark ferric iron minerals "
                   "(hematite, goethite, limonite, jarosite), typical of gossans and oxidised sulphide zones "
                   "that can accompany gold mineralisation.",
        "note": "Moderate and Strong classes are the ones to follow up. Bare red laterite soils and ironstone "
                "also give high values, so compare with geology and the clay/hydroxyl map.",
    },
    "clay": {
        "name": "Clay / hydroxyl", "ratio": "B11/B12", "cmap": "Purples", "colors": indices.CLAY_COLORS,
        "class_title": "Clay / hydroxyl alteration classes", "file": "clay_hydroxyl",
        "caption": "Clay / Hydroxyl Index = SWIR-1 / SWIR-2 (Sentinel-2 B11/B12). Al-OH and Mg-OH minerals "
                   "(sericite/muscovite, kaolinite, illite, smectite, chlorite) absorb near 2.2 µm (B12), so high "
                   "values mark phyllic and argillic alteration around gold-bearing quartz veins and shear zones.",
        "note": "Moderate and Strong classes outline possible hydrothermal alteration. Weathered clay-rich soils, "
                "dry grass and some granites also score high, so check against geology and the iron oxide map.",
    },
}


@fragment
def render(aoi, kind):
    cfg = CFG[kind]
    st.caption(cfg["caption"])
    sc = get_scene(aoi)
    if sc is None:
        need_scene()
        return
    bands, tr, epsg, inside, clear, aoi_utm = header(sc, aoi)
    c1, c2 = st.columns([2, 3])
    method = c1.radio("Method", ["Band ratio", "Crósta PCA"], horizontal=True, key=f"rs_{kind}_method",
                      help="Crósta PCA (feature-oriented principal components) separates the mineral signal from "
                           "brightness and topography; band ratio is simpler and easier to compare between scenes.")
    hide_veg = c2.checkbox(f"Hide dense vegetation (NDVI above 0.4)", key=f"rs_{kind}_veg",
                           help="Vegetation masks the rock signal underneath.")
    arr = index_layer(sc, kind, method)
    valid = usable_mask(sc, hide_veg)
    cls, thr = indices.classify_sigma(arr, valid)
    px_ha = sc["data"]["res"] ** 2 / 1e4
    n_valid = int((cls >= 0).sum())
    if n_valid < 20:
        st.warning("Too few clear, unvegetated pixels in the area. Untick the vegetation option or choose "
                   "another scene.")
        return
    unit = cfg["ratio"] if method == "Band ratio" else "PC score"
    label = f"{cfg['name']} Index ({cfg['ratio']})" if method == "Band ratio" else crosta(
        sc, "hydroxyl" if kind == "clay" else "iron")["label"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Scene date", sc["data"]["date"])
    k2.metric(f"Mean ({unit})", f"{thr[0]:.3f}")
    k3.metric("Strong anomaly area", f"{(cls == 3).sum() * px_ha:,.1f} ha")
    k4.metric("Area mapped", f"{100 * n_valid / max(inside.sum(), 1):.0f}% of AOI")
    m1, m2 = st.columns(2)
    with m1:
        show_fig(export.raster_figure(arr, tr, aoi_utm, f"{label} · {sc['data']['date']}", cfg["cmap"],
                                      valid=valid, cbar_label=unit), f"{cfg['file']}_index_map.png",
                 key=f"{kind}_png_index")
    with m2:
        show_fig(export.class_figure(cls, tr, indices.IRON_CLASSES, cfg["colors"], aoi_utm,
                                     f"{cfg['class_title']} · {sc['data']['date']}"),
                 f"{cfg['file']}_classes_map.png", key=f"{kind}_png_classes")
    if method != "Band ratio":
        cr = crosta(sc, "hydroxyl" if kind == "clay" else "iron")
        st.caption(f"{cr['info']} Selected: PC{cr['pc_index'] + 1}"
                   f"{' (sign inverted so bright = mineral)' if cr['sign'] < 0 else ''}.")
    ranges = [f"< {thr[0]:.3f}", f"{thr[0]:.3f} – {thr[1]:.3f}", f"{thr[1]:.3f} – {thr[2]:.3f}", f"> {thr[2]:.3f}"]
    tbl = pd.DataFrame({"Class": indices.IRON_CLASSES,
                        "Rule": ["below mean", "mean to +1σ", "+1σ to +2σ", "above +2σ"],
                        f"Range ({unit})": ranges,
                        "Area (ha)": [round(float((cls == i).sum() * px_ha), 2) for i in range(4)],
                        "% of mapped area": [round(100 * float((cls == i).sum()) / n_valid, 1) for i in range(4)]})
    st.dataframe(tbl, hide_index=True, width="stretch")
    st.download_button(f"{cfg['name']} index + classes (GeoTIFF)",
                       lambda: export.geotiff_bytes({f"{cfg['file']}_index": np.where(valid, arr, np.nan),
                                                     f"{cfg['file']}_class": np.where(cls >= 0, cls, np.nan)
                                                    .astype(np.float32)}, tr, epsg),
                       f"{cfg['file']}_{sc['data']['date']}.tif", "image/tiff", on_click="ignore", key=f"{kind}_tif")
    st.caption(cfg["note"])
