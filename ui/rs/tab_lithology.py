"""Lithological (spectral unit) classification map."""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from geoai import export, lithology

from .common import get_scene, header, lith, need_scene, show_fig

WAVELENGTH = {"B02": 490, "B03": 560, "B04": 665, "B08": 842, "B11": 1610, "B12": 2190}


def spectra_figure(means, colors):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [WAVELENGTH[b] for b in lithology.FEATURE_BANDS]
    for (name, r), c in zip(means.iterrows(), colors):
        ax.plot(x, [r[b] for b in lithology.FEATURE_BANDS], "-o", color=c, label=name, ms=4)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean reflectance")
    ax.set_title("Average spectrum of each unit", fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


@st.fragment
def render(aoi):
    st.caption("Unsupervised K-means classification of Sentinel-2 reflectance and mineral indices. Pixels with a "
               "similar spectrum are grouped into units that usually follow rock type and its weathering cover. "
               "Units are spectral, so name them from field mapping or a geological map.")
    sc = get_scene(aoi)
    if sc is None:
        need_scene()
        return
    bands, tr, epsg, inside, clear, aoi_utm = header(sc, aoi)
    k = st.slider("Number of units", 3, 12, key="rs_lith_k",
                  help="Start with 5-8. More units split the ground more finely.")
    try:
        with st.spinner("Classifying…"):
            res = lith(sc, k)
    except ValueError as e:
        st.warning(str(e))
        return
    labels, means, colors = res["labels"], res["means"], res["colors"]
    px_ha = sc["data"]["res"] ** 2 / 1e4
    inside_lab = np.where(inside, labels, -1)
    legend = [f"{u}: {h}" for u, h in zip(means.index, means["hint"])]
    left, right = st.columns([3, 2])
    with left:
        show_fig(export.class_figure(inside_lab, tr, legend, colors, aoi_utm,
                                     f"Lithological (spectral) classification · {sc['data']['date']}"),
                 "lithology_classes.png", key="lith_png_map")
    with right:
        show_fig(spectra_figure(means, colors), "lithology_spectra.png", key="lith_png_spec")
    n_in = max(int((inside_lab >= 0).sum()), 1)
    tbl = means[["hint", "clay_hydroxyl", "iron_oxide", "ndvi"]].copy()
    tbl.insert(1, "Area (ha)", [round(float((inside_lab == i).sum() * px_ha), 2) for i in range(k)])
    tbl.insert(2, "% of AOI", [round(100 * float((inside_lab == i).sum()) / n_in, 1) for i in range(k)])
    tbl.columns = ["Spectral interpretation", "Area (ha)", "% of AOI", "Clay index (B11/B12)",
                   "Iron index (B4/B2)", "NDVI"]
    st.dataframe(tbl.round(3), width="stretch")
    c1, c2 = st.columns(2)
    c1.download_button("Units table (CSV)", tbl.to_csv(), "lithology_units.csv", "text/csv", on_click="ignore",
                       key="lith_csv")
    c2.download_button("Classification (GeoTIFF)",
                       lambda: export.geotiff_bytes({"unit": np.where(labels >= 0, labels + 1, np.nan)
                                                    .astype(np.float32)}, tr, epsg),
                       f"lithology_{sc['data']['date']}.tif", "image/tiff", on_click="ignore", key="lith_tif")
    st.caption("Interpretations are automatic hints from each unit's average spectrum, not confirmed rock types.")
