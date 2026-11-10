"""PCA images: standard PCA of six bands and Crósta feature-oriented PCs for clay and iron."""
import numpy as np
import pandas as pd
import streamlit as st

from geoai import export, indices, pca

from .common import crosta, get_scene, header, need_scene, show_fig, std_pca


@st.fragment
def render(aoi):
    st.caption("Principal Component Analysis turns correlated bands into uncorrelated components. PC1 holds "
               "overall brightness and topography; the later PCs hold the subtle spectral differences between "
               "rocks, soils and alteration minerals.")
    sc = get_scene(aoi)
    if sc is None:
        need_scene()
        return
    bands, tr, epsg, inside, clear, aoi_utm = header(sc, aoi)
    date = sc["data"]["date"]
    try:
        res = std_pca(sc)
    except ValueError as e:
        st.warning(str(e))
        return
    pcs, ex = res["pcs"], res["explained"]
    n = len(ex)

    st.markdown("**PCA colour composite**")
    combos = ["PC1-PC2-PC3", "PC2-PC3-PC4", "PC3-PC4-PC5", "PC4-PC5-PC6"]
    combo = st.selectbox("Components (R-G-B)", combos, key="rs_pca_rgb")
    idx = [int(p[2:]) - 1 for p in combo.split("-")]
    rgb = np.dstack([indices.stretch(pcs[i], clear) for i in idx])
    rgb[~clear] = 0
    left, right = st.columns([3, 2])
    with left:
        show_fig(export.rgb_figure(rgb, tr, aoi_utm, f"PCA composite {combo} · {date}"), "pca_composite.png",
                 key="pca_png_rgb")
    with right:
        show_fig(export.loadings_figure(res["loadings"], res["names"], ex, "Loadings (eigenvectors)"),
                 "pca_loadings.png", key="pca_png_load")
        st.caption("Each row shows how much each band contributes to a PC. A PC with large opposite loadings on "
                   "two bands highlights materials that differ between those bands.")
        st.download_button("Composite GeoTIFF", lambda: export.rgb_geotiff_bytes(rgb, tr, epsg),
                           "pca_composite.tif", "image/tiff", on_click="ignore", key="pca_tif_rgb")

    st.markdown("**Principal component images**")
    show_fig(export.panels_figure([pcs[i] for i in range(n)],
                                  [f"PC{i + 1} ({100 * ex[i]:.1f}% of variance)" for i in range(n)],
                                  ["gray"] * n, tr, valid=clear, aoi_utm=aoi_utm, ncols=3,
                                  suptitle=f"Standardised PCA of B2, B3, B4, B8, B11, B12 · {date}"),
             "pca_components.png", key="pca_png_all")
    st.download_button("All PCs (GeoTIFF)",
                       lambda: export.geotiff_bytes({f"PC{i + 1}": pcs[i] for i in range(n)}, tr, epsg),
                       f"pca_{date}.tif", "image/tiff", on_click="ignore", key="pca_tif_all")

    st.markdown("**Crósta technique: mineral-specific PCs** (bright = target mineral group)")
    cols = st.columns(2)
    for col, kind, cmap in [(cols[0], "hydroxyl", "magma"), (cols[1], "iron", "YlOrRd")]:
        with col:
            try:
                cr = crosta(sc, kind)
            except ValueError as e:
                st.warning(str(e))
                continue
            show_fig(export.raster_figure(cr["image"], tr, aoi_utm,
                                          f"{cr['label']} = PC{cr['pc_index'] + 1}", cmap, valid=clear,
                                          cbar_label="PC score"), f"crosta_{kind}.png", key=f"pca_png_{kind}")
            st.caption(cr["info"])
            st.dataframe(pd.DataFrame(np.round(cr["loadings"], 3), columns=cr["names"],
                                      index=[f"PC{i + 1} ({100 * e:.1f}%)" for i, e in enumerate(cr["explained"])]),
                         width="stretch")
