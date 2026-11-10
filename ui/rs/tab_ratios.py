"""Band ratio images: all diagnostic ratios + ratio colour composites."""
import numpy as np
import streamlit as st

from geoai import export, indices

from .common import get_scene, header, need_scene, show_fig, ss


@st.fragment
def render(aoi):
    st.caption("Band ratios divide one band by another. This cancels most of the brightness changes caused by "
               "topography and shadow, and leaves the colour differences produced by minerals.")
    sc = get_scene(aoi)
    if sc is None:
        need_scene()
        return
    bands, tr, epsg, inside, clear, aoi_utm = header(sc, aoi)
    date = sc["data"]["date"]
    keys = list(indices.RATIOS)
    arrays = [indices.compute_ratio(bands, k) for k in keys]
    titles = [f"{k} · {indices.RATIOS[k][0]}" for k in keys]
    cmaps = [indices.RATIOS[k][2] for k in keys]

    st.markdown("**Ratio colour composite**")
    presets = list(indices.RATIO_PRESETS) + ["Custom"]
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    preset = c1.selectbox("Combination", presets, key="rs_ratio_preset")
    if preset == "Custom":
        rgb_keys = (c2.selectbox("Red", keys, key="rs_ratio_r"), c3.selectbox("Green", keys, key="rs_ratio_g"),
                    c4.selectbox("Blue", keys, key="rs_ratio_b"))
    else:
        rgb_keys = indices.RATIO_PRESETS[preset]
    rgb = indices.ratio_rgb(bands, rgb_keys, valid=clear)
    left, right = st.columns([3, 2])
    with left:
        show_fig(export.rgb_figure(rgb, tr, aoi_utm, f"R={rgb_keys[0]}  G={rgb_keys[1]}  B={rgb_keys[2]} · {date}"),
                 "ratio_composite.png", key="ratio_png_rgb")
    with right:
        st.markdown(
            "**How to read it**\n\n"
            "- In **Sabins** and **Alteration** composites, red/magenta = clay (B11/B12), green/yellow = iron "
            "oxide (B4/B2). Yellow-white areas have both: the strongest alteration signal.\n"
            "- In **Kaufmann**, vegetation is green and altered rock pink to red.\n"
            "- Use **Custom** to build any combination.")
        st.download_button("Composite GeoTIFF", lambda: export.rgb_geotiff_bytes(rgb, tr, epsg),
                           "ratio_composite.tif", "image/tiff", on_click="ignore", key="ratio_tif_rgb")

    st.markdown("**All band ratio images** (cloud and no-data masked)")
    show_fig(export.panels_figure(arrays, titles, cmaps, tr, valid=clear, aoi_utm=aoi_utm, ncols=4,
                                  suptitle=f"Sentinel-2 band ratios · {date}"), "band_ratios.png", key="ratio_png_all")
    st.download_button("All ratios (GeoTIFF, one band each)",
                       lambda: export.geotiff_bytes({k.replace("/", "_"): np.where(clear, a, np.nan)
                                                     for k, a in zip(keys, arrays)}, tr, epsg),
                       f"band_ratios_{date}.tif", "image/tiff", on_click="ignore", key="ratio_tif_all")
