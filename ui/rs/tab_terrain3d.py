"""3-D terrain tab: the landform, with a map draped over it."""
import numpy as np
import streamlit as st

from geoai import indices, terrain, terrain3d
from geoai.imagery import dem as dem_mod

from .common import get_scene, header, need_scene
from ui.state import fragment

# What to drape over the terrain: ("picture", composite name) or ("value", grid name).
DRAPES = {
    "Shaded relief": ("picture", None),
    "Satellite true colour": ("picture", "True colour (B4-B3-B2)"),
    "Geology false colour": ("picture", "Geology false colour (B12-B11-B2)"),
    "Alteration ratios (clay / iron / ferric)":
        ("picture", "Alteration ratios (R=B11/B12 clay, G=B4/B2 iron, B=B11/B8A ferric)"),
    "Elevation": ("value", "elevation"),
    "Slope": ("value", "slope"),
    "Clay / hydroxyl alteration": ("value", "clay_hydroxyl"),
    "Iron oxide": ("value", "iron_oxide"),
}
# colour scale, colour-bar label, and whether the scale runs the wrong way for this quantity
SCALES = {"elevation": ("Earth", "Elevation (m)", True), "slope": ("YlOrBr", "Slope (°)", False),
          "clay_hydroxyl": ("YlOrRd", "Clay / hydroxyl", False), "iron_oxide": ("YlOrRd", "Iron oxide", False)}
DETAIL = {"Standard (fastest)": 250, "High": 400, "Very high (slower)": 600}


@fragment
def render(aoi):
    st.caption("The landform in 3-D, with any of the maps draped over it. Drag to rotate, scroll to zoom, "
               "right-drag to pan. Seeing the maps on the ground tells you which anomalies sit on ridges of "
               "resistant rock, which follow valleys eroded along faults, and which are only slope or shadow.")
    sc = get_scene(aoi)
    if sc is None:
        need_scene()
        return
    if sc.get("dem") is None:
        st.warning(f"Elevation data is not available for this scene: {sc.get('dem_error')}")
        return
    bands, tr, epsg, inside, clear, aoi_utm = header(sc, aoi)
    dem, res_m = sc["dem"], sc["data"]["res"]
    if not np.isfinite(dem[inside]).any():
        st.warning("No elevation values inside the area.")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    name = c1.selectbox("Drape over the terrain", list(DRAPES), key="rs_d3_drape",
                        help="A picture (shaded relief, satellite, false colour) keeps its own colours. A "
                             "measured grid (elevation, slope, an index) gets a colour bar and shows its value "
                             "when you hover.")
    exag = c2.slider("Vertical exaggeration", 1.0, 30.0, step=0.5, key="rs_d3_exag",
                     help="Most ground is far wider than it is high, so relief needs stretching before it can "
                          "be seen. Elevations on the axis and in the hover stay true. Raise it on flat ground, "
                          "lower it in steep country.")
    detail = c3.selectbox("Detail", list(DETAIL), key="rs_d3_detail",
                          help="Grid points along the longer side. More detail is sharper but slower to rotate.")
    with st.expander("Lighting"):
        s1, s2 = st.columns(2)
        az = s1.slider("Sun azimuth (°)", 0, 360, step=15, key="rs_d3_az",
                       help="Direction the light comes from. Features perpendicular to it stand out most.")
        alt = s2.slider("Sun altitude (°)", 5, 85, step=5, key="rs_d3_alt")
        st.caption("Shades the picture drapes (shaded relief, satellite, false colour) so the relief reads "
                   "through them. The measured grids are lit by their own colour scale instead.")

    hs = terrain.hillshade(dem, res_m, az, alt, 1.0)
    kind, which = DRAPES[name]
    try:
        if kind == "picture":
            rgb = terrain3d.shaded_rgb(dem, hs) if which is None else \
                terrain3d.shade_rgb(indices.composite(bands, which, clear), hs)
            fig, step, cell = terrain3d.figure(dem, res_m, tr, rgb, exaggeration=exag, max_side=DETAIL[detail],
                                               aoi_utm=aoi_utm, title=f"{name} in 3-D")
        else:
            grid = {"elevation": dem, "slope": terrain.slope_deg(dem, res_m)}.get(which)
            if grid is None:
                grid = np.where(clear, indices.compute_index(bands, which), np.nan)
            scale, label, rev = SCALES[which]
            fig, step, cell = terrain3d.figure(dem, res_m, tr, None, exaggeration=exag, max_side=DETAIL[detail],
                                               aoi_utm=aoi_utm, title=f"{name} in 3-D", scalar=grid,
                                               scalar_label=label, colourscale=scale, reverse=rev)
    except Exception as e:
        st.error(f"Could not build the 3-D view: {e}")
        return
    st.plotly_chart(fig, width="stretch", key="rs_terrain_3d")
    st.download_button("3-D view (interactive HTML)",
                       lambda f=fig: f.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
                       "terrain_3d.html", "text/html", on_click="ignore", key="d3_html",
                       help="Opens in any browser and can still be rotated. Needs an internet connection the "
                            "first time it is opened.")
    st.caption(f"{dem_mod.ATTRIBUTION}. Elevation is real, stretched {exag:g}× for viewing; the axis and the "
               f"hover still read true metres. Drawn on a {cell:,.0f} m grid"
               + (f", averaged from the {res_m:g} m analysis grid" if step > 1 else "")
               + ". The red line is your area of interest.")
