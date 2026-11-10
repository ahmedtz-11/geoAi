import streamlit as st

st.title("GeoAI Platform")
st.caption("Tools to help geologists and geophysicists find underground minerals, with a focus on gold.")

c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.subheader("🛰️ 1 · Remote sensing")
    st.markdown(
        "**Input:** CSV or Excel file with the 4 corner coordinates of your licence or study area "
        "(lat/long or UTM).\n\n"
        "**You get:**\n"
        "- the area on a map, with its size in km² and hectares\n"
        "- a high-resolution satellite image of the area\n"
        "- the latest Sentinel-2 scene: colour composites and gold-alteration maps "
        "(clay/sericite, iron oxide, gossan)\n"
        "- candidate alteration zones to check in the field\n"
        "- downloads: PNG, GeoTIFF, KML, GeoJSON, CSV"
    )
    st.page_link("views/remote_sensing.py", label="Open remote sensing", icon="➡️")
with c2, st.container(border=True):
    st.subheader("🧭 2 · Interpretation & gold prediction")
    st.markdown(
        "**Input:** magnetic, IP, geological and geochemical data (.xls/.xlsx, .dat, .csv, images).\n\n"
        "**You get:** interpreted maps, trends (strike), depth estimates and gold prediction.\n\n"
        "*In development: gridding, residual, gradient, AI anomaly and multi-dataset target tools "
        "are available now.*"
    )
    st.page_link("views/interpretation.py", label="Open interpretation", icon="➡️")

st.info("Uploaded data stays in memory for your session only. Nothing is saved on the server.")
