import streamlit as st

st.title("GeoAI Platform")
st.caption("Tools to help geologists and geophysicists find underground minerals, with a focus on gold. "
           "Use any sections you have data for; the last one combines them.")

CARDS = [
    ("🛰️ 1 · Remote sensing", "views/remote_sensing.py",
     "**Input:** the 4 corner coordinates of your area (CSV / Excel).\n\n"
     "**You get:** satellite image, iron oxide and clay alteration, band ratios, PCA, lithological "
     "classification, lineaments, DEM / hillshade and a prospectivity map."),
    ("🧲 2 · Magnetic", "views/magnetic.py",
     "**Input:** one magnetic survey file (.dat, .csv, .xls/.xlsx, GeoTIFF or a map picture).\n\n"
     "**You get:** TMI, RTP/RTE, 1VD, 2VD, analytic signal, tilt, THDR, lineaments, upward continuation, "
     "SPI and Euler depths, and a magnetic target map."),
    ("⚡ 3 · IP", "views/ip.py",
     "**Input:** one file per survey line (.csv / .txt / Res2DInv .dat).\n\n"
     "**You get:** one pseudosection image per line, with anomaly position and depth."),
    ("🪨 4 · Geology", "views/geology.py",
     "**Input:** mapping points with lithology, strike and dip, or a geological map.\n\n"
     "**You get:** lithology map, favourable host rocks, strike rose and stereonet."),
    ("⚗️ 5 · Geochemistry", "views/geochemistry.py",
     "**Input:** soil / rock / stream samples with Au and pathfinder elements.\n\n"
     "**You get:** anomaly thresholds and classes, correlations, a gold index and anomalous samples."),
    ("⭐ 6 · Integrated gold prediction", "views/interpretation.py",
     "**Input:** whatever you submitted in sections 1–5.\n\n"
     "**You get:** combined gold prospectivity map, ranked targets with estimated depth and trend, and a report."),
    ("⛏️ Pit mapping", "views/pit_mapping.py",
     "**Input:** pit coordinates (Excel / CSV / TXT), plus an optional picture and field notes.\n\n"
     "**You get:** the mineralisation trend, what potential each pit still has, proposed neighbour pits with "
     "coordinates, and a pit register kept for future use."),
]

for row in (CARDS[:3], CARDS[3:6], CARDS[6:]):
    cols = st.columns(3)
    for col, (title, path, text) in zip(cols, row):
        with col, st.container(border=True):
            st.subheader(title)
            st.markdown(text)
            st.page_link(path, label="Open", icon="➡️")

st.info("Uploaded data stays in memory for your session only. The one exception is the pit register, which is "
        "saved in the GeoAI folder when you press Save on the Pit mapping page.")
