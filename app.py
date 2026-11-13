"""GeoAI Platform - entry point.

Run:  streamlit run app.py
"""
import streamlit as st

from ui import state
from ui.rs.common import keep_widget_state as rs_defaults

st.set_page_config(page_title="GeoAI Platform", page_icon="⛏️", layout="wide")

# keep choices made on one page / tab when moving to another (see ui/state.py)
state.restore()
rs_defaults()

home = st.Page("views/home.py", title="Home", icon="🏠", default=True)
remote = st.Page("views/remote_sensing.py", title="1 · Remote sensing", icon="🛰️", url_path="remote-sensing")
magnetic = st.Page("views/magnetic.py", title="2 · Magnetic", icon="🧲", url_path="magnetic")
ip = st.Page("views/ip.py", title="3 · IP", icon="⚡", url_path="ip")
geology = st.Page("views/geology.py", title="4 · Geology", icon="🪨", url_path="geology")
geochem = st.Page("views/geochemistry.py", title="5 · Geochemistry", icon="⚗️", url_path="geochemistry")
predict = st.Page("views/interpretation.py", title="6 · Integrated gold prediction", icon="⭐",
                  url_path="prediction")

pit = st.Page("views/pit_mapping.py", title="Pit mapping", icon="⛏️", url_path="pit-mapping")

nav = st.navigation({"GeoAI": [home],
                     "Sections": [remote, magnetic, ip, geology, geochem],
                     "Integration": [predict],
                     "Field tools": [pit]})
try:
    nav.run()
finally:
    state.remember()
