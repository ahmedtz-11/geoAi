"""GeoAI Platform - entry point.

Run on your own machine:   streamlit run app.py
Behind the portal:         nginx forwards signed-in people here (see deploy/README.md)
"""
import streamlit as st

from ui import gate, state
from ui.rs.common import keep_widget_state as rs_defaults

st.set_page_config(page_title="GeoAI Platform", page_icon="⛏️", layout="wide")

# Who is using the app. With no portal this is a local developer with everything open; behind the portal
# it is the signed-in subscriber, and anyone else is stopped right here.
me = gate.require()

# keep choices made on one page / tab when moving to another (see ui/state.py)
state.restore()
rs_defaults()

PAGES = {
    "home": st.Page("views/home.py", title="Home", icon="🏠", default=True),
    "remote-sensing": st.Page("views/remote_sensing.py", title="1 · Remote sensing", icon="🛰️",
                              url_path="remote-sensing"),
    "magnetic": st.Page("views/magnetic.py", title="2 · Magnetic", icon="🧲", url_path="magnetic"),
    "ip": st.Page("views/ip.py", title="3 · IP", icon="⚡", url_path="ip"),
    "geology": st.Page("views/geology.py", title="4 · Geology", icon="🪨", url_path="geology"),
    "geochemistry": st.Page("views/geochemistry.py", title="5 · Geochemistry", icon="⚗️",
                            url_path="geochemistry"),
    "prediction": st.Page("views/interpretation.py", title="6 · Integrated gold prediction", icon="⭐",
                          url_path="prediction"),
    "pit-mapping": st.Page("views/pit_mapping.py", title="Pit mapping", icon="⛏️", url_path="pit-mapping"),
}
open_to_me = gate.allowed_pages(PAGES)
GROUPS = {"GeoAI": ["home"],
          "Sections": ["remote-sensing", "magnetic", "ip", "geology", "geochemistry"],
          "Integration": ["prediction"],
          "Field tools": ["pit-mapping"]}

nav_spec = {}
for title, keys in GROUPS.items():
    pages = [open_to_me[k] for k in keys if k in open_to_me]
    if pages:
        nav_spec[title] = pages
if not nav_spec:                                 # a plan that opens nothing should still land somewhere
    nav_spec = {"GeoAI": [PAGES["home"]]}

nav = st.navigation(nav_spec)
try:
    nav.run()
finally:
    gate.sidebar_badge()
    locked = [k for k in PAGES if k not in open_to_me]
    if locked and me.gated:
        st.sidebar.caption(f"{len(locked)} more section(s) come with the higher plans.")
    state.remember()
