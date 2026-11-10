"""GeoAI Platform - entry point.

Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="GeoAI Platform", page_icon="⛏️", layout="wide")

home = st.Page("views/home.py", title="Home", icon="🏠", default=True)
remote = st.Page("views/remote_sensing.py", title="1 · Remote sensing", icon="🛰️", url_path="remote-sensing")
interp = st.Page("views/interpretation.py", title="2 · Interpretation & prediction", icon="🧭",
                 url_path="interpretation")

st.navigation({"GeoAI": [home], "Sections": [remote, interp]}).run()
