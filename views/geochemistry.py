"""Section 4 - Geochemistry."""
import streamlit as st

from geoai import geochem
from ui.interp import page, tab_geochem


def settings():
    with st.expander("Advanced settings"):
        m = st.selectbox("Default anomaly thresholds", geochem.METHODS,
                         help="Can also be changed on the results below.")
    return {"geochem_method": m}


page.data_page(
    "geochem", "5 · Geochemistry",
    "Soil, rock-chip and stream samples: statistics with below-detection handling, anomaly thresholds and classes, "
    "element correlations, a multi-element gold index and the anomalous samples to follow up.",
    tab_geochem.render, settings)
