"""The Streamlit side of the gate: who is using the app, and what their plan lets them do.

nginx puts a signed token on every request it forwards here. This module reads it, checks the signature
against the shared secret, and hands the rest of the app a small record of the signed-in person.

Running `streamlit run app.py` on your own machine sets no secret, so the gate stays open and you work as
before. Deployment sets GEOAI_SECRET_KEY, and from that moment a request without a valid token is turned
away. The lock arrives with the deployment, so there is no way to put the app on the internet with the
gate accidentally off.
"""
import os
from dataclasses import dataclass, field

import streamlit as st

HEADER = "x-geoai-token"
SECRET = os.environ.get("GEOAI_SECRET_KEY", "")
# Local running only. nginx puts the token in a header; on a laptop there is no nginx, so the portal
# passes it in the link instead. A URL is written to logs and history, so this stays off unless asked
# for: deploy/env.example never sets it, and the VPS therefore never accepts a token from a URL.
ALLOW_URL_TOKEN = os.environ.get("GEOAI_ALLOW_TOKEN_IN_URL", "0") == "1"
SALT = "geoai-gate"
MAX_AGE = int(os.environ.get("GEOAI_SESSION_HOURS", "12")) * 3600
PORTAL = os.environ.get("GEOAI_PORTAL_URL", "/")

ALL_SECTIONS = ["home", "remote-sensing", "magnetic", "ip", "geology", "geochemistry", "prediction",
                "pit-mapping"]

# which section each page file belongs to, so a link can be checked before it is drawn
SECTION_OF = {"views/home.py": "home", "views/remote_sensing.py": "remote-sensing",
              "views/magnetic.py": "magnetic", "views/ip.py": "ip", "views/geology.py": "geology",
              "views/geochemistry.py": "geochemistry", "views/interpretation.py": "prediction",
              "views/pit_mapping.py": "pit-mapping"}


@dataclass
class Visitor:
    email: str = "you@localhost"
    name: str = "Local user"
    org: str = ""
    admin: bool = True
    plan: str = "dev"
    plan_name: str = "Developer (no portal)"
    sections: list = field(default_factory=lambda: list(ALL_SECTIONS))
    download: bool = True
    days_left: object = None
    gated: bool = False          # True when a real portal token was read

    def may_open(self, section):
        return self.admin or section in self.sections

    def may_download(self):
        return bool(self.download)


def _token():
    try:
        header = st.context.headers.get(HEADER)
    except Exception:
        header = None
    if header:
        return header
    if ALLOW_URL_TOKEN:
        # The token stays in the address bar on purpose: Streamlit clears the session on a page refresh,
        # and without it in the URL every refresh would lock you out mid-test.
        token = st.query_params.get("t")
        if token:
            st.session_state["_url_token"] = token
        return token or st.session_state.get("_url_token")
    return None


@st.cache_resource(show_spinner=False)
def _serializer(secret):
    from itsdangerous import URLSafeTimedSerializer
    return URLSafeTimedSerializer(secret, salt=SALT)


def visitor():
    """Who is using the app right now. Cached per session so the token is only verified once."""
    if "_visitor" in st.session_state:
        return st.session_state["_visitor"]
    v = Visitor()
    if SECRET:
        data = None
        raw = _token()
        if raw:
            try:
                data = _serializer(SECRET).loads(raw, max_age=MAX_AGE)
            except Exception:
                data = None
        if data is None:
            v = None                                    # locked: deployment is on and the token is no good
        else:
            v = Visitor(email=data.get("email", ""), name=data.get("name", ""), org=data.get("org", ""),
                        admin=bool(data.get("admin")), plan=data.get("plan", "none"),
                        plan_name=data.get("plan_name", ""), sections=list(data.get("sections", [])),
                        download=bool(data.get("download")), days_left=data.get("days_left"), gated=True)
    st.session_state["_visitor"] = v
    return v


def require():
    """Call once at the top of the app. Stops the run for anyone who should not be here."""
    v = visitor()
    if v is None:
        st.title("Please sign in")
        st.error("Your session has expired, or this page was opened without signing in.")
        st.link_button("Go to the sign-in page", PORTAL, type="primary")
        st.stop()
    return v


def allowed_pages(pages):
    """Keep only the pages this visitor's plan opens. `pages` maps a section key to an st.Page."""
    v = visitor()
    if v is None:
        return {}
    return {k: p for k, p in pages.items() if v.may_open(k)}


def page_link(path, **kwargs):
    """st.page_link, but silent when the plan does not open that page.

    st.page_link raises StreamlitPageNotFoundError for a page that never reached st.navigation, which on
    a limited plan would stop the page it is drawn on. Use this everywhere instead.
    """
    v = visitor()
    section = SECTION_OF.get(str(path).replace("\\", "/"))
    if v is None or (section and not v.may_open(section)):
        return
    st.page_link(path, **kwargs)


def sidebar_badge():
    """Who is signed in, what they are on, and the way out. Nothing when there is no portal."""
    v = visitor()
    if v is None or not v.gated:
        return
    with st.sidebar:
        st.divider()
        st.caption(f"**{v.name}**  \n{v.email}")
        left = f" · {v.days_left} days left" if v.days_left is not None else ""
        st.caption(f"{v.plan_name}{left}")
        if v.days_left is not None and v.days_left <= 7:
            st.warning("Your subscription ends soon. Renew it on your account page to keep access.")
        st.link_button("My account", PORTAL.rstrip("/") + "/account", width="stretch")


def download_guard(section=""):
    """True when this visitor may download. Shows the upsell once per page when they may not."""
    v = visitor()
    if v is None:
        return False
    if v.may_download():
        return True
    st.info("Downloads are part of the paid plans. You can read every result on screen; upgrade on your "
            "account page to save the maps, grids and tables.")
    return False


def note_usage(kind, section="", detail=""):
    """Tell the portal what happened, for the monthly cap and the admin's activity list. Never fails."""
    v = visitor()
    if v is None or not v.gated:
        return
    try:
        import sqlite3
        from datetime import datetime, timezone
        path = os.environ.get("GEOAI_USAGE_DB", "")
        if not path:
            return
        with sqlite3.connect(path, timeout=2) as con:
            uid = con.execute("SELECT id FROM users WHERE email = ?", (v.email,)).fetchone()
            if uid:
                con.execute("INSERT INTO usage (user_id, kind, section, detail, at) VALUES (?,?,?,?,?)",
                            (uid[0], kind, section, detail[:255],
                             datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")))
    except Exception:
        pass
