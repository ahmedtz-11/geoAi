"""The gate between the portal and the Streamlit app.

nginx asks this blueprint, on every request to /app, whether the browser's session may pass. The answer
carries a short-lived signed token naming the user and what their plan allows; nginx copies it into the
request it forwards to Streamlit, overwriting anything the browser tried to send under the same name.

Two independent things have to hold for this to be safe, and the deployment guide sets up both:
  1. Streamlit listens on 127.0.0.1 only, so the one way in is through nginx.
  2. The token is signed, so even a header that reached Streamlit another way would not be believed.
"""
from flask import Blueprint, current_app, redirect, request, url_for
from flask_login import current_user, login_required
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .models import record, db

bp = Blueprint("gate", __name__)
HEADER = "X-GeoAI-Token"


def serializer(app=None):
    app = app or current_app
    return URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="geoai-gate")


def issue(user, app=None):
    """A signed statement of who this is and what they may do, good for a minute."""
    return serializer(app).dumps({
        "email": user.email, "name": user.name or user.email.split("@")[0],
        "org": user.organisation, "admin": bool(user.is_admin),
        "plan": (user.plan.code if user.plan else "none"),
        "plan_name": user.plan_name,
        "sections": user.allowed_sections(),
        "download": user.may_download(),
        "days_left": user.days_left,
    })


def read(token, app=None, max_age=None):
    """Verify a token and return what it says, or None. Never raises."""
    if not token:
        return None
    app = app or current_app
    try:
        return serializer(app).loads(token, max_age=max_age or app.config["SESSION_HOURS"] * 3600)
    except (BadSignature, SignatureExpired, Exception):
        return None


@bp.route("/_gate/check")
def check():
    """nginx auth_request target. 200 lets the request through, 401 sends it to the sign-in page."""
    if not current_user.is_authenticated:
        return "", 401
    if not current_user.is_active_account:
        return "", 401
    resp = current_app.response_class("", status=200)
    resp.headers[HEADER] = issue(current_user)
    resp.headers["X-GeoAI-Email"] = current_user.email
    return resp


@bp.route("/app-signin")
@login_required
def app_signin():
    """The one link that opens the app, from wherever in the portal.

    Behind nginx it is just a redirect: the gate adds the token to the request itself. Running locally
    there is no nginx, so the token travels in the link and GEOAI_ALLOW_TOKEN_IN_URL lets Streamlit
    accept it.
    """
    record("open_app", current_user)
    db.session.commit()
    local = current_app.config.get("APP_URL")
    if local:
        return redirect(f"{local.rstrip('/')}/?t={issue(current_user)}")
    return redirect(current_app.config["APP_PATH"] + "/")


@bp.route("/_gate/denied")
def denied():
    return redirect(url_for("auth.login", next=request.args.get("from", current_app.config["APP_PATH"])))
