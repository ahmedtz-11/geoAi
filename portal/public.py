"""The pages anyone can see."""
from flask import Blueprint, current_app, redirect, render_template, url_for
from flask_login import current_user

from .models import SECTIONS, Plan

bp = Blueprint("public", __name__)


@bp.route("/")
def index():
    plans = Plan.query.filter_by(is_public=True).order_by(Plan.sort).all()
    return render_template("index.html", plans=plans, sections=SECTIONS)


@bp.route("/app")
@bp.route("/app/")
def app_redirect():
    """Reached only when nginx is not in front (local development). Send people the right way."""
    if current_user.is_authenticated:
        return redirect(url_for("gate.app_signin"))
    return redirect(url_for("auth.login", next=current_app.config["APP_PATH"]))
