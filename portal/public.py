"""The pages anyone can see."""
from flask import Blueprint, render_template

from .models import SECTIONS, Plan

bp = Blueprint("public", __name__)


@bp.route("/")
def index():
    plans = Plan.query.filter_by(is_public=True).order_by(Plan.sort).all()
    return render_template("index.html", plans=plans, sections=SECTIONS)
