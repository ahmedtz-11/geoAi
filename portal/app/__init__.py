"""The workbench: the seven GeoAI sections, served by the same Flask application as the portal.

There is no gate any more. The sections are routes behind @login_required in the application that owns
the session, so a person who may not open a section simply has no route to it — nothing is signed,
forwarded or trusted between two servers.
"""
import functools

from flask import (Blueprint, abort, current_app, flash, redirect, render_template, request,
                   send_file, url_for)
from flask_login import current_user, login_required

from . import jobs, store

bp = Blueprint("app", __name__, url_prefix="/app")

# key must match portal.models.SECTIONS, which is what the plans are written in terms of
SECTIONS = [
    {"key": "remote-sensing", "title": "1 · Remote sensing", "icon": "🛰️", "endpoint": "app.remote_sensing",
     "blurb": "The four corner coordinates of your area give you satellite imagery, alteration indices, "
              "band ratios, PCA, lithological classification, lineaments, DEM and 3D terrain, and a "
              "prospectivity map."},
    {"key": "magnetic", "title": "2 · Magnetic", "icon": "🧲", "endpoint": "app.magnetic",
     "blurb": "One magnetic survey file becomes TMI, RTP/RTE, vertical derivatives, analytic signal, "
              "tilt, THDR, lineaments, upward continuation, SPI and Euler depths, and a target map."},
    {"key": "ip", "title": "3 · IP", "icon": "⚡", "endpoint": "app.ip",
     "blurb": "One file per survey line gives inverse model chargeability and resistivity sections in "
              "Res2DInv style, with anomaly positions and depths."},
    {"key": "geology", "title": "4 · Geology", "icon": "🪨", "endpoint": "app.geology",
     "blurb": "Mapping points with lithology, strike and dip, and/or a QDS sheet: lithology map, "
              "favourable host rocks, strike rose, stereonet and structural sets."},
    {"key": "geochemistry", "title": "5 · Geochemistry", "icon": "⚗️", "endpoint": "app.geochemistry",
     "blurb": "Soil samples with gold assays: anomaly thresholds, populations, interpolated surfaces "
              "and the zones worth following up."},
    {"key": "prediction", "title": "6 · Integrated gold prediction", "icon": "⭐", "endpoint": "app.prediction",
     "blurb": "Everything you have submitted in the other sections, combined into one prospectivity "
              "map with ranked targets, estimated depth and trend, and a report."},
    {"key": "pit-mapping", "title": "Pit mapping", "icon": "⛏️", "endpoint": "app.pit_mapping",
     "blurb": "Pit coordinates, a picture and field notes: the mineralisation trend, what each pit "
              "still has, proposed neighbour pits, and a register kept for next time."},
]
BY_KEY = {s["key"]: s for s in SECTIONS}


# ------------------------------------------------------------------ access
def section(key):
    """Mark a view as belonging to a section, and refuse it to a plan that does not include it."""
    def wrap(view):
        @functools.wraps(view)
        @login_required
        def guarded(*a, **kw):
            if key not in current_user.allowed_sections():
                flash(f"{BY_KEY[key]['title']} is not part of your plan.", "error")
                return redirect(url_for("billing.pricing"))
            return view(*a, **kw)
        guarded._section = key
        return guarded
    return wrap


def may_download():
    if not current_user.may_download():
        abort(403, "Downloads are not part of your plan.")


def note_usage(kind, section="", detail=""):
    """Record what someone did. This is what the monthly upload cap counts, and what the admin
    dashboard shows as activity."""
    from ..models import UsageEvent, db
    try:
        db.session.add(UsageEvent(user_id=current_user.id, kind=kind, section=section,
                                  detail=str(detail)[:255]))
        db.session.commit()
    except Exception:                      # usage accounting must never lose someone's work
        db.session.rollback()


def upload_allowance():
    """(used, limit, left) for this month. An administrator is never capped."""
    if current_user.is_admin:
        return 0, None, None
    used, limit = current_user.upload_allowance()
    return used, limit, None if limit is None else max(limit - used, 0)


@bp.context_processor
def _nav():
    """What the sidebar needs, on every page of the workbench."""
    if not current_user.is_authenticated:
        return {"app_sections": [], "locked_count": 0, "allowance": (0, None, None)}
    allowed = set(current_user.allowed_sections())
    have = set(store.sections_with_results(current_user.id))
    items = [{**s, "open": s["key"] in allowed, "has_result": s["key"] in have} for s in SECTIONS]
    return {"app_sections": items, "locked_count": sum(0 if i["open"] else 1 for i in items),
            "allowance": upload_allowance()}


# ------------------------------------------------------------------ the front of the workbench
@bp.route("/")
@login_required
def home():
    return render_template("app/home.html", section_key="home", section_title="Workbench")


@bp.route("/vendor/plotly.min.js")
def plotly_js():
    """Served from the installed plotly package rather than a CDN: the VPS may have no outbound
    internet, and this way the library can never drift from the version that drew the figure."""
    import os

    import plotly
    path = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="application/javascript", max_age=60 * 60 * 24 * 30,
                     conditional=True)


# ------------------------------------------------------------------ jobs
@bp.route("/job/<job_id>")
@login_required
def job_status(job_id):
    job = jobs.get(job_id, current_user.id)
    if job is None:
        return {"state": "missing", "progress": 0, "message": "That job is no longer running.",
                "error": "The server restarted, or it finished more than half an hour ago."}, 404
    return job.as_dict()


@bp.route("/job/<job_id>/cancel", methods=["POST"])
@login_required
def job_cancel(job_id):
    job = jobs.get(job_id, current_user.id)
    if job is None:
        abort(404)
    job.cancel.set()
    return {"ok": True}


@bp.route("/wait/<section_key>/<job_id>")
@login_required
def wait(section_key, job_id):
    """The page that watches a job and sends the browser on when it finishes."""
    if section_key not in BY_KEY:
        abort(404)
    job = jobs.get(job_id, current_user.id)
    if job is None:
        flash("That run is no longer in progress.", "error")
        return redirect(url_for(BY_KEY[section_key]["endpoint"]))
    if job.state == "done":
        return redirect(url_for(BY_KEY[section_key]["endpoint"]))
    return render_template("app/wait.html", job=job, section_key=section_key,
                           section=BY_KEY[section_key], section_title=BY_KEY[section_key]["title"])


# ------------------------------------------------------------------ serving what a run produced
@bp.route("/fig/<section_key>/<run_id>/<name>.png")
@login_required
def figure(section_key, run_id, name):
    """A picture from one of this person's own runs. Never anybody else's: the path is built from
    current_user.id, so another person's run id simply does not resolve."""
    run = store.Run.load(current_user.id, section_key, run_id)
    if run is None:
        abort(404)
    path = run.fig_path(name)
    if not path.is_file():
        drawn = _draw_on_demand(run, name)
        if not drawn:
            abort(404)
        path = drawn
    return send_file(path, mimetype="image/png", max_age=0, conditional=True)


def _draw_on_demand(run, name):
    """A picture that was not pre-rendered — a Normal/Filled variant, say — drawn when first asked for."""
    from .sections import redraw
    return redraw(run, name, request.args)


@bp.route("/download/<section_key>/<run_id>/<kind>/<name>")
@login_required
def download(section_key, run_id, kind, name):
    may_download()
    run = store.Run.load(current_user.id, section_key, run_id)
    if run is None:
        abort(404)
    from .sections import deliver
    note_usage("download", section_key, f"{kind}:{name}")
    return deliver(run, kind, name, request.args)


@bp.route("/run/<section_key>/<run_id>/delete", methods=["POST"])
@login_required
def delete_run(section_key, run_id):
    run = store.Run.load(current_user.id, section_key, run_id)
    if run is None:
        abort(404)
    run.delete()
    flash("Results cleared.", "ok")
    return redirect(url_for(BY_KEY[section_key]["endpoint"]))


_attached = False


def register(app):
    """Attach the section views, then the blueprint. Called by the application factory.

    The blueprint is a module-level object, so its routes are added once however many applications
    are built — a test suite makes a fresh one per test, and Flask refuses new rules on a blueprint
    that is already registered somewhere.
    """
    global _attached
    from . import sections
    from .render import fmt, stats_row, table_html
    if not _attached:
        sections.attach(bp)
        _attached = True
    app.register_blueprint(bp)
    app.config.setdefault("KEEP_RUNS", store.KEEP_RUNS)
    app.jinja_env.filters["grid_stats"] = stats_row
    app.jinja_env.filters["num"] = fmt
    app.jinja_env.filters["table"] = table_html
    return bp
