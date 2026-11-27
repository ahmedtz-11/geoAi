"""GeoAI: the public site, accounts and subscriptions, the admin dashboard, and the workbench itself.

One Flask application serves all of it, so a section of the workbench is an ordinary route behind
@login_required rather than a second server that has to be told who is signed in.
"""
import os
import pathlib
import traceback
from datetime import datetime, timezone

from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import LoginManager
from werkzeug.exceptions import HTTPException

from . import icons
from .app.store import ResultUnreadable
from .config import DATA, Config, check
from .models import Plan, User, db, seed_plans

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please sign in to continue."


def section_name(key):
    """The plain name of a section, without the step number the workbench numbers its pages with:
    the rail wants "2 · Magnetic", a plan's feature list just wants "Magnetic"."""
    from .models import SECTIONS
    return SECTIONS.get(key, str(key).replace("-", " ").capitalize()).split("\u00b7")[-1].strip()


def create_app(config=None):
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(Config)
    if config:
        app.config.update(config)
    DATA.mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    icons.init_app(app)

    from .admin import bp as admin_bp
    from .auth import bp as auth_bp
    from .billing import bp as billing_bp
    from .public import bp as public_bp
    for bp in (public_bp, auth_bp, billing_bp, admin_bp):
        app.register_blueprint(bp)

    from .app import register as register_workbench
    register_workbench(app)

    @login_manager.user_loader
    def load_user(uid):
        return db.session.get(User, int(uid))

    @app.context_processor
    def inject():
        from .app.uploads import ACCEPT_ATTR
        return {"BRAND": app.config["BRAND"], "SUPPORT_EMAIL": app.config["SUPPORT_EMAIL"],
                "section_name": section_name,
                "accept_attr": ACCEPT_ATTR,
                "accept_text": "CSV, text, Geosoft XYZ (.dat), Excel, GeoTIFF, or a picture of a map."}

    @app.after_request
    def do_not_cache_pages(response):
        """Signing out has to mean the pages are gone, and the browser's Back button serves whatever
        it has in its history cache unless it is told not to keep one. Pictures and the vendored
        Plotly set their own caching and are left alone; every page is no-store, which is also what
        keeps one person's results out of a shared browser after they sign out."""
        if response.mimetype == "text/html":
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"          # for HTTP/1.0 proxies
            response.headers["Expires"] = "0"
            response.headers["Vary"] = "Cookie"
        return response

    @app.errorhandler(403)
    def forbidden(_):
        return render_template("error.html", code=403, message="You do not have access to that page."), 403

    @app.errorhandler(404)
    def missing(_):
        return render_template("error.html", code=404, message="That page does not exist."), 404

    @app.errorhandler(413)
    def too_big(_):
        mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return render_template("error.html", code=413,
                               message=f"That file is larger than {mb} MB, which is the most this "
                                       f"server accepts. Split the survey, or ask the administrator "
                                       f"to raise GEOAI_MAX_UPLOAD_MB."), 413

    @app.errorhandler(ResultUnreadable)
    def result_gone(exc):
        """A finished run whose result cannot be read back is not an error the person can act on,
        so clear it and send them to the section's upload step with an explanation."""
        from .app import BY_KEY
        try:
            exc.run.delete()
        except OSError:
            pass
        flash("Those results could not be read back, so they have been cleared. "
              "Upload the file again to re-run the interpretation.", "error")
        section = BY_KEY.get(exc.run.section)
        return redirect(url_for(section["endpoint"]) if section else url_for("app.home"))

    @app.errorhandler(Exception)
    def crashed(exc):
        """Anything unhandled. The traceback goes to the log and to data/last-error.log, because a
        browser showing "Internal Server Error" tells whoever is running this nothing at all."""
        if isinstance(exc, HTTPException):
            return exc
        app.logger.exception("unhandled error at %s", request.path)
        try:
            log = pathlib.Path(app.config["DATA_DIR"]) / "last-error.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(
                f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  "
                f"{request.method} {request.full_path}\n\n{traceback.format_exc()}",
                encoding="utf-8")
        except OSError:
            pass
        return render_template("error.html", code=500,
                               message="Something went wrong on the server. The details were "
                                       "written to data/last-error.log."), 500

    with app.app_context():
        db.create_all()
        seed_plans()
        for problem in check(app):
            app.logger.warning(problem)
    return app


def create_admin(app, email, password, name="Administrator"):
    """Make or promote the first administrator. Used by `flask --app wsgi admin-user`."""
    with app.app_context():
        u = User.query.filter_by(email=email.lower().strip()).first()
        if u is None:
            u = User(email=email.lower().strip(), name=name)
            db.session.add(u)
        u.set_password(password)
        u.is_admin = True
        u.is_active_account = True
        pro = Plan.query.filter_by(code="pro").first()
        db.session.commit()
        return u
