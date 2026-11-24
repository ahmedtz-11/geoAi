"""The GeoAI portal: accounts, subscriptions, an admin dashboard, and the gate to the Streamlit app."""
import os

from flask import Flask, render_template
from flask_login import LoginManager

from .config import DATA, Config, check
from .models import Plan, User, db, seed_plans

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please sign in to continue."


def create_app(config=None):
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(Config)
    if config:
        app.config.update(config)
    DATA.mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    from .admin import bp as admin_bp
    from .auth import bp as auth_bp
    from .billing import bp as billing_bp
    from .gate import bp as gate_bp
    from .public import bp as public_bp
    for bp in (public_bp, auth_bp, billing_bp, admin_bp, gate_bp):
        app.register_blueprint(bp)

    @login_manager.user_loader
    def load_user(uid):
        return db.session.get(User, int(uid))

    @app.context_processor
    def inject():
        return {"BRAND": app.config["BRAND"], "APP_PATH": app.config["APP_PATH"],
                "SUPPORT_EMAIL": app.config["SUPPORT_EMAIL"]}

    @app.errorhandler(403)
    def forbidden(_):
        return render_template("error.html", code=403, message="You do not have access to that page."), 403

    @app.errorhandler(404)
    def missing(_):
        return render_template("error.html", code=404, message="That page does not exist."), 404

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
