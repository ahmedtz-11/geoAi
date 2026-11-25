"""Portal settings. Everything that differs between your laptop and the VPS comes from the environment."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("GEOAI_DATA", ROOT / "data"))


class Config:
    SECRET_KEY = os.environ.get("GEOAI_SECRET_KEY", "dev-only-change-me")
    SQLALCHEMY_DATABASE_URI = os.environ.get("GEOAI_DATABASE_URL", f"sqlite:///{DATA / 'portal.db'}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    # where a person's uploads and results are kept (portal/app/store.py)
    DATA_DIR = str(DATA)
    # the biggest upload the app accepts; a GeoTIFF of a whole licence block is the usual worst case
    MAX_CONTENT_LENGTH = int(os.environ.get("GEOAI_MAX_UPLOAD_MB", "200")) * 1024 * 1024
    # how many finished runs of each section to keep per person before the oldest is dropped
    KEEP_RUNS = int(os.environ.get("GEOAI_KEEP_RUNS", "5"))

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.environ.get("GEOAI_HTTPS", "0") == "1"
    REMEMBER_COOKIE_SECURE = SESSION_COOKIE_SECURE
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 24 * 14

    BRAND = os.environ.get("GEOAI_BRAND", "GeoAI")
    SUPPORT_EMAIL = os.environ.get("GEOAI_SUPPORT_EMAIL", "support@example.com")
    CURRENCY = os.environ.get("GEOAI_CURRENCY", "TZS")

    # payment gateway: left off until the client's merchant account exists (see portal/billing.py)
    PAYMENT_PROVIDER = os.environ.get("GEOAI_PAYMENT_PROVIDER", "manual")
    PAYMENT_WEBHOOK_SECRET = os.environ.get("GEOAI_PAYMENT_WEBHOOK_SECRET", "")


def check(app):
    """Refuse to run in the open with the development secret."""
    problems = []
    if not app.debug and app.config["SECRET_KEY"] == "dev-only-change-me":
        problems.append("GEOAI_SECRET_KEY is still the development value: set it before going live.")
    if not app.debug and not app.config["SESSION_COOKIE_SECURE"]:
        problems.append("GEOAI_HTTPS is not 1, so session cookies may travel unencrypted.")
    return problems
