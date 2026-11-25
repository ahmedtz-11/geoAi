"""Run GeoAI on your own machine.

    python run_local.py

One server, on http://localhost:8000: the public site, the sign-in, the admin dashboard and the
workbench. Makes an administrator the first time it runs and prints the sign-in details. Ctrl+C stops
it.

This is for your laptop. It uses a fixed development secret, so it is not for a server —
deploy/README.md covers that.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ADMIN_EMAIL = os.environ.get("GEOAI_DEV_ADMIN", "admin@local")
ADMIN_PASSWORD = os.environ.get("GEOAI_DEV_PASSWORD", "local-admin-password")
PORT = int(os.environ.get("GEOAI_PORT", "8000"))


def main():
    DATA.mkdir(exist_ok=True)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("GEOAI_SECRET_KEY", "local-development-only-not-for-a-server")
    os.environ.setdefault("GEOAI_DATA", str(DATA))
    os.environ.setdefault("GEOAI_DATABASE_URL", f"sqlite:///{DATA / 'portal.db'}")

    from portal import create_admin, create_app
    app = create_app()
    create_admin(app, ADMIN_EMAIL, ADMIN_PASSWORD, "Local Admin")

    print("\n" + "=" * 68)
    print(f"  GeoAI        http://localhost:{PORT}")
    print(f"  Workbench    http://localhost:{PORT}/app/")
    print(f"  Admin        http://localhost:{PORT}/admin")
    print(f"  Sign in as   {ADMIN_EMAIL}  /  {ADMIN_PASSWORD}")
    print("=" * 68)
    print("""
  Try the whole flow:
    1. Open the address above and press "Create account" — that is a new customer.
       They land on the Free plan, so the workbench shows Remote sensing and nothing else.
    2. Sign in as the administrator above in a private window, open Admin → Users,
       press Manage on the new account and record a payment against a plan.
    3. Back in the customer's window, open the Workbench: the sections their plan pays
       for are now there, and the others are locked.

  Ctrl+C stops it.
""")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
