"""Run the whole thing on your own machine: the portal, the app, and the gate between them.

    python run_local.py

Starts the portal on http://localhost:8000 and the Streamlit app on http://localhost:8501, with the gate
switched on so you see exactly what a subscriber sees. Makes an administrator the first time it runs and
prints the sign-in details. Ctrl+C stops both.

This is for your laptop, not for the VPS. It uses a fixed development secret and lets the gate token
travel in a link, because there is no nginx here to put it in a header. deploy/README.md is the real
thing; nothing here is used there.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ADMIN_EMAIL = os.environ.get("GEOAI_DEV_ADMIN", "admin@local")
ADMIN_PASSWORD = os.environ.get("GEOAI_DEV_PASSWORD", "local-admin-password")

ENV = {
    **os.environ,
    "GEOAI_SECRET_KEY": "local-development-only-not-for-a-server",
    "GEOAI_DATA": str(DATA),
    "GEOAI_DATABASE_URL": f"sqlite:///{DATA / 'portal.db'}",
    "GEOAI_USAGE_DB": str(DATA / "portal.db"),
    "GEOAI_PORTAL_URL": "http://localhost:8000",
    "GEOAI_APP_URL": "http://localhost:8501",      # no nginx here, so the app has its own address
    "GEOAI_ALLOW_TOKEN_IN_URL": "1",               # …and the token has to travel in the link
    "GEOAI_HTTPS": "0",
    "PYTHONUNBUFFERED": "1",
}


def main():
    DATA.mkdir(exist_ok=True)
    sys.path.insert(0, str(ROOT))
    os.environ.update({k: v for k, v in ENV.items() if k.startswith("GEOAI_")})

    from portal import create_admin, create_app
    app = create_app()
    create_admin(app, ADMIN_EMAIL, ADMIN_PASSWORD, "Local Admin")

    print("\n" + "=" * 68)
    print("  Portal      http://localhost:8000")
    print("  Admin       http://localhost:8000/admin")
    print(f"  Sign in as  {ADMIN_EMAIL}  /  {ADMIN_PASSWORD}")
    print("=" * 68)
    print("""
  Try the whole flow:
    1. Open http://localhost:8000 and press "Create account" — that is a new customer.
       They land on the Free plan and the app shows only Home and Remote sensing.
    2. Sign in as the administrator above in a private window, open Admin → Users,
       press Manage on the new account, and record a payment against a plan.
    3. Go back to the customer's window, reload, press "Open the app": the sections
       their plan pays for are now there, and the others are gone.

  Ctrl+C stops both.
""")
    procs = [
        subprocess.Popen([sys.executable, "-m", "flask", "--app", "wsgi", "run", "--port", "8000"],
                         cwd=ROOT, env=ENV),
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py",
                          "--server.port", "8501", "--server.headless", "true",
                          "--browser.gatherUsageStats", "false"], cwd=ROOT, env=ENV),
    ]
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
        for p in procs:
            if p.poll() not in (None, 0):
                print(f"\nOne of the two stopped with code {p.returncode}; shutting the other down.")
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        print("Stopped.")


if __name__ == "__main__":
    main()
