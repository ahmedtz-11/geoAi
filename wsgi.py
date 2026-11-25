"""What gunicorn runs, and the few commands you need from the command line.

  flask --app wsgi admin-user  you@example.com 'a long password'   make the first administrator
  flask --app wsgi show-config                                     check the settings before going live
"""
import click

from portal import create_admin, create_app
from portal.config import check

app = create_app()


@app.cli.command("admin-user")
@click.argument("email")
@click.argument("password")
@click.option("--name", default="Administrator")
def admin_user(email, password, name):
    """Create the first administrator, or reset an existing one's password."""
    u = create_admin(app, email, password, name)
    click.echo(f"{u.email} is an administrator.")


@app.cli.command("show-config")
def show_config():
    """Print the settings that matter and anything unsafe about them."""
    for key in ("SECRET_KEY", "SQLALCHEMY_DATABASE_URI", "DATA_DIR", "MAX_CONTENT_LENGTH",
                "KEEP_RUNS", "PAYMENT_PROVIDER", "SESSION_COOKIE_SECURE", "BRAND",
                "SUPPORT_EMAIL"):
        value = app.config.get(key)
        if key == "SECRET_KEY":
            value = "(set)" if value != "dev-only-change-me" else "DEVELOPMENT VALUE"
        click.echo(f"{key:28s} {value}")
    problems = check(app)
    for p in problems:
        click.echo(f"  warning: {p}")
    if not problems:
        click.echo("  no problems found.")


if __name__ == "__main__":
    app.run(debug=True, port=8000)
