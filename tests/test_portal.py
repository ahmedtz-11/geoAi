"""The portal: accounts, plans, the admin dashboard, and the gate to the Streamlit app.

The gate tests matter most. Everything else here is a convenience that can be fixed later; a hole in the
gate means anyone on the internet uses the app for nothing.
"""
import os
import pathlib
import sys
from datetime import timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portal import create_app                                        # noqa: E402
from portal.models import (Payment, Plan, Subscription, User, db, now, seed_plans)  # noqa: E402

SECRET = "test-secret-not-the-real-one"


@pytest.fixture()
def app(tmp_path):
    app = create_app({"TESTING": True, "SECRET_KEY": SECRET, "WTF_CSRF_ENABLED": False,
                      "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
                      # a test must never write into the real data folder
                      "DATA_DIR": str(tmp_path / "data")})
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


def signup(client, email="geo@example.com", pw="a-long-password-9"):
    return client.post("/register", data={"name": "Geo", "email": email, "password": pw,
                                          "password2": pw, "organisation": "Acme", "phone": "+255"},
                       follow_redirects=True)


def make_admin(app, email="boss@example.com", pw="another-long-pass-7"):
    with app.app_context():
        u = User(email=email, name="Boss", is_admin=True)
        u.set_password(pw)
        db.session.add(u)
        db.session.commit()
    return email, pw


# ---------------------------------------------------------------- accounts
def test_a_new_account_is_free_and_sees_only_the_free_sections(app, client):
    assert signup(client).status_code == 200
    with app.app_context():
        u = User.query.filter_by(email="geo@example.com").first()
        assert u.plan.code == "free"
        assert u.subscription is None
        assert u.allowed_sections() == ["home", "remote-sensing"]
        assert u.may_download() is False
        assert u.password_hash != "a-long-password-9" and u.check_password("a-long-password-9")


def test_passwords_are_checked_and_duplicates_refused(client):
    short = client.post("/register", data={"email": "a@b.com", "password": "short", "password2": "short"},
                        follow_redirects=True)
    assert b"at least 10 characters" in short.data.lower()
    signup(client)
    client.get("/logout")
    again = signup(client)
    assert b"already exists" in again.data
    bad = client.post("/login", data={"email": "geo@example.com", "password": "wrong-password-x"},
                      follow_redirects=True)
    assert b"do not match an account" in bad.data


def test_a_suspended_account_cannot_sign_in(app, client):
    signup(client)
    client.get("/logout")
    with app.app_context():
        u = User.query.filter_by(email="geo@example.com").first()
        u.is_active_account = False
        db.session.commit()
    r = client.post("/login", data={"email": "geo@example.com", "password": "a-long-password-9"},
                    follow_redirects=True)
    assert b"suspended" in r.data


def test_the_open_redirect_is_closed(client):
    signup(client)
    client.get("/logout")
    r = client.post("/login?next=https://evil.example.com",
                    data={"email": "geo@example.com", "password": "a-long-password-9",
                          "next": "https://evil.example.com"})
    assert "evil.example.com" not in r.headers.get("Location", "")


# ---------------------------------------------------------------- plans and subscriptions
def test_activating_a_plan_opens_its_sections_and_expires_on_time(app, client):
    signup(client)
    with app.app_context():
        from portal.billing import activate
        u = User.query.filter_by(email="geo@example.com").first()
        pro = Plan.query.filter_by(code="pro").first()
        activate(u, pro, method="manual", reference="RCPT-1", by="boss@example.com")
        u = User.query.filter_by(email="geo@example.com").first()
        assert u.plan.code == "pro" and u.may_download()
        assert "geochemistry" in u.allowed_sections() and u.days_left == 29
        assert Payment.query.count() == 1 and Payment.query.first().reference == "RCPT-1"

        s = u.subscription                                    # wind the clock past the end
        s.ends_at = now() - timedelta(days=1)
        db.session.commit()
        u = User.query.filter_by(email="geo@example.com").first()
        assert u.subscription is None, "an expired subscription must stop counting"
        assert u.plan.code == "free" and not u.may_download()
        assert u.allowed_sections() == ["home", "remote-sensing"]


def test_renewing_the_same_plan_adds_days_rather_than_stacking(app):
    with app.app_context():
        from portal.billing import activate
        u = User(email="r@example.com")
        u.set_password("a-long-password-9")
        db.session.add(u)
        basic = Plan.query.filter_by(code="basic").first()
        activate(u, basic, by="admin")
        first_end = u.subscription.ends_at
        activate(u, basic, by="admin")
        assert len([s for s in u.subscriptions if s.status == "active"]) == 1
        assert (u.subscription.ends_at - first_end).days == 30

        pro = Plan.query.filter_by(code="pro").first()        # a different plan replaces, not stacks
        activate(u, pro, by="admin")
        live = [s for s in u.subscriptions if s.is_live]
        assert len(live) == 1 and live[0].plan.code == "pro"


def test_a_plan_is_data_so_changing_it_changes_access(app):
    with app.app_context():
        from portal.billing import activate
        u = User(email="d@example.com")
        u.set_password("a-long-password-9")
        db.session.add(u)
        basic = Plan.query.filter_by(code="basic").first()
        activate(u, basic, by="admin")
        assert "geochemistry" not in u.allowed_sections()
        basic.sections = basic.sections + ",geochemistry"     # the admin edits the plan
        db.session.commit()
        assert "geochemistry" in u.allowed_sections()


# ---------------------------------------------------------------- the gate
def test_a_stranger_is_sent_to_sign_in_not_into_a_section(client):
    for url in ("/app/", "/app/magnetic", "/app/geochemistry"):
        r = client.get(url)
        assert r.status_code == 302 and "/login" in r.headers["Location"], url


def test_a_plan_decides_which_sections_open(app, client):
    """The whole subscription model rests on this: a section the plan does not include has no route
    the person can reach, in the same application that holds their session."""
    signup(client)                                        # lands on Free
    assert client.get("/app/remote-sensing").status_code == 200
    for closed in ("/app/magnetic", "/app/ip", "/app/geochemistry", "/app/prediction"):
        r = client.get(closed)
        assert r.status_code == 302 and "/pricing" in r.headers["Location"], closed

    with app.app_context():
        from portal.billing import activate
        u = User.query.filter_by(email="geo@example.com").first()
        activate(u, Plan.query.filter_by(code="basic").first(), by="admin")

    assert client.get("/app/magnetic").status_code == 200          # basic includes it
    assert client.get("/app/ip").status_code == 200
    r = client.get("/app/geochemistry")                            # pro only
    assert r.status_code == 302 and "/pricing" in r.headers["Location"]


def test_one_person_cannot_read_another_persons_results(app, client):
    """Run folders are addressed from the signed-in user's own id, so another person's run id simply
    does not resolve — there is no id to guess your way into."""
    signup(client, "a@example.com")
    with app.app_context():
        from portal.app import store
        other = User(email="b@example.com")
        other.set_password("a-long-password-9")
        db.session.add(other)
        db.session.commit()
        run = store.Run.create(other.id, "magnetic")
        run.write_meta({"state": "done"})
        run.fig_path("overview").write_bytes(b"not really a png")
        stolen = run.id

    assert client.get(f"/app/fig/magnetic/{stolen}/overview.png").status_code == 404
    assert client.get(f"/app/download/magnetic/{stolen}/png/overview").status_code in (403, 404)
    r = client.post(f"/app/run/magnetic/{stolen}/delete")
    assert r.status_code == 404
    with app.app_context():
        from portal.app import store
        assert store.Run.load(other.id, "magnetic", stolen) is not None, "it must still be there"


def test_a_job_belongs_to_the_person_who_started_it(app, client):
    from portal.app import jobs
    signup(client)
    with app.app_context():
        job = jobs.Job(user_id=99999, section="magnetic", run_id="r1")
        jobs._jobs[job.id] = job
    assert client.get(f"/app/job/{job.id}").status_code == 404
    assert client.post(f"/app/job/{job.id}/cancel").status_code == 404
    assert not job.cancel.is_set()


def test_an_admin_sees_everything_whatever_their_plan(app):
    with app.app_context():
        u = User(email="a@example.com", is_admin=True)
        u.set_password("a-long-password-9")
        db.session.add(u)
        db.session.commit()
        from portal.models import SECTIONS
        assert u.allowed_sections() == list(SECTIONS) and u.may_download()


# ---------------------------------------------------------------- the admin dashboard
def test_the_dashboard_is_closed_to_ordinary_users(app, client):
    signup(client)
    for url in ("/admin/", "/admin/users", "/admin/plans", "/admin/payments", "/admin/audit"):
        assert client.get(url).status_code == 403, url


def test_an_admin_can_activate_suspend_and_is_recorded(app, client):
    signup(client)
    client.get("/logout")
    email, pw = make_admin(app)
    client.post("/login", data={"email": email, "password": pw})
    with app.app_context():
        uid = User.query.filter_by(email="geo@example.com").first().id

    r = client.post(f"/admin/users/{uid}/activate",
                    data={"plan": "pro", "days": "90", "amount": "300000", "method": "mpesa",
                          "reference": "MP-77", "note": "paid in cash"}, follow_redirects=True)
    assert r.status_code == 200
    with app.app_context():
        u = db.session.get(User, uid)
        assert u.plan.code == "pro" and u.days_left == 89
        p = Payment.query.filter_by(user_id=uid).first()
        assert p.amount == 300000 and p.method == "mpesa" and p.reference == "MP-77"
        from portal.models import AuditLog
        assert AuditLog.query.filter_by(action="activate", target="geo@example.com").count() == 1

    client.post(f"/admin/users/{uid}/suspend", follow_redirects=True)
    with app.app_context():
        assert db.session.get(User, uid).is_active_account is False

    client.post(f"/admin/users/{uid}/cancel", follow_redirects=True)
    with app.app_context():
        assert db.session.get(User, uid).subscription is None


def test_an_admin_cannot_lock_themselves_out(app, client):
    email, pw = make_admin(app)
    client.post("/login", data={"email": email, "password": pw})
    with app.app_context():
        me = User.query.filter_by(email=email).first().id
    client.post(f"/admin/users/{me}/suspend", follow_redirects=True)
    client.post(f"/admin/users/{me}/admin", follow_redirects=True)
    with app.app_context():
        u = db.session.get(User, me)
        assert u.is_active_account and u.is_admin


def test_an_admin_can_create_a_plan_from_the_dashboard(app, client):
    email, pw = make_admin(app)
    client.post("/login", data={"email": email, "password": pw})
    client.post("/admin/plans", data={"code": "student", "name": "Student", "price": "25000",
                                      "currency": "TZS", "period_days": "30", "seats": "1",
                                      "max_uploads_per_month": "10",
                                      "sections": ["home", "geochemistry"], "is_public": "on"},
                follow_redirects=True)
    with app.app_context():
        p = Plan.query.filter_by(code="student").first()
        assert p and p.price == 25000 and p.section_list == ["home", "geochemistry"]
        assert p.can_download is False                 # the box was not ticked


def test_bad_numbers_in_the_admin_form_do_not_crash(app, client):
    signup(client)
    client.get("/logout")
    email, pw = make_admin(app)
    client.post("/login", data={"email": email, "password": pw})
    with app.app_context():
        uid = User.query.filter_by(email="geo@example.com").first().id
    r = client.post(f"/admin/users/{uid}/activate", data={"plan": "pro", "days": "many"},
                    follow_redirects=True)
    assert r.status_code == 200 and b"must be numbers" in r.data


# ---------------------------------------------------------------- pages render
def test_every_public_page_renders(app, client):
    for url in ("/", "/pricing", "/login", "/register"):
        assert client.get(url).status_code == 200, url
    signup(client)
    assert client.get("/account").status_code == 200
    client.get("/logout")
    email, pw = make_admin(app)
    client.post("/login", data={"email": email, "password": pw})
    for url in ("/admin/", "/admin/users", "/admin/plans", "/admin/payments", "/admin/audit"):
        assert client.get(url).status_code == 200, url


def test_seed_plans_is_safe_to_run_twice(app):
    with app.app_context():
        before = Plan.query.count()
        assert seed_plans() == []
        assert Plan.query.count() == before == 3


def test_every_section_has_a_route_and_a_plan_can_name_it(app):
    """The subscription model is written in terms of section keys, so the workbench's sections and
    the keys the plans use must not drift apart."""
    from portal.app import SECTIONS as NAV
    from portal.app.sections import MODULES
    from portal.models import SECTIONS as PLAN_SECTIONS

    keys = [s["key"] for s in NAV]
    assert len(keys) == len(set(keys)), "two sections share a key"
    assert set(keys) <= set(PLAN_SECTIONS), \
        f"a plan cannot name {set(keys) - set(PLAN_SECTIONS)}"
    assert set(MODULES) == set(keys), "a section has no module, or a module has no section"

    endpoints = {r.endpoint for r in app.url_map.iter_rules()}
    for s in NAV:
        assert s["endpoint"] in endpoints, f"{s['key']} has no route"


def test_the_workbench_store_keeps_people_apart(app):
    """Run folders are built from the owner's id, so one person's path can never be another's."""
    from portal.app import store
    with app.app_context():
        a = store.Run.create(1, "magnetic")
        b = store.Run.create(2, "magnetic")
        assert store.home(1) != store.home(2)
        assert str(a.dir).startswith(str(store.home(1)))
        assert str(b.dir).startswith(str(store.home(2)))
        assert store.Run.load(2, "magnetic", a.id) is None
        a.write_meta({"state": "done"})
        assert [r.id for r in store.Run.list(1, "magnetic")] == [a.id]
        assert store.sections_with_results(2) == {}


def test_a_draft_is_not_mistaken_for_a_run(app):
    from portal.app import store
    with app.app_context():
        store.add_draft(7, "ip", "line1.csv", b"a,b\n1,2\n")
        assert [p.name for p in store.draft_files(7, "ip")] == ["line1.csv"]
        assert store.Run.list(7, "ip") == []
        store.clear_draft(7, "ip")
        assert store.draft_files(7, "ip") == []


def test_the_monthly_run_cap_is_enforced(app, client):
    """A plan's upload allowance has to actually stop a run, or the tier means nothing."""
    from portal.models import UsageEvent
    signup(client)                                    # Free: 5 runs a month
    with app.app_context():
        u = User.query.filter_by(email="geo@example.com").first()
        used, limit = u.upload_allowance()
        assert (used, limit) == (0, 5)
        for _ in range(5):
            db.session.add(UsageEvent(user_id=u.id, kind="upload", section="remote-sensing"))
        db.session.commit()
        assert u.upload_allowance() == (5, 5)

    client.post("/app/remote-sensing", data={              # an area to run against
        "do": "typed",
        "corners": "corner,latitude,longitude\nC1,-2.85,32.15\nC2,-2.85,32.19\n"
                   "C3,-2.89,32.19\nC4,-2.89,32.15\n"})
    r = client.post("/app/remote-sensing", data={"do": "load", "auto": "1"})
    assert r.status_code == 302 and "/pricing" in r.headers["Location"], \
        f"a run past the monthly cap must be refused, got {r.headers.get('Location')}"


def test_an_admin_is_never_capped(app, client):
    from portal.app import upload_allowance
    email, pw = make_admin(app)
    client.post("/login", data={"email": email, "password": pw})
    with app.test_request_context():
        from flask_login import login_user
        with app.app_context():
            login_user(User.query.filter_by(email=email).first())
            assert upload_allowance() == (0, None, None)


# ---------------------------------------------------------------- the icon set
def test_every_icon_the_templates_ask_for_exists_in_the_sprite():
    """icon() raises on an unknown name, but only when that branch of a template runs. Check the
    whole set up front so a typo in a rarely-seen panel cannot reach anybody."""
    import re

    from portal import icons

    have = set(icons.names())
    root = pathlib.Path(__file__).resolve().parents[1]
    pattern = re.compile(r"""icon\(\s*['"]([a-z0-9-]+)['"]""")
    asked = {}
    for path in list((root / "portal" / "templates").rglob("*.html")) + \
            list((root / "portal").rglob("*.py")):
        for name in pattern.findall(path.read_text(encoding="utf-8")):
            asked.setdefault(name, path.relative_to(root))
    assert asked, "no icon() calls found — has the sprite been wired up?"
    missing = {n: str(p) for n, p in asked.items() if n not in have}
    assert not missing, f"icons used but not in the sprite: {missing}"


def test_the_section_icons_are_icon_names_not_emojis():
    from portal import icons
    from portal.app import SECTIONS

    have = set(icons.names())
    for s in SECTIONS:
        assert s["icon"] in have, f"{s['key']} has icon {s['icon']!r}, which is not in the sprite"


def test_no_emojis_are_left_in_the_templates():
    """The icon sprite replaced every emoji. Typographic marks are fine; pictographs are not."""
    allowed = set("\u2014\u2013\u00b7\u2026\u2018\u2019\u201c\u201d\u00d7\u00b0"
                  "\u2032\u2033\u2265\u2264\u2192\u221e\u2318\u00a0")
    root = pathlib.Path(__file__).resolve().parents[1] / "portal" / "templates"
    bad = []
    for path in root.rglob("*.html"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(ord(ch) > 0x2000 and ch not in allowed for ch in line):
                bad.append(f"{path.name}:{n}")
    assert not bad, f"emoji or other pictographs left in: {bad}"


# ---------------------------------------------------------------- failure handling
def test_a_run_whose_result_cannot_be_read_is_cleared_instead_of_crashing(client, app):
    """A finished run with no readable run.pkl — a restored folder, a disk that filled mid-save,
    an upgrade past an old pickle — must not take the section down with it."""
    import json

    from portal.app import store

    signup(client)
    client.get("/logout")
    email, pw = make_admin(app)                      # an administrator may open every section
    client.post("/login", data={"email": email, "password": pw})
    with app.app_context():
        uid = User.query.filter_by(email=email).first().id
        run = store.Run.create(uid, "magnetic")
        (run.dir / "meta.json").write_text(json.dumps(
            {"id": run.id, "section": "magnetic", "state": "done", "label": "x"}), encoding="utf-8")
        gone = run.dir
    r = client.get("/app/magnetic", follow_redirects=True)
    assert r.status_code == 200
    assert "could not be read back" in r.get_data(as_text=True)
    assert not gone.exists(), "the unreadable run should have been cleared"


def test_an_unhandled_error_is_written_to_the_data_folder(app, client):
    """Whoever runs this needs the traceback, and a browser saying "Internal Server Error" has none."""
    import pathlib

    @app.route("/_boom_for_test")
    def boom():
        raise RuntimeError("deliberate")

    assert client.get("/_boom_for_test").status_code == 500
    log = pathlib.Path(app.config["DATA_DIR"]) / "last-error.log"
    assert log.is_file()
    text = log.read_text(encoding="utf-8")
    assert "RuntimeError: deliberate" in text and "/_boom_for_test" in text


def test_a_missing_page_is_still_a_plain_404(client):
    assert client.get("/no-such-page").status_code == 404
