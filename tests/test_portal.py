"""The portal: accounts, plans, the admin dashboard, and the gate to the Streamlit app.

The gate tests matter most. Everything else here is a convenience that can be fixed later; a hole in the
gate means anyone on the internet uses the app for nothing.
"""
import os
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
                      "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}"})
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
def test_the_gate_refuses_a_stranger_and_admits_a_subscriber(app, client):
    assert client.get("/_gate/check").status_code == 401     # nobody signed in
    signup(client)
    r = client.get("/_gate/check")
    assert r.status_code == 200 and r.headers.get("X-GeoAI-Token")


def test_a_gate_token_cannot_be_forged_or_replayed_after_a_change(app):
    from itsdangerous import URLSafeTimedSerializer

    from portal.gate import issue, read
    from ui import gate as sgate
    with app.app_context():
        u = User(email="t@example.com", name="T")
        u.set_password("a-long-password-9")
        db.session.add(u)
        db.session.commit()
        token = issue(u)
        assert read(token)["email"] == "t@example.com"

        # the Streamlit side must reach the same verdict with only the shared secret
        same = URLSafeTimedSerializer(SECRET, salt=sgate.SALT).loads(token, max_age=3600)
        assert same["sections"] == ["home", "remote-sensing"] and same["download"] is False

        assert read(token + "x") is None                      # tampered
        assert read("") is None and read(None) is None
        other = URLSafeTimedSerializer("a-different-secret", salt=sgate.SALT).dumps({"email": "evil"})
        assert read(other) is None, "a token signed with another key must never be believed"
        assert read(token, max_age=-1) is None                # expired


def test_the_streamlit_side_reads_the_token_and_gates_pages(app, monkeypatch):
    from portal.gate import issue
    from ui import gate as sgate
    with app.app_context():
        from portal.billing import activate
        u = User(email="s@example.com", name="S")
        u.set_password("a-long-password-9")
        db.session.add(u)
        activate(u, Plan.query.filter_by(code="basic").first(), by="admin")
        token = issue(u)

    monkeypatch.setattr(sgate, "SECRET", SECRET)
    data = sgate._serializer(SECRET).loads(token, max_age=3600)
    v = sgate.Visitor(email=data["email"], admin=data["admin"], sections=data["sections"],
                      download=data["download"], gated=True)
    assert v.may_open("magnetic") and not v.may_open("geochemistry")
    assert v.may_download()
    pages = {k: object() for k in sgate.ALL_SECTIONS}
    monkeypatch.setitem(__import__("streamlit").session_state, "_visitor", v) \
        if hasattr(__import__("streamlit"), "session_state") else None
    assert set(k for k in pages if v.may_open(k)) == set(data["sections"])


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


def test_every_page_link_in_the_app_is_guarded():
    """A link to a page the plan does not open stops the page it is drawn on, so every st.page_link
    must go through gate.page_link, and gate must know which section each page file belongs to."""
    import pathlib
    import re

    from ui.gate import ALL_SECTIONS, SECTION_OF
    root = pathlib.Path(__file__).resolve().parent.parent

    pages = set(re.findall(r'st\.Page\("([^"]+)"', (root / "app.py").read_text(encoding="utf-8")))
    assert pages, "app.py should declare its pages with st.Page"
    assert pages <= set(SECTION_OF), f"gate.SECTION_OF is missing {pages - set(SECTION_OF)}"
    assert set(SECTION_OF.values()) == set(ALL_SECTIONS)

    bare = []
    for f in list((root / "views").rglob("*.py")) + list((root / "ui").rglob("*.py")):
        if f.name == "gate.py":
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\bst\.page_link\(", line):
                bare.append(f"{f.relative_to(root)}:{n}")
    assert not bare, "use gate.page_link instead of st.page_link at " + ", ".join(bare)
