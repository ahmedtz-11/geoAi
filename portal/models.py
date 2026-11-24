"""Accounts, plans and subscriptions for the GeoAI portal.

One idea runs through this file: a plan is data, not code. Which sections a subscriber may open and
whether they may download results are rows in the database, so the client can add a tier or change what
it includes from the admin dashboard without anyone touching Python.
"""
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

db = SQLAlchemy()
_hasher = PasswordHasher()

# every part of the app a plan can grant. The keys match the url_path of each Streamlit page.
SECTIONS = {
    "home": "Home",
    "remote-sensing": "1 · Remote sensing",
    "magnetic": "2 · Magnetic",
    "ip": "3 · IP",
    "geology": "4 · Geology",
    "geochemistry": "5 · Geochemistry",
    "prediction": "6 · Integrated gold prediction",
    "pit-mapping": "Pit mapping",
}
FREE_SECTIONS = ["home", "remote-sensing"]


def now():
    return datetime.now(timezone.utc)


def _aware(dt):
    """Times come back from SQLite without a zone; treat them as UTC so comparisons never crash."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Plan(db.Model):
    __tablename__ = "plans"
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(32), unique=True, nullable=False)        # free, basic, pro
    name = db.Column(db.String(80), nullable=False)
    price = db.Column(db.Float, nullable=False, default=0.0)
    currency = db.Column(db.String(8), nullable=False, default="TZS")
    period_days = db.Column(db.Integer, nullable=False, default=30)
    sections = db.Column(db.Text, nullable=False, default="")           # comma-separated keys of SECTIONS
    can_download = db.Column(db.Boolean, nullable=False, default=False)
    max_uploads_per_month = db.Column(db.Integer, nullable=False, default=0)   # 0 = no limit
    seats = db.Column(db.Integer, nullable=False, default=1)
    is_public = db.Column(db.Boolean, nullable=False, default=True)     # shown on the pricing page
    sort = db.Column(db.Integer, nullable=False, default=0)

    @property
    def section_list(self):
        return [s for s in (self.sections or "").split(",") if s]

    def allows(self, section):
        return section in self.section_list

    @property
    def price_label(self):
        if self.price <= 0:
            return "Free"
        per = {30: "month", 31: "month", 90: "quarter", 365: "year"}.get(self.period_days,
                                                                         f"{self.period_days} days")
        return f"{self.currency} {self.price:,.0f} / {per}"


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(190), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False, default="")
    organisation = db.Column(db.String(160), nullable=False, default="")
    phone = db.Column(db.String(40), nullable=False, default="")
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)
    is_active_account = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=now)
    last_login = db.Column(db.DateTime)

    subscriptions = db.relationship("Subscription", back_populates="user", cascade="all, delete-orphan",
                                    order_by="Subscription.id.desc()")
    payments = db.relationship("Payment", back_populates="user", cascade="all, delete-orphan")

    # ---- passwords
    def set_password(self, raw):
        self.password_hash = _hasher.hash(raw)

    def check_password(self, raw):
        try:
            _hasher.verify(self.password_hash, raw)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False
        if _hasher.check_needs_rehash(self.password_hash):              # argon2 parameters moved on
            self.set_password(raw)
        return True

    # ---- flask-login
    @property
    def is_active(self):
        return self.is_active_account

    # ---- subscription
    @property
    def subscription(self):
        """The live subscription, or None. A subscription is live when it is active and not past its end."""
        for s in self.subscriptions:
            if s.is_live:
                return s
        return None

    @property
    def plan(self):
        s = self.subscription
        return s.plan if s else Plan.query.filter_by(code="free").first()

    @property
    def plan_name(self):
        p = self.plan
        return p.name if p else "No plan"

    @property
    def expires_at(self):
        s = self.subscription
        return _aware(s.ends_at) if s else None

    @property
    def days_left(self):
        e = self.expires_at
        return None if e is None else max((e - now()).days, 0)

    def allowed_sections(self):
        """Which sections this user may open: their plan's, or the free ones when they have no plan."""
        if self.is_admin:
            return list(SECTIONS)
        p = self.plan
        return p.section_list if p else list(FREE_SECTIONS)

    def may_download(self):
        p = self.plan
        return bool(self.is_admin or (p and p.can_download))

    def uploads_this_month(self):
        start = now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return db.session.query(func.count(UsageEvent.id)).filter(
            UsageEvent.user_id == self.id, UsageEvent.kind == "upload",
            UsageEvent.at >= start).scalar() or 0

    def upload_allowance(self):
        """(used, limit) for this calendar month; limit None means no limit."""
        p = self.plan
        limit = p.max_uploads_per_month if p and p.max_uploads_per_month else None
        return self.uploads_this_month(), limit


class Subscription(db.Model):
    __tablename__ = "subscriptions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    plan_id = db.Column(db.Integer, db.ForeignKey("plans.id"), nullable=False)
    started_at = db.Column(db.DateTime, nullable=False, default=now)
    ends_at = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(24), nullable=False, default="active")   # active, cancelled, expired
    note = db.Column(db.String(255), nullable=False, default="")
    created_by = db.Column(db.String(190), nullable=False, default="")    # admin email, or the gateway

    user = db.relationship("User", back_populates="subscriptions")
    plan = db.relationship("Plan")

    @property
    def is_live(self):
        return self.status == "active" and _aware(self.ends_at) > now()

    @property
    def days_left(self):
        return max((_aware(self.ends_at) - now()).days, 0)

    @staticmethod
    def grant(user, plan, days=None, note="", by=""):
        """Give a user a plan. Extends an existing live subscription on the same plan instead of stacking."""
        days = int(days or plan.period_days)
        live = user.subscription
        if live and live.plan_id == plan.id:
            live.ends_at = _aware(live.ends_at) + timedelta(days=days)
            live.note = note or live.note
            return live
        if live:
            live.status = "cancelled"                                      # one plan at a time
        s = Subscription(user=user, plan=plan, started_at=now(), ends_at=now() + timedelta(days=days),
                         note=note, created_by=by)
        db.session.add(s)
        return s


class Payment(db.Model):
    """A payment recorded by hand, or reported by a gateway. Kept even when a subscription is replaced."""
    __tablename__ = "payments"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    plan_id = db.Column(db.Integer, db.ForeignKey("plans.id"))
    amount = db.Column(db.Float, nullable=False, default=0.0)
    currency = db.Column(db.String(8), nullable=False, default="TZS")
    method = db.Column(db.String(40), nullable=False, default="manual")   # manual, mpesa, flutterwave…
    reference = db.Column(db.String(120), nullable=False, default="")     # the gateway's or the receipt no.
    status = db.Column(db.String(24), nullable=False, default="paid")     # paid, pending, failed, refunded
    at = db.Column(db.DateTime, nullable=False, default=now)
    note = db.Column(db.String(255), nullable=False, default="")

    user = db.relationship("User", back_populates="payments")
    plan = db.relationship("Plan")


class UsageEvent(db.Model):
    """What people actually do, for the monthly cap and so the client can see the app being used."""
    __tablename__ = "usage"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), index=True)
    kind = db.Column(db.String(32), nullable=False)                       # login, upload, run, download
    section = db.Column(db.String(40), nullable=False, default="")
    detail = db.Column(db.String(255), nullable=False, default="")
    at = db.Column(db.DateTime, nullable=False, default=now, index=True)


class AuditLog(db.Model):
    """Every change an admin makes to someone's account, so there is always an answer to 'who did that'."""
    __tablename__ = "audit"
    id = db.Column(db.Integer, primary_key=True)
    actor = db.Column(db.String(190), nullable=False)
    action = db.Column(db.String(60), nullable=False)
    target = db.Column(db.String(190), nullable=False, default="")
    detail = db.Column(db.String(500), nullable=False, default="")
    at = db.Column(db.DateTime, nullable=False, default=now, index=True)


def record(kind, user=None, section="", detail=""):
    db.session.add(UsageEvent(user_id=user.id if user else None, kind=kind, section=section,
                              detail=detail[:255]))


def audit(actor, action, target="", detail=""):
    db.session.add(AuditLog(actor=actor, action=action, target=target, detail=detail[:500]))


DEFAULT_PLANS = [
    dict(code="free", name="Free", price=0, period_days=3650, sort=0, can_download=False,
         max_uploads_per_month=5, sections=",".join(FREE_SECTIONS)),
    dict(code="basic", name="Basic", price=120000, period_days=30, sort=1, can_download=True,
         max_uploads_per_month=50,
         sections="home,remote-sensing,magnetic,ip,pit-mapping"),
    dict(code="pro", name="Professional", price=300000, period_days=30, sort=2, can_download=True,
         max_uploads_per_month=0, seats=3, sections=",".join(SECTIONS)),
]


def seed_plans():
    """Create the starting plans once. Editing them afterwards is the admin's job, not this function's."""
    made = []
    for spec in DEFAULT_PLANS:
        if not Plan.query.filter_by(code=spec["code"]).first():
            db.session.add(Plan(**spec))
            made.append(spec["code"])
    db.session.commit()
    return made
