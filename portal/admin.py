"""The admin dashboard: who has signed up, who is paying, and who may use what."""
from datetime import timedelta
from functools import wraps

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy import func, or_

from .billing import PROVIDERS, activate
from .models import (SECTIONS, AuditLog, Payment, Plan, Subscription, UsageEvent, User, audit, db, now)

bp = Blueprint("admin", __name__, url_prefix="/admin")


def admin_only(f):
    @wraps(f)
    @login_required
    def wrapper(*a, **kw):
        if not current_user.is_admin:
            abort(403)
        return f(*a, **kw)
    return wrapper


@bp.route("/")
@admin_only
def dashboard():
    month = now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    live = [s for s in Subscription.query.filter_by(status="active").all() if s.is_live]
    stats = {
        "users": User.query.count(),
        "active": len(live),
        "new_this_month": User.query.filter(User.created_at >= month).count(),
        "revenue_month": db.session.query(func.coalesce(func.sum(Payment.amount), 0.0))
        .filter(Payment.at >= month, Payment.status == "paid").scalar(),
        "revenue_total": db.session.query(func.coalesce(func.sum(Payment.amount), 0.0))
        .filter(Payment.status == "paid").scalar(),
        "expiring": sorted([s for s in live if s.days_left <= 7], key=lambda s: s.ends_at)[:10],
    }
    by_plan = {}
    for s in live:
        by_plan[s.plan.name] = by_plan.get(s.plan.name, 0) + 1
    recent = UsageEvent.query.order_by(UsageEvent.at.desc()).limit(15).all()
    users = {u.id: u for u in User.query.all()}
    return render_template("admin/dashboard.html", stats=stats, by_plan=by_plan, recent=recent,
                           users=users, currency=(live[0].plan.currency if live else "TZS"),
                           audit=AuditLog.query.order_by(AuditLog.at.desc()).limit(10).all())


@bp.route("/users")
@admin_only
def users():
    q = request.args.get("q", "").strip()
    query = User.query
    if q:
        like = f"%{q}%"
        query = query.filter(or_(User.email.ilike(like), User.name.ilike(like),
                                 User.organisation.ilike(like)))
    show = request.args.get("show", "all")
    people = query.order_by(User.created_at.desc()).all()
    if show == "paid":
        people = [u for u in people if u.subscription]
    elif show == "free":
        people = [u for u in people if not u.subscription]
    elif show == "suspended":
        people = [u for u in people if not u.is_active_account]
    return render_template("admin/users.html", people=people, q=q, show=show,
                           plans=Plan.query.order_by(Plan.sort).all())


@bp.route("/users/<int:uid>")
@admin_only
def user_detail(uid):
    u = db.session.get(User, uid) or abort(404)
    used, limit = u.upload_allowance()
    return render_template("admin/user.html", u=u, plans=Plan.query.order_by(Plan.sort).all(),
                           used=used, limit=limit, providers=PROVIDERS,
                           events=UsageEvent.query.filter_by(user_id=uid)
                           .order_by(UsageEvent.at.desc()).limit(25).all())


@bp.route("/users/<int:uid>/activate", methods=["POST"])
@admin_only
def activate_user(uid):
    u = db.session.get(User, uid) or abort(404)
    plan = Plan.query.filter_by(code=request.form.get("plan", "")).first() or abort(400)
    try:
        days = int(request.form.get("days") or plan.period_days)
        amount = float(request.form.get("amount") or plan.price)
    except ValueError:
        flash("Days and amount must be numbers.", "error")
        return redirect(url_for("admin.user_detail", uid=uid))
    activate(u, plan, amount=amount, method=request.form.get("method", "manual"),
             reference=request.form.get("reference", "").strip(), by=current_user.email, days=days,
             note=request.form.get("note", "").strip())
    flash(f"{u.email} is on {plan.name} for {days} days.", "ok")
    return redirect(url_for("admin.user_detail", uid=uid))


@bp.route("/users/<int:uid>/cancel", methods=["POST"])
@admin_only
def cancel(uid):
    u = db.session.get(User, uid) or abort(404)
    s = u.subscription
    if s is None:
        flash("That account has no live subscription.", "error")
    else:
        s.status = "cancelled"
        audit(current_user.email, "cancel", u.email, f"{s.plan.code}, {s.days_left} days unused")
        db.session.commit()
        flash(f"Subscription cancelled for {u.email}.", "ok")
    return redirect(url_for("admin.user_detail", uid=uid))


@bp.route("/users/<int:uid>/suspend", methods=["POST"])
@admin_only
def suspend(uid):
    u = db.session.get(User, uid) or abort(404)
    if u.id == current_user.id:
        flash("You cannot suspend your own account.", "error")
        return redirect(url_for("admin.user_detail", uid=uid))
    u.is_active_account = not u.is_active_account
    audit(current_user.email, "suspend" if not u.is_active_account else "restore", u.email)
    db.session.commit()
    flash(f"{u.email} is now {'active' if u.is_active_account else 'suspended'}.", "ok")
    return redirect(url_for("admin.user_detail", uid=uid))


@bp.route("/users/<int:uid>/admin", methods=["POST"])
@admin_only
def toggle_admin(uid):
    u = db.session.get(User, uid) or abort(404)
    if u.id == current_user.id:
        flash("You cannot remove your own administrator rights.", "error")
        return redirect(url_for("admin.user_detail", uid=uid))
    u.is_admin = not u.is_admin
    audit(current_user.email, "grant_admin" if u.is_admin else "revoke_admin", u.email)
    db.session.commit()
    flash(f"{u.email} is {'now an administrator' if u.is_admin else 'no longer an administrator'}.", "ok")
    return redirect(url_for("admin.user_detail", uid=uid))


@bp.route("/plans", methods=["GET", "POST"])
@admin_only
def plans():
    if request.method == "POST":
        code = request.form.get("code", "").strip().lower()
        p = Plan.query.filter_by(code=code).first()
        if p is None:
            if not code:
                flash("A plan needs a short code.", "error")
                return redirect(url_for("admin.plans"))
            p = Plan(code=code)
            db.session.add(p)
        try:
            p.name = request.form.get("name", "").strip() or code.title()
            p.price = float(request.form.get("price") or 0)
            p.currency = request.form.get("currency", "TZS").strip() or "TZS"
            p.period_days = int(request.form.get("period_days") or 30)
            p.seats = int(request.form.get("seats") or 1)
            p.max_uploads_per_month = int(request.form.get("max_uploads_per_month") or 0)
        except ValueError:
            flash("Price, period, seats and the upload cap must be numbers.", "error")
            return redirect(url_for("admin.plans"))
        p.sections = ",".join(s for s in request.form.getlist("sections") if s in SECTIONS)
        p.can_download = bool(request.form.get("can_download"))
        p.is_public = bool(request.form.get("is_public"))
        audit(current_user.email, "plan_saved", code, f"{p.price:,.0f} {p.currency}, {p.sections}")
        db.session.commit()
        flash(f"Plan “{p.name}” saved.", "ok")
        return redirect(url_for("admin.plans"))
    return render_template("admin/plans.html", plans=Plan.query.order_by(Plan.sort).all(),
                           sections=SECTIONS)


@bp.route("/payments")
@admin_only
def payments():
    rows = Payment.query.order_by(Payment.at.desc()).limit(300).all()
    total = sum(p.amount for p in rows if p.status == "paid")
    return render_template("admin/payments.html", rows=rows, total=total)


@bp.route("/audit")
@admin_only
def audit_log():
    return render_template("admin/audit.html",
                           rows=AuditLog.query.order_by(AuditLog.at.desc()).limit(300).all())
