"""Signing up, signing in, and the account page."""
import re

from flask import (Blueprint, current_app, flash, redirect, render_template, request, session,
                   url_for)
from flask_login import current_user, login_required, login_user, logout_user

from .models import User, audit, db, now, record

bp = Blueprint("auth", __name__)
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def safe_next(target, default="/"):
    """Only ever redirect inside this site: an absolute URL from the query string is an open redirect."""
    if not target or "//" in target or ":" in target or not target.startswith("/"):
        return default
    return target


def password_problem(pw):
    if len(pw) < 10:
        return "Use at least 10 characters."
    if pw.isdigit() or pw.isalpha():
        return "Mix letters with numbers or punctuation."
    if pw.lower() in ("password12", "geoai12345", "1234567890"):
        return "That password is too easy to guess."
    return None


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("auth.account"))
    form = {}
    if request.method == "POST":
        form = {k: request.form.get(k, "").strip() for k in ("name", "email", "organisation", "phone")}
        email = form["email"].lower()
        pw, pw2 = request.form.get("password", ""), request.form.get("password2", "")
        error = None
        if not EMAIL.match(email):
            error = "Enter a valid email address."
        elif User.query.filter_by(email=email).first():
            error = "An account with that email already exists. Sign in instead."
        elif pw != pw2:
            error = "The two passwords do not match."
        else:
            error = password_problem(pw)
        if error:
            flash(error, "error")
        else:
            u = User(email=email, name=form["name"] or email.split("@")[0],
                     organisation=form["organisation"], phone=form["phone"])
            u.set_password(pw)
            db.session.add(u)
            db.session.flush()
            record("signup", u)
            audit(email, "register", email, "self-service sign-up")
            db.session.commit()
            login_user(u)
            flash("Welcome. Your account is on the Free plan; choose a plan when you are ready.", "ok")
            return redirect(url_for("billing.pricing"))
    return render_template("register.html", form=form)


@bp.route("/login", methods=["GET", "POST"])
def login():
    nxt = safe_next(request.args.get("next"), url_for("auth.account"))
    if current_user.is_authenticated:
        return redirect(nxt)
    email = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        u = User.query.filter_by(email=email).first()
        if u is None or not u.check_password(request.form.get("password", "")):
            flash("That email and password do not match an account.", "error")
        elif not u.is_active_account:
            flash("This account is suspended. Please contact support.", "error")
        else:
            login_user(u, remember=bool(request.form.get("remember")))
            u.last_login = now()
            record("login", u)
            db.session.commit()
            return redirect(safe_next(request.form.get("next"), nxt))
    return render_template("login.html", email=email, next=nxt)


@bp.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()            # drop everything, not only the login keys, and issue a new cookie
    flash("Signed out.", "ok")
    return redirect(url_for("public.index"))


@bp.route("/account")
@login_required
def account():
    used, limit = current_user.upload_allowance()
    return render_template("account.html", used=used, limit=limit,
                           history=current_user.subscriptions[:10],
                           payments=sorted(current_user.payments, key=lambda p: p.at, reverse=True)[:10])


@bp.route("/account/password", methods=["POST"])
@login_required
def change_password():
    old, new, new2 = (request.form.get(k, "") for k in ("current", "password", "password2"))
    if not current_user.check_password(old):
        flash("Your current password is not right.", "error")
    elif new != new2:
        flash("The two new passwords do not match.", "error")
    else:
        problem = password_problem(new)
        if problem:
            flash(problem, "error")
        else:
            current_user.set_password(new)
            audit(current_user.email, "password_change", current_user.email)
            db.session.commit()
            flash("Password changed.", "ok")
    return redirect(url_for("auth.account"))


@bp.route("/account/details", methods=["POST"])
@login_required
def change_details():
    for field in ("name", "organisation", "phone"):
        setattr(current_user, field, request.form.get(field, "").strip())
    db.session.commit()
    flash("Details saved.", "ok")
    return redirect(url_for("auth.account"))
