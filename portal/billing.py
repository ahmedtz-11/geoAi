"""Plans, pricing and payments.

Payment is deliberately in two halves. `manual` is complete and working: a subscriber pays however the
client likes, an administrator records it, and access starts. A gateway (Flutterwave, Pesapal, DPO,
Selcom, M-Pesa) plugs into `start_checkout` and `handle_webhook` without anything else changing, because
both halves end at the same place: `activate`, which records the payment and grants the subscription.

One thing to tell the client before they choose a gateway: mobile money in Tanzania generally cannot keep
a card on file and charge it silently each month, so a renewal is usually a request the subscriber
approves. The expiry-and-renew model here suits that, rather than assuming a silent auto-charge.
"""
from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from .models import Payment, Plan, Subscription, audit, db, record

bp = Blueprint("billing", __name__)
PROVIDERS = {
    "manual": "Recorded by an administrator",
    "flutterwave": "Flutterwave (cards and mobile money)",
    "pesapal": "Pesapal (cards and mobile money)",
    "dpo": "DPO Pay",
    "selcom": "Selcom",
    "mpesa": "Vodacom M-Pesa direct",
}


def activate(user, plan, amount=None, method="manual", reference="", by="", days=None, note=""):
    """The one place a subscription begins. Both manual and gateway payments arrive here."""
    pay = Payment(user=user, plan=plan, amount=plan.price if amount is None else amount,
                  currency=plan.currency, method=method, reference=reference, status="paid", note=note)
    db.session.add(pay)
    sub = Subscription.grant(user, plan, days=days, note=note, by=by or method)
    record("subscribe", user, detail=f"{plan.code} via {method}")
    audit(by or method, "activate", user.email,
          f"{plan.code} for {days or plan.period_days} days, {pay.currency} {pay.amount:,.0f}, ref {reference}")
    db.session.commit()
    return sub


@bp.route("/pricing")
def pricing():
    plans = Plan.query.filter_by(is_public=True).order_by(Plan.sort).all()
    return render_template("pricing.html", plans=plans, provider=current_app.config["PAYMENT_PROVIDER"])


@bp.route("/subscribe/<code>", methods=["GET", "POST"])
@login_required
def subscribe(code):
    plan = Plan.query.filter_by(code=code).first_or_404()
    provider = current_app.config["PAYMENT_PROVIDER"]
    if plan.price <= 0:
        flash("The Free plan needs no payment; it is already yours.", "ok")
        return redirect(url_for("auth.account"))
    if provider == "manual":
        if request.method == "POST":
            record("request_plan", current_user, detail=plan.code)
            audit(current_user.email, "request_plan", current_user.email, plan.code)
            db.session.commit()
            flash("Thank you. Your request has reached the administrator; access opens as soon as your "
                  "payment is confirmed.", "ok")
            return redirect(url_for("auth.account"))
        return render_template("subscribe_manual.html", plan=plan)
    return redirect(start_checkout(current_user, plan))


# ---------------------------------------------------------------- gateway hooks
def start_checkout(user, plan):
    """Hand the subscriber to the gateway. Fill this in once the client's merchant account exists.

    Each provider wants roughly the same thing: create an order with an amount, a currency, your own
    reference, and a URL to come back to, then send the browser to the page the provider returns.
    """
    raise NotImplementedError(
        f"The {current_app.config['PAYMENT_PROVIDER']} gateway is not connected yet. Set "
        "GEOAI_PAYMENT_PROVIDER=manual until the merchant account and API keys exist.")


@bp.route("/payments/webhook/<provider>", methods=["POST"])
def webhook(provider):
    """Where a gateway reports a payment.

    Nothing is trusted until the signature is checked: a webhook URL is public, so anyone can post to it.
    """
    if provider not in PROVIDERS:
        abort(404)
    secret = current_app.config["PAYMENT_WEBHOOK_SECRET"]
    if not secret:
        current_app.logger.warning("webhook received with no GEOAI_PAYMENT_WEBHOOK_SECRET set; ignored")
        abort(503)
    if not verify_signature(provider, request, secret):
        current_app.logger.warning("webhook with a bad signature from %s", request.remote_addr)
        abort(400)
    raise NotImplementedError("Connect the gateway, then map its payload to activate().")


def verify_signature(provider, req, secret):
    """Each gateway signs differently; this is where that goes. Returning False is the safe default."""
    return False
