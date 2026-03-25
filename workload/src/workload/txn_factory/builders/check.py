"""Check transaction builders (CheckCreate, CheckCash, CheckCancel)."""

import time
from random import choice

from xrpl.models.transactions import CheckCancel, CheckCash, CheckCreate

from workload.constants import TxIntent
from workload.randoms import random, randrange
from workload.txn_factory.context import TxnContext, choice_omit

RIPPLE_EPOCH_OFFSET = 946684800


def build_check_create(ctx: TxnContext, intent: TxIntent) -> dict:
    """Build a CheckCreate transaction.

    Creates a deferred payment (Check) that the destination can later cash.
    Randomly uses XRP or IOU for send_max.
    """
    src = ctx.rand_account()
    dst = choice_omit(ctx.wallets, [src])

    use_xrp = random() < 0.5
    if use_xrp or not ctx.currencies:
        send_max = str(randrange(1_000_000, 100_000_000))  # 1-100 XRP in drops
    else:
        currency = ctx.rand_currency()
        send_max = {
            "currency": currency.currency,
            "issuer": currency.issuer,
            "value": str(randrange(10, 1000)),
        }

    tx: dict = {
        "TransactionType": "CheckCreate",
        "Account": src.address,
        "Destination": dst.address,
        "SendMax": send_max,
    }

    # ~50% chance of setting an expiration (1-24 hours from now)
    if random() < 0.5:
        ripple_now = int(time.time()) - RIPPLE_EPOCH_OFFSET
        tx["Expiration"] = ripple_now + randrange(3600, 86400)

    return tx


def build_check_cash(ctx: TxnContext, intent: TxIntent) -> dict | None:
    """Build a CheckCash transaction to redeem an existing Check.

    The destination of the Check cashes it for the exact send_max amount.
    When forced_account is set, only picks checks destined for that account.
    Pops the check from tracking at build time to prevent double-use.
    """
    if not ctx.checks:
        return None

    # Filter to checks this account can cash (destination only)
    if ctx.forced_account:
        eligible = {k: v for k, v in ctx.checks.items() if v["destination"] == ctx.forced_account.address}
    else:
        eligible = dict(ctx.checks)

    if not eligible:
        return None

    check_id, check_data = choice(list(eligible.items()))
    del ctx.checks[check_id]

    return {
        "TransactionType": "CheckCash",
        "Account": check_data["destination"],
        "CheckID": check_id,
        "Amount": check_data["send_max"],
    }


def build_check_cancel(ctx: TxnContext, intent: TxIntent) -> dict | None:
    """Build a CheckCancel transaction to void an existing Check.

    Either the sender or destination can cancel a Check.
    When forced_account is set, only picks checks where that account is sender or destination.
    Pops the check from tracking at build time to prevent double-use.
    """
    if not ctx.checks:
        return None

    # Filter to checks this account can cancel (sender or destination)
    if ctx.forced_account:
        addr = ctx.forced_account.address
        eligible = {k: v for k, v in ctx.checks.items() if v["sender"] == addr or v["destination"] == addr}
    else:
        eligible = dict(ctx.checks)

    if not eligible:
        return None

    check_id, check_data = choice(list(eligible.items()))
    del ctx.checks[check_id]

    # Use the forced account, or pick sender/destination randomly
    if ctx.forced_account:
        canceller = ctx.forced_account.address
    else:
        canceller = choice([check_data["sender"], check_data["destination"]])

    return {
        "TransactionType": "CheckCancel",
        "Account": canceller,
        "CheckID": check_id,
    }


BUILDERS = {
    "CheckCreate": (build_check_create, CheckCreate),
    "CheckCash": (build_check_cash, CheckCash),
    "CheckCancel": (build_check_cancel, CheckCancel),
}


def _is_eligible_check_cash(wallet, ctx) -> bool:
    """Account must be the destination of at least one Check."""
    return bool(ctx.checks and any(c["destination"] == wallet.address for c in ctx.checks.values()))


def _is_eligible_check_cancel(wallet, ctx) -> bool:
    """Account must be sender or destination of at least one Check."""
    return bool(
        ctx.checks
        and any(c["sender"] == wallet.address or c["destination"] == wallet.address for c in ctx.checks.values())
    )


ELIGIBILITY = {
    "CheckCash": _is_eligible_check_cash,
    "CheckCancel": _is_eligible_check_cancel,
}


# ---------------------------------------------------------------------------
# Tainting strategies
# ---------------------------------------------------------------------------


def _check_cash_bad_id(tx: dict) -> dict:
    """CheckCash with nonexistent check — tecNO_ENTRY."""
    tx["CheckID"] = "0" * 64
    return tx


def _check_cancel_bad_id(tx: dict) -> dict:
    """CheckCancel with nonexistent check — tecNO_ENTRY."""
    tx["CheckID"] = "0" * 64
    return tx


TAINTERS = {
    "CheckCash": [_check_cash_bad_id],
    "CheckCancel": [_check_cancel_bad_id],
}
