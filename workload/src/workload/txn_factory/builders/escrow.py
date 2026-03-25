"""Escrow transaction builders (EscrowCreate, EscrowFinish, EscrowCancel)."""

import time
from random import choice

from xrpl.models.transactions import EscrowCancel, EscrowCreate, EscrowFinish

from workload.constants import TxIntent
from workload.randoms import randrange
from workload.txn_factory.context import TxnContext, choice_omit

RIPPLE_EPOCH_OFFSET = 946684800


def build_escrow_create(ctx: TxnContext, intent: TxIntent) -> dict:
    """Build an EscrowCreate transaction.

    Creates a time-locked escrow of XRP. No crypto-conditions — just time-based.
    finish_after = now + 1s (cashable almost immediately)
    cancel_after = now + 300s (cancellable after 5 minutes)
    """
    src = ctx.rand_account()
    dst = choice_omit(ctx.wallets, [src])

    ripple_now = int(time.time()) - RIPPLE_EPOCH_OFFSET
    finish_after = ripple_now + 1
    cancel_after = ripple_now + 300

    return {
        "TransactionType": "EscrowCreate",
        "Account": src.address,
        "Destination": dst.address,
        "Amount": str(randrange(1_000_000, 50_000_000)),  # 1-50 XRP in drops
        "FinishAfter": finish_after,
        "CancelAfter": cancel_after,
    }


def build_escrow_finish(ctx: TxnContext, intent: TxIntent) -> dict | None:
    """Build an EscrowFinish to release a matured escrow.

    Only picks escrows whose finish_after has passed. Anyone can finish
    a time-based escrow (no condition/fulfillment needed).
    Pops the escrow from tracking at build time to prevent double-use.
    """
    if not ctx.escrows:
        return None

    ripple_now = int(time.time()) - RIPPLE_EPOCH_OFFSET
    finishable = {k: v for k, v in ctx.escrows.items() if v["finish_after"] <= ripple_now}
    if not finishable:
        return None

    escrow_id, escrow_data = choice(list(finishable.items()))
    del ctx.escrows[escrow_id]

    # Anyone can finish a time-based escrow
    finisher = ctx.rand_account()

    return {
        "TransactionType": "EscrowFinish",
        "Account": finisher.address,
        "Owner": escrow_data["owner"],
        "OfferSequence": escrow_data["sequence"],
    }


def build_escrow_cancel(ctx: TxnContext, intent: TxIntent) -> dict | None:
    """Build an EscrowCancel to cancel an expired escrow.

    Only picks escrows whose cancel_after has passed. Only the owner can cancel.
    When forced_account is set, only picks escrows owned by that account.
    Pops the escrow from tracking at build time to prevent double-use.
    """
    if not ctx.escrows:
        return None

    ripple_now = int(time.time()) - RIPPLE_EPOCH_OFFSET
    cancellable = {k: v for k, v in ctx.escrows.items() if v["cancel_after"] <= ripple_now}

    # Filter to escrows this account owns
    if ctx.forced_account:
        cancellable = {k: v for k, v in cancellable.items() if v["owner"] == ctx.forced_account.address}

    if not cancellable:
        return None

    escrow_id, escrow_data = choice(list(cancellable.items()))
    del ctx.escrows[escrow_id]

    return {
        "TransactionType": "EscrowCancel",
        "Account": escrow_data["owner"],
        "Owner": escrow_data["owner"],
        "OfferSequence": escrow_data["sequence"],
    }


BUILDERS = {
    "EscrowCreate": (build_escrow_create, EscrowCreate),
    "EscrowFinish": (build_escrow_finish, EscrowFinish),
    "EscrowCancel": (build_escrow_cancel, EscrowCancel),
}


def _is_eligible_escrow_cancel(wallet, ctx) -> bool:
    """Account must own at least one escrow."""
    return bool(ctx.escrows and any(e["owner"] == wallet.address for e in ctx.escrows.values()))


ELIGIBILITY = {
    "EscrowCancel": _is_eligible_escrow_cancel,
}


# ---------------------------------------------------------------------------
# Tainting strategies
# ---------------------------------------------------------------------------


def _escrow_finish_bad_seq(tx: dict) -> dict:
    """EscrowFinish with invalid sequence — tecNO_TARGET."""
    tx["OfferSequence"] = 999999999
    return tx


def _escrow_cancel_bad_seq(tx: dict) -> dict:
    """EscrowCancel with invalid sequence — tecNO_TARGET."""
    tx["OfferSequence"] = 999999999
    return tx


TAINTERS = {
    "EscrowFinish": [_escrow_finish_bad_seq],
    "EscrowCancel": [_escrow_cancel_bad_seq],
}
