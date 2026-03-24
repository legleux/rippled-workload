"""Payment, TrustSet, and AccountSet transaction builders."""

from random import choice

from xrpl.models.transactions import AccountSet, Payment, TrustSet

from workload.constants import TxIntent
from workload.randoms import random
from workload.txn_factory.context import TxnContext


def build_payment(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build a Payment transaction with random source and destination."""
    src = ctx.rand_account()
    dst = ctx.rand_account(omit=[src.address])

    use_xrp = random() < ctx.config.get("amm", {}).get("xrp_chance", 0.1)

    if use_xrp or not ctx.currencies:
        amount = str(ctx.config["transactions"]["payment"]["amount"])
    else:
        available_currencies = ctx.get_account_currencies(src)

        issuer_currencies = [c for c in ctx.currencies if c.issuer == src.address]

        sendable_currencies = list(set(available_currencies + issuer_currencies))

        if sendable_currencies:
            currency = choice(sendable_currencies)
            amount = {
                "currency": currency.currency,
                "issuer": currency.issuer,
                "value": "100",  # 100 units of the currency
            }
        else:
            amount = str(ctx.config["transactions"]["payment"]["amount"])

    result = {
        "TransactionType": "Payment",
        "Account": src.address,
        "Destination": dst.address,
        "Amount": amount,
    }

    return result


def build_trustset(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build a TrustSet transaction with random account and currency.

    Picks a currency where:
      1. issuer != src.address (prevents temDST_IS_SRC)
      2. currency not in src's existing trustlines (creates useful new trustlines)
    """
    src = ctx.rand_account()

    existing_trustlines = ctx.get_account_currencies(src)
    existing_keys = {(c.currency, c.issuer) for c in existing_trustlines}

    available = [c for c in ctx.currencies if c.issuer != src.address and (c.currency, c.issuer) not in existing_keys]

    if not available:
        available = [c for c in ctx.currencies if c.issuer != src.address]
        if not available:
            raise RuntimeError(f"No currencies available for {src.address} to trust")

    cur = choice(available)

    result = {
        "TransactionType": "TrustSet",
        "Account": src.address,
        "LimitAmount": {
            "currency": cur.currency,
            "issuer": cur.issuer,
            "value": str(ctx.config["transactions"]["trustset"]["limit"]),  # From config
        },
    }

    return result


def build_accountset(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an AccountSet transaction with random account."""
    src = ctx.rand_account()
    return {
        "TransactionType": "AccountSet",
        "Account": src.address,
    }


BUILDERS = {
    "Payment": (build_payment, Payment),
    "TrustSet": (build_trustset, TrustSet),
    "AccountSet": (build_accountset, AccountSet),
}


# ---------------------------------------------------------------------------
# Tainting strategies for intentionally invalid transactions
# ---------------------------------------------------------------------------


def _payment_self_send(tx: dict) -> dict:
    """Payment to self — temDST_IS_SRC."""
    tx["Destination"] = tx["Account"]
    return tx


def _payment_zero_amount(tx: dict) -> dict:
    """Payment with zero amount — temBAD_AMOUNT."""
    tx["Amount"] = "0"
    return tx


def _payment_overspend(tx: dict) -> dict:
    """Payment exceeding balance — tecUNFUNDED_PAYMENT."""
    tx["Amount"] = "999999999999999999"
    return tx


def _trustset_self_trust(tx: dict) -> dict:
    """TrustSet to yourself — temDST_IS_SRC."""
    if isinstance(tx.get("LimitAmount"), dict):
        tx["LimitAmount"]["issuer"] = tx["Account"]
    return tx


TAINTERS = {
    "Payment": [_payment_self_send, _payment_zero_amount, _payment_overspend],
    "TrustSet": [_trustset_self_trust],
}
