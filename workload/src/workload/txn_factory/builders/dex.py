"""DEX-related transaction builders: OfferCreate, OfferCancel, AMM*."""

from random import choice, sample

from xrpl.models.transactions import (
    AMMCreate,
    AMMDeposit,
    AMMWithdraw,
    OfferCancel,
    OfferCreate,
)
from xrpl.models.transactions.amm_deposit import AMMDepositFlag
from xrpl.models.transactions.amm_withdraw import AMMWithdrawFlag

from workload.constants import TxIntent
from workload.randoms import random, randrange
from workload.txn_factory.context import TxnContext


def build_offer_create(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an OfferCreate transaction to trade currencies on the DEX.

    Creates offers to exchange XRP/IOU or IOU/IOU pairs.
    """
    src = ctx.rand_account()

    use_xrp = random() < 0.5

    if use_xrp or not ctx.currencies:
        currency = ctx.rand_currency() if ctx.currencies else None
        if currency:
            if random() < 0.5:
                taker_pays = str(randrange(1_000_000, 100_000_000))  # XRP in drops
                taker_gets = {
                    "currency": currency.currency,
                    "issuer": currency.issuer,
                    "value": str(randrange(10, 1000)),
                }
            else:
                taker_pays = {
                    "currency": currency.currency,
                    "issuer": currency.issuer,
                    "value": str(randrange(10, 1000)),
                }
                taker_gets = str(randrange(1_000_000, 100_000_000))  # XRP in drops
        else:
            taker_pays = str(randrange(1_000_000, 100_000_000))
            taker_gets = str(randrange(1_000_000, 100_000_000))
    else:
        if len(ctx.currencies) >= 2:
            cur1, cur2 = sample(ctx.currencies, 2)
        else:
            cur1 = cur2 = ctx.rand_currency()

        taker_pays = {
            "currency": cur1.currency,
            "issuer": cur1.issuer,
            "value": str(randrange(10, 1000)),
        }
        taker_gets = {
            "currency": cur2.currency,
            "issuer": cur2.issuer,
            "value": str(randrange(10, 1000)),
        }

    return {
        "TransactionType": "OfferCreate",
        "Account": src.address,
        "TakerPays": taker_pays,
        "TakerGets": taker_gets,
    }


def build_offer_cancel(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an OfferCancel transaction to cancel an existing offer.

    Requires at least one IOU offer to exist in tracking.
    """
    if not ctx.offers:
        raise RuntimeError("No offers available to cancel")

    iou_offers = {k: v for k, v in ctx.offers.items() if v.get("type") == "IOUOffer"}
    if not iou_offers:
        raise RuntimeError("No IOU offers available to cancel")

    offer_id, offer_data = choice(list(iou_offers.items()))

    return {
        "TransactionType": "OfferCancel",
        "Account": offer_data["owner"],
        "OfferSequence": offer_data["sequence"],  # Sequence number when offer was created
    }


def build_amm_create(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an AMMCreate transaction with random currency pair.

    NOTE: Fee will be set to owner_reserve in build_sign_and_track based on TransactionType.
    """
    src = ctx.rand_account()

    max_attempts = 10
    amount_xrp = "1000000000"  # 1000 XRP (in drops)

    for _attempt in range(max_attempts):
        currency = ctx.rand_currency()
        amount_iou = {
            "currency": currency.currency,
            "issuer": currency.issuer,
            "value": str(ctx.config["amm"]["default_amm_token_deposit"]),
        }

        if not ctx.amm_pool_exists(amount_xrp, amount_iou):
            return {
                "TransactionType": "AMMCreate",
                "Account": src.address,
                "Amount": amount_xrp,
                "Amount2": amount_iou,
                "TradingFee": ctx.config["amm"]["trading_fee"],  # From config
            }

    return {
        "TransactionType": "AMMCreate",
        "Account": src.address,
        "Amount": amount_xrp,
        "Amount2": amount_iou,
        "TradingFee": ctx.config["amm"]["trading_fee"],
    }


def build_amm_deposit(ctx: TxnContext, intent: TxIntent) -> dict | None:  # TODO: handle TxIntent.INVALID
    """Build an AMMDeposit transaction to add liquidity to an existing AMM pool.

    Uses TF_TWO_ASSET flag for dual-asset deposit.
    Only picks pools where src account has the required assets.
    """
    src = ctx.rand_account()
    account_currencies = {(c.currency, c.issuer) for c in ctx.get_account_currencies(src)}
    has_balance_data = bool(account_currencies)

    eligible_pools = []
    for p in ctx.amm_pool_registry or []:
        a1, a2 = p["asset1"], p["asset2"]
        if a1.get("currency") == "XRP":
            # XRP/IOU pool: account always has XRP; if we have balance data, verify IOU too
            if not has_balance_data or (a2["currency"], a2["issuer"]) in account_currencies:
                eligible_pools.append(p)
        else:
            # IOU/IOU pool: only proceed if we can confirm both assets
            if (
                has_balance_data
                and (a1["currency"], a1["issuer"]) in account_currencies
                and (a2["currency"], a2["issuer"]) in account_currencies
            ):
                eligible_pools.append(p)

    if not eligible_pools:
        return None
    pool = choice(eligible_pools)

    asset1 = pool["asset1"]
    asset2 = pool["asset2"]

    amm_cfg = ctx.config.get("amm", {})

    if asset1.get("currency") == "XRP":
        amount1 = amm_cfg.get("deposit_amount_xrp", "1000000000")
        amount2 = {
            "currency": asset2["currency"],
            "issuer": asset2["issuer"],
            "value": amm_cfg.get("deposit_amount_iou", "500"),
        }
        asset1_field = {"currency": "XRP"}
        asset2_field = {"currency": asset2["currency"], "issuer": asset2["issuer"]}
    else:
        amount1 = {
            "currency": asset1["currency"],
            "issuer": asset1["issuer"],
            "value": amm_cfg.get("deposit_amount_iou", "500"),
        }
        amount2 = {
            "currency": asset2["currency"],
            "issuer": asset2["issuer"],
            "value": amm_cfg.get("deposit_amount_iou", "500"),
        }
        asset1_field = {"currency": asset1["currency"], "issuer": asset1["issuer"]}
        asset2_field = {"currency": asset2["currency"], "issuer": asset2["issuer"]}

    return {
        "TransactionType": "AMMDeposit",
        "Account": src.address,
        "Asset": asset1_field,
        "Asset2": asset2_field,
        "Amount": amount1,
        "Amount2": amount2,
        "Flags": AMMDepositFlag.TF_TWO_ASSET,
    }


def build_amm_withdraw(ctx: TxnContext, intent: TxIntent) -> dict | None:  # TODO: handle TxIntent.INVALID
    """Build an AMMWithdraw transaction to remove liquidity from an existing AMM pool.

    Uses TF_TWO_ASSET flag for proportional dual-asset withdrawal.
    Withdraws 10% of deposit amounts to keep pools healthy.
    Only picks pools where src holds LP tokens — returns None if no eligible pool found.
    """
    lp_holders: set[str] = set()
    for p in ctx.amm_pool_registry or []:
        for addr in p.get("lp_holders", [p.get("creator", "")]):
            lp_holders.add(addr)
    if not lp_holders:
        return None
    src = ctx.rand_owner(lp_holders)
    if src is None:
        return None
    eligible_pools = [
        p for p in (ctx.amm_pool_registry or []) if src.address in p.get("lp_holders", [p.get("creator", "")])
    ]
    if not eligible_pools:
        return None
    pool = choice(eligible_pools)

    asset1 = pool["asset1"]
    asset2 = pool["asset2"]

    amm_cfg = ctx.config.get("amm", {})
    withdraw_xrp = str(int(int(amm_cfg.get("deposit_amount_xrp", "1000000000")) * 0.1))
    withdraw_iou = str(float(amm_cfg.get("deposit_amount_iou", "500")) * 0.1)

    if asset1.get("currency") == "XRP":
        amount1 = withdraw_xrp
        amount2 = {
            "currency": asset2["currency"],
            "issuer": asset2["issuer"],
            "value": withdraw_iou,
        }
        asset1_field = {"currency": "XRP"}
        asset2_field = {"currency": asset2["currency"], "issuer": asset2["issuer"]}
    else:
        amount1 = {
            "currency": asset1["currency"],
            "issuer": asset1["issuer"],
            "value": withdraw_iou,
        }
        amount2 = {
            "currency": asset2["currency"],
            "issuer": asset2["issuer"],
            "value": withdraw_iou,
        }
        asset1_field = {"currency": asset1["currency"], "issuer": asset1["issuer"]}
        asset2_field = {"currency": asset2["currency"], "issuer": asset2["issuer"]}

    return {
        "TransactionType": "AMMWithdraw",
        "Account": src.address,
        "Asset": asset1_field,
        "Asset2": asset2_field,
        "Amount": amount1,
        "Amount2": amount2,
        "Flags": AMMWithdrawFlag.TF_TWO_ASSET,
    }


BUILDERS = {
    "OfferCreate": (build_offer_create, OfferCreate),
    "OfferCancel": (build_offer_cancel, OfferCancel),
    "AMMCreate": (build_amm_create, AMMCreate),
    "AMMDeposit": (build_amm_deposit, AMMDeposit),
    "AMMWithdraw": (build_amm_withdraw, AMMWithdraw),
}


# ---------------------------------------------------------------------------
# Per-account eligibility predicates
# ---------------------------------------------------------------------------


def _is_eligible_offer_cancel(wallet, ctx) -> bool:
    """Wallet must own at least one IOU offer."""
    return bool(
        ctx.offers
        and any(v.get("type") == "IOUOffer" and v.get("owner") == wallet.address for v in ctx.offers.values())
    )


def _is_eligible_amm_withdraw(wallet, ctx) -> bool:
    """Wallet must hold LP tokens in at least one pool."""
    return any(wallet.address in p.get("lp_holders", [p.get("creator", "")]) for p in (ctx.amm_pool_registry or []))


def _is_eligible_amm_deposit(wallet, ctx) -> bool:
    """Wallet must have balance in both assets of at least one pool."""
    account_currencies = {(c.currency, c.issuer) for c in ctx.get_account_currencies(wallet)}
    has_balance_data = bool(account_currencies)
    return any(
        (
            p["asset1"].get("currency") == "XRP"
            and (not has_balance_data or (p["asset2"].get("currency"), p["asset2"].get("issuer")) in account_currencies)
        )
        or (
            has_balance_data
            and "issuer" in p["asset1"]
            and "issuer" in p["asset2"]
            and (p["asset1"]["currency"], p["asset1"]["issuer"]) in account_currencies
            and (p["asset2"]["currency"], p["asset2"]["issuer"]) in account_currencies
        )
        for p in (ctx.amm_pool_registry or [])
    )


ELIGIBILITY = {
    "OfferCancel": _is_eligible_offer_cancel,
    "AMMWithdraw": _is_eligible_amm_withdraw,
    "AMMDeposit": _is_eligible_amm_deposit,
}


# ---------------------------------------------------------------------------
# Tainting strategies
# ---------------------------------------------------------------------------


def _offer_create_zero_pays(tx: dict) -> dict:
    """OfferCreate with zero TakerPays — temBAD_OFFER."""
    if isinstance(tx.get("TakerPays"), str):
        tx["TakerPays"] = "0"
    elif isinstance(tx.get("TakerPays"), dict):
        tx["TakerPays"]["value"] = "0"
    return tx


def _offer_cancel_bad_seq(tx: dict) -> dict:
    """OfferCancel with nonexistent sequence — temBAD_SEQUENCE."""
    tx["OfferSequence"] = 999_999_999
    return tx


def _amm_deposit_zero(tx: dict) -> dict:
    """AMMDeposit with zero amount — temBAD_AMM_TOKENS."""
    if isinstance(tx.get("Amount"), str):
        tx["Amount"] = "0"
    elif isinstance(tx.get("Amount"), dict):
        tx["Amount"]["value"] = "0"
    return tx


TAINTERS = {
    "OfferCreate": [_offer_create_zero_pays],
    "OfferCancel": [_offer_cancel_bad_seq],
    "AMMDeposit": [_amm_deposit_zero],
}
