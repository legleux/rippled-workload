"""NFToken transaction builders."""

from random import choice

from xrpl.models.transactions import (
    NFTokenAcceptOffer,
    NFTokenBurn,
    NFTokenCancelOffer,
    NFTokenCreateOffer,
    NFTokenMint,
    Memo,
)

from workload.constants import TxIntent
from workload.randoms import random, randrange
from workload.txn_factory.context import TxnContext


def build_nftoken_mint(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an NFTokenMint transaction with random account."""
    src = ctx.rand_account()
    memo_msg = "Some really cool info no doubt"
    memo = Memo(memo_data=memo_msg.encode("utf-8").hex())
    return {
        "TransactionType": "NFTokenMint",
        "Account": src.address,
        "NFTokenTaxon": 0,
        "memos": [memo],
    }


def build_nftoken_burn(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an NFTokenBurn transaction to burn a random NFT.

    Requires at least one NFT to exist in tracking.
    """
    if not ctx.nfts:
        raise RuntimeError("No NFTs available to burn")

    nft_id, owner = choice(list(ctx.nfts.items()))

    return {
        "TransactionType": "NFTokenBurn",
        "Account": owner,
        "NFTokenID": nft_id,
    }


def build_nftoken_create_offer(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an NFTokenCreateOffer transaction to create a sell or buy offer.

    Randomly creates either:
    - Sell offer: owner offers to sell their NFT
    - Buy offer: non-owner offers to buy someone's NFT
    """
    is_sell_offer = random() < 0.5

    if is_sell_offer:
        if not ctx.nfts:
            raise RuntimeError("No NFTs available to create sell offer")

        nft_id, owner = choice(list(ctx.nfts.items()))

        return {
            "TransactionType": "NFTokenCreateOffer",
            "Account": owner,
            "NFTokenID": nft_id,
            "Amount": str(randrange(1_000_000, 100_000_000)),  # 1-100 XRP in drops
            "Flags": 1,  # tfSellNFToken flag
        }
    else:
        if not ctx.nfts:
            raise RuntimeError("No NFTs available to create buy offer")

        nft_id, _owner = choice(list(ctx.nfts.items()))
        buyer = ctx.rand_account()

        return {
            "TransactionType": "NFTokenCreateOffer",
            "Account": buyer.address,
            "NFTokenID": nft_id,
            "Amount": str(randrange(1_000_000, 100_000_000)),  # 1-100 XRP in drops
            "Owner": _owner,  # Owner of the NFT (required for buy offers)
        }


def build_nftoken_cancel_offer(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an NFTokenCancelOffer transaction to cancel an existing offer.

    Requires at least one NFT offer to exist in tracking.
    """
    if not ctx.offers:
        raise RuntimeError("No NFT offers available to cancel")

    nft_offers = {k: v for k, v in ctx.offers.items() if v.get("type") == "NFTokenOffer"}
    if not nft_offers:
        raise RuntimeError("No NFT offers available to cancel")

    offer_id, offer_data = choice(list(nft_offers.items()))

    return {
        "TransactionType": "NFTokenCancelOffer",
        "Account": offer_data["owner"],
        "NFTokenOffers": [offer_id],  # Can cancel multiple offers in one txn
    }


def build_nftoken_accept_offer(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an NFTokenAcceptOffer transaction to accept an existing offer.

    Requires at least one NFT offer to exist in tracking.
    """
    if not ctx.offers:
        raise RuntimeError("No NFT offers available to accept")

    nft_offers = {k: v for k, v in ctx.offers.items() if v.get("type") == "NFTokenOffer"}
    if not nft_offers:
        raise RuntimeError("No NFT offers available to accept")

    offer_id, offer_data = choice(list(nft_offers.items()))

    if offer_data.get("is_sell_offer"):
        acceptor = ctx.rand_account()
        return {
            "TransactionType": "NFTokenAcceptOffer",
            "Account": acceptor.address,
            "NFTokenSellOffer": offer_id,
        }
    else:
        nft_id = offer_data.get("nft_id")
        if nft_id and nft_id in (ctx.nfts or {}):
            owner = ctx.nfts[nft_id]
            return {
                "TransactionType": "NFTokenAcceptOffer",
                "Account": owner,
                "NFTokenBuyOffer": offer_id,
            }
        else:
            acceptor = ctx.rand_account()
            return {
                "TransactionType": "NFTokenAcceptOffer",
                "Account": acceptor.address,
                "NFTokenBuyOffer": offer_id,
            }


BUILDERS = {
    "NFTokenMint": (build_nftoken_mint, NFTokenMint),
    "NFTokenBurn": (build_nftoken_burn, NFTokenBurn),
    "NFTokenCreateOffer": (build_nftoken_create_offer, NFTokenCreateOffer),
    "NFTokenCancelOffer": (build_nftoken_cancel_offer, NFTokenCancelOffer),
    "NFTokenAcceptOffer": (build_nftoken_accept_offer, NFTokenAcceptOffer),
}


# ---------------------------------------------------------------------------
# Tainting strategies
# ---------------------------------------------------------------------------


def _nftoken_burn_bad_id(tx: dict) -> dict:
    """NFTokenBurn with nonexistent token — tecNO_ENTRY."""
    tx["NFTokenID"] = "0" * 64
    return tx


TAINTERS = {
    "NFTokenBurn": [_nftoken_burn_bad_id],
}
