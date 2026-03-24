"""MPToken transaction builders."""

import json
from random import choice

from xrpl.models.transactions import (
    MPTokenAuthorize,
    MPTokenIssuanceCreate,
    MPTokenIssuanceDestroy,
    MPTokenIssuanceSet,
)

from workload.constants import TxIntent
from workload.txn_factory.context import TxnContext, token_metadata


def build_mptoken_issuance_create(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an MPTokenIssuanceCreate transaction with random account."""
    src = ctx.rand_account()
    metadata_hex = json.dumps(choice(token_metadata)).encode("utf-8").hex()
    return {
        "TransactionType": "MPTokenIssuanceCreate",
        "Account": src.address,
        "MPTokenMetadata": metadata_hex,
    }


def build_mptoken_issuance_set(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an MPTokenIssuanceSet transaction to modify MPToken properties."""
    src = ctx.rand_account()
    mpt_id = ctx.rand_mptoken_id()

    return {
        "TransactionType": "MPTokenIssuanceSet",
        "Account": src.address,
        "MPTokenIssuanceID": mpt_id,
    }


def build_mptoken_authorize(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an MPTokenAuthorize transaction to authorize/unauthorize holder."""
    src = ctx.rand_account()
    mpt_id = ctx.rand_mptoken_id()

    return {
        "TransactionType": "MPTokenAuthorize",
        "Account": src.address,
        "MPTokenIssuanceID": mpt_id,
    }


def build_mptoken_issuance_destroy(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build an MPTokenIssuanceDestroy transaction to destroy an MPToken issuance."""
    src = ctx.rand_account()
    mpt_id = ctx.rand_mptoken_id()

    return {
        "TransactionType": "MPTokenIssuanceDestroy",
        "Account": src.address,
        "MPTokenIssuanceID": mpt_id,
    }


BUILDERS = {
    "MPTokenIssuanceCreate": (build_mptoken_issuance_create, MPTokenIssuanceCreate),
    "MPTokenIssuanceSet": (build_mptoken_issuance_set, MPTokenIssuanceSet),
    "MPTokenAuthorize": (build_mptoken_authorize, MPTokenAuthorize),
    "MPTokenIssuanceDestroy": (build_mptoken_issuance_destroy, MPTokenIssuanceDestroy),
}


# ---------------------------------------------------------------------------
# Tainting strategies
# ---------------------------------------------------------------------------


def _mptoken_destroy_bad_id(tx: dict) -> dict:
    """MPTokenIssuanceDestroy with nonexistent ID — tecOBJECT_NOT_FOUND."""
    tx["MPTokenIssuanceID"] = "0" * 64
    return tx


TAINTERS = {
    "MPTokenIssuanceDestroy": [_mptoken_destroy_bad_id],
}
