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
    """Build an MPTokenIssuanceSet transaction to modify MPToken properties.

    Only the issuer can call MPTokenIssuanceSet — eligibility predicate ensures
    forced_account is an issuer, so we pick one of their issuances.
    """
    src = ctx.forced_account or ctx.rand_account()
    own_ids = [mid for mid, iss in ctx.mptoken_issuance_ids.items() if iss == src.address]
    if not own_ids:
        return None
    mpt_id = choice(own_ids)

    # Must set at least one flag — alternate between lock (0x01) and unlock (0x02)
    flag = choice([0x00000001, 0x00000002])  # tfMPTLock / tfMPTUnlock
    return {
        "TransactionType": "MPTokenIssuanceSet",
        "Account": src.address,
        "MPTokenIssuanceID": mpt_id,
        "Flags": flag,
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
    """Build an MPTokenIssuanceDestroy transaction to destroy an MPToken issuance.

    Only the issuer can destroy — eligibility predicate ensures forced_account is an issuer.
    """
    src = ctx.forced_account or ctx.rand_account()
    own_ids = [mid for mid, iss in ctx.mptoken_issuance_ids.items() if iss == src.address]
    if not own_ids:
        return None
    mpt_id = choice(own_ids)

    return {
        "TransactionType": "MPTokenIssuanceDestroy",
        "Account": src.address,
        "MPTokenIssuanceID": mpt_id,
    }


def _is_mpt_issuer(wallet, ctx: TxnContext) -> bool:
    """Account has at least one MPToken issuance."""
    return any(iss == wallet.address for iss in (ctx.mptoken_issuance_ids or {}).values())


ELIGIBILITY = {
    "MPTokenIssuanceSet": _is_mpt_issuer,
    "MPTokenIssuanceDestroy": _is_mpt_issuer,
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
