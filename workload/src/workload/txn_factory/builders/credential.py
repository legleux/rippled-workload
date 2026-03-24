"""Credential and Delegation transaction builders."""

from random import choice, sample

from xrpl.models.transactions import (
    CredentialAccept,
    CredentialCreate,
    CredentialDelete,
    DelegateSet,
)
from xrpl.models.transactions.delegate_set import GranularPermission
from xrpl.models.transactions.deposit_preauth import Credential as XRPLCredential  # noqa: F401

from workload.constants import TxIntent
from workload.randoms import randrange
from workload.txn_factory.context import TxnContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_hex(n: int) -> str:
    """Generate a random hex string of *n* bytes."""
    return bytes(randrange(256) for _ in range(n)).hex()


def _random_credential_type(cfg: dict) -> str:
    """Random hex-encoded credential type, length from config."""
    max_bytes = cfg.get("transactions", {}).get("credential_create", {}).get("credential_type_max_bytes", 64)
    return _random_hex(randrange(1, max_bytes + 1))


def _random_credential_uri(cfg: dict) -> str:
    """Random hex-encoded URI for credentials, length from config."""
    max_bytes = cfg.get("transactions", {}).get("credential_create", {}).get("uri_max_bytes", 256)
    return _random_hex(randrange(10, max_bytes + 1))


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


def build_delegate_set(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build a DelegateSet transaction to delegate permissions to another account.

    Uses GranularPermission enum values from xrpl-py, matching upstream branch.
    Picks 1-3 random permissions from the 12 available granular permissions.
    """
    src, delegate = ctx.rand_accounts(2)
    all_perms = list(GranularPermission)
    max_p = ctx.config.get("transactions", {}).get("delegate_set", {}).get("max_permissions", 3)
    num_perms = min(len(all_perms), randrange(1, max_p + 1))
    selected = sample(all_perms, num_perms)
    permissions = [{"Permission": {"PermissionValue": p.value}} for p in selected]
    return {
        "TransactionType": "DelegateSet",
        "Account": src.address,
        "Authorize": delegate.address,
        "Permissions": permissions,
    }


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def build_credential_create(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build a CredentialCreate — issuer attests about a subject.

    Includes Expiration (1 hour – 30 days from now) and URI, matching upstream params.py ranges.
    """
    import time

    issuer = ctx.rand_account()
    subject = ctx.rand_account(omit=[issuer.address])
    ccfg = ctx.config.get("transactions", {}).get("credential_create", {})
    # Expiration: Ripple epoch = Unix epoch - 946684800
    ripple_epoch_offset = 946684800
    exp_min = ccfg.get("expiration_min_offset", 3600)
    exp_max = ccfg.get("expiration_max_offset", 2592000)
    expiration = int(time.time()) - ripple_epoch_offset + randrange(exp_min, exp_max + 1)
    return {
        "TransactionType": "CredentialCreate",
        "Account": issuer.address,
        "Subject": subject.address,
        "CredentialType": _random_credential_type(ctx.config),
        "Expiration": expiration,
        "URI": _random_credential_uri(ctx.config),
    }


def build_credential_accept(ctx: TxnContext, intent: TxIntent) -> dict | None:  # TODO: handle TxIntent.INVALID
    """Build a CredentialAccept — subject accepts an issued credential."""
    if not ctx.credentials:
        return None
    unaccepted = [c for c in ctx.credentials if not c.get("accepted")]
    if not unaccepted:
        return None
    subjects = {c["subject"] for c in unaccepted}
    src = ctx.rand_owner(subjects)
    if src is None:
        return None
    eligible = [c for c in unaccepted if c["subject"] == src.address]
    if not eligible:
        return None
    cred = choice(eligible)
    return {
        "TransactionType": "CredentialAccept",
        "Account": src.address,
        "Issuer": cred["issuer"],
        "CredentialType": cred["credential_type"],
    }


def build_credential_delete(ctx: TxnContext, intent: TxIntent) -> dict | None:  # TODO: handle TxIntent.INVALID
    """Build a CredentialDelete — issuer or subject removes a credential."""
    if not ctx.credentials:
        return None
    participants = {c["issuer"] for c in ctx.credentials} | {c["subject"] for c in ctx.credentials}
    src = ctx.rand_owner(participants)
    if src is None:
        return None
    eligible = [c for c in ctx.credentials if c["issuer"] == src.address or c["subject"] == src.address]
    if not eligible:
        return None
    cred = choice(eligible)
    return {
        "TransactionType": "CredentialDelete",
        "Account": src.address,
        "Subject": cred["subject"],
        "Issuer": cred["issuer"],
        "CredentialType": cred["credential_type"],
    }


BUILDERS = {
    "DelegateSet": (build_delegate_set, DelegateSet),
    "CredentialCreate": (build_credential_create, CredentialCreate),
    "CredentialAccept": (build_credential_accept, CredentialAccept),
    "CredentialDelete": (build_credential_delete, CredentialDelete),
}


# ---------------------------------------------------------------------------
# Per-account eligibility predicates
# ---------------------------------------------------------------------------


def _is_eligible_credential_accept(wallet, ctx) -> bool:
    """Wallet must be subject of an unaccepted credential."""
    return bool(
        ctx.credentials and any(c["subject"] == wallet.address and not c.get("accepted") for c in ctx.credentials)
    )


def _is_eligible_credential_delete(wallet, ctx) -> bool:
    """Wallet must be issuer or subject of a credential."""
    return bool(
        ctx.credentials
        and any(c["issuer"] == wallet.address or c["subject"] == wallet.address for c in ctx.credentials)
    )


ELIGIBILITY = {
    "CredentialAccept": _is_eligible_credential_accept,
    "CredentialDelete": _is_eligible_credential_delete,
}


# ---------------------------------------------------------------------------
# Tainting strategies
# ---------------------------------------------------------------------------


def _credential_create_self(tx: dict) -> dict:
    """CredentialCreate with subject=self — temMALFORMED."""
    tx["Subject"] = tx["Account"]
    return tx


TAINTERS = {
    "CredentialCreate": [_credential_create_self],
}
