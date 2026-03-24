"""Permissioned Domain transaction builders."""

from random import choice

from xrpl.models.transactions import (
    PermissionedDomainDelete,
    PermissionedDomainSet,
)
from xrpl.models.transactions.deposit_preauth import Credential as XRPLCredential  # noqa: F401

from workload.constants import TxIntent
from workload.randoms import random, randrange
from workload.txn_factory.builders.credential import _random_credential_type, _random_hex  # noqa: F401
from workload.txn_factory.context import TxnContext


def build_permissioned_domain_set(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build a PermissionedDomainSet — create or update a permissioned domain.

    Accepts 1-10 credential definitions (matching upstream params.domain_credential_count).
    Each credential references a random account as issuer with a random credential type.
    """
    src = ctx.rand_account()
    dcfg = ctx.config.get("transactions", {}).get("permissioned_domain_set", {})
    max_creds = dcfg.get("max_credentials", 10)
    num_creds = randrange(1, max_creds + 1)
    accepted = [
        {
            "Credential": {
                "Issuer": ctx.rand_account().address,
                "CredentialType": _random_credential_type(ctx.config),
            }
        }
        for _ in range(num_creds)
    ]
    result = {
        "TransactionType": "PermissionedDomainSet",
        "Account": src.address,
        "AcceptedCredentials": accepted,
    }
    # Optionally update an existing domain owned by src
    if ctx.domains and random() < 0.3:
        owned = [d for d in ctx.domains if d["owner"] == src.address]
        if owned:
            result["DomainID"] = choice(owned)["domain_id"]
    return result


def build_permissioned_domain_delete(ctx: TxnContext, intent: TxIntent) -> dict | None:
    # TODO: handle TxIntent.INVALID
    """Build a PermissionedDomainDelete — owner removes a domain."""
    if not ctx.domains:
        return None
    owners = {d["owner"] for d in ctx.domains}
    src = ctx.rand_owner(owners)
    if src is None:
        return None
    owned = [d for d in ctx.domains if d["owner"] == src.address]
    if not owned:
        return None
    domain = choice(owned)
    return {
        "TransactionType": "PermissionedDomainDelete",
        "Account": src.address,
        "DomainID": domain["domain_id"],
    }


BUILDERS = {
    "PermissionedDomainSet": (build_permissioned_domain_set, PermissionedDomainSet),
    "PermissionedDomainDelete": (build_permissioned_domain_delete, PermissionedDomainDelete),
}


# ---------------------------------------------------------------------------
# Per-account eligibility predicates
# ---------------------------------------------------------------------------


def _is_eligible_domain_delete(wallet, ctx) -> bool:
    """Wallet must own a permissioned domain."""
    return bool(ctx.domains and any(d["owner"] == wallet.address for d in ctx.domains))


ELIGIBILITY = {
    "PermissionedDomainDelete": _is_eligible_domain_delete,
}
