"""Vault transaction builders."""

from __future__ import annotations

from random import choice
from typing import TYPE_CHECKING

from xrpl.models.transactions import (
    VaultClawback,
    VaultCreate,
    VaultDelete,
    VaultDeposit,
    VaultSet,
    VaultWithdraw,
)

from workload.constants import TxIntent
from workload.randoms import random, randrange
from workload.txn_factory.context import TxnContext

if TYPE_CHECKING:
    from xrpl.wallet import Wallet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_hex(n: int) -> str:
    """Generate a random hex string of *n* bytes."""
    return bytes(randrange(256) for _ in range(n)).hex()


def _random_vault_asset(ctx: TxnContext) -> dict:
    """Pick a random asset suitable for vault creation: IOU, MPT, or XRP."""
    roll = random()
    if ctx.currencies and roll < 0.5:
        cur = ctx.rand_currency()
        return {"currency": cur.currency, "issuer": cur.issuer}
    if ctx.mptoken_issuance_ids and roll < 0.8:
        return {"mpt_issuance_id": ctx.rand_mptoken_id()}
    if ctx.currencies:
        cur = ctx.rand_currency()
        return {"currency": cur.currency, "issuer": cur.issuer}
    return {"currency": "XRP"}


def _vault_amount_for_asset(asset: dict, cfg: dict) -> str | dict:
    """Create an Amount matching a vault's asset type, ranges from config."""
    vcfg = cfg.get("transactions", {}).get("vault_deposit", {})
    if "mpt_issuance_id" in asset:
        lo = vcfg.get("mpt_amount_min", 1)
        hi = vcfg.get("mpt_amount_max", 10_000)
        return {"mpt_issuance_id": asset["mpt_issuance_id"], "value": str(randrange(lo, hi + 1))}
    if asset.get("currency") and asset.get("issuer"):
        lo = vcfg.get("iou_amount_min", 1)
        hi = vcfg.get("iou_amount_max", 10_000)
        return {"currency": asset["currency"], "issuer": asset["issuer"], "value": str(randrange(lo, hi + 1))}
    # XRP — drops
    lo = vcfg.get("xrp_drops_min", 1_000_000)
    hi = vcfg.get("xrp_drops_max", 100_000_000)
    return str(randrange(lo, hi + 1))


def _eligible_vaults_for_account(wallet: Wallet, ctx: TxnContext) -> list[dict]:
    """Return vaults this account can deposit to (has the vault's asset)."""
    if not ctx.vaults:
        return []
    account_currencies = {(c.currency, c.issuer) for c in ctx.get_account_currencies(wallet)}
    eligible = []
    for v in ctx.vaults:
        asset = v.get("asset", {})
        if "mpt_issuance_id" in asset:
            eligible.append(v)  # No MPT balance tracking yet — assume eligible
        elif asset.get("currency") and asset.get("issuer"):
            # IOU vault: account needs a trust line with balance
            if (asset["currency"], asset["issuer"]) in account_currencies:
                eligible.append(v)
        else:
            eligible.append(v)  # XRP vault: any account can deposit
    return eligible


def _ineligible_vault_for_account(wallet: Wallet, ctx: TxnContext) -> dict | None:
    """Return a vault this account CANNOT deposit to, or None if all are eligible."""
    if not ctx.vaults:
        return None
    eligible_ids = {id(v) for v in _eligible_vaults_for_account(wallet, ctx)}
    ineligible = [v for v in ctx.vaults if id(v) not in eligible_ids]
    return choice(ineligible) if ineligible else None


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_vault_create(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build a VaultCreate — open a new vault for an asset.

    Includes AssetsMaximum (100M-10B drops) and Data (1-256 bytes hex),
    matching upstream params.py ranges.
    """
    src = ctx.rand_account()
    asset = _random_vault_asset(ctx)
    vcfg = ctx.config.get("transactions", {}).get("vault_create", {})
    am_min = vcfg.get("assets_maximum_min", 100_000_000)
    am_max = vcfg.get("assets_maximum_max", 10_000_000_000)
    data_max = vcfg.get("data_max_bytes", 256)
    return {
        "TransactionType": "VaultCreate",
        "Account": src.address,
        "Asset": asset,
        "AssetsMaximum": str(randrange(am_min, am_max + 1)),
        "Data": _random_hex(randrange(1, data_max + 1)),
    }


def build_vault_set(ctx: TxnContext, intent: TxIntent) -> dict | None:  # TODO: handle TxIntent.INVALID
    """Build a VaultSet — owner updates vault settings.

    Includes AssetsMaximum and Data, matching upstream params.py ranges.
    """
    if not ctx.vaults:
        return None
    vault_owners = {v["owner"] for v in ctx.vaults}
    src = ctx.rand_owner(vault_owners)
    if src is None:
        return None
    owned = [v for v in ctx.vaults if v["owner"] == src.address]
    if not owned:
        return None
    vault = choice(owned)
    vcfg = ctx.config.get("transactions", {}).get("vault_create", {})
    am_min = vcfg.get("assets_maximum_min", 100_000_000)
    am_max = vcfg.get("assets_maximum_max", 10_000_000_000)
    data_max = vcfg.get("data_max_bytes", 256)
    return {
        "TransactionType": "VaultSet",
        "Account": src.address,
        "VaultID": vault["vault_id"],
        "AssetsMaximum": str(randrange(am_min, am_max + 1)),
        "Data": _random_hex(randrange(1, data_max + 1)),
    }


def build_vault_delete(ctx: TxnContext, intent: TxIntent) -> dict | None:  # TODO: handle TxIntent.INVALID
    """Build a VaultDelete — owner removes a vault."""
    if not ctx.vaults:
        return None
    vault_owners = {v["owner"] for v in ctx.vaults}
    src = ctx.rand_owner(vault_owners)
    if src is None:
        return None
    owned = [v for v in ctx.vaults if v["owner"] == src.address]
    if not owned:
        return None
    vault = choice(owned)
    return {
        "TransactionType": "VaultDelete",
        "Account": src.address,
        "VaultID": vault["vault_id"],
    }


def build_vault_deposit(ctx: TxnContext, intent: TxIntent) -> dict | None:
    """Build a VaultDeposit — deposit assets into an existing vault.

    VALID: picks a vault the account can actually deposit to (has the asset).
    INVALID: deliberately picks a vault the account has no trust line for.
    """
    if not ctx.vaults:
        return None
    src = ctx.rand_account()
    match intent:
        case TxIntent.VALID:
            eligible = _eligible_vaults_for_account(src, ctx)
            if not eligible:
                return None
            vault = choice(eligible)
        case TxIntent.INVALID:
            vault = _ineligible_vault_for_account(src, ctx)
            if vault is None:
                vault = choice(ctx.vaults)  # fallback: can't find mismatch
    return {
        "TransactionType": "VaultDeposit",
        "Account": src.address,
        "VaultID": vault["vault_id"],
        "Amount": _vault_amount_for_asset(vault.get("asset", {}), ctx.config),
    }


def build_vault_withdraw(ctx: TxnContext, intent: TxIntent) -> dict | None:  # TODO: handle TxIntent.INVALID
    """Build a VaultWithdraw — owner withdraws from a vault.

    Amount matches vault's asset type, same ranges as deposit.
    """
    if not ctx.vaults:
        return None
    vault_owners = {v["owner"] for v in ctx.vaults}
    src = ctx.rand_owner(vault_owners)
    if src is None:
        return None
    owned = [v for v in ctx.vaults if v["owner"] == src.address]
    if not owned:
        return None
    vault = choice(owned)
    return {
        "TransactionType": "VaultWithdraw",
        "Account": src.address,
        "VaultID": vault["vault_id"],
        "Amount": _vault_amount_for_asset(vault.get("asset", {}), ctx.config),
    }


def build_vault_clawback(ctx: TxnContext, intent: TxIntent) -> dict | None:  # TODO: handle TxIntent.INVALID
    """Build a VaultClawback — vault owner claws back from a holder.

    Matches upstream branch: Account = vault owner, Holder = random other account.
    No Amount field (claws back all).
    """
    if not ctx.vaults:
        return None
    vault_owners = {v["owner"] for v in ctx.vaults}
    src = ctx.rand_owner(vault_owners)
    if src is None:
        return None
    owned = [v for v in ctx.vaults if v["owner"] == src.address]
    if not owned:
        return None
    vault = choice(owned)
    holder = ctx.rand_account(omit=[src.address])
    return {
        "TransactionType": "VaultClawback",
        "Account": src.address,
        "VaultID": vault["vault_id"],
        "Holder": holder.address,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BUILDERS = {
    "VaultCreate": (build_vault_create, VaultCreate),
    "VaultSet": (build_vault_set, VaultSet),
    "VaultDelete": (build_vault_delete, VaultDelete),
    "VaultDeposit": (build_vault_deposit, VaultDeposit),
    "VaultWithdraw": (build_vault_withdraw, VaultWithdraw),
    "VaultClawback": (build_vault_clawback, VaultClawback),
}


# ---------------------------------------------------------------------------
# Per-account eligibility predicates
# ---------------------------------------------------------------------------


def _is_eligible_vault_owner_op(wallet, ctx) -> bool:
    """Wallet must own a vault."""
    return bool(ctx.vaults and any(v["owner"] == wallet.address for v in ctx.vaults))


def _is_eligible_vault_deposit(wallet, ctx) -> bool:
    """Wallet must be able to deposit to at least one vault."""
    return bool(_eligible_vaults_for_account(wallet, ctx))


ELIGIBILITY = {
    "VaultSet": _is_eligible_vault_owner_op,
    "VaultDelete": _is_eligible_vault_owner_op,
    "VaultWithdraw": _is_eligible_vault_owner_op,
    "VaultClawback": _is_eligible_vault_owner_op,
    "VaultDeposit": _is_eligible_vault_deposit,
}


# ---------------------------------------------------------------------------
# Tainting strategies
# ---------------------------------------------------------------------------


def _vault_deposit_bad_id(tx: dict) -> dict:
    """VaultDeposit with nonexistent vault — tecOBJECT_NOT_FOUND."""
    tx["VaultID"] = "0" * 64
    return tx


def _vault_set_bad_id(tx: dict) -> dict:
    """VaultSet with nonexistent vault — tecOBJECT_NOT_FOUND."""
    tx["VaultID"] = "0" * 64
    return tx


TAINTERS = {
    "VaultDeposit": [_vault_deposit_bad_id],
    "VaultSet": [_vault_set_bad_id],
}
