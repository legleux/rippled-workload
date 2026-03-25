"""Transaction builder registry — collects builders from all builder modules.

Provides the public API for transaction generation:
  - build_txn_dict(): build a transaction dict from type name
  - txn_model_cls(): get the xrpl-py model class for a type
  - generate_txn(): build + convert to xrpl-py model
  - pick_eligible_txn_type(): legacy convenience wrapper
  - global_eligible_types(): global capability filters (called once per submission set)
  - is_account_eligible(): per-account eligibility check for a specific type
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from random import choices, shuffle
from typing import Any

from workload.randoms import random

from xrpl.models.transactions import Transaction
from xrpl.transaction import transaction_json_to_binary_codec_form
from xrpl.wallet import Wallet

from workload.constants import TxIntent
from workload.txn_factory.builders import batch, check, credential, dex, domain, escrow, mptoken, nft, payment, vault
from workload.txn_factory.context import TxnContext, deep_update

log = logging.getLogger("workload.txn")

# ---------------------------------------------------------------------------
# Collect BUILDERS from all builder modules
# ---------------------------------------------------------------------------

BuilderFn = Callable[[TxnContext, TxIntent], dict | None]

_BUILDERS: dict[str, tuple[BuilderFn, type[Transaction]]] = {}
_ELIGIBILITY: dict[str, Callable[[Wallet, TxnContext], bool]] = {}

_MODULES = [payment, dex, nft, mptoken, vault, credential, domain, batch, check, escrow]

for _mod in _MODULES:
    _BUILDERS.update(_mod.BUILDERS)
    _ELIGIBILITY.update(getattr(_mod, "ELIGIBILITY", {}))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_txn_dict(txn_type: str, ctx: TxnContext, intent: TxIntent = TxIntent.VALID) -> dict | None:
    """Call the synchronous builder for txn_type and return the raw dict (or None if ineligible)."""
    builder_fn, _ = _BUILDERS[txn_type]
    return builder_fn(ctx, intent)


def txn_model_cls(txn_type: str) -> type[Transaction]:
    """Return the xrpl-py model class for txn_type."""
    _, model_cls = _BUILDERS[txn_type]
    return model_cls


def global_eligible_types(ctx: TxnContext) -> list[str]:
    """Return transaction types that pass global capability filters.

    Called once per submission set — does NOT check per-account eligibility.
    """
    candidates = list(_BUILDERS.keys())

    disabled = (
        ctx.disabled_types
        if ctx.disabled_types is not None
        else set(ctx.config.get("transactions", {}).get("disabled", []))
    )
    if disabled:
        candidates = [t for t in candidates if t not in disabled]

    if not ctx.mptoken_issuance_ids:
        mpt_ops = {"MPTokenAuthorize", "MPTokenIssuanceSet", "MPTokenIssuanceDestroy"}
        candidates = [t for t in candidates if t not in mpt_ops]

    if not ctx.nfts:
        candidates = [t for t in candidates if t not in {"NFTokenBurn", "NFTokenCreateOffer"}]

    has_nft_offers = ctx.offers and any(v.get("type") == "NFTokenOffer" for v in ctx.offers.values())
    if not has_nft_offers:
        candidates = [t for t in candidates if t not in {"NFTokenCancelOffer", "NFTokenAcceptOffer"}]

    if not ctx.amm_pool_registry:
        candidates = [t for t in candidates if t not in {"AMMDeposit", "AMMWithdraw"}]

    if not ctx.credentials:
        candidates = [t for t in candidates if t not in {"CredentialAccept", "CredentialDelete"}]

    if not ctx.domains:
        candidates = [t for t in candidates if t != "PermissionedDomainDelete"]

    if not ctx.vaults:
        vault_ops = {"VaultSet", "VaultDelete", "VaultDeposit", "VaultWithdraw", "VaultClawback"}
        candidates = [t for t in candidates if t not in vault_ops]

    if not ctx.tickets:
        candidates = [t for t in candidates if t != "TicketUse"]

    if not ctx.checks:
        candidates = [t for t in candidates if t not in {"CheckCash", "CheckCancel"}]

    if not ctx.escrows:
        candidates = [t for t in candidates if t not in {"EscrowFinish", "EscrowCancel"}]

    # Batch uses async builder with multi-seq allocation — not supported in sync path
    candidates = [t for t in candidates if t != "Batch"]

    return candidates


def is_account_eligible(wallet: Wallet, txn_type: str, ctx: TxnContext) -> bool:
    """Check if a specific wallet can submit a specific transaction type.

    Returns True for types with no per-account requirements.
    Delegates to ELIGIBILITY predicates from builder modules.
    """
    pred = _ELIGIBILITY.get(txn_type)
    if pred is None:
        return True  # No per-account filter for this type
    return pred(wallet, ctx)


def _compute_weights(eligible_types: list[str], config: dict) -> list[float]:
    """Compute weighted sampling weights for eligible types from config percentages."""
    percentages = config.get("transactions", {}).get("percentages", {})
    defined_total = sum(percentages.get(t, 0) for t in eligible_types)
    remaining = 1.0 - defined_total
    undefined_types = [t for t in eligible_types if t not in percentages]
    per_undefined = remaining / len(undefined_types) if undefined_types else 0
    return [percentages.get(t, per_undefined) for t in eligible_types]


def compose_submission_set(
    free_accounts: list[str],
    clean_accounts: list[str],
    target: int,
    ctx: TxnContext,
    config: dict,
) -> list[tuple[str, str, TxIntent]]:
    """Compose a submission set using type-first assignment.

    1. Determine set size = min(target, len(free_accounts))
    2. Roll N types from weighted distribution (global filters applied)
    3. For each type, roll intent using per-type invalid ratio
    4. Match accounts: INVALID → clean account only (0 pending); VALID → any free account

    Args:
        free_accounts: All accounts below max_pending threshold (clean + partial).
        clean_accounts: Accounts with 0 pending txns. Only these can receive
            INVALID-intent txns — tainted txns must not queue behind in-flight
            txns or they cause tefPAST_SEQ cascades.

    Returns list of (account_address, txn_type, intent) assignments.
    """
    eligible_types = global_eligible_types(ctx)
    if not eligible_types:
        return []

    set_size = min(target, len(free_accounts))
    if set_size == 0:
        return []

    weights = _compute_weights(eligible_types, config)
    rolled_types = choices(eligible_types, weights=weights, k=set_size)

    # Per-type intent ratios
    intent_cfg = config.get("transactions", {}).get("intent", {})
    global_invalid = intent_cfg.get("invalid", 0.10)
    per_type_ratios = intent_cfg.get("per_type", {})

    available = list(free_accounts)
    shuffle(available)
    # Clean pool for INVALID intent — accounts with 0 pending txns only
    clean_set = set(clean_accounts)
    assignments: list[tuple[str, str, TxIntent]] = []

    for txn_type in rolled_types:
        if not available:
            break

        invalid_ratio = per_type_ratios.get(txn_type, global_invalid)
        intent = TxIntent.INVALID if random() < invalid_ratio else TxIntent.VALID

        if intent == TxIntent.INVALID:
            # Only clean accounts (0 pending) for invalid txns — prevents tefPAST_SEQ cascade
            picked = None
            for i, addr in enumerate(available):
                if addr in clean_set:
                    picked = available.pop(i)
                    clean_set.discard(picked)
                    break
            if picked:
                assignments.append((picked, txn_type, intent))
            # If no clean account available, skip this invalid roll (don't downgrade to valid)
        else:
            # Find account eligible for this type
            for i, addr in enumerate(available):
                wallet = next((w for w in ctx.wallets if w.address == addr), None)
                if wallet and is_account_eligible(wallet, txn_type, ctx):
                    assignments.append((available.pop(i), txn_type, intent))
                    break

    return assignments


def pick_eligible_txn_type(wallet: Wallet, ctx: TxnContext, intent: TxIntent = TxIntent.VALID) -> str | None:
    """Return a weight-sampled eligible transaction type for wallet, or None if none available.

    Legacy convenience wrapper combining global_eligible_types + is_account_eligible.
    When intent is INVALID, per-account filters are skipped.
    """
    candidates = global_eligible_types(ctx)

    # Per-account filters: only apply for VALID intent
    if intent == TxIntent.VALID:
        candidates = [t for t in candidates if is_account_eligible(wallet, t, ctx)]

    if not candidates:
        return None

    percentages = ctx.config.get("transactions", {}).get("percentages", {})
    defined_total = sum(percentages.get(t, 0) for t in candidates)
    remaining = 1.0 - defined_total
    undefined_types = [t for t in candidates if t not in percentages]
    per_undefined = remaining / len(undefined_types) if undefined_types else 0
    weights = [percentages.get(t, per_undefined) for t in candidates]
    return choices(candidates, weights=weights, k=1)[0]


async def generate_txn(
    ctx: TxnContext, txn_type: str | None = None, intent: TxIntent = TxIntent.VALID, **overrides: Any
) -> Transaction | None:
    """Generate a transaction with sane defaults.

    Args:
        ctx: Transaction context with wallets, currencies, and defaults
        txn_type: Transaction type name (e.g., "Payment", "TrustSet").
                 If None, picks a random available type.
        **overrides: Additional fields to override in the transaction

    Returns:
        A fully formed Transaction model ready to sign and submit, or None if ineligible.
    """
    if txn_type is None:
        configured_types = global_eligible_types(ctx)
        if not configured_types:
            raise RuntimeError("No transaction types available to generate")

        percentages = ctx.config.get("transactions", {}).get("percentages", {})
        defined_total = sum(percentages.get(t, 0) for t in configured_types)
        remaining = 1.0 - defined_total
        undefined_types = [t for t in configured_types if t not in percentages]
        per_undefined = remaining / len(undefined_types) if undefined_types else 0
        weights = [percentages.get(t, per_undefined) for t in configured_types]
        txn_type = choices(configured_types, weights=weights, k=1)[0]
    else:
        if txn_type not in _BUILDERS:
            for builder_type in _BUILDERS:
                if builder_type.lower() == str(txn_type).lower():
                    txn_type = builder_type
                    break

    log.debug("Generating %s txn", txn_type)

    builder_spec = _BUILDERS.get(txn_type)
    if not builder_spec:
        raise ValueError(f"Unsupported txn_type: {txn_type}")

    builder_fn, model_cls = builder_spec

    if inspect.iscoroutinefunction(builder_fn):
        composed = await builder_fn(ctx, intent)
    else:
        composed = builder_fn(ctx, intent)

    if composed is None:
        log.debug("Builder for %s returned None (account ineligible) — skipping", txn_type)
        return None

    if overrides:
        deep_update(composed, transaction_json_to_binary_codec_form(overrides))

    log.debug("Created %s", txn_type)
    return model_cls.from_xrpl(composed)


# ---------------------------------------------------------------------------
# Convenience factories (used by API endpoints)
# ---------------------------------------------------------------------------


async def create_payment(ctx: TxnContext, **overrides: Any) -> Transaction | None:
    """Create a Payment transaction with sane defaults."""
    return await generate_txn(ctx, "Payment", **overrides)


async def create_xrp_payment(ctx: TxnContext, **overrides: Any) -> Transaction | None:
    """Create an XRP-only Payment transaction."""
    from random import sample as stdlib_sample

    wl = list(ctx.wallets)
    if len(wl) >= 2:
        src, dst = stdlib_sample(wl, 2)
    else:
        src = wl[0] if wl else ctx.funding_wallet
        dst = ctx.funding_wallet if ctx.funding_wallet is not src else src

    amount = str(ctx.config["transactions"]["payment"]["amount"])
    return await generate_txn(ctx, "Payment", Account=src.address, Destination=dst.address, Amount=amount, **overrides)


async def create_trustset(ctx: TxnContext, **overrides: Any) -> Transaction | None:
    return await generate_txn(ctx, "TrustSet", **overrides)


async def create_accountset(ctx: TxnContext, **overrides: Any) -> Transaction | None:
    return await generate_txn(ctx, "AccountSet", **overrides)


async def create_nftoken_mint(ctx: TxnContext, **overrides: Any) -> Transaction | None:
    return await generate_txn(ctx, "NFTokenMint", **overrides)


async def create_mptoken_issuance_create(ctx: TxnContext, **overrides: Any) -> Transaction | None:
    return await generate_txn(ctx, "MPTokenIssuanceCreate", **overrides)


async def create_batch(ctx: TxnContext, **overrides: Any) -> Transaction | None:
    return await generate_txn(ctx, "Batch", **overrides)


async def create_amm_create(ctx: TxnContext, **overrides: Any) -> Transaction | None:
    return await generate_txn(ctx, "AMMCreate", **overrides)


def update_transaction(transaction: Transaction, **kwargs) -> Transaction:
    """Update an existing transaction with new fields."""
    payload = transaction.to_xrpl()
    payload.update(kwargs)
    return type(transaction).from_xrpl(payload)
