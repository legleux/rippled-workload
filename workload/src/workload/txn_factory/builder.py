from collections.abc import Sequence, Iterable, Callable, Awaitable
from dataclasses import dataclass, replace
from random import choice, sample
from typing import TypeVar, Any
import json
from xrpl.wallet import Wallet
from xrpl.models import IssuedCurrency, TransactionFlag
from xrpl.models.amounts import IssuedCurrencyAmount
from xrpl.transaction import transaction_json_to_binary_codec_form
from xrpl.models.transactions import (
    AccountSet,
    AMMCreate,
    Batch,
    BatchFlag,
    MPTokenIssuanceCreate,
    MPTokenIssuanceSet,
    MPTokenAuthorize,
    MPTokenIssuanceDestroy,
    NFTokenMint,
    Transaction,
    Memo,
    Payment,
    TrustSet,
)

from workload.randoms import randrange
import logging

log = logging.getLogger("workload.txn")

T = TypeVar("T")


# Transaction types are registered in _BUILDERS (single source of truth)
# Config can disable specific types via transactions.disabled = [...]


def choice_omit(seq: Sequence[T], omit: Iterable[T]) -> T:
    pool = [x for x in seq if x not in omit]
    if not pool:
        raise ValueError("No options left after excluding omits!")
    return choice(pool)


# Async helpers
AwaitInt = Callable[[], Awaitable[int]]
AwaitSeq = Callable[[str], Awaitable[int]]


@dataclass(slots=True)
class TxnContext:
    funding_wallet: "Wallet"
    wallets: Sequence["Wallet"]  # <-- sequence, not dict    currencies: Sequence[IssuedCurrency]
    currencies: Sequence[IssuedCurrency]
    config: dict  # Full config dict from config.toml
    base_fee_drops: "AwaitInt"
    next_sequence: "AwaitSeq"
    mptoken_issuance_ids: list[str] | None = None  # Track created MPToken issuance IDs

    def rand_account(self, omit: Wallet | None = None) -> "Wallet":
        return choice_omit(self.wallets, omit=[omit] if omit else [])

    def rand_currency(self) -> IssuedCurrency:
        if not self.currencies:
            raise RuntimeError("No currencies configured")
        return choice(self.currencies)

    def rand_mptoken_id(self) -> str:
        """Get a random MPToken issuance ID from tracked IDs."""
        if not self.mptoken_issuance_ids:
            raise RuntimeError("No MPToken issuance IDs available")
        return choice(self.mptoken_issuance_ids)

    def derive(self, **overrides) -> "TxnContext":
        return replace(self, **overrides)

    @classmethod
    def build(
        cls,
        *,
        funding_wallet: Wallet,
        wallets: Sequence[Wallet],
        currencies: Sequence[IssuedCurrency],
        config: dict,
        base_fee_drops: AwaitInt,
        next_sequence: AwaitSeq,
    ) -> "TxnContext":
        return cls(
            wallets=wallets,
            currencies=currencies,
            funding_wallet=funding_wallet,
            config=config,
            base_fee_drops=base_fee_drops,
            next_sequence=next_sequence,
        )


token_metadata = [
    dict(
        ticker="GOOSE",
        name="goosecoin",
        icon="https://🪿.com",  # This might not work...
        # icon="https://xn--n28h.com",
        asset_class="rwa",
        asset_subclass="commodity",
        issuer_name="Mother Goose",
    ),
]


def sample_omit(seq: Sequence[T], omit: T, k: int) -> list[T]:
    return sample([x for x in seq if x != omit], k)


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


# =============================================================================
# Internal builder functions - one per transaction type
# =============================================================================


def _build_payment(ctx: TxnContext) -> dict:
    """Build a Payment transaction with random source and destination."""
    wl = list(ctx.wallets)
    if len(wl) >= 2:
        src, dst = sample(wl, 2)
    else:
        # one or zero wallets: use funding wallet as dst; allow self as last resort
        src = wl[0] if wl else ctx.funding_wallet
        dst = ctx.funding_wallet if ctx.funding_wallet is not src else src

    # Randomly choose between XRP and issued currency
    # Use xrp_chance from config (default behavior: mostly issued currencies)
    from random import random

    use_xrp = random() < ctx.config.get("amm", {}).get("xrp_chance", 0.1)

    if use_xrp or not ctx.currencies:
        # Send XRP (in drops)
        amount = str(ctx.config["transactions"]["payment"]["amount"])
    else:
        # Send issued currency
        currency = ctx.rand_currency()
        amount = {
            "currency": currency.currency,
            "issuer": currency.issuer,
            "value": "100",  # 100 units of the currency
        }

    result = {
        "TransactionType": "Payment",
        "Account": src.address,
        "Destination": dst.address,
        "Amount": amount,
    }

    return result


def _build_trustset(ctx: TxnContext) -> dict:
    """Build a TrustSet transaction with random account and currency."""
    cur = ctx.rand_currency()
    src = ctx.rand_account()

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


def _build_accountset(ctx: TxnContext) -> dict:
    """Build an AccountSet transaction with random account."""
    src = ctx.rand_account()
    return {
        "TransactionType": "AccountSet",
        "Account": src.address,
    }


def _build_nftoken_mint(ctx: TxnContext) -> dict:
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


def _build_mptoken_issuance_create(ctx: TxnContext) -> dict:
    """Build an MPTokenIssuanceCreate transaction with random account."""
    src = ctx.rand_account()
    metadata_hex = json.dumps(choice(token_metadata)).encode("utf-8").hex()
    return {
        "TransactionType": "MPTokenIssuanceCreate",
        "Account": src.address,
        "MPTokenMetadata": metadata_hex,
    }


def _build_mptoken_issuance_set(ctx: TxnContext) -> dict:
    """Build an MPTokenIssuanceSet transaction to modify MPToken properties."""
    src = ctx.rand_account()
    mpt_id = ctx.rand_mptoken_id()

    return {
        "TransactionType": "MPTokenIssuanceSet",
        "Account": src.address,
        "MPTokenIssuanceID": mpt_id,
        # Optionally set holder to lock/unlock for specific account
        # "Holder": ctx.rand_account().address,
    }


def _build_mptoken_authorize(ctx: TxnContext) -> dict:
    """Build an MPTokenAuthorize transaction to authorize/unauthorize holder."""
    src = ctx.rand_account()
    mpt_id = ctx.rand_mptoken_id()

    return {
        "TransactionType": "MPTokenAuthorize",
        "Account": src.address,
        "MPTokenIssuanceID": mpt_id,
        # Holder can be specified to authorize a specific account
        # If omitted, authorizes the Account itself
    }


def _build_mptoken_issuance_destroy(ctx: TxnContext) -> dict:
    """Build an MPTokenIssuanceDestroy transaction to destroy an MPToken issuance."""
    src = ctx.rand_account()
    mpt_id = ctx.rand_mptoken_id()

    return {
        "TransactionType": "MPTokenIssuanceDestroy",
        "Account": src.address,
        "MPTokenIssuanceID": mpt_id,
    }


async def _build_batch(ctx: TxnContext) -> dict:
    """Build a Batch transaction with random inner transactions of various types."""
    from random import random

    src = ctx.rand_account()

    # 1. Random count (2-8 inner txns) - Batch requires minimum 2
    num_inner = randrange(2, 9)

    # 2. Allocate sequences: Batch gets first, inner txns get next N
    # IMPORTANT: Batch uses seq N, inner txns use N+1, N+2, N+3, ...
    batch_seq = await ctx.next_sequence(src.address)
    inner_sequences = [await ctx.next_sequence(src.address) for _ in range(num_inner)]

    # 3. Build random inner txns of different types
    inner_txns = []
    for seq in inner_sequences:
        # Pick random inner txn type
        txn_type = choice(["Payment", "TrustSet", "AccountSet", "NFTokenMint"])

        # Build based on type - ALL must have: fee="0", signing_pub_key="", TF_INNER_BATCH_TXN flag
        if txn_type == "Payment":
            # Mix XRP and issued currencies
            use_xrp = random() < 0.5
            if use_xrp or not ctx.currencies:
                amount = str(randrange(1_000_000, 100_000_000))  # 1-100 XRP in drops
            else:
                currency = ctx.rand_currency()
                amount = IssuedCurrencyAmount(
                    currency=currency.currency,
                    issuer=currency.issuer,
                    value=str(randrange(10, 1000)),
                )

            inner_tx = Payment(
                account=src.address,
                destination=choice_omit(ctx.wallets, [src]).address,
                amount=amount,
                fee="0",
                signing_pub_key="",
                flags=TransactionFlag.TF_INNER_BATCH_TXN,
                sequence=seq,
            )

        elif txn_type == "TrustSet":
            cur = ctx.rand_currency()
            inner_tx = TrustSet(
                account=src.address,
                limit_amount=IssuedCurrencyAmount(
                    currency=cur.currency,
                    issuer=cur.issuer,
                    value=str(ctx.config["transactions"]["trustset"]["limit"]),
                ),
                fee="0",
                signing_pub_key="",
                flags=TransactionFlag.TF_INNER_BATCH_TXN,
                sequence=seq,
            )

        elif txn_type == "AccountSet":
            inner_tx = AccountSet(
                account=src.address,
                fee="0",
                signing_pub_key="",
                flags=TransactionFlag.TF_INNER_BATCH_TXN,
                sequence=seq,
            )

        elif txn_type == "NFTokenMint":
            memo = Memo(memo_data="Batch NFT".encode("utf-8").hex())
            inner_tx = NFTokenMint(
                account=src.address,
                nftoken_taxon=0,
                fee="0",
                signing_pub_key="",
                flags=TransactionFlag.TF_INNER_BATCH_TXN,
                sequence=seq,
                memos=[memo],
            )

        inner_txns.append({"RawTransaction": inner_tx})

    # Randomly pick a batch mode for testing variety
    # tfAllOrNothing: all must succeed or batch fails
    # tfOnlyOne: first success wins, rest skipped
    # tfUntilFailure: apply until first failure
    # tfIndependent: all execute regardless of failures
    batch_mode = choice([
        BatchFlag.TF_ALL_OR_NOTHING,
        BatchFlag.TF_ONLY_ONE,
        BatchFlag.TF_UNTIL_FAILURE,
        BatchFlag.TF_INDEPENDENT,
    ])

    return {
        "TransactionType": "Batch",
        "Account": src.address,
        "Sequence": batch_seq,  # Explicitly set so build_sign_and_track won't allocate a new one
        "Flags": batch_mode,
        "RawTransactions": inner_txns,
    }


def _build_amm_create(ctx: TxnContext) -> dict:
    """Build an AMMCreate transaction with random currency pair.

    NOTE: Fee will be set to owner_reserve in build_sign_and_track based on TransactionType.
    """
    src = ctx.rand_account()
    currency = ctx.rand_currency()

    # AMM needs two assets - one XRP, one issued currency
    # Use values from config, but allow overrides
    return {
        "TransactionType": "AMMCreate",
        "Account": src.address,
        "Amount": "1000000000",  # 1000 XRP (in drops) - can be overridden
        "Amount2": {
            "currency": currency.currency,
            "issuer": currency.issuer,
            "value": str(ctx.config["amm"]["default_amm_token_deposit"]),  # From config
        },
        "TradingFee": ctx.config["amm"]["trading_fee"],  # From config
        # NOTE: Do NOT set Fee here - it must equal owner_reserve, which is set in build_sign_and_track
    }


# =============================================================================
# Dispatch table - maps transaction type to (builder_fn, model_class)
# To add a new transaction type:
#   1. Write a _build_newtype() function above
#   2. Add an entry here: "NewType": (_build_newtype, NewType),
#   3. Optionally add a create_newtype() convenience function below
# =============================================================================

_BUILDERS: dict[str, tuple[Callable[[TxnContext], dict], type[Transaction]]] = {
    "Payment": (_build_payment, Payment),
    "TrustSet": (_build_trustset, TrustSet),
    "AccountSet": (_build_accountset, AccountSet),
    "NFTokenMint": (_build_nftoken_mint, NFTokenMint),
    "MPTokenIssuanceCreate": (_build_mptoken_issuance_create, MPTokenIssuanceCreate),
    "MPTokenIssuanceSet": (_build_mptoken_issuance_set, MPTokenIssuanceSet),
    "MPTokenAuthorize": (_build_mptoken_authorize, MPTokenAuthorize),
    "MPTokenIssuanceDestroy": (_build_mptoken_issuance_destroy, MPTokenIssuanceDestroy),
    "AMMCreate": (_build_amm_create, AMMCreate),
    "Batch": (_build_batch, Batch),
}


# =============================================================================
# Public API - these are the only functions clients should call
# =============================================================================


async def generate_txn(ctx: TxnContext, txn_type: str | None = None, **overrides: Any) -> Transaction:
    """Generate a transaction with sane defaults.

    Args:
        ctx: Transaction context with wallets, currencies, and defaults
        txn_type: Transaction type name (e.g., "Payment", "TrustSet").
                 If None, picks a random available type.
        **overrides: Additional fields to override in the transaction

    Returns:
        A fully formed Transaction model ready to sign and submit

    Raises:
        ValueError: If txn_type is not supported
    """
    import inspect

    # Choose or normalize the type name
    if txn_type is None:
        # Start with ALL transaction types from _BUILDERS (single source of truth)
        configured_types = list(_BUILDERS.keys())

        # Remove any types disabled in config
        disabled_types = ctx.config.get("transactions", {}).get("disabled", [])
        if disabled_types:
            configured_types = [t for t in configured_types if t not in disabled_types]
            log.debug("Disabled transaction types: %s", disabled_types)

        # MPToken types that require existing issuance IDs
        requires_mpt_id = {"MPTokenAuthorize", "MPTokenIssuanceSet", "MPTokenIssuanceDestroy"}

        # Filter out MPToken types that need IDs if none are available
        if not ctx.mptoken_issuance_ids:
            configured_types = [t for t in configured_types if t not in requires_mpt_id]
            log.debug("No MPToken IDs available, excluding: %s", requires_mpt_id)

        if not configured_types:
            raise RuntimeError("No transaction types available to generate")

        txn_type = choice(configured_types)
    else:
        # Normalize case: try exact match first, then case-insensitive
        if txn_type not in _BUILDERS:
            # Try case-insensitive lookup
            for builder_type in _BUILDERS.keys():
                if builder_type.lower() == str(txn_type).lower():
                    txn_type = builder_type
                    break

    log.debug("Generating %s txn", txn_type)

    builder_spec = _BUILDERS.get(txn_type)
    if not builder_spec:
        raise ValueError(f"Unsupported txn_type: {txn_type}")

    builder_fn, model_cls = builder_spec

    # Build base transaction with defaults (handle async builders)
    if inspect.iscoroutinefunction(builder_fn):
        composed = await builder_fn(ctx)
    else:
        composed = builder_fn(ctx)

    # Apply user overrides
    if overrides:
        deep_update(composed, transaction_json_to_binary_codec_form(overrides))

    # Debug: dump transaction dict before converting to model
    log.debug(f"Transaction dict for {txn_type}: {composed}")

    log.debug(f"Created {txn_type}")
    return model_cls.from_xrpl(composed)


async def create_payment(ctx: TxnContext, **overrides: Any) -> Payment:
    """Create a Payment transaction with sane defaults."""
    return await generate_txn(ctx, "Payment", **overrides)


async def create_trustset(ctx: TxnContext, **overrides: Any) -> TrustSet:
    """Create a TrustSet transaction with sane defaults."""
    return await generate_txn(ctx, "TrustSet", **overrides)


async def create_accountset(ctx: TxnContext, **overrides: Any) -> AccountSet:
    """Create an AccountSet transaction with sane defaults."""
    return await generate_txn(ctx, "AccountSet", **overrides)


async def create_nftoken_mint(ctx: TxnContext, **overrides: Any) -> NFTokenMint:
    """Create an NFTokenMint transaction with sane defaults."""
    return await generate_txn(ctx, "NFTokenMint", **overrides)


async def create_mptoken_issuance_create(ctx: TxnContext, **overrides: Any) -> MPTokenIssuanceCreate:
    """Create an MPTokenIssuanceCreate transaction with sane defaults."""
    return await generate_txn(ctx, "MPTokenIssuanceCreate", **overrides)


async def create_batch(ctx: TxnContext, **overrides: Any) -> Batch:
    """Create a Batch transaction with sane defaults."""
    return await generate_txn(ctx, "Batch", **overrides)


async def create_amm_create(ctx: TxnContext, **overrides: Any) -> AMMCreate:
    """Create an AMMCreate transaction with sane defaults."""
    return await generate_txn(ctx, "AMMCreate", **overrides)


def update_transaction(transaction: Transaction, **kwargs) -> Transaction:
    """Update an existing transaction with new fields."""
    payload = transaction.to_xrpl()
    payload.update(kwargs)
    return type(transaction).from_xrpl(payload)
