from collections.abc import Sequence, Iterable, Callable, Awaitable
from dataclasses import dataclass, replace
from random import choice, sample
from typing import TypeVar, Any
import json
from xrpl.wallet import Wallet
from xrpl.models import IssuedCurrency, TransactionFlag
from xrpl.transaction import transaction_json_to_binary_codec_form
from xrpl.models.transactions import (
    AccountSet,
    Batch,
    BatchFlag,
    MPTokenIssuanceCreate,
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


# Need to map the class names to lowercase for when the come in from API for now. Might be a better way.
available_txns = {
    t.lower(): t
    for t in [
        "MPTokenIssuanceCreate",
        "Payment",
        "NFTokenMint",
        "TrustSet",
        "AccountSet",
        # "Batch",  # enable when ready
    ]
}


def choice_omit(seq: Sequence[T], omit: Iterable[T]) -> T:
    pool = [x for x in seq if x not in omit]
    if not pool:
        raise ValueError("No options left after excluding omits!")
    return choice(pool)


# Async helpers
AwaitInt = Callable[[], Awaitable[int]]
AwaitSeq = Callable[[str], Awaitable[int]]


class TxnDefaults:
    # conservative defaults; strings in XRPL JSON format
    max_fee_drops: int = 500
    value: str = "100000000000"
    amount: str = "1000000000"
    xrp_amount_drops: str = "10000000"  # used in Payment fallback
    trust_limit_value: str = "100000000000"  # used in TrustSet fallback
    payment: dict | None = {"Amount": "10000000"}  # string drops
    trust_set: dict | None = {"LimitAmount": {"value": "100000000000"}}


@dataclass(slots=True)
class TxnContext:
    funding_wallet: "Wallet"
    wallets: Sequence["Wallet"]  # <-- sequence, not dict    currencies: Sequence[IssuedCurrency]
    currencies: Sequence[IssuedCurrency]
    defaults: "TxnDefaults"
    base_fee_drops: "AwaitInt"
    next_sequence: "AwaitSeq"

    def rand_account(self, omit: Wallet | None = None) -> "Wallet":
        return choice_omit(self.wallets, omit=[omit] if omit else [])

    def rand_currency(self) -> IssuedCurrency:
        if not self.currencies:
            raise RuntimeError("No currencies configured")
        return choice(self.currencies)

    def derive(self, **overrides) -> "TxnContext":
        return replace(self, **overrides)

    @classmethod
    def build(
        cls,
        *,
        funding_wallet: Wallet,
        wallets: Sequence[Wallet],
        currencies: Sequence[IssuedCurrency],
        base_fee_drops: AwaitInt,
        next_sequence: AwaitSeq,
        defaults: TxnDefaults | None = None,
    ) -> "TxnContext":
        return cls(
            wallets=wallets,
            currencies=currencies,
            funding_wallet=funding_wallet,
            defaults=defaults or TxnDefaults(),
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

    result = {
        "TransactionType": "Payment",
        "Account": src.address,
        "Destination": dst.address,
    }

    # Apply defaults
    if ctx.defaults.payment:
        deep_update(result, ctx.defaults.payment)

    # Ensure Amount is set
    if "Amount" not in result:
        result["Amount"] = ctx.defaults.xrp_amount_drops

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
        },
    }

    # Apply defaults
    if ctx.defaults.trust_set:
        deep_update(result, ctx.defaults.trust_set)

    # Ensure value is set in LimitAmount
    la = result.setdefault("LimitAmount", {})
    if not la.get("value"):
        la["value"] = ctx.defaults.trust_limit_value

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


def _build_batch(ctx: TxnContext) -> dict:
    """Build a Batch transaction with random inner payments."""
    src = ctx.rand_account()
    payments = [
        {
            "RawTransaction": Payment(
                account=src.address,
                destination=choice_omit(ctx.wallets, [src]).address,
                amount="10000000",
                fee="0",
                flags=TransactionFlag.TF_INNER_BATCH_TXN,
                sequence=0,
                signing_pub_key="",
            )
        }
        for _ in range(randrange(1, 9))
    ]

    return {
        "TransactionType": "Batch",
        "Account": src.address,
        "Flags": BatchFlag,
        "RawTransactions": payments,
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
    # Choose or normalize the type name
    if txn_type is None:
        txn_type = choice(list(available_txns.values()))
    else:
        txn_type = available_txns.get(str(txn_type).lower(), txn_type)

    log.info("Generating %s txn", txn_type)

    builder_spec = _BUILDERS.get(txn_type)
    if not builder_spec:
        raise ValueError(f"Unsupported txn_type: {txn_type}")

    builder_fn, model_cls = builder_spec

    # Build base transaction with defaults
    composed = builder_fn(ctx)

    # Apply user overrides
    if overrides:
        deep_update(composed, transaction_json_to_binary_codec_form(overrides))

    log.info(f"Created {txn_type}")
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


def update_transaction(transaction: Transaction, **kwargs) -> Transaction:
    """Update an existing transaction with new fields."""
    payload = transaction.to_xrpl()
    payload.update(kwargs)
    return type(transaction).from_xrpl(payload)
