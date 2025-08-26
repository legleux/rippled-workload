import json

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Sequence

from xrpl.models import IssuedCurrency
from xrpl.models.transactions.transaction import Memo, Transaction
from xrpl.models.transactions import (
    AccountSet,
    MPTokenIssuanceCreate,
    NFTokenMint,
    Payment,
    TrustSet,
)
from xrpl.models import IssuedCurrency

from workload import logger
from workload.randoms import choice
from workload.transactions.mptoken import metadata as token_metadata
from workload.utils import choice_omit

@dataclass
class TxnSpec:
    model: type
    builder: Callable[["TxnContext"], dict]

REGISTRY: Dict[str, TxnSpec] = {}

def register_txn(model_cls: type):
    """
    Decorator to register a txn builder against its XRPL model.
    """
    def wrap(fn: Callable[["TxnContext"], dict]):
        REGISTRY[model_cls.__name__] = TxnSpec(model=model_cls, builder=fn)
        return fn
    return wrap


def deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

@dataclass
class TxnContext:
    accounts: Sequence[str]
    currencies: Sequence[IssuedCurrency]
    fee: str
    defaults: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def rand_account(self, omit: str | None = None) -> str:
        return choice_omit(self.accounts, omit=omit) if omit else choice(self.accounts)

    def rand_currency(self) -> IssuedCurrency:
        return choice(self.currencies)

@register_txn(Payment)
def build_payment(ctx: TxnContext) -> dict:
    acct = ctx.rand_account()
    dest = ctx.rand_account(omit=acct)
    return {
        "TransactionType": "Payment",
        "Account": acct,
        "Destination": dest,
        # Amount is either the default or call-specific
    }

@register_txn(TrustSet)
def build_trustset(ctx: TxnContext) -> dict:
    cur = ctx.rand_currency()
    acct = ctx.rand_account(omit=cur.issuer)
    return {
        "TransactionType": "TrustSet",
        "Account": acct,
        "LimitAmount": {
            "currency": cur.currency,
            "issuer": cur.issuer,
            # no "value" here — defaults/overrides will supply it
        },
    }

@register_txn(AccountSet)
def build_accountset(ctx: TxnContext) -> dict:
    acct = ctx.rand_account()
    return {
        "TransactionType": "AccountSet",
        "Account": acct,
        # SetFlag #TODO: Figure out if SetFlag can even be randomized meaningfully.
    }

@register_txn(NFTokenMint)
def build_nftoken_mint(ctx: TxnContext) -> dict:
    acct = ctx.rand_account()
    memo_msg = "Some really cool info no doubt"
    memo = Memo(memo_data=memo_msg.encode("utf-8").hex())
    return {
        "TransactionType": "NFTokenMint",
        "Account": acct,
        "NFTokenTaxon": 0,
        "memos": [memo],
    }

@register_txn(MPTokenIssuanceCreate)
def build_mptoken_issuance_create(ctx: TxnContext) -> dict:
    acct = ctx.rand_account()
    metadata_hex = json.dumps(choice(token_metadata)).encode("utf-8").hex()
    return {
        "TransactionType": "MPTokenIssuanceCreate",
        "Account": acct,
        "MPTokenMetadata": metadata_hex,
    }

def update_transaction(transaction: Transaction, **kwargs) -> Transaction:
    payload = transaction.to_xrpl()
    payload.update(kwargs)
    return type(transaction).from_xrpl(payload)

async def generate_txn(txn_type: str, ctx: TxnContext, **overrides: Any) -> dict:
    spec = REGISTRY.get(txn_type)
    if not spec:
        raise ValueError(f"Unsupported txn_type: {txn_type!r}")

    logger.info("Building %s", txn_type)

    # Start with defaults (global + per-type)
    composed: dict = {}
    deep_update(composed, ctx.defaults.get("*", {}))
    deep_update(composed, ctx.defaults.get(txn_type, {}))

    # Add the builder’s random fields
    derived = spec.builder(ctx)
    deep_update(composed, derived)

    # Merge any explicit overrides
    if overrides:
        deep_update(composed, overrides)
        logger.info("Overriding: %s", {k: v for k, v in overrides.items()})

    if composed.get("Fee") is None:
        composed["Fee"] = ctx.fee

    for k, v in composed.items():
        logger.info("%s: %s", k, v)

    txn = spec.model.from_xrpl(composed)
    logger.info("Generated %s", txn_type)
    logger.info(json.dumps(txn.to_xrpl(), indent=2))
    return txn
