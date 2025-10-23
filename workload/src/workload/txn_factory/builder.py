from collections.abc import Sequence, Iterable, Callable, Awaitable
from dataclasses import dataclass
from random import choice, sample
from typing import TYPE_CHECKING, TypeVar, Any
import json
from dataclasses import dataclass
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
available_txns = {t.lower():t for t in [
    # "Batch",
    # "MPTokenCreate",
    "Payment",
    "NFTokenMint",
    "TrustSet",
    "AccountSet"
]}

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
    xrp_amount_drops: str = "10000000"           # used in Payment fallback
    trust_limit_value: str = "100000000000"      # used in TrustSet fallback
    payment: dict | None = {"Amount": "10000000"}  # string drops
    trust_set: dict | None = {"LimitAmount": {"value": "100000000000"}}

@dataclass(slots=True)
class TxnContext:
    funding_wallet: "Wallet"
    wallets: dict[str, "Wallet"]
    currencies: Sequence[IssuedCurrency]
    defaults: TxnDefaults
    base_fee_drops: AwaitInt
    next_sequence: AwaitSeq

    def rand_account(self, omit: Wallet | None = None) -> Wallet:
        return choice_omit(self.wallets, omit=[omit] if omit else [])

    def rand_currency(self) -> IssuedCurrency:
        if not self.currencies:
            raise RuntimeError("No currencies configured")
        return choice(self.currencies)

    def derive(self, **overrides) -> "TxnContext":
        return self.model_copy(update=overrides)

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
        icon="https://🪿.com", # This might not work...
        # icon="https://xn--n28h.com",
        asset_class="rwa",
        asset_subclass="commodity",
        issuer_name="Mother Goose",
    ),
]

def sample_omit(seq: Sequence[T], omit: T, k: int) -> list[T]:
    return sample([x for x in seq if x != omit], k)

@dataclass
class TxnSpec:
    model: type[Transaction]
    builder: Callable[[TxnContext], dict]

REGISTRY: dict[str, TxnSpec] = {}

def register_txn(model_cls: type[Transaction]):
    def wrap(fn: Callable[[TxnContext], dict]):
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

@register_txn(Payment)
def build_payment(ctx: TxnContext) -> dict:
    acct = ctx.rand_account()
    dest = ctx.rand_account(omit=acct)
    return {
        "TransactionType": "Payment",
        "Account": acct,
        "Destination": dest,
    }

@register_txn(TrustSet)
def build_trustset(ctx: TxnContext) -> dict:
    cur = ctx.rand_currency()
    acct = ctx.rand_account()
    return {
        "TransactionType": "TrustSet",
        "Account": acct,
        "LimitAmount": {"currency": cur.currency, "issuer": cur.issuer},
    }

@register_txn(AccountSet)
def build_accountset(ctx: TxnContext) -> dict:
    acct = ctx.rand_account()
    return {"TransactionType": "AccountSet", "Account": acct}

@register_txn(NFTokenMint)
def build_nftoken_mint(ctx: TxnContext) -> dict:
    acct = ctx.rand_account()
    memo_msg = "Some really cool info no doubt"
    memo = Memo(memo_data=memo_msg.encode("utf-8").hex())
    return {"TransactionType": "NFTokenMint", "Account": acct, "NFTokenTaxon": 0, "memos": [memo]}

@register_txn(MPTokenIssuanceCreate)
def build_mptoken_issuance_create(ctx: TxnContext) -> dict:
    acct = ctx.rand_account()
    metadata_hex = json.dumps(choice(token_metadata)).encode("utf-8").hex()
    return {"TransactionType": "MPTokenIssuanceCreate", "Account": acct, "MPTokenMetadata": metadata_hex}

@register_txn(Batch)
def build_batch(ctx: TxnContext) -> dict:
    """This might be easier to pass back as dict (since txns are frozen) to to fill in sequences of the inner txns
    otherwise we need a client in the txn_factory, which may not be such a bad idea anyway.
    """
    acct = ctx.rand_account()
    potential_inner = ["Payment"]
    # compose some random inner txns, let's only deal with payments for now.
    # and make them all from the same account
    payments = [{"RawTransaction":Payment(
        account=acct.address,
        destination=choice_omit(ctx.wallets, acct),
        amount="10000000",
        fee="0",
        flags=TransactionFlag.TF_INNER_BATCH_TXN,
        sequence=0,
        signing_pub_key="")} for _ in range(randrange(1, 9))]
    return {
        "TransactionType": "Batch",
        "Account": acct,
        "Flags": BatchFlag,
        "RawTransactions": payments,
        }

def update_transaction(transaction: Transaction, **kwargs) -> Transaction:
    payload = transaction.to_xrpl()
    payload.update(kwargs)
    return type(transaction).from_xrpl(payload)

async def generate_txn(ctx: TxnContext, txn_type: str | None = None, **overrides: Any) -> Transaction:
    txn_type = available_txns.get(txn_type) if txn_type not in available_txns.values() else txn_type
    # That's a little convoluted...it's either non-existent and we'll get a random one or you wanted a random one.
    txn_type = txn_type or choice(list(available_txns.values())) # NOTE: probably make this an enum for easier lookup
    spec = REGISTRY.get(txn_type)
    if not spec:
        raise ValueError(f"Unsupported txn_type: {txn_type}")

    composed: dict = {}
    if txn_type == "Payment" and ctx.defaults.payment:
        deep_update(composed, ctx.defaults.payment)
    elif txn_type == "TrustSet" and ctx.defaults.trust_set:
        deep_update(composed, ctx.defaults.trust_set)

    derived = spec.builder(ctx)
    deep_update(composed, derived)

    if overrides:
        deep_update(composed, transaction_json_to_binary_codec_form(overrides))

    if txn_type == "TrustSet":
        la = composed.setdefault("LimitAmount", {})
        if not la.get("value"):
            la["value"] = ctx.defaults.trust_limit_value
    elif txn_type == "Payment":
        if "Amount" not in composed:
            composed["Amount"] = ctx.defaults.xrp_amount_drops

    log.info(f"Created {txn_type}")
    return spec.model.from_xrpl(composed)
