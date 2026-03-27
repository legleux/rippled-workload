"""Transaction context and shared utilities for builder modules.

TxnContext provides wallets, currencies, tracking state, and config to all
builder functions. Utility functions (choice_omit, sample_omit, deep_update)
are shared across builder modules.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from random import choice, sample
from typing import TypeVar

from xrpl.models import IssuedCurrency
from xrpl.wallet import Wallet

T = TypeVar("T")

log = logging.getLogger("workload.txn")

AwaitInt = Callable[[], Awaitable[int]]
AwaitSeq = Callable[[str], Awaitable[int]]


def choice_omit(seq: Sequence[T], omit: Iterable[T]) -> T:
    pool = [x for x in seq if x not in omit]
    if not pool:
        raise ValueError("No options left after excluding omits!")
    return choice(pool)


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


token_metadata = [
    dict(
        ticker="GOOSE",
        name="goosecoin",
        icon="https://🪿.com",  # This might not work...
        asset_class="rwa",
        asset_subclass="commodity",
        issuer_name="Mother Goose",
    ),
]


@dataclass(slots=True)
class TxnContext:
    funding_wallet: Wallet
    wallets: Sequence[Wallet]
    currencies: Sequence[IssuedCurrency]
    config: dict
    base_fee_drops: AwaitInt
    next_sequence: AwaitSeq
    mptoken_issuance_ids: dict[str, str] | None = None  # {mpt_id: issuer_address}
    amm_pools: set[frozenset[str]] | None = None
    amm_pool_registry: list[dict] | None = None
    nfts: dict[str, str] | None = None
    offers: dict[str, dict] | None = None
    tickets: dict[str, set[int]] | None = None
    checks: dict[str, dict] | None = None  # {check_id: {sender, destination, send_max}}
    escrows: dict[str, dict] | None = None  # {escrow_id: {owner, sequence, destination, finish_after, cancel_after}}
    balances: dict[str, dict[str | tuple[str, str], float]] | None = None
    disabled_types: set[str] | None = None
    forced_account: Wallet | None = None
    credentials: list[dict] | None = None
    vaults: list[dict] | None = None
    domains: list[dict] | None = None

    def rand_accounts(self, n: int, omit: list[str] | None = None) -> list[Wallet]:
        """Pick n unique random accounts, optionally excluding addresses.

        When forced_account is set, it is guaranteed to be the first element
        (unless its address is in omit, in which case it is excluded normally).
        """
        omit_set = set(omit) if omit else set()
        if self.forced_account is not None and self.forced_account.address not in omit_set:
            rest = [w for w in self.wallets if w.address != self.forced_account.address and w.address not in omit_set]
            if len(rest) < n - 1:
                raise ValueError(f"Need {n} accounts but only {len(rest) + 1} available after excluding {omit}")
            return [self.forced_account] + sample(rest, n - 1)
        available = [w for w in self.wallets if w.address not in omit_set]
        if len(available) < n:
            raise ValueError(f"Need {n} accounts but only {len(available)} available after excluding {omit}")
        return sample(available, n)

    def rand_account(self, omit: list[str] | None = None) -> Wallet:
        """Pick a single random account, optionally excluding addresses.

        When forced_account is set, returns it unless its address is in omit.
        """
        return self.rand_accounts(1, omit)[0]

    def rand_owner(self, owner_addresses: set[str]) -> Wallet | None:
        """Pick a random account that is in the owner set.

        When forced_account is set, returns it only if it's an owner (else None).
        When not forced, picks randomly from the intersection of wallets and owners.
        """
        if self.forced_account is not None:
            return self.forced_account if self.forced_account.address in owner_addresses else None
        candidates = [w for w in self.wallets if w.address in owner_addresses]
        return choice(candidates) if candidates else None

    def get_account_currencies(self, account: Wallet) -> list[IssuedCurrency]:
        """Get list of IOU currencies this account has a non-zero balance of.

        Returns currencies the account can actually send (has positive balance).
        Useful for avoiding tecPATH_DRY errors.
        """
        if not self.balances or account.address not in self.balances:
            return []

        account_balances = self.balances[account.address]
        currencies_with_balance = []

        for key, balance in account_balances.items():
            if isinstance(key, tuple) and balance > 0:
                currency_code, issuer = key
                currencies_with_balance.append(IssuedCurrency(currency=currency_code, issuer=issuer))

        return currencies_with_balance

    def rand_currency(self) -> IssuedCurrency:
        if not self.currencies:
            raise RuntimeError("No currencies configured")
        return choice(self.currencies)

    def rand_mptoken_id(self) -> str:
        """Get a random MPToken issuance ID from tracked IDs."""
        if not self.mptoken_issuance_ids:
            raise RuntimeError("No MPToken issuance IDs available")
        return choice(list(self.mptoken_issuance_ids.keys()))

    def rand_mptoken_with_issuer(self) -> tuple[str, str]:
        """Get a random (mpt_id, issuer_address) pair."""
        if not self.mptoken_issuance_ids:
            raise RuntimeError("No MPToken issuance IDs available")
        mpt_id = choice(list(self.mptoken_issuance_ids.keys()))
        return mpt_id, self.mptoken_issuance_ids[mpt_id]

    def _asset_id(self, amount: str | dict) -> str:
        """Convert an Amount (XRP drops or IOU) to a unique asset identifier."""
        if isinstance(amount, str):
            return "XRP"
        else:
            return f"{amount['currency']}.{amount['issuer']}"

    def amm_pool_exists(self, asset1: str | dict, asset2: str | dict) -> bool:
        """Check if an AMM pool for this asset pair already exists."""
        if not self.amm_pools:
            return False
        id1 = self._asset_id(asset1)
        id2 = self._asset_id(asset2)
        pool_id = frozenset([id1, id2])
        return pool_id in self.amm_pools

    def rand_amm_pool(self) -> dict:
        """Pick a random AMM pool from the registry."""
        if not self.amm_pool_registry:
            raise RuntimeError("No AMM pools available")
        return choice(self.amm_pool_registry)

    def derive(self, **overrides) -> TxnContext:
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
    ) -> TxnContext:
        return cls(
            wallets=wallets,
            currencies=currencies,
            funding_wallet=funding_wallet,
            config=config,
            base_fee_drops=base_fee_drops,
            next_sequence=next_sequence,
        )
