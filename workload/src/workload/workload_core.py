import asyncio
import hashlib
import json
import logging
import multiprocessing
import sys
import time
from collections import Counter, deque
from typing import Protocol, Any

from dataclasses import dataclass, field
from enum import StrEnum, auto

# Antithesis assertions for critical failures
try:
    from antithesis.assertions import always, sometimes, reachable
    ANTITHESIS_AVAILABLE = True
except ImportError:
    ANTITHESIS_AVAILABLE = False
    # No-op fallbacks
    def always(condition, message, details=None):
        pass
    def sometimes(condition, message, details=None):
        pass
    def reachable(message, details=None):
        pass

import httpx
import xrpl
from xrpl.core.binarycodec import encode, encode_for_signing
from xrpl.core.keypairs import sign
from xrpl.asyncio.clients import AsyncJsonRpcClient
from xrpl.models import IssuedCurrency, Transaction, SubmitOnly
from xrpl.models.amounts import IssuedCurrencyAmount
from xrpl.asyncio.ledger import get_latest_validated_ledger_sequence
from xrpl.wallet import Wallet
from xrpl.models.transactions import (
    AccountSet,
    AccountSetAsfFlag,
    Payment,
    TrustSet,
)
from xrpl.models.requests import (
    AccountInfo,
    AccountLines,
    AccountObjects,
    Ledger,
    Tx,
    ServerState,
)


from workload.txn_factory.builder import TxnContext, generate_txn
import workload.constants as C

num_cpus = multiprocessing.cpu_count()

log = logging.getLogger("workload.core")


# ============================================================================
# TODO: IMPLEMENT FEE ESCALATION SUPPORT
# ============================================================================
# See ../FeeEscalation.md for detailed documentation on how fee escalation works.
#
# PRIORITY IMPROVEMENTS:
# 1. Add get_fee_info() method using xrpl.models.requests.Fee command to get:
#    - current_queue_size, max_queue_size
#    - expected_ledger_size (dynamic limit)
#    - open_ledger_fee (current escalated fee to skip queue)
#    - median_fee (lastLedgerMedianFeeLevel from previous ledger)
#
# 2. Implement dynamic fee adjustment in build_sign_and_track():
#    - Check if open_ledger_fee > base_fee
#    - If escalated, calculate required fee using formula:
#      fee_level = median_fee * (txns_in_open)^2 / (expected_limit)^2
#    - Set Fee field accordingly to ensure transaction gets into open ledger
#
# 3. Add queue monitoring and backpressure:
#    - Check current_queue_size vs max_queue_size before submitting
#    - If queue is full (minimum_fee > base_fee), delay or increase fees
#    - Track per-account queue depth (max 10 txns per account)
#
# 4. Expose fee/queue metrics via API endpoints:
#    - GET /state/fees - current fee escalation state
#    - GET /state/queue - transaction queue status
#
# NOTES:
# - Queue can hold expected_ledger_size * 20 transactions (min 2000)
# - Per-account queue limit: 10 transactions
# - Queued transactions drain when ledger closes (highest fee first)
# - Expected ledger size grows by 20% when healthy, drops 50% when unhealthy
# ============================================================================


# # TODO: Temporary constant store
# C = SimpleNamespace(
#     default_create_amount=int(100 * 1e6),
#     max_create_amount=int(100e6 * 1e6), # alot?
#     horizon=20,  # If it's not validated/failed after 20 ledgers it's gone...
#     rpc_timeout=2.0,
#     submit_timeout=20,
#     lock_timeout=2.0,
# )

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

log = logging.getLogger("workload")

# FINAL_STATES = {"VALIDATED", "REJECTED", "EXPIRED"}
TERMINAL_STATE = {C.TxState.VALIDATED, C.TxState.REJECTED, C.TxState.EXPIRED}


@dataclass(slots=True)
class PendingTx:
    tx_hash: str
    signed_blob_hex: str
    account: str
    tx_json: dict
    sequence: int | None
    last_ledger_seq: int
    transaction_type: C.TxType | None
    created_ledger: int
    wallet: Wallet | None = None
    state: C.TxState = C.TxState.CREATED
    attempts: int = 0
    engine_result_first: str | None = None
    validated_ledger: int | None = None
    meta_txn_result: str | None = None
    created_at: float = field(default_factory=time.time)
    finalized_at: float | None = None

    def __str__(self):
        return f"{self.transaction_type} -- {self.account} -- {self.state}"


class Store(Protocol):
    async def upsert(self, p: PendingTx) -> None: ...
    async def get(self, tx_hash: str) -> PendingTx | None: ...
    async def mark(self, tx_hash: str, **fields) -> None: ...
    async def rekey(self, old_hash: str, new_hash: str) -> None: ...
    async def find_by_state(self, *states: C.TxState) -> list[PendingTx]: ...
    async def all(self) -> list[PendingTx]: ...


class InMemoryStore:
    """Current snapshot of transaction states and validation history for metrics."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._records: dict[str, dict] = {}
        self.validations: deque[ValidationRecord] = deque(maxlen=5000)
        self.count_by_state: dict[str, int] = {}
        self.count_by_type: dict[str, int] = {}
        self.validated_by_source: dict[str, int] = {}

    def _recount(self) -> None:
        # per-state tallies
        self.count_by_state = Counter(rec.get("state", "UNKNOWN") for rec in self._records.values())
        # per-type tallies
        self.count_by_type = Counter(rec.get("transaction_type", "UNKNOWN") for rec in self._records.values())
        # how many VALIDATEDs came from which path
        self.validated_by_source = Counter(v.src for v in self.validations)

    async def update_record(self, tx: dict) -> None:
        """Insert or update a flat transaction record and recompute metrics."""
        txh = tx.get("tx_hash")
        log.debug("update_record %s", txh)
        if not txh:
            raise ValueError("update_record() requires 'tx_hash'")
        async with self._lock:
            self._records[txh] = tx
            self._recount()

    async def get(self, tx_hash: str) -> dict | None:
        async with self._lock:
            return self._records.get(tx_hash)

    async def mark(self, tx_hash: str, *, source: str | None = None, **fields) -> None:
        """
        Update or insert a transaction record.

        tx_hash:
            Unique transaction identifier.
        source:
            Origin of the update ("ws", "poll", etc.).
        **fields:
            Additional fields to merge (e.g., state, validated_ledger, meta_txn_result).

        Behavior:
        - Creates/updates the flat record under tx_hash.
        - On first transition to a final state (VALIDATED/REJECTED/EXPIRED), stamps 'finalized_at'.
        - When state == VALIDATED, appends a ValidationRecord(txn, ledger, src) exactly once per (txn, ledger).
        - Recomputes per-state and per-source counters.
        """
        log.debug("Mark %s", tx_hash)
        async with self._lock:
            rec = self._records.get(tx_hash, {})

            prev_state = rec.get("state")
            rec_before = dict(rec)
            rec.update(fields)
            rec_after = dict(rec)

            if set(rec_after.items()) - set(rec_before.items()):
                d = set(rec_after.items()) - set(rec_before.items())
                log.debug("After has more diff %s", d)
            elif set(rec_before.items()) - set(rec_after.items()):
                d = set(rec_before.items()) - set(rec_after.items())
                log.debug("Before has more diff %s", d)

            if source is not None:
                rec["source"] = source

            state = rec.get("state")
            if isinstance(state, C.TxState):  # normalize enum to string
                state = state.name
                rec["state"] = state

            # Terminal handling
            if state in TERMINAL_STATE:
                rec.setdefault("finalized_at", time.time())

                # Only VALIDATED gets validation history — and only on the first transition to VALIDATED
                if state == "VALIDATED" and prev_state != "VALIDATED":
                    seq = rec.get("validated_ledger") or 0
                    src = source or rec.get("source", "unknown")
                    # De-dupe defensively by (txn, seq)
                    if not any(v.txn == tx_hash and v.seq == seq for v in self.validations):
                        log.debug("%s ValidationRecord for in %s by %s -- %s", state, seq, src, tx_hash)
                        self.validations.append(ValidationRecord(txn=tx_hash, seq=seq, src=src))

            self._records[tx_hash] = rec
            self._recount()
            log.debug("%s --> %s  %s", prev_state, state, tx_hash)

    async def rekey(self, old_hash: str, new_hash: str) -> None:
        """
        Replace a record's key when a tx's canonical hash changes.
        - Moves the flat record from old_hash -> new_hash.
        - Updates inner 'tx_hash' field if present.
        - No-op if old_hash doesn't exist.
        """
        async with self._lock:
            rec = self._records.pop(old_hash, None)
            if rec is None:
                return
            rec["tx_hash"] = new_hash
            self._records[new_hash] = rec
            # (no recount needed unless you want to)

    # Snapshot-oriented queries (flat dicts). Live PendingTx queries belong on Workload.
    async def find_by_state(self, *states: C.TxState | str) -> list[dict]:
        """Return flat records whose 'state' matches any of the given states."""
        wanted = {s.name if isinstance(s, C.TxState) else s for s in states}
        async with self._lock:
            return [rec for rec in self._records.values() if rec.get("state") in wanted]

    async def all_records(self) -> list[dict]:
        async with self._lock:
            return list(self._records.values())

    def snapshot_stats(self) -> dict:
        return {
            "by_state": dict(self.count_by_state),
            "by_type": dict(self.count_by_type),
            "validated_by_source": dict(self.validated_by_source),
            "total_tracked": len(self._records),
            "recent_validations": len(self.validations),
        }


@dataclass
class AccountRecord:
    lock: asyncio.Lock
    next_seq: int | None = None


class ValidationSrc(StrEnum):
    POLL = auto()
    WS = auto()


@dataclass
class ValidationRecord:
    txn: str
    seq: int
    src: str


def _sha512half(b: bytes) -> bytes:
    return hashlib.sha512(b).digest()[:32]


def _txid_from_signed_blob_hex(signed_blob_hex: str) -> str:
    # XRPL txid = SHA512Half(0x54584E00 || signed_bytes)
    return _sha512half(bytes.fromhex("54584E00") + bytes.fromhex(signed_blob_hex)).hex().upper()


def issue_currencies(issuer: str, currency_code: list[str]) -> list[IssuedCurrency]:
    issued_currencies = [IssuedCurrency.from_dict(dict(issuer=issuer, currency=cc)) for cc in currency_code]
    return issued_currencies


async def debug_last_tx(client: AsyncJsonRpcClient, account: str):
    # ai = await client.request({"method": "account_info", "params": [{"account": account, "ledger_index": "validated"}]})
    ai = await client.request(AccountInfo(account=account, ledger_index="validated"))
    try:
        log.debug(
            "acct %s seq=%s bal=%s",
            account,
            ai.result["account_data"]["Sequence"],
            ai.result["account_data"]["Balance"],
        )
    except KeyError as e:
        pass


class Workload:
    def __init__(self, config: dict, client: AsyncJsonRpcClient, *, store: Store | None = None):
        self.config = config
        self.client = client

        # TODO: Load from pre-generated accounts.json
        self.funding_wallet = Wallet.from_seed(
            "snoPBrXtMeMyMHUVTgbuqAfg1SUTb", algorithm=xrpl.CryptoAlgorithm.SECP256K1
        )

        self.accounts: dict[str, AccountRecord] = {}
        self.wallets: dict[str, Wallet] = {}
        self.gateways: list[Wallet] = []
        self.users: list[Wallet] = []

        # Live txns that are going on. Not finalized yet. Go to self.store after we figure it out.
        self.pending: dict[str, PendingTx] = {}

        # tracks the recorded state of all transactions (past and present)—
        self.store: Store = store or InMemoryStore()

        self._fee_cache: int | None = None
        self._fee_lock = asyncio.Lock()

        # Currencies will be created after gateways are initialized
        self._currencies: list[IssuedCurrency] = []

        # Track MPToken issuance IDs for MPToken transactions
        self._mptoken_issuance_ids: list[str] = []

        # Track created AMM pools (asset pairs) to avoid duplicate creation
        # Each entry is a frozenset of two asset identifiers (XRP or "currency.issuer")
        self._amm_pools: set[frozenset[str]] = set()

        # Track NFTs: {nft_id: owner_address}
        self._nfts: dict[str, str] = {}

        # Track offers (generic for NFT, IOU, MPToken): {offer_id: {type, owner, ...}}
        self._offers: dict[str, dict] = {}

        # Track tickets: {account: {ticket_seq1, ticket_seq2, ...}}
        self._tickets: dict[str, set[int]] = {}

        # Heartbeat tracking - separate from normal workload metrics
        # Maps ledger_index -> {tx_hash, submitted_at, validated_at, status}
        self.heartbeats: dict[int, dict] = {}
        self.last_heartbeat_ledger: int | None = None
        self.missed_heartbeats: list[int] = []  # Ledger indices where we failed to submit

        # In-memory balance tracking - our own state, independent of ledger
        # During fuzzing, we can't trust the ledger state, so we track what we send/receive
        # Structure: {account: {"XRP": drops, ("CUR", "issuer"): value}}
        self.balances: dict[str, dict[str | tuple[str, str], float]] = {}

        # Finally, set up the txn_context for generic txn use.
        self.ctx = self.configure_txn_context(
            wallets=self.wallets,
            funding_wallet=self.funding_wallet,
            config=self.config,
        )

    # Set up the txn_context if we want random transactons
    def configure_txn_context(
        self,
        *,
        funding_wallet: "Wallet",
        wallets: dict[str, "Wallet"] | list["Wallet"],
        currencies: list["IssuedCurrency"] | None = None,
        config: dict | None = None,
    ) -> TxnContext:
        currs = currencies if currencies is not None else self._currencies
        # Note: currencies can be empty initially, will be populated during init_participants()
        # accept dict or list; normalize to list
        wl = list(wallets.values()) if isinstance(wallets, dict) else list(wallets)
        ctx = TxnContext.build(
            funding_wallet=funding_wallet,
            wallets=wl,
            currencies=currs,
            config=config or self.config,
            base_fee_drops=self._open_ledger_fee,
            next_sequence=self.alloc_seq,
        )
        # Add MPToken issuance IDs and AMM pools
        ctx.mptoken_issuance_ids = self._mptoken_issuance_ids
        ctx.amm_pools = self._amm_pools
        # Add NFTs, offers, and tickets
        ctx.nfts = self._nfts
        ctx.offers = self._offers
        ctx.tickets = self._tickets
        # Add in-memory balance tracking
        ctx.balances = self.balances
        return ctx

    # Will it be sufficient to do this every time an account is created? or intermittently and mark some accounts as
    # not usable yet?
    def update_txn_context(self):
        self.ctx = self.configure_txn_context(
            wallets=list(self.wallets.values()),
            funding_wallet=self.funding_wallet,
        )

    def get_all_account_addresses(self) -> list[str]:
        """Return all account addresses we're tracking (for WebSocket subscription).

        Returns empty list if no accounts initialized yet (WS will fall back to transaction stream).
        """
        # Don't subscribe until we have actual accounts (after init_participants)
        if not self.wallets:
            return []

        # Include funding wallet + heartbeat wallet + all tracked wallets (gateways + users)
        addresses = [self.funding_wallet.address]

        # CRITICAL: Include heartbeat wallet so we receive validation events for heartbeats
        if hasattr(self, 'heartbeat_wallet') and self.heartbeat_wallet:
            addresses.append(self.heartbeat_wallet.address)

        addresses.extend(self.wallets.keys())
        return addresses

    # =========================================================================
    # AMM Pool Tracking
    # =========================================================================

    def _asset_id(self, amount: str | dict) -> str:
        """Convert an Amount (XRP drops or IOU) to a unique asset identifier."""
        if isinstance(amount, str):
            return "XRP"  # XRP is always represented as "XRP"
        else:
            # IOU: "currency.issuer"
            return f"{amount['currency']}.{amount['issuer']}"

    def _amm_pool_id(self, asset1: str | dict, asset2: str | dict) -> frozenset[str]:
        """Create a unique AMM pool identifier from two assets (order-independent)."""
        id1 = self._asset_id(asset1)
        id2 = self._asset_id(asset2)
        return frozenset([id1, id2])

    def amm_pool_exists(self, asset1: str | dict, asset2: str | dict) -> bool:
        """Check if an AMM pool for this asset pair already exists."""
        pool_id = self._amm_pool_id(asset1, asset2)
        return pool_id in self._amm_pools

    def register_amm_pool(self, asset1: str | dict, asset2: str | dict):
        """Register a newly created AMM pool."""
        pool_id = self._amm_pool_id(asset1, asset2)
        if pool_id not in self._amm_pools:
            self._amm_pools.add(pool_id)
            id1, id2 = sorted(pool_id)  # Sort for consistent display
            log.debug(f"Registered AMM pool: {id1} / {id2}")

    # =========================================================================
    # NFT, Offer, and Ticket tracking - for transaction generation
    # =========================================================================

    def random_nft(self) -> tuple[str, str] | None:
        """Get random NFT. Returns (nft_id, owner) or None."""
        from random import choice
        return choice(list(self._nfts.items())) if self._nfts else None

    def random_offer(self, offer_type: str | None = None) -> dict | None:
        """Get random offer, optionally filtered by type (NFTokenOffer, IOU, MPToken)."""
        from random import choice
        offers = [o for o in self._offers.values()
                  if offer_type is None or o.get("type") == offer_type]
        return choice(offers) if offers else None

    def random_ticket_for_account(self, account: str) -> int | None:
        """Get random ticket sequence for account."""
        from random import choice
        tickets = self._tickets.get(account, set())
        return choice(list(tickets)) if tickets else None

    def has_tickets(self, account: str) -> bool:
        """Check if account has any tickets."""
        return account in self._tickets and len(self._tickets[account]) > 0

    def consume_ticket(self, account: str, ticket_seq: int):
        """Mark a ticket as consumed."""
        if account in self._tickets:
            self._tickets[account].discard(ticket_seq)
            if not self._tickets[account]:
                del self._tickets[account]
            log.debug(f"Consumed ticket {ticket_seq} for {account[:8]}")

    # =========================================================================
    # State persistence - load and save workload state from SQLite
    # =========================================================================

    def load_state_from_store(self) -> bool:
        """Load workload state from SQLite store if available.

        Returns:
            True if state was loaded, False otherwise
        """
        from workload.sqlite_store import SQLiteStore

        if not isinstance(self.store, SQLiteStore):
            log.debug("Store is not SQLiteStore, cannot load state")
            return False

        if not self.store.has_state():
            log.debug("No persisted state found in database")
            return False

        log.debug("Loading workload state from database...")

        # Load wallets
        wallet_data = self.store.load_wallets()
        for address, (wallet, is_gateway, is_user) in wallet_data.items():
            self.wallets[address] = wallet
            self._record_for(address)

            if is_gateway:
                self.gateways.append(wallet)
            if is_user:
                self.users.append(wallet)

        # Load currencies
        currencies = self.store.load_currencies()
        self._currencies = currencies

        log.debug(
            f"Loaded state: {len(self.wallets)} wallets "
            f"({len(self.gateways)} gateways, {len(self.users)} users), "
            f"{len(self._currencies)} currencies"
        )

        if len(self.gateways) > 0 and len(self._currencies) == 0:
            log.debug("Incomplete state detected: gateways exist but no currencies found. Rejecting loaded state.")
            self.wallets.clear()
            self.gateways.clear()
            self.users.clear()
            self._currencies = []
            return False

        # Update transaction context with loaded wallets
        self.update_txn_context()

        return True

    def save_wallet_to_store(self, wallet: Wallet, is_gateway: bool = False, is_user: bool = False) -> None:
        """Save a wallet to the persistent store."""
        from workload.sqlite_store import SQLiteStore

        if isinstance(self.store, SQLiteStore):
            self.store.save_wallet(wallet, is_gateway=is_gateway, is_user=is_user)

    def save_currencies_to_store(self) -> None:
        """Save all currencies to the persistent store."""
        from workload.sqlite_store import SQLiteStore

        if isinstance(self.store, SQLiteStore):
            for currency in self._currencies:
                self.store.save_currency(currency)

    # Use Fee for dynamic value when submitting txns with possibility of fee-escalation.
    # async def _open_ledger_fee(self) -> int:
    #     async with self._fee_lock:
    #         r = await self.client.request(Fee())
    #         self._fee_cache = int(r.result["drops"]["open_ledger_fee"])
    #         return self._fee_cache

    async def _latest_validated_ledger(self) -> int:
        return await get_latest_validated_ledger_sequence(client=self.client)

    def _record_for(self, addr: str) -> AccountRecord:
        rec = self.accounts.get(addr)
        if rec is None:
            log.debug("_record for %s", addr)
            rec = AccountRecord(lock=asyncio.Lock(), next_seq=None)
            self.accounts[addr] = rec
        return rec

    def _get_balance(self, account: str, currency: str, issuer: str | None = None) -> float:
        """Get tracked balance for an account.

        Args:
            account: Account address
            currency: "XRP" or IOU currency code
            issuer: Issuer address for IOUs (None for XRP)

        Returns:
            Balance value (drops for XRP, units for IOUs)
        """
        if account not in self.balances:
            return 0.0

        if currency == "XRP":
            return self.balances[account].get("XRP", 0.0)
        else:
            key = (currency, issuer) if issuer else currency
            return self.balances[account].get(key, 0.0)

    def _set_balance(self, account: str, currency: str, value: float, issuer: str | None = None):
        """Set tracked balance for an account.

        Args:
            account: Account address
            currency: "XRP" or IOU currency code
            value: New balance value
            issuer: Issuer address for IOUs (None for XRP)
        """
        if account not in self.balances:
            self.balances[account] = {}

        if currency == "XRP":
            self.balances[account]["XRP"] = value
        else:
            key = (currency, issuer) if issuer else currency
            self.balances[account][key] = value

    def _update_balance(self, account: str, currency: str, delta: float, issuer: str | None = None):
        """Update (credit/debit) tracked balance for an account.

        Args:
            account: Account address
            currency: "XRP" or IOU currency code
            delta: Amount to add (positive) or subtract (negative)
            issuer: Issuer address for IOUs (None for XRP)
        """
        current = self._get_balance(account, currency, issuer)
        self._set_balance(account, currency, current + delta, issuer)

    async def _rpc(self, req, *, t=C.RPC_TIMEOUT):
        return await asyncio.wait_for(self.client.request(req), timeout=t)

    async def alloc_seq(self, addr: str) -> int:
        rec = self._record_for(addr)

        async with rec.lock:
            if rec.next_seq is None:
                ai = await self._rpc(AccountInfo(account=addr, ledger_index="current", strict=True))
                if not ai.is_successful() or "account_data" not in ai.result:
                    raise ValueError(f"Account {addr} does not exist on ledger - database/network mismatch")
                rec.next_seq = ai.result["account_data"]["Sequence"]

            s = rec.next_seq
            rec.next_seq += 1
            return s

    async def _open_ledger_fee(self) -> int:
        ss = await self._rpc(ServerState(), t=2.0)
        base = float(ss.result["state"]["validated_ledger"]["base_fee"])
        return int(base * 10)  # TODO: Tie this down, need to be able to handle fee elevation.

    async def _owner_reserve(self) -> int:
        """Get the owner reserve in drops (required for AMM creates, NFT pages, etc.)."""
        ss = await self._rpc(ServerState(), t=2.0)
        # owner_reserve is in XRP, convert to drops
        reserve_xrp = int(ss.result["state"]["validated_ledger"]["reserve_inc"])
        return reserve_xrp  # Already in drops in the response

    async def _last_ledger_sequence_offset(self, off: int) -> int:
        ss = await self._rpc(ServerState(), t=2.0)
        return ss.result["state"]["validated_ledger"]["seq"] + off

    async def _current_ledger_index(self) -> int:
        """Get the latest validated ledger index."""
        ss = await self._rpc(ServerState(), t=2.0)
        return ss.result["state"]["validated_ledger"]["seq"]

    async def server_info(self) -> dict:
        """Get current server info from rippled. Result structure changes, see XRPL docs."""
        from xrpl.models.requests import ServerInfo
        r = await self.client.request(ServerInfo())
        return r.result

    async def _expected_ledger_size(self) -> int:
        """Get the expected number of transactions per ledger from the server."""
        from xrpl.models.requests import Fee
        fee_result = await self._rpc(Fee())
        # expected_ledger_size is a string in the fee response
        return int(fee_result.result.get("expected_ledger_size", 40))

    async def record_created(self, p: PendingTx) -> None:
        # store pending txn keyed by local hash
        self.pending[p.tx_hash] = p
        p.state = C.TxState.CREATED
        log.debug("Creating record %s for %s", p.state, p.tx_hash)
        await self.store.update_record(
            {
                "tx_hash": p.tx_hash,
                "state": p.state,  # or p.state.name?
                "created_ledger": p.created_ledger,
            }
        )

    async def record_submitted(self, p: PendingTx, engine_result: str | None, srv_txid: str | None):
        if p.state in TERMINAL_STATE:
            pass  # Don't overwrite terminal states. This should probably be an exception.
            return
        old = p.tx_hash
        new_hash = srv_txid or old
        if srv_txid and srv_txid != old:
            log.debug(f"TX HASH CHANGED: {old[:8]} -> {new_hash[:8]} (server returned different hash)")
            self.pending[new_hash] = self.pending.pop(old, p)
            p.tx_hash = new_hash
            await self.store.rekey(old, new_hash)
        p.state = C.TxState.SUBMITTED
        p.engine_result_first = p.engine_result_first or engine_result
        self.pending[new_hash] = p
        log.debug(f"TX SUBMITTED: {new_hash[:8]} ({p.transaction_type}) - {engine_result}")
        await self.store.mark(new_hash, state=C.TxState.SUBMITTED, engine_result_first=p.engine_result_first)

    async def _update_account_balances(self, account: str) -> None:
        """Fetch and store current balances for an account from the ledger."""
        from workload.sqlite_store import SQLiteStore

        if not isinstance(self.store, SQLiteStore):
            return  # Balance tracking only works with SQLiteStore

        try:
            # Fetch XRP balance from account info
            acc_info = await self._rpc(AccountInfo(account=account, ledger_index="validated"), t=2.0)
            if not acc_info.is_successful():
                log.debug(f"AccountInfo failed for {account}: {acc_info.result}")
                return

            xrp_balance = acc_info.result.get("account_data", {}).get("Balance")
            if xrp_balance:
                self.store.update_balance(account, "XRP", xrp_balance)

            # Fetch trust line balances (IOUs)
            acc_lines = await self._rpc(AccountLines(account=account, ledger_index="validated"), t=2.0)
            if acc_lines.is_successful():
                for line in acc_lines.result.get("lines", []):
                    currency = line.get("currency")
                    issuer = line.get("account")  # The counterparty is the issuer
                    balance = line.get("balance")
                    if currency and issuer and balance:
                        self.store.update_balance(account, "IOU", balance, currency=currency, issuer=issuer)

            # Fetch MPToken balances
            acc_objects = await self._rpc(
                AccountObjects(account=account, ledger_index="validated", type="mptoken"), t=2.0
            )
            if acc_objects.is_successful():
                for obj in acc_objects.result.get("account_objects", []):
                    if obj.get("LedgerEntryType") == "MPToken":
                        # MPToken ID identifies the token uniquely
                        mpt_id = obj.get("MPTokenIssuanceID")
                        # Outstanding balance for this MPToken
                        balance = obj.get("MPTAmount", "0")
                        if mpt_id:
                            # Store MPToken balance using issuance ID as currency identifier
                            self.store.update_balance(account, "MPToken", balance, currency=mpt_id, issuer=None)

            log.debug(f"Updated balances for {account}")
        except asyncio.CancelledError:
            # Expected during shutdown, don't log as warning
            raise
        except asyncio.TimeoutError:
            log.debug(f"Balance update timed out for {account} (expected during heavy load)")
        except Exception as e:
            log.debug(f"Failed to update balances for {account}: {type(e).__name__}: {e}")

    async def record_validated(self, rec: ValidationRecord, meta_result: str | None = None) -> dict:
        p_live = self.pending.get(rec.txn)  # keep this reference

        # DIAGNOSTIC LOGGING: Track validation source and potential race conditions
        log.debug(
            "🔍 RECORD_VALIDATED called: tx=%s | ledger=%s | source=%s | in_pending=%s | "
            "current_state=%s | meta_result=%s",
            rec.txn[:8],
            rec.seq,
            rec.src,
            p_live is not None,
            p_live.state.name if p_live else "N/A",
            meta_result
        )

        # Check for potential race condition: already validated by another source
        if p_live and p_live.state == C.TxState.VALIDATED:
            log.warning(
                "⚠️  RACE CONDITION: tx %s already VALIDATED (previous source may have beaten us) | "
                "current_source=%s | validated_ledger=%s",
                rec.txn[:8],
                rec.src,
                p_live.validated_ledger
            )

        # Distinguish transaction outcomes (XRPL Reliable Submission Best Practice)
        # - tesSUCCESS: Transaction succeeded
        # - tec*: Transaction included in ledger and cost burned, but failed to achieve effect
        # - ter*: Should not appear in validated ledgers (retryable)
        if meta_result == "tesSUCCESS":
            log.debug(f"✓ TX SUCCESS: {rec.txn[:8]} in ledger {rec.seq} - operation succeeded")
        elif meta_result and meta_result.startswith("tec"):
            log.warning(f"⚠ TX VALIDATED BUT FAILED: {rec.txn[:8]} in ledger {rec.seq} - result={meta_result}")
            log.warning(f"   Cost burned but operation failed. Common causes: insufficient balance, no path, etc.")
        elif meta_result:
            log.warning(f"⚠ TX VALIDATED WITH UNEXPECTED CODE: {rec.txn[:8]} in ledger {rec.seq} - result={meta_result}")

        if p_live:
            p_live.state = C.TxState.VALIDATED
            p_live.validated_ledger = rec.seq
            p_live.meta_txn_result = meta_result
        else:
            log.debug("record_validated: tx not in pending (race or already finalized): %s", rec.txn)

        await self.store.mark(
            rec.txn,
            state=C.TxState.VALIDATED,
            validated_ledger=rec.seq,
            meta_txn_result=meta_result,
            source=rec.src,
        )

        p_live = self.pending.get(rec.txn)
        w = getattr(p_live, "wallet", None)
        if w is not None:
            self.wallets[w.address] = w
            self._record_for(w.address)
            self.users.append(w)
            self.save_wallet_to_store(w, is_user=True)  # Persist newly created user wallet
            self.update_txn_context()
            log.debug("Adopted new account after validation: %s", w.address)

        # Update balances for the account involved in the transaction
        # IN-MEMORY balance tracking - no RPC calls, our own state for fuzzing
        if p_live and meta_result == "tesSUCCESS" and p_live.transaction_type == C.TxType.PAYMENT:
            try:
                tx_json = p_live.tx_json
                if tx_json:
                    sender = tx_json.get("Account")
                    destination = tx_json.get("Destination")
                    amount = tx_json.get("Amount")

                    if sender and destination and amount:
                        if isinstance(amount, str):
                            # XRP payment (drops)
                            amount_val = float(amount)
                            self._update_balance(sender, "XRP", -amount_val)  # Debit sender
                            self._update_balance(destination, "XRP", amount_val)  # Credit destination
                            log.debug(f"Balance: {sender[:8]} sent {amount_val} drops XRP to {destination[:8]}")
                        elif isinstance(amount, dict):
                            # IOU payment
                            currency = amount.get("currency")
                            issuer = amount.get("issuer")
                            value = float(amount.get("value", 0))

                            if currency and issuer:
                                # Issuers have infinite balance, don't track for them
                                if sender != issuer:
                                    self._update_balance(sender, currency, -value, issuer)  # Debit sender
                                if destination != issuer:
                                    self._update_balance(destination, currency, value, issuer)  # Credit destination
                                log.debug(f"Balance: {sender[:8]} sent {value} {currency}/{issuer[:8]} to {destination[:8]}")
            except Exception as e:
                log.debug(f"Failed to update in-memory balances for {rec.txn}: {e}")

        # Track MPToken issuance IDs from MPTokenIssuanceCreate transactions
        if p_live and p_live.transaction_type == C.TxType.MPTOKEN_ISSUANCE_CREATE:
            try:
                # Fetch full transaction with metadata to get mpt_issuance_id
                tx_result = await self._rpc(Tx(transaction=rec.txn))
                mpt_id = tx_result.result.get("mpt_issuance_id")
                if mpt_id and mpt_id not in self._mptoken_issuance_ids:
                    self._mptoken_issuance_ids.append(mpt_id)
                    self.update_txn_context()  # Refresh context with new MPToken ID
                    log.debug("Tracked new MPToken issuance ID: %s", mpt_id)
            except Exception as e:
                log.debug(f"Failed to extract MPToken issuance ID from {rec.txn}: {e}")

        # Track AMM pools from AMMCreate transactions
        if p_live and p_live.transaction_type == C.TxType.AMM_CREATE:
            try:
                # Extract assets from the transaction JSON
                tx_json = p_live.tx_json
                if tx_json and "Amount" in tx_json and "Amount2" in tx_json:
                    self.register_amm_pool(tx_json["Amount"], tx_json["Amount2"])
                    self.update_txn_context()  # Refresh context with new AMM pool
            except Exception as e:
                log.debug(f"Failed to track AMM pool from {rec.txn}: {e}")

        # Track NFTs from NFTokenMint - calculate ID deterministically, no metadata parsing!
        if p_live and p_live.transaction_type == C.TxType.NFTOKEN_MINT and meta_result == "tesSUCCESS":
            try:
                from workload.nft_utils import encode_nftoken_id
                nft_id = encode_nftoken_id(
                    flags=p_live.tx_json.get("Flags", 0),
                    transfer_fee=p_live.tx_json.get("TransferFee", 0),
                    issuer=p_live.account,
                    taxon=p_live.tx_json.get("NFTokenTaxon", 0),
                    sequence=p_live.sequence,
                )
                self._nfts[nft_id] = p_live.account
                self.update_txn_context()
                log.debug(f"Tracked NFT mint: {nft_id[:16]}... by {p_live.account[:8]}")
            except Exception as e:
                log.debug(f"Failed to track NFT mint: {e}")

        # Track NFT burns
        if p_live and p_live.transaction_type == C.TxType.NFTOKEN_BURN and meta_result == "tesSUCCESS":
            try:
                nft_id = p_live.tx_json.get("NFTokenID")
                if nft_id and nft_id in self._nfts:
                    del self._nfts[nft_id]
                    self.update_txn_context()
                    log.debug(f"Tracked NFT burn: {nft_id[:16]}...")
            except Exception as e:
                log.debug(f"Failed to track NFT burn: {e}")

        # Track NFT offer creation
        if p_live and p_live.transaction_type == C.TxType.NFTOKEN_CREATE_OFFER and meta_result == "tesSUCCESS":
            try:
                # TODO: Consider refactoring this metadata parsing - offer IDs are ledger-generated
                # and may not be deterministically calculable like NFTokenID, but we should verify.

                # Extract offer ID from metadata - look for CreatedNode with NFTokenOffer type
                meta = rec.meta
                if meta and isinstance(meta, dict):
                    affected_nodes = meta.get("AffectedNodes", [])
                    for node in affected_nodes:
                        if "CreatedNode" in node:
                            created = node["CreatedNode"]
                            if created.get("LedgerEntryType") == "NFTokenOffer":
                                # Extract offer index (the offer ID)
                                offer_id = created.get("LedgerIndex")
                                if offer_id:
                                    # Determine if it's a sell or buy offer
                                    is_sell = bool(p_live.tx_json.get("Flags", 0) & 1)  # tfSellNFToken = 1
                                    nft_id = p_live.tx_json.get("NFTokenID")

                                    self._offers[offer_id] = {
                                        "type": "NFTokenOffer",
                                        "owner": p_live.account,
                                        "nft_id": nft_id,
                                        "is_sell_offer": is_sell,
                                        "amount": p_live.tx_json.get("Amount"),
                                    }
                                    self.update_txn_context()
                                    log.debug(f"Tracked NFT {'sell' if is_sell else 'buy'} offer: {offer_id[:16]}... by {p_live.account[:8]}")
                                    break
            except Exception as e:
                log.debug(f"Failed to track NFT offer creation: {e}")

        # Track NFT offer cancellation
        if p_live and p_live.transaction_type == C.TxType.NFTOKEN_CANCEL_OFFER and meta_result == "tesSUCCESS":
            try:
                # Remove cancelled offers from tracking
                offer_ids = p_live.tx_json.get("NFTokenOffers", [])
                for offer_id in offer_ids:
                    if offer_id in self._offers:
                        del self._offers[offer_id]
                        log.debug(f"Tracked NFT offer cancellation: {offer_id[:16]}...")
                if offer_ids:
                    self.update_txn_context()
            except Exception as e:
                log.debug(f"Failed to track NFT offer cancellation: {e}")

        # Track NFT offer acceptance (removes offer and transfers NFT)
        if p_live and p_live.transaction_type == C.TxType.NFTOKEN_ACCEPT_OFFER and meta_result == "tesSUCCESS":
            try:
                # Remove accepted offer from tracking
                sell_offer = p_live.tx_json.get("NFTokenSellOffer")
                buy_offer = p_live.tx_json.get("NFTokenBuyOffer")

                if sell_offer and sell_offer in self._offers:
                    offer_data = self._offers[sell_offer]
                    nft_id = offer_data.get("nft_id")
                    # Update NFT owner
                    if nft_id and nft_id in self._nfts:
                        self._nfts[nft_id] = p_live.account
                        log.debug(f"Tracked NFT transfer: {nft_id[:16]}... to {p_live.account[:8]}")
                    del self._offers[sell_offer]
                    log.debug(f"Tracked NFT sell offer acceptance: {sell_offer[:16]}...")

                if buy_offer and buy_offer in self._offers:
                    offer_data = self._offers[buy_offer]
                    nft_id = offer_data.get("nft_id")
                    # Update NFT owner to the buyer (from the offer)
                    buyer = offer_data.get("owner")
                    if nft_id and nft_id in self._nfts and buyer:
                        self._nfts[nft_id] = buyer
                        log.debug(f"Tracked NFT transfer: {nft_id[:16]}... to {buyer[:8]}")
                    del self._offers[buy_offer]
                    log.debug(f"Tracked NFT buy offer acceptance: {buy_offer[:16]}...")

                if sell_offer or buy_offer:
                    self.update_txn_context()
            except Exception as e:
                log.debug(f"Failed to track NFT offer acceptance: {e}")

        # Track ticket creation
        if p_live and p_live.transaction_type == C.TxType.TICKET_CREATE and meta_result == "tesSUCCESS":
            try:
                # TODO: Refactor this deeply nested metadata parsing - tickets might be deterministically
                # calculable from transaction fields like NFTokenID. Check rippled source for ticket
                # sequence allocation algorithm and move to a ticket_utils.py module if possible.

                # Extract ticket sequences from metadata
                meta = rec.meta
                if meta and isinstance(meta, dict):
                    affected_nodes = meta.get("AffectedNodes", [])
                    ticket_seqs = []
                    for node in affected_nodes:
                        if "CreatedNode" in node:
                            created = node["CreatedNode"]
                            if created.get("LedgerEntryType") == "Ticket":
                                # Extract ticket sequence from NewFields
                                new_fields = created.get("NewFields", {})
                                ticket_seq = new_fields.get("TicketSequence")
                                if ticket_seq is not None:
                                    ticket_seqs.append(ticket_seq)

                    if ticket_seqs:
                        account = p_live.account
                        if account not in self._tickets:
                            self._tickets[account] = set()
                        self._tickets[account].update(ticket_seqs)
                        self.update_txn_context()
                        log.debug(f"Tracked {len(ticket_seqs)} tickets for {account[:8]}: {ticket_seqs}")
            except Exception as e:
                log.debug(f"Failed to track ticket creation: {e}")

        # Track IOU offer creation (DEX trading)
        if p_live and p_live.transaction_type == C.TxType.OFFER_CREATE and meta_result == "tesSUCCESS":
            try:
                # TODO: Refactor this deeply nested metadata parsing - offer IDs are likely deterministically
                # calculable from ledger objects. Check rippled source for offer index calculation
                # (probably hash of account + sequence or similar) and move to offer_utils.py if possible.

                # Extract offer index from metadata - look for CreatedNode with Offer type
                meta = rec.meta
                if meta and isinstance(meta, dict):
                    affected_nodes = meta.get("AffectedNodes", [])
                    for node in affected_nodes:
                        if "CreatedNode" in node:
                            created = node["CreatedNode"]
                            if created.get("LedgerEntryType") == "Offer":
                                # Extract offer index (the offer ID) and offer details
                                offer_id = created.get("LedgerIndex")
                                new_fields = created.get("NewFields", {})

                                if offer_id and p_live.sequence is not None:
                                    self._offers[offer_id] = {
                                        "type": "IOUOffer",
                                        "owner": p_live.account,
                                        "sequence": p_live.sequence,  # Used for OfferCancel
                                        "taker_pays": new_fields.get("TakerPays"),
                                        "taker_gets": new_fields.get("TakerGets"),
                                    }
                                    self.update_txn_context()
                                    log.debug(f"Tracked IOU offer: {offer_id[:16]}... by {p_live.account[:8]} seq={p_live.sequence}")
                                    break
            except Exception as e:
                log.debug(f"Failed to track IOU offer creation: {e}")

        # Track IOU offer cancellation
        if p_live and p_live.transaction_type == C.TxType.OFFER_CANCEL and meta_result == "tesSUCCESS":
            try:
                # Remove cancelled offer from tracking
                # Find the offer by owner + sequence
                offer_sequence = p_live.tx_json.get("OfferSequence")
                if offer_sequence is not None:
                    # Find offer by owner and sequence
                    for offer_id, offer_data in list(self._offers.items()):
                        if (offer_data.get("type") == "IOUOffer" and
                            offer_data.get("owner") == p_live.account and
                            offer_data.get("sequence") == offer_sequence):
                            del self._offers[offer_id]
                            self.update_txn_context()
                            log.debug(f"Tracked IOU offer cancellation: {offer_id[:16]}... seq={offer_sequence}")
                            break
            except Exception as e:
                log.debug(f"Failed to track IOU offer cancellation: {e}")

        # CRITICAL: Sequence collision cleanup
        # When a transaction validates with sequence N, any other pending transactions
        # from the same account with sequence ≤ N are now IMPOSSIBLE to validate.
        # This prevents tefPAST_SEQ errors from retry logic attempting to resubmit
        # transactions that have already been consumed by the ledger.
        if p_live and p_live.sequence is not None and p_live.account:
            collision_count = 0
            for other_hash, other_p in list(self.pending.items()):
                if (other_hash != rec.txn and  # Don't invalidate the transaction that just validated
                    other_p.account == p_live.account and
                    other_p.sequence is not None and
                    other_p.sequence <= p_live.sequence and
                    other_p.state not in TERMINAL_STATE):
                    log.warning(f"⚠️ SEQUENCE COLLISION: tx {other_hash[:8]} seq={other_p.sequence} invalidated because "
                                f"tx {rec.txn[:8]} seq={p_live.sequence} already validated")
                    other_p.state = C.TxState.REJECTED
                    await self.store.mark(other_hash, state=C.TxState.REJECTED,
                                         engine_result_first=other_p.engine_result_first or "sequence_collision",
                                         engine_result_final="sequence_collision")
                    collision_count += 1

            if collision_count > 0:
                log.warning(f"  Invalidated {collision_count} transactions from {p_live.account[:8]} with sequences ≤ {p_live.sequence}")

        log.debug("txn %s validated at ledger %s via %s", rec.txn, rec.seq, rec.src)
        return {"tx_hash": rec.txn, "ledger_index": rec.seq, "source": rec.src, "meta_result": meta_result}

    async def record_expired(self, tx_hash: str):
        if tx_hash in self.pending:
            p = self.pending[tx_hash]
            p.state = C.TxState.EXPIRED
            await self.store.mark(tx_hash, state=C.TxState.EXPIRED)

            # CRITICAL: Reset sequence tracking when transaction expires
            # If a txn with sequence N expires, the ledger still expects sequence N
            # but our tracking has moved to N+1. We must reset to match the ledger.
            if p.account and p.account in self.accounts and p.sequence is not None:
                log.debug(f"⚠ TX EXPIRED: {tx_hash[:8]} seq={p.sequence} - resetting sequence tracking for {p.account[:8]}")

                # Reset tracking to force re-query from ledger
                self.accounts[p.account].next_seq = None

                # Expire all pending transactions from this account with higher sequences
                # They're now invalid because they depend on this sequence being filled
                expired_count = 0
                for other_hash, other_p in list(self.pending.items()):
                    if (other_p.account == p.account and
                        other_p.sequence is not None and
                        other_p.sequence > p.sequence and
                        other_p.state not in TERMINAL_STATE):
                        log.debug(f"  ⚠ Cascading expiry: {other_hash[:8]} seq={other_p.sequence} (depends on {p.sequence})")
                        other_p.state = C.TxState.EXPIRED
                        await self.store.mark(other_hash, state=C.TxState.EXPIRED)
                        expired_count += 1

                if expired_count > 0:
                    log.debug(f"  Expired {expired_count} additional transactions from {p.account[:8]} with sequences > {p.sequence}")
            # self.pending.pop(tx_hash, None) # see if this gets out of hand?

    def find_by_state(self, *states: C.TxState) -> list[PendingTx]:
        return [p for p in self.pending.values() if p.state in set(states)]

    async def build_sign_and_track(self, txn: Transaction, wallet: Wallet, horizon: int = C.HORIZON) -> PendingTx:
        created_li = (await self._rpc(ServerState(), t=2.0)).result["state"]["validated_ledger"][
            "seq"
        ]  # TODO: Constant
        lls = created_li + horizon
        tx = txn.to_xrpl()
        if tx.get("Flags") == 0:
            del tx["Flags"]

        need_seq = "TicketSequence" not in tx and not tx.get("Sequence")
        need_fee = not tx.get("Fee")

        seq = await self.alloc_seq(wallet.address) if need_seq else tx.get("Sequence")

        # AMMCreate requires fee = owner_reserve, not base fee
        if need_fee:
            if tx.get("TransactionType") == "AMMCreate":
                fee = await self._owner_reserve()
                log.debug(f"AMMCreate fee set to owner_reserve: {fee} drops ({fee/1_000_000} XRP)")
            else:
                fee = await self._open_ledger_fee()
        else:
            fee = int(tx["Fee"])

        if need_seq:
            tx["Sequence"] = seq
        if need_fee:
            tx["Fee"] = str(fee)
        tx["SigningPubKey"] = wallet.public_key
        tx["LastLedgerSequence"] = lls

        signing_blob = encode_for_signing(tx)
        to_sign = signing_blob if isinstance(signing_blob, str) else signing_blob.hex()
        tx["TxnSignature"] = sign(to_sign, wallet.private_key)
        signed_blob_hex = encode(tx)
        local_txid = _txid_from_signed_blob_hex(signed_blob_hex)

        p = PendingTx(
            tx_hash=local_txid,
            signed_blob_hex=signed_blob_hex,
            account=tx["Account"],
            tx_json=tx,
            sequence=tx.get("Sequence"),
            last_ledger_seq=lls,
            transaction_type=tx.get("TransactionType"),
            created_ledger=created_li,
        )
        await self.record_created(p)
        return p

    async def submit_pending(self, p: PendingTx, timeout: float = C.SUBMIT_TIMEOUT) -> dict | None:
        # If the txn is in this state already we've got nothing to do but why are we here in the first place?
        if p.state in TERMINAL_STATE:
            log.debug("%s not active txn!", p)
            return None

        try:
            # Get current ledger to check if transaction is expired
            current_ledger = await self._current_ledger_index()
            is_expired = current_ledger > p.last_ledger_seq if p.last_ledger_seq else False
            ledger_age = current_ledger - p.created_ledger if p.created_ledger else 0

            p.attempts += 1

            # DIAGNOSTIC LOGGING: Track all submissions, especially old/expired transactions
            log_level = logging.WARNING if is_expired or ledger_age > 100 else logging.DEBUG
            log.log(
                log_level,
                "🔍 SUBMIT_PENDING called: tx=%s | type=%s | state=%s | seq=%s | account=%s | "
                "attempts=%d | created_ledger=%s | last_ledger_seq=%s | current_ledger=%s | "
                "EXPIRED=%s | age=%d ledgers",
                p.tx_hash[:8] if p.tx_hash else "None",
                p.transaction_type,
                p.state.name if p.state else "None",
                p.sequence,
                p.account[:12] if p.account else "None",
                p.attempts,
                p.created_ledger,
                p.last_ledger_seq,
                current_ledger,
                is_expired,
                ledger_age
            )

            # Stack trace for expired/old transactions to track where they're coming from
            if is_expired or ledger_age > 100:
                import traceback
                log.warning("⚠️  SUBMITTING EXPIRED/OLD TRANSACTION - Call stack:\n%s",
                           ''.join(traceback.format_stack()[-5:]))
            if p.transaction_type == "AccountSet":
                pass
            resp = await asyncio.wait_for(self.client.request(SubmitOnly(tx_blob=p.signed_blob_hex)), timeout=timeout)
            res = resp.result
            er = res.get("engine_result")

            log.debug(f"SUBMIT RESPONSE for {p.tx_hash[:8]} ({p.transaction_type}): engine_result={er} | full response: {res}")

            if p.engine_result_first is None:
                p.engine_result_first = er

            if isinstance(er, str) and er.startswith(("tem", "tef")):
                # terminal reject: mark and stop
                p.state = C.TxState.REJECTED
                self.pending[p.tx_hash] = p
                await self.store.mark(
                    p.tx_hash,
                    state=p.state,
                    engine_result_first=p.engine_result_first,
                    engine_result_final=er,
                )
                log.debug("************* Terminal Rejection ***********************")
                log.debug("%s by %s was %s %s", p.transaction_type, p.account, p.state, p.engine_result_first)
                log.debug("********************************************************")
                return res

            # Handle ter* codes (retryable errors like terPRE_SEQ)
            if isinstance(er, str) and er.startswith("ter"):
                # Mark as RETRYABLE instead of SUBMITTED so finality checker knows to retry
                p.state = C.TxState.RETRYABLE
                p.engine_result_first = p.engine_result_first or er
                self.pending[p.tx_hash] = p
                await self.store.mark(
                    p.tx_hash,
                    state=p.state,
                    engine_result_first=p.engine_result_first,
                )
                log.debug(f"RETRYABLE: {p.tx_hash[:8]} ({p.transaction_type}) - {er}")
                return res

            # tesSUCCESS or other success codes - mark as SUBMITTED
            srv_txid = res.get("tx_json", {}).get("hash")
            if isinstance(srv_txid, str) and srv_txid and srv_txid != p.tx_hash:
                self.pending[srv_txid] = self.pending.pop(p.tx_hash, p)
                p.tx_hash = srv_txid
            await self.record_submitted(p, engine_result=er, srv_txid=srv_txid)
            return res

        except asyncio.TimeoutError:
            p.state = C.TxState.FAILED_NET
            self.pending[p.tx_hash] = p
            log.error("timeout")
            await self.store.mark(p.tx_hash, state=C.TxState.FAILED_NET, engine_result_first=p.engine_result_first)
            return {"engine_result": "timeout"}

        except Exception as e:
            p.state = C.TxState.FAILED_NET
            self.pending[p.tx_hash] = p
            log.error("submit error tx=%s: %s", p.tx_hash, e)
            # NEW: persist state transition
            await self.store.mark(p.tx_hash, state=C.TxState.FAILED_NET, message=str(e))
            return {"engine_result": "error", "message": str(e)}

    def log_validation(self, tx_hash, ledger_index, result, validation_src):
        log.debug(
            "Validated via %s tx=%s li=%s result=%s", validation_src, tx_hash, ledger_index, result
        )  # FIX: DEbug only...

    # TODO: Default constants
    async def check_finality(self, p: PendingTx, grace: int = 2) -> C.TxState:
        """Check if a submitted transaction has reached finality (validated, expired, or rejected).

        Returns the transaction's current state. Ledger index and meta result are stored in the PendingTx object.
        """
        try:
            txr = await self.client.request(Tx(transaction=p.tx_hash))

            # Only log if actually validated - skip the noise of pending transactions
            if txr.is_successful() and txr.result.get("validated"):
                li = int(txr.result["ledger_index"])
                result = txr.result["meta"]["TransactionResult"]

                log.debug(f"TX VALIDATED: {p.tx_hash[:8]} in ledger {li} with result {result}")

                p.state = C.TxState.VALIDATED
                p.validated_ledger = li
                p.meta_txn_result = result
                await self.record_validated(ValidationRecord(p.tx_hash, li, ValidationSrc.POLL), result)
                return p.state
            else:
                # Transaction found but not validated yet (in open ledger) - normal, will check again later
                log.debug(f"TX {p.tx_hash[:8]} in open ledger, not validated yet")
        except Exception as e:
            # Transaction not found yet - this is normal for newly submitted txns
            log.debug(f"TX {p.tx_hash[:8]} not found in any ledger yet")
            pass

        latest_val = await self._latest_validated_ledger()
        if latest_val > (p.last_ledger_seq + grace):
            # XRPL Best Practice: Sequence Collision Detection
            # Before marking expired, check if account sequence advanced past this transaction's sequence
            # If yes, a DIFFERENT transaction with the same sequence was included (malleability, concurrent submission, etc.)
            if p.sequence is not None and p.account:
                try:
                    acc_info = await self._rpc(AccountInfo(account=p.account, ledger_index="validated"), t=2.0)
                    if acc_info.is_successful():
                        ledger_seq = acc_info.result["account_data"]["Sequence"]

                        if ledger_seq > p.sequence:
                            # Sequence was consumed by a DIFFERENT transaction!
                            log.error(f"⚠️ SEQUENCE COLLISION: tx {p.tx_hash[:8]} seq={p.sequence} never included, "
                                      f"but account sequence is now {ledger_seq}. Different tx consumed this sequence!")
                            log.error(f"   Possible causes: transaction malleability, concurrent submission, resubmission with modifications")

                            # Mark as REJECTED, not EXPIRED - this is a different failure mode
                            # Do NOT reset sequence tracking - the ledger sequence is correct
                            p.state = C.TxState.REJECTED
                            await self.store.mark(p.tx_hash, state=C.TxState.REJECTED,
                                                  message=f"Sequence collision - seq {p.sequence} consumed by different tx")
                            return p.state
                except Exception as e:
                    log.debug(f"Failed to check account sequence for collision detection: {e}")
                    # Fall through to normal expiry handling

            # Normal expiry - transaction never made it and sequence gap exists on ledger
            log.debug(f"TX EXPIRED: {p.tx_hash[:8]} - latest_ledger={latest_val} > last_ledger_seq={p.last_ledger_seq} + grace={grace}")
            await self.record_expired(p.tx_hash)
            return C.TxState.EXPIRED

        # Transaction is still within LastLedgerSequence window - keep current state
        # RETRYABLE stays RETRYABLE (terPRE_SEQ waiting for prior seq to validate)
        # SUBMITTED stays SUBMITTED (tesSUCCESS waiting for validation)
        return p.state

    async def submit_signed_tx_blobs(self, items: list):
        """
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ⚠️  DEAD CODE - NOT CURRENTLY USED                                    ┃
        ┃                                                                        ┃
        ┃ This function was written for fire-and-forget bulk submission         ┃
        ┃ without state tracking overhead. All current workload patterns        ┃
        ┃ now use build_sign_and_track() + submit_pending() for proper          ┃
        ┃ tracking, validation checking, and sequence management.               ┃
        ┃                                                                        ┃
        ┃ If you need untracked bulk submission in the future, this is here,    ┃
        ┃ but be aware it bypasses ALL state management and recording.          ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        """
        def _to_blob(x):
            if isinstance(x, str):
                return x
            if isinstance(x, (tuple, list)):
                return x[0]
            blob = getattr(x, "signed_blob_hex", None)
            if isinstance(blob, str):
                return blob
            raise TypeError(f"unsupported tx item: {type(x)}")

        blobs = [_to_blob(i) for i in items]
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self.client.request(SubmitOnly(tx_blob=b))) for b in blobs]
        return [t.result().result for t in tasks]

    async def _is_account_active(self, address: str) -> bool:
        try:
            r = await self.client.request(AccountInfo(account=address, ledger_index="validated"))
            return r.is_successful()
        except Exception:
            return False  # TODO: Say something?

    async def _ensure_funded(self, wallet: Wallet, amt_drops: str):
        """Fund a wallet from the workload funding_wallet account if it hasn't been created yet."""
        if await self._is_account_active(wallet.address):
            return
        amt_drops = str(amt_drops)
        fund_tx = Payment(
            account=self.funding_wallet.address,
            destination=wallet.address,
            amount=amt_drops,
        )

        p = await self.build_sign_and_track(fund_tx, self.funding_wallet)

        await debug_last_tx(self.client, p.account)
        await self.submit_pending(p)
        log.debug(f"Funding {wallet.address} with {int(xrpl.utils.drops_to_xrp(amt_drops))} XRP - waiting for validation...")

        # Wait for the funding transaction to validate before continuing
        # This ensures the account exists on ledger before we try to set flags
        await self.wait_for_validation(p.tx_hash)
        log.debug(f"✓ Funded {wallet.address} - account now active on ledger")

        await debug_last_tx(self.client, p.account)

    async def _submit_funding_no_wait(self, wallet: Wallet, amt_drops: str) -> str | None:
        """
        Submit funding transaction for wallet without waiting for validation.

        Returns tx_hash if submitted, None if account already exists.
        This allows batch submission of multiple funding transactions to fill ledgers efficiently.
        """
        if await self._is_account_active(wallet.address):
            return None

        amt_drops = str(amt_drops)
        fund_tx = Payment(
            account=self.funding_wallet.address,
            destination=wallet.address,
            amount=amt_drops,
        )

        p = await self.build_sign_and_track(fund_tx, self.funding_wallet)
        await self.submit_pending(p)

        log.debug(f"Submitted funding for {wallet.address[:8]}... ({int(xrpl.utils.drops_to_xrp(amt_drops))} XRP) - tx: {p.tx_hash[:8]}...")
        return p.tx_hash

    async def _acctset_flags(self, wallet: Wallet, *, require_auth=False, default_ripple=True):
        flags = []
        if require_auth:
            flags.append(AccountSetAsfFlag.ASF_REQUIRE_AUTH)
        if default_ripple:
            flags.append(AccountSetAsfFlag.ASF_DEFAULT_RIPPLE)
        for f in flags:
            t = AccountSet(account=wallet.address, set_flag=f)
            p = await self.build_sign_and_track(t, wallet)
            log.debug("Submitting AccountSet")
            await self.submit_pending(p)
            log.debug("Submitted AccountSet %s", p.tx_json)
            log.debug(json.dumps(p.tx_json))

    # TODO: Default constants
    async def wait_for_validation(self, tx_hash: str, *, overall: float = 15.0, per_rpc: float = 2.0) -> dict:
        from xrpl.models.requests import Tx

        try:
            async with asyncio.timeout(overall):
                while True:
                    r = await asyncio.wait_for(self.client.request(Tx(transaction=tx_hash)), timeout=per_rpc)
                    if r.result.get("validated"):
                        return r.result
                    await asyncio.sleep(0.5)
        except TimeoutError:
            return {"validated": False, "timeout": True}

    # TODO: Default constants
    async def bootstrap_gateway(self, w, *, drops=1_000_000_000, require_auth=False, default_ripple=False):
        fund = Payment(account=w.address, destination=w.address, amount=str(drops))  # or from funder→w
        p0 = await self.build_sign_and_track(fund, self.funding_wallet)
        await self.submit_pending(p0)

        flags = []
        if require_auth:
            flags.append(AccountSetAsfFlag.ASF_REQUIRE_AUTH)
        if default_ripple:
            flags.append(AccountSetAsfFlag.ASF_DEFAULT_RIPPLE)

        pendings = []
        for f in flags:
            tx = AccountSet(account=w.address, set_flag=f)
            p = await self.build_sign_and_track(tx, w)  # allocator hands next Sequence
            pendings.append(p)
            await self.submit_pending(p)

        for p in [p0, *pendings]:
            _ = await self.wait_for_validation(p.tx_hash, overall=15.0)

    async def _apply_gateway_flags(self, *, req_auth: bool, def_ripple: bool) -> dict[str, Any]:
        """Apply per-gateway account flags. One AccountSet per asf flag.

        IMPORTANT: Waits for all AccountSet transactions to validate before returning,
        ensuring flags are active before subsequent operations (TrustSets, Payments).
        """
        flags: list[AccountSetAsfFlag] = []
        if req_auth:
            flags.append(AccountSetAsfFlag.ASF_REQUIRE_AUTH)
        if def_ripple:
            flags.append(AccountSetAsfFlag.ASF_DEFAULT_RIPPLE)

        if not flags or not self.gateways:
            return {"applied": 0, "results": []}

        results: list[dict[str, Any]] = []
        tx_hashes: list[str] = []

        for w in self.gateways:
            addr = w.classic_address
            for f in flags:
                tx = AccountSet(account=addr, set_flag=f)

                # must sign with this wallet and use its sequence
                p = await self.build_sign_and_track(tx, w)

                # give startup breathing room
                res = await self.submit_pending(p, timeout=max(getattr(self, "rpc_timeout", 3.0), 15.0))

                er = (res or {}).get("engine_result")
                txh = (res or {}).get("tx_json", {}).get("hash") if res else None

                results.append(
                    {
                        "address": addr,
                        "flag": f.name,
                        "engine_result": er,
                        "tx_hash": txh,
                        "state": p.state.name,
                    }
                )

                if isinstance(er, str) and er != "tesSUCCESS":
                    log.error("AccountSet failed addr=%s flag=%s res=%s", addr, f.name, res)
                elif txh:
                    tx_hashes.append(txh)

        # CRITICAL: Wait for all AccountSet transactions to validate before proceeding
        # This ensures gateway flags are active before TrustSets and token distribution
        if tx_hashes:
            log.debug(f"Waiting for {len(tx_hashes)} AccountSet (gateway flags) transactions to validate...")
            max_wait_ledgers = 10
            start_ledger = await self._current_ledger_index()

            for ledger_offset in range(1, max_wait_ledgers + 1):
                target_ledger = start_ledger + ledger_offset
                while await self._current_ledger_index() < target_ledger:
                    await asyncio.sleep(0.5)

                validated_count = sum(
                    1 for tx_hash in tx_hashes
                    if self.pending.get(tx_hash, PendingTx(tx_hash="", signed_blob_hex="", account="",
                        tx_json={}, sequence=None, last_ledger_seq=0, transaction_type=None,
                        created_ledger=0)).state == C.TxState.VALIDATED
                )

                if validated_count == len(tx_hashes):
                    log.debug(f"✓ All {len(tx_hashes)} gateway flag AccountSets validated after {ledger_offset} ledger(s)")
                    break
                log.debug(f"  After ledger {target_ledger}: {validated_count}/{len(tx_hashes)} gateway flags validated")
            else:
                # If flags don't validate, this could cause issues with require_auth
                log.error(f"⚠ CRITICAL: only {validated_count}/{len(tx_hashes)} gateway flags validated after {max_wait_ledgers} ledgers")
                log.error(f"⚠ Aborting - proceeding without validated flags could cause authorization issues!")
                raise RuntimeError(f"Gateway flag validation timeout: {validated_count}/{len(tx_hashes)} validated")

        return {"applied": len(flags) * len(self.gateways), "results": results}

    # TODO: Default constants
    async def wait_until_validated(
        self, tx_hash: str, *, overall: float = 15.0, per_rpc: float = 2.0
    ) -> dict[str, Any]:
        """Block until tx validated, rejected, or timeout. Returns the final Tx result dict.

        Invariants on success:
          - result["validated"] is True
          - result["ledger_index"] is an int
          - result["meta"]["TransactionResult"] is a str
        """
        try:
            async with asyncio.timeout(overall):
                while True:
                    r = await asyncio.wait_for(
                        self.client.request(Tx(transaction=tx_hash)),
                        timeout=per_rpc,
                    )
                    result: dict[str, Any] = r.result

                    if not result.get("validated"):
                        await asyncio.sleep(0.5)
                        continue

                    # Enforce invariants
                    meta = result.get("meta")
                    if not isinstance(meta, dict) or "TransactionResult" not in meta:
                        raise RuntimeError(f"Validated response missing meta.TransactionResult for {tx_hash}")
                    ledger_index = result.get("ledger_index")
                    if not isinstance(ledger_index, int):
                        raise RuntimeError(f"Validated response missing integer ledger_index for {tx_hash}")

                    meta_result: str = meta["TransactionResult"]

                    # Single recording path; store can attach meta_result for counts/diagnostics
                    await self.record_validated(
                        ValidationRecord(txn=tx_hash, seq=ledger_index, src=ValidationSrc.POLL),
                        meta_result=meta_result,
                    )
                    return result

        except TimeoutError:
            log.debug("Validation timeout tx=%s after %.1fs", tx_hash, overall)
            return {"validated": False, "timeout": True}

    async def submit_random_txn(self, n: int | None = None):
        txn = await generate_txn(self.ctx)
        pending_txn = await self.build_sign_and_track(txn, self.wallets[txn.account])
        x = await self.submit_pending(pending_txn)
        log.debug(f"Submitting random {txn.transaction_type.name.title().replace('_', ' ')} txn.")
        log.debug(x)
        return x

    async def create_transaction(self, transaction: str):
        """
        Build, sign, track, and submit a transaction.
        If auto_expand_on_payment is True and this is a Payment, we replace the destination
        with a new wallet we control, then adopt it into the pool after validation.
        """
        txn = await generate_txn(self.ctx, transaction)
        pending = await self.build_sign_and_track(txn, self.wallets[txn.account])
        return await self.submit_pending(pending)

    async def create_account(self, initial_xrp_drops: str | None = None, wait=True) -> dict[str, Any]:
        """
        Randomly generate a wallet, fund it from funding_wallet, and adopt it
        into the pool *after* validation via record_validated().
        Returns basic submission info (tx hash) so callers can monitor if desired.
        """
        if not getattr(self, "funding_wallet", None):
            raise RuntimeError("No funding_wallet configured")

        # amount to fund (drops)
        amount = initial_xrp_drops or self.config["users"]["default_balance"]

        # 1) mint a brand-new wallet (keys we control)
        w = Wallet.create()

        # 2) funding payment
        fund_txn = Payment(
            account=self.funding_wallet.address,
            destination=w.address,
            amount=str(amount),
        )

        # 3) sign/track/submit
        pending = await self.build_sign_and_track(fund_txn, self.funding_wallet)
        pending.wallet = w  # stash for *post-validation* adoption in record_validated()
        submit_res = await self.submit_pending(pending)

        # 4) return minimal facts; adoption happens later on validation
        return {
            "address": w.address,
            "tx_hash": pending.tx_hash,
            "submitted": True,
            "engine_result": (submit_res or {}).get("engine_result"),
            "funding_drops": int(amount),
        }

    # ============================================== #
    # ============ Initialization Stuff============= #
    # ============================================== #

    async def _establish_trust_lines(self) -> None:
        """Create TrustSet transactions from all users to all configured currencies.

        Submits all TrustSets in parallel (order doesn't matter), then waits for all to validate.
        """
        if not self.users or not self._currencies:
            log.debug("No users or currencies to establish trust lines")
            return

        trust_limit = str(self.config["transactions"]["trustset"]["limit"])

        log.debug(f"\033[96m{'='*80}\033[0m")
        log.debug(f"\033[96mEstablishing trust lines: {len(self.users)} users × {len(self._currencies)} currencies = {len(self.users) * len(self._currencies)} TrustSets\033[0m")
        log.debug(f"\033[96mSubmitting all TrustSets in parallel (order doesn't matter)\033[0m")
        log.debug(f"\033[96m{'='*80}\033[0m")

        trustset_count = 0
        result_counts = {}
        trustset_hashes = []

        # Build list of work to do (user, currency pairs)
        work_items = []
        for user in self.users:
            for currency in self._currencies:
                work_items.append((user, currency))

        # Submit in batches, building just before submission to avoid expiry
        ledger_size = await self._expected_ledger_size()
        batch_size = max(10, ledger_size // 2)  # Submit 1/3 of ledger capacity at a time, min 10
        total_batches = (len(work_items) + batch_size - 1) // batch_size

        log.debug(f"Will create {len(work_items)} TrustSets, submitting in {total_batches} batches of ~{batch_size}...")
        log.debug(f"Ledger capacity: {ledger_size} txns, batch size: {batch_size} txns")

        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(work_items))
            batch_work = work_items[batch_start:batch_end]

            log.debug(f"  Building and submitting batch {batch_num + 1}/{total_batches}: {len(batch_work)} TrustSets...")

            batch = []
            for user, currency in batch_work:
                trustset_count += 1
                trust_tx = TrustSet(
                    account=user.address,
                    limit_amount=IssuedCurrencyAmount(
                        currency=currency.currency,
                        issuer=currency.issuer,
                        value=trust_limit,
                    ),
                )
                pending = await self.build_sign_and_track(trust_tx, user)
                batch.append(pending)
                trustset_hashes.append(pending.tx_hash)

            async with asyncio.TaskGroup() as tg:
                submit_tasks = [
                    tg.create_task(self.submit_pending(p))
                    for p in batch
                ]

            for task in submit_tasks:
                try:
                    result = task.result()
                    engine_result = result.get('engine_result') if result else 'None'
                    result_counts[engine_result] = result_counts.get(engine_result, 0) + 1
                except Exception as e:
                    log.error(f"TrustSet submission failed: {e}")
                    result_counts['ERROR'] = result_counts.get('ERROR', 0) + 1

            if batch_num < total_batches - 1:
                current_ledger = await self._current_ledger_index()
                next_ledger = current_ledger + 3
                log.debug(f"  Waiting for ledger {next_ledger} (current: {current_ledger}) before next batch...")
                while await self._current_ledger_index() < next_ledger:
                    await asyncio.sleep(1.0)

        for result_code, count in sorted(result_counts.items()):
            color = "\033[92m" if result_code == "tesSUCCESS" else "\033[91m"
            log.debug(f"\033[96m  {color}{result_code}: {count}\033[0m")
        log.debug(f"\033[96m{'='*80}\033[0m")

        # Wait for all TrustSets to validate before proceeding to token distribution
        log.debug(f"\033[96mWaiting for all TrustSets to validate (checking after each ledger close)...\033[0m")
        start_ledger = await self._current_ledger_index()
        max_ledgers = 20  # Wait up to 20 ledgers (enough for 640 TrustSets at ~50/ledger)

        for ledger_offset in range(1, max_ledgers + 1):
            # Wait for next ledger to close
            target_ledger = start_ledger + ledger_offset
            while await self._current_ledger_index() < target_ledger:
                await asyncio.sleep(0.5)

            validated_count = sum(
                1 for tx_hash in trustset_hashes
                if self.pending.get(tx_hash, PendingTx(tx_hash="", signed_blob_hex="", account="",
                    tx_json={}, sequence=None, last_ledger_seq=0, transaction_type=None,
                    created_ledger=0)).state == C.TxState.VALIDATED
            )

            if validated_count == len(trustset_hashes):
                log.debug(f"\033[92m✓ All {len(trustset_hashes)} TrustSets validated after {ledger_offset} ledger(s) (ledger {target_ledger})\033[0m")
                break
            log.debug(f"\033[93m  After ledger {target_ledger}: {validated_count}/{len(trustset_hashes)} TrustSets validated\033[0m")
        else:
            # CRITICAL: If TrustSets don't validate, token distribution will fail with tecPATH_DRY
            # Users cannot receive tokens without established trust lines
            log.error(f"\033[91m⚠ CRITICAL: only {validated_count}/{len(trustset_hashes)} TrustSets validated after {max_ledgers} ledgers\033[0m")
            log.error(f"\033[91m⚠ Aborting - token distribution would fail without trust lines!\033[0m")
            raise RuntimeError(f"TrustSet validation timeout: {validated_count}/{len(trustset_hashes)} validated")

    async def _distribute_initial_tokens(self) -> None:
        """Distribute tokens using CASCADE/TREE pattern for speed.

        Instead of gateways sending to all users sequentially:
        - Round 1: 2 gateways → 2 users (2 txns in parallel)
        - Round 2: 2 users → 4 users (2 txns in parallel)
        - Round 3: 4 users → 8 users (4 txns in parallel)
        - etc.

        This is O(log n) rounds instead of O(n) sequential transactions!
        """
        if not self.users or not self._currencies:
            log.debug("No users or currencies to distribute tokens")
            return

        initial_amount = float(self.config.get("currencies", {}).get("token_distribution", 1_000_000))
        recipients_per_sender = 2  # Binary tree: each sender sends to 2 recipients

        log.debug(f"\033[95m{'='*80}\033[0m")
        log.debug(f"\033[95m🌳 CASCADE TOKEN DISTRIBUTION (tree pattern)\033[0m")
        log.debug(f"\033[95m   {len(self._currencies)} currencies × {len(self.users)} users\033[0m")
        log.debug(f"\033[95m   Initial amount: {initial_amount:,.0f} (halves each round)\033[0m")
        log.debug(f"\033[95m{'='*80}\033[0m")

        total_payments = 0
        result_counts = {}

        # Distribute each currency separately using cascade pattern
        for currency in self._currencies:
            issuer_wallet = self.wallets.get(currency.issuer)
            if not issuer_wallet:
                log.error(f"\033[91m✗ Cannot find wallet for issuer {currency.issuer}\033[0m")
                continue

            log.debug(f"\033[93m📊 Distributing {currency.currency} from {currency.issuer[:12]}...\033[0m")

            # Start with issuer as the only account with tokens
            # Track how much each account has/will receive
            account_balances = {}  # Will track expected balances after validation
            have_tokens = [issuer_wallet]
            need_tokens = list(self.users)  # All users need tokens
            round_num = 0

            while need_tokens:
                round_num += 1
                round_payments = []
                round_tx_hashes = []

                # Calculate amount for THIS round: halves each round
                # Round 1: initial_amount (1M)
                # Round 2: initial_amount / 2 (500k)
                # Round 3: initial_amount / 4 (250k)
                # This ensures senders have enough to distribute
                round_amount = initial_amount / (recipients_per_sender ** (round_num - 1))
                round_amount_str = str(int(round_amount))

                # Each account that has tokens sends to up to 2 accounts that don't
                senders_this_round = have_tokens.copy()
                new_recipients = []

                log.warning(f"\n🌊 Round {round_num} for {currency.currency}:")
                log.warning(f"   Amount per payment: {round_amount:,.0f}")
                log.warning(f"   Senders who have tokens: {len(senders_this_round)}")
                log.warning(f"   Recipients who need tokens: {len(need_tokens)}")

                for sender in senders_this_round:
                    # Send to up to 2 recipients
                    for _ in range(min(recipients_per_sender, len(need_tokens))):
                        if not need_tokens:
                            break

                        recipient = need_tokens.pop(0)
                        new_recipients.append(recipient)

                        # Track expected balance for recipient
                        account_balances[recipient.address] = round_amount

                        # Build payment transaction
                        payment_tx = Payment(
                            account=sender.address,
                            destination=recipient.address,
                            amount=IssuedCurrencyAmount(
                                currency=currency.currency,
                                issuer=currency.issuer,
                                value=round_amount_str,
                            ),
                        )
                        pending = await self.build_sign_and_track(payment_tx, sender)
                        round_payments.append(pending)
                        round_tx_hashes.append(pending.tx_hash)
                        log.warning(f"   📤 {sender.address[:8]}... → {recipient.address[:8]}... | tx: {pending.tx_hash[:8]}")

                # Submit all payments for this round in parallel
                async with asyncio.TaskGroup() as tg:
                    submit_tasks = [
                        tg.create_task(self.submit_pending(p))
                        for p in round_payments
                    ]

                # Collect results
                submit_results = {}
                for i, task in enumerate(submit_tasks):
                    try:
                        result = task.result()
                        engine_result = result.get('engine_result') if result else 'None'
                        result_counts[engine_result] = result_counts.get(engine_result, 0) + 1
                        submit_results[round_tx_hashes[i]] = engine_result
                        total_payments += 1

                        if engine_result != 'tesSUCCESS':
                            log.warning(f"   ⚠️  {round_tx_hashes[i][:8]} submitted with {engine_result}")
                    except Exception as e:
                        log.error(f"Payment submission failed: {e}")
                        result_counts['ERROR'] = result_counts.get('ERROR', 0) + 1

                # CRITICAL: Wait for ALL transactions in this round to VALIDATE before proceeding
                if need_tokens:
                    log.warning(f"   ⏳ Waiting for {len(round_tx_hashes)} transactions to validate...")
                    max_wait_ledgers = 10
                    start_ledger = await self._current_ledger_index()
                    validated_count = 0

                    for wait_round in range(max_wait_ledgers):
                        await asyncio.sleep(1.0)  # Wait for ledger close
                        current = await self._current_ledger_index()

                        # Check validation status of all round transactions
                        validated_count = 0
                        failed_count = 0
                        for tx_hash in round_tx_hashes:
                            p = self.pending.get(tx_hash)
                            if p and p.state == C.TxState.VALIDATED:
                                validated_count += 1
                            elif p and p.state in {C.TxState.REJECTED, C.TxState.EXPIRED}:
                                failed_count += 1
                                log.error(f"   ❌ {tx_hash[:8]} FAILED with state {p.state.name}")

                        if validated_count == len(round_tx_hashes):
                            log.warning(f"   ✅ All {validated_count} transactions validated (ledger {current})")
                            break
                        elif validated_count + failed_count == len(round_tx_hashes):
                            log.error(f"   ⚠️  {validated_count} validated, {failed_count} failed - stopping cascade")
                            need_tokens.clear()  # Abort cascade
                            break
                        else:
                            log.warning(f"   ⏳ Ledger {current}: {validated_count}/{len(round_tx_hashes)} validated...")
                    else:
                        # Timeout - not all validated
                        log.error(f"   ⏰ TIMEOUT: Only {validated_count}/{len(round_tx_hashes)} validated after {max_wait_ledgers} ledgers")
                        log.error(f"   ⚠️  Aborting cascade for {currency.currency} to prevent tecPATH_DRY")
                        need_tokens.clear()  # Abort to prevent cascade failures

                # Only add recipients to have_tokens if their transactions validated
                for i, recipient in enumerate(new_recipients):
                    tx_hash = round_tx_hashes[i]
                    p = self.pending.get(tx_hash)
                    if p and p.state == C.TxState.VALIDATED:
                        have_tokens.append(recipient)
                    else:
                        log.error(f"   ❌ Skipping {recipient.address[:8]} (tx {tx_hash[:8]} not validated)")

            log.debug(f"\033[92m  ✓ {currency.currency} distribution complete in {round_num} rounds\033[0m")

        log.debug(f"\033[95m{'='*80}\033[0m")
        log.debug(f"\033[95m✓ Token distribution complete: {total_payments} total payments\033[0m")
        for result_code, count in sorted(result_counts.items()):
            color = "\033[92m" if result_code == "tesSUCCESS" else "\033[91m"
            log.debug(f"\033[95m  {color}{result_code}: {count}\033[0m")
        log.debug(f"\033[95m{'='*80}\033[0m")

    async def init_participants(self, *, gateway_cfg: dict[str, Any], user_cfg: dict[str, Any]) -> dict:
        out_gw, out_us = [], []
        req_auth = gateway_cfg["require_auth"]
        def_ripple = gateway_cfg["default_ripple"]

        # ============================================================
        # BATCH GATEWAY CREATION & FUNDING
        # ============================================================
        g = gateway_cfg['number']
        log.debug(f"Creating {g} gateway wallets...")

        # Step 1: Create all gateway wallets
        gateway_wallets = []
        for _ in range(g):
            w = Wallet.create()
            self.wallets[w.address] = w
            self._record_for(w.address)  # BUG: Only record address in workload after validated on ledger
            self.gateways.append(w)
            gateway_wallets.append(w)
            out_gw.append(w.address)

        # Step 2: Build, sign, and submit all funding transactions
        # Sequence allocation happens sequentially (required for same source account)
        # but network submissions happen in parallel for efficiency
        log.info(f"🚀 Building and submitting funding for {len(gateway_wallets)} gateways...")
        gateway_funding_hashes = []

        # Build and sign all transactions (allocates sequences sequentially)
        pending_submissions = []
        for w in gateway_wallets:
            if await self._is_account_active(w.address):
                continue

            amt_drops = str(self.config["gateways"]["default_balance"])
            fund_tx = Payment(
                account=self.funding_wallet.address,
                destination=w.address,
                amount=amt_drops,
            )

            p = await self.build_sign_and_track(fund_tx, self.funding_wallet)
            pending_submissions.append(p)
            gateway_funding_hashes.append(p.tx_hash)
            log.debug(f"Built funding tx for {w.address[:8]}... - {p.tx_hash[:8]}... seq={p.sequence}")

        # Submit all transactions in parallel
        log.info(f"🚀 Submitting {len(pending_submissions)} gateway funding transactions in parallel...")
        submit_tasks = [self.submit_pending(p) for p in pending_submissions]
        await asyncio.gather(*submit_tasks)

        # Step 3: Wait for all gateway funding to validate
        log.info(f"⏳ Waiting for {len(gateway_funding_hashes)} gateway funding transactions to validate...")
        await asyncio.gather(*[self.wait_for_validation(h) for h in gateway_funding_hashes])
        log.info(f"✓ All {len(gateway_wallets)} gateways funded and validated")

        # Step 4: Persist gateway wallets
        for w in gateway_wallets:
            self.save_wallet_to_store(w, is_gateway=True)

        # Create currencies issued by each gateway
        currency_codes = self.config["currencies"]["codes"][:2]
        for gateway in self.gateways:
            gateway_currencies = issue_currencies(gateway.address, currency_codes)
            self._currencies.extend(gateway_currencies)
            log.debug(f"Gateway {gateway.address[:8]}... will issue {len(gateway_currencies)} currencies: {currency_codes}")

        # Update context with new currencies
        self.update_txn_context()

        # ============================================================
        # BATCH USER CREATION & FUNDING
        # ============================================================
        u = user_cfg['number']
        log.debug(f"Creating {u} user wallets...")

        # Step 1: Create all user wallets
        user_wallets = []
        for _ in range(u):
            w = Wallet.create()
            self.wallets[w.address] = w
            self._record_for(w.address)
            self.users.append(w)
            user_wallets.append(w)
            out_us.append(w.address)

        # Step 2: Build, sign, and submit all user funding transactions
        # Sequence allocation happens sequentially (required for same source account)
        # but network submissions happen in parallel for efficiency
        log.info(f"🚀 Building and submitting funding for {len(user_wallets)} users...")
        user_funding_hashes = []

        # Build and sign all transactions (allocates sequences sequentially)
        pending_submissions = []
        for w in user_wallets:
            if await self._is_account_active(w.address):
                continue

            amt_drops = str(self.config["users"]["default_balance"])
            fund_tx = Payment(
                account=self.funding_wallet.address,
                destination=w.address,
                amount=amt_drops,
            )

            p = await self.build_sign_and_track(fund_tx, self.funding_wallet)
            pending_submissions.append(p)
            user_funding_hashes.append(p.tx_hash)
            log.debug(f"Built funding tx for {w.address[:8]}... - {p.tx_hash[:8]}... seq={p.sequence}")

        # Submit all transactions in parallel
        log.info(f"🚀 Submitting {len(pending_submissions)} user funding transactions in parallel...")
        submit_tasks = [self.submit_pending(p) for p in pending_submissions]
        await asyncio.gather(*submit_tasks)

        # Step 3: Wait for all user funding to validate
        log.info(f"⏳ Waiting for {len(user_funding_hashes)} user funding transactions to validate...")
        await asyncio.gather(*[self.wait_for_validation(h) for h in user_funding_hashes])
        log.info(f"✓ All {len(user_wallets)} users funded and validated")

        # Step 4: Persist user wallets
        for w in user_wallets:
            self.save_wallet_to_store(w, is_user=True)

        if req_auth or def_ripple:
            await self._apply_gateway_flags(req_auth=req_auth, def_ripple=def_ripple)

        # Establish trust lines: users create TrustSet for all currencies
        log.debug(f"Establishing trust lines for {len(self.users)} users across {len(self._currencies)} currencies")
        await self._establish_trust_lines()

        # Distribute initial tokens: gateways send tokens to users
        log.debug(f"Distributing initial tokens from {len(self.gateways)} gateways to {len(self.users)} users")
        await self._distribute_initial_tokens()

        # Persist currencies after initialization
        self.save_currencies_to_store()

        return {"gateways": out_gw, "users": out_us}

    # ============================================== #
    # =============== Utility Stuff ================ #
    # ============================================== #

    async def _post(self, url: str, payload: dict):
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(url, json=payload)
                response = resp.json()
            except Exception as e:
                pass
                # responses.append({"error": str(e)})
        try:
            response = resp.json()
        except:
            pass
        finally:
            return response

    async def validator_state(self, n: int):
        # FIX: this to work locally, in Docker we'd use the ${val_name}{index}
        import subprocess

        val = f"val{n}"
        cmd = ["docker", "inspect", "-f", "'{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "]
        cmd.append(val)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        val_ip = result.stdout.strip().replace("'", "")
        rpc_port = 5005
        return f"http://{val_ip}:{rpc_port}"

    def snapshot_pending(self, *, open_only: bool = True) -> list[dict]:
        OPEN_STATES = {C.TxState.CREATED, C.TxState.SUBMITTED, C.TxState.RETRYABLE, C.TxState.FAILED_NET}  # TODO: Move
        out = []
        for txh, p in self.pending.items():
            if open_only and p.state not in OPEN_STATES:
                continue
            out.append(
                {
                    "tx_hash": txh,
                    "state": p.state.name,
                    "account": p.account,
                    "sequence": p.sequence,
                    "last_ledger_seq": p.last_ledger_seq,
                    "created_ledger": p.created_ledger,
                    "attempts": p.attempts,
                    "engine_result_first": p.engine_result_first,
                    "validated_ledger": p.validated_ledger,
                    "meta_txn_result": p.meta_txn_result,
                }
            )
        return out

    def get_accounts_with_pending_txns(self) -> set[str]:
        """Get set of account addresses that have pending (non-terminal) transactions.

        Returns:
            Set of account addresses with CREATED, SUBMITTED, or RETRYABLE transactions
        """
        PENDING_STATES = {C.TxState.CREATED, C.TxState.SUBMITTED, C.TxState.RETRYABLE}
        accounts = set()
        for p in self.pending.values():
            if p.state in PENDING_STATES and p.account:
                accounts.add(p.account)
        return accounts

    def snapshot_finalized(self) -> list[dict]:
        # FINAL_STATES = {C.TxState.VALIDATED, C.TxState.REJECTED, C.TxState.EXPIRED}
        return [r for r in self.snapshot_pending(open_only=False) if r["state"] in {s.name for s in TERMINAL_STATE}]

    def snapshot_failed(self) -> list[dict[str, Any]]:
        failed_states = {"REJECTED", "EXPIRED", "FAILED_NET"}
        return [r for r in self.snapshot_pending(open_only=False) if r["state"] in failed_states]

    def snapshot_stats(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for p in self.pending.values():
            state = p.state.name
            by_state[state] = by_state.get(state, 0) + 1

        return {
            "total_tracked": len(self.pending),
            "by_state": by_state,
            "gateways": len(self.gateways),
            "users": len(self.users),
        }

    def snapshot_tx(self, tx_hash: str) -> dict[str, Any]:
        p = self.pending.get(tx_hash)
        ws_port = 6006  # TODO: Use the real ws port
        if not p:
            return {}
        return {
            "tx_hash": p.tx_hash,
            "state": p.state.name,
            "account": p.account,
            "sequence": p.sequence,
            "last_ledger_seq": p.last_ledger_seq,
            "created_ledger": p.created_ledger,
            "attempts": p.attempts,
            "engine_result_first": p.engine_result_first,
            "validated_ledger": p.validated_ledger,
            "meta_txn_result": p.meta_txn_result,
            "link": f"https://custom.xrpl.org/localhost:{ws_port}/transactions/{tx_hash}",
        }

    async def submit_heartbeat(self, ledger_index: int) -> dict | None:
        """Submit a 1-drop heartbeat payment for the given ledger.

        This is our canary - we should see exactly ONE heartbeat per ledger.
        If we miss heartbeats, something is wrong with the network or our connection.

        Args:
            ledger_index: The ledger index this heartbeat is for (from ledger close event)

        Returns:
            Submission result or None on failure
        """
        import time
        from xrpl.models.transactions import Payment

        try:
            # Track attempt
            self.last_heartbeat_ledger = ledger_index

            # Build and sign heartbeat directly WITHOUT tracking in store (to avoid polluting metrics)
            # Get sequence number for heartbeat wallet
            seq = await self.alloc_seq(self.heartbeat_wallet.address)

            # Get current ledger for LastLedgerSequence
            lls = await self._last_ledger_sequence_offset(C.HORIZON)

            # Build transaction dict
            from xrpl.core.binarycodec import encode, encode_for_signing
            from xrpl.core.keypairs import sign

            heartbeat_tx_dict = {
                "TransactionType": "Payment",
                "Account": self.heartbeat_wallet.address,
                "Destination": self.heartbeat_destination,
                "Amount": "1",  # 1 drop
                "Sequence": seq,
                "Fee": str(await self._open_ledger_fee()),
                "LastLedgerSequence": lls,
                "SigningPubKey": self.heartbeat_wallet.public_key,
            }

            # Sign transaction
            signing_blob = encode_for_signing(heartbeat_tx_dict)
            to_sign = signing_blob if isinstance(signing_blob, str) else signing_blob.hex()
            heartbeat_tx_dict["TxnSignature"] = sign(to_sign, self.heartbeat_wallet.private_key)

            # Encode signed transaction
            signed_blob_hex = encode(heartbeat_tx_dict)

            # Compute transaction hash
            from xrpl.core.addresscodec import decode_classic_address
            from hashlib import sha512

            def _txid_from_signed_blob_hex(blob_hex: str) -> str:
                """Compute transaction ID from signed blob."""
                blob_bytes = bytes.fromhex(blob_hex)
                hash_prefix = b'\x54\x58\x4E\x00'  # "TXN\0"
                full = hash_prefix + blob_bytes
                h = sha512(full).digest()
                return h[:32].hex().upper()

            tx_hash = _txid_from_signed_blob_hex(signed_blob_hex)

            # Track heartbeat (but NOT in store/pending)
            self.heartbeats[ledger_index] = {
                "tx_hash": tx_hash,
                "submitted_at": time.time(),
                "status": "submitted",
                "sequence": seq,
            }

            log.info(f"💓 HEARTBEAT #{ledger_index}: {tx_hash[:8]}... seq={seq}")

            # Submit directly to ledger (bypass normal tracking)
            from xrpl.models.requests import SubmitOnly
            resp = await asyncio.wait_for(
                self.client.request(SubmitOnly(tx_blob=signed_blob_hex)),
                timeout=C.SUBMIT_TIMEOUT
            )

            result = resp.result
            engine_result = result.get("engine_result")

            # Update heartbeat status
            self.heartbeats[ledger_index]["engine_result"] = engine_result
            self.heartbeats[ledger_index]["status"] = "submitted_ok" if engine_result == "tesSUCCESS" else f"error:{engine_result}"

            # Antithesis assertion - heartbeat success should happen
            if ANTITHESIS_AVAILABLE and engine_result == "tesSUCCESS":
                sometimes(
                    True,
                    "Heartbeat transaction should succeed on every ledger",
                    {
                        "ledger_index": ledger_index,
                        "tx_hash": tx_hash,
                        "sequence": seq,
                        "total_heartbeats": len(self.heartbeats),
                    }
                )

            if engine_result != "tesSUCCESS":
                log.error(f"❌ HEARTBEAT FAILED #{ledger_index}: {engine_result}")
                self.missed_heartbeats.append(ledger_index)

                # Antithesis assertion - missing 10+ heartbeats means we're bonked
                if ANTITHESIS_AVAILABLE and len(self.missed_heartbeats) >= 10:
                    always(
                        False,
                        "Missed 10+ heartbeats - system is bonked",
                        {
                            "ledger_index": ledger_index,
                            "tx_hash": tx_hash,
                            "engine_result": engine_result,
                            "sequence": seq,
                            "missed_count": len(self.missed_heartbeats),
                            "missed_ledgers": self.missed_heartbeats[-10:],  # Last 10 misses
                        }
                    )

            return result

        except Exception as e:
            log.error(f"❌ HEARTBEAT EXCEPTION #{ledger_index}: {e}")
            self.missed_heartbeats.append(ledger_index)
            if ledger_index in self.heartbeats:
                self.heartbeats[ledger_index]["status"] = f"exception:{e}"

            # Antithesis assertion - 10+ exceptions means we're bonked
            if ANTITHESIS_AVAILABLE and len(self.missed_heartbeats) >= 10:
                always(
                    False,
                    "Missed 10+ heartbeats due to exceptions - system is bonked",
                    {
                        "ledger_index": ledger_index,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "missed_count": len(self.missed_heartbeats),
                        "missed_ledgers": self.missed_heartbeats[-10:],  # Last 10 misses
                    }
                )

            return None

    def snapshot_heartbeat(self) -> dict:
        """Get heartbeat status for monitoring.

        Returns:
            Dict with heartbeat stats: last ledger, missed beats, recent beats
        """
        recent_count = 20
        recent_ledgers = sorted(self.heartbeats.keys())[-recent_count:]
        recent_beats = {li: self.heartbeats[li] for li in recent_ledgers}

        return {
            "last_heartbeat_ledger": self.last_heartbeat_ledger,
            "total_heartbeats": len(self.heartbeats),
            "missed_heartbeats": self.missed_heartbeats[-20:],  # Last 20 misses
            "missed_count": len(self.missed_heartbeats),
            "recent_heartbeats": recent_beats,
        }


async def periodic_state_monitor(w: Workload, stop: asyncio.Event, interval: int = 10):
    """Periodically print state summary to monitor workload health.

    Args:
        w: Workload instance
        stop: Event to signal shutdown
        interval: Seconds between state prints (default 10)
    """
    import sys

    while not stop.is_set():
        try:
            await asyncio.sleep(interval)

            # Get state summary
            stats = w.snapshot_stats()
            store_stats = w.store.snapshot_stats()

            # Print compact summary
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"STATE SUMMARY @ {asyncio.get_event_loop().time():.1f}s", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            print(f"Tracked: {stats['total_tracked']} | Gateways: {stats['gateways']} | Users: {stats['users']}", file=sys.stderr)
            print(f"By State: {stats['by_state']}", file=sys.stderr)
            print(f"Store Total: {store_stats['total_tracked']} | Recent Validations: {store_stats['recent_validations']}", file=sys.stderr)
            print(f"Store by State: {store_stats['by_state']}", file=sys.stderr)
            print(f"Store by Type: {store_stats['by_type']}", file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
            sys.stderr.flush()

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[state_monitor] Error printing state summary")
            await asyncio.sleep(1)


async def periodic_finality_check(w: Workload, stop: asyncio.Event, interval: int = 5, max_concurrent: int = 20):
    """Check finality of submitted transactions and retry failed ones with controlled concurrency.

    Args:
        w: Workload instance
        stop: Event to signal shutdown
        interval: Seconds between check cycles
        max_concurrent: Maximum number of concurrent RPC calls (default 20)
    """
    # Semaphore to limit concurrent RPC calls
    sem = asyncio.Semaphore(max_concurrent)

    async def check_with_limit(p):
        """Check finality with semaphore to limit concurrency."""
        async with sem:
            return await w.check_finality(p)

    async def retry_with_limit(p):
        """Retry submission with semaphore to limit concurrency."""
        async with sem:
            return await w.submit_pending(p)

    while not stop.is_set():
        try:
            # Check finality of SUBMITTED transactions
            submitted = list(w.find_by_state(C.TxState.SUBMITTED))
            if len(submitted) > 0:
                log.debug(f"[finality] Checking {len(submitted)} SUBMITTED transactions (max {max_concurrent} concurrent)")

                # Check with controlled concurrency using semaphore
                tasks = [check_with_limit(p) for p in submitted]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Log any failures
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        log.debug(f"[finality] check failed for {submitted[i].tx_hash[:8]}: {result}")

            # Check RETRYABLE transactions for expiry
            # terPRE_SEQ transactions should be checked for expiry (may never validate if prior seq is gone)
            # Other retryable errors can be retried
            retryable = list(w.find_by_state(C.TxState.RETRYABLE))

            # Separate terPRE_SEQ (check for expiry only) from other retryable (can resubmit)
            pre_seq_txns = [p for p in retryable if p.engine_result_first == "terPRE_SEQ"]
            other_retryable = [p for p in retryable if p.engine_result_first != "terPRE_SEQ"]

            # Check terPRE_SEQ for expiry/validation
            if len(pre_seq_txns) > 0:
                log.debug(f"[finality] Checking {len(pre_seq_txns)} terPRE_SEQ transactions for expiry")
                pre_seq_tasks = [check_with_limit(p) for p in pre_seq_txns]
                await asyncio.gather(*pre_seq_tasks, return_exceptions=True)

            # Retry non-terPRE_SEQ retryable transactions
            if len(other_retryable) > 0:
                log.debug(f"[finality] Retrying {len(other_retryable)} RETRYABLE transactions (non-terPRE_SEQ)")

                # Retry with controlled concurrency
                retry_tasks = [retry_with_limit(p) for p in other_retryable]
                retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

                # Log retry results
                for i, result in enumerate(retry_results):
                    if isinstance(result, Exception):
                        log.debug(f"[finality] retry failed for {other_retryable[i].tx_hash[:8]}: {result}")
                    elif isinstance(result, dict):
                        engine_result = result.get("engine_result")
                        if engine_result and engine_result != "tesSUCCESS":
                            log.debug(f"[finality] retry {other_retryable[i].tx_hash[:8]}: {engine_result}")

            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[finality] outer loop error; continuing")
            await asyncio.sleep(0.5)


async def heartbeat_listener(w: Workload, stop: asyncio.Event, ws_url: str):
    """Listen to WebSocket ledger events and submit heartbeat on every ledger close.

    This is our canary - we should see exactly ONE heartbeat transaction per ledger.
    If we miss heartbeats, it indicates network issues, WS disconnection, or other problems.

    Features:
    - Auto-reconnect on WS disconnection
    - RPC fallback if WS is down for too long
    - Tracks missed heartbeats separately from workload metrics

    Args:
        w: Workload instance
        stop: Event to signal shutdown
        ws_url: WebSocket URL (e.g., "ws://localhost:6006")
    """
    import websockets
    from xrpl.models.requests import Subscribe

    last_ledger_seen = None
    ws_failures = 0
    max_ws_failures = 3
    rpc_fallback_active = False

    while not stop.is_set():
        try:
            log.info(f"[heartbeat] Connecting to WebSocket: {ws_url}")
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as websocket:
                # Subscribe to ledger stream
                subscribe_msg = Subscribe(streams=["ledger"])
                await websocket.send(subscribe_msg.to_json())

                log.info(f"[heartbeat] 💓 WebSocket connected, listening for ledger events...")
                ws_failures = 0  # Reset failure count on successful connection
                rpc_fallback_active = False

                # Listen for messages
                async for message in websocket:
                    if stop.is_set():
                        break

                    try:
                        import json
                        msg = json.loads(message)

                        # Handle ledger close events
                        if msg.get("type") == "ledgerClosed":
                            ledger_index = msg.get("ledger_index")

                            if ledger_index is not None:
                                # Submit heartbeat for this ledger
                                log.debug(f"[heartbeat] 💓 Ledger #{ledger_index} closed, submitting heartbeat...")
                                await w.submit_heartbeat(ledger_index)
                                last_ledger_seen = ledger_index

                    except Exception as e:
                        log.error(f"[heartbeat] Error processing message: {e}")
                        continue

        except websockets.exceptions.ConnectionClosed:
            ws_failures += 1
            log.warning(f"[heartbeat] WebSocket connection closed (failures: {ws_failures}/{max_ws_failures})")

            if ws_failures >= max_ws_failures and not rpc_fallback_active:
                log.error(f"[heartbeat] Too many WS failures, activating RPC fallback...")
                rpc_fallback_active = True

        except Exception as e:
            ws_failures += 1
            log.error(f"[heartbeat] WebSocket error (failures: {ws_failures}/{max_ws_failures}): {e}")

        # If WS is failing, use RPC fallback polling
        if rpc_fallback_active or ws_failures > 0:
            try:
                log.warning(f"[heartbeat] Using RPC fallback to check for new ledgers...")

                # Poll for new ledgers via RPC
                from xrpl.models.requests import ServerState

                server_state = await w.client.request(ServerState())
                current_ledger = server_state.result["state"]["validated_ledger"]["seq"]

                # If we see a new ledger, submit heartbeat
                if last_ledger_seen is None or current_ledger > last_ledger_seen:
                    log.warning(f"[heartbeat] 💓 RPC detected new ledger #{current_ledger}, submitting heartbeat...")
                    await w.submit_heartbeat(current_ledger)
                    last_ledger_seen = current_ledger

            except Exception as e:
                log.error(f"[heartbeat] RPC fallback error: {e}")

        # Wait before retry
        await asyncio.sleep(2 if not stop.is_set() else 0)

    log.info("[heartbeat] Heartbeat listener shutting down...")
