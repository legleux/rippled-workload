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
        # TODO: Let's try to ensure we never get "UNKNOWN"
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
        self.gateway_names: dict[str, str] = {}  # address -> name mapping

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
        # Add MPToken issuance IDs
        ctx.mptoken_issuance_ids = self._mptoken_issuance_ids
        # Add in-memory balance tracking for fan-out
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

        # Include funding wallet + all tracked wallets (gateways + users)
        addresses = [self.funding_wallet.address]
        addresses.extend(self.wallets.keys())
        return addresses

    def get_account_display_name(self, address: str) -> str:
        """Get display name for an account (gateway name if available, else address)."""
        return self.gateway_names.get(address, address)

    # =========================================================================
    # State persistence - load and save workload state from SQLite
    # =========================================================================

    def load_state_from_store(self) -> bool:
        """Load workload state from SQLite store if available.

        On hot-reload, this allows us to skip re-creating accounts and TrustSets.
        We clear any pending transactions (stale from previous session) and
        reset sequence numbers from on-chain state.

        Returns:
            True if state was loaded, False otherwise
        """
        from workload.sqlite_store import SQLiteStore

        if not isinstance(self.store, SQLiteStore):
            log.warning("Store is not SQLiteStore, cannot load state")
            return False

        if not self.store.has_state():
            log.debug("No persisted state found in database")
            return False

        log.debug("Loading workload state from database...")

        # Clear any stale pending transactions from previous session
        self.pending.clear()

        # Load wallets
        gateway_names_from_config = self.config.get("gateways", {}).get("names", [])
        wallet_data = self.store.load_wallets()
        gateway_idx = 0
        for address, (wallet, is_gateway, is_user) in wallet_data.items():
            self.wallets[address] = wallet
            self._record_for(address)

            if is_gateway:
                self.gateways.append(wallet)
                # Restore gateway name association
                if gateway_idx < len(gateway_names_from_config):
                    self.gateway_names[address] = gateway_names_from_config[gateway_idx]
                    gateway_idx += 1
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

        # Validate state consistency: if we have gateways but no currencies, state is incomplete
        if len(self.gateways) > 0 and len(self._currencies) == 0:
            log.warning("Incomplete state detected: gateways exist but no currencies found. Rejecting loaded state.")
            # Clear the partial state
            self.wallets.clear()
            self.gateways.clear()
            self.users.clear()
            self.gateway_names.clear()
            self._currencies = []
            return False

        # Update transaction context with loaded wallets
        self.update_txn_context()

        return True

    def save_wallet_to_store(self, wallet: Wallet, is_gateway: bool = False, is_user: bool = False, funded_ledger_index: int | None = None) -> None:
        """Save a wallet to the persistent store."""
        from workload.sqlite_store import SQLiteStore

        if isinstance(self.store, SQLiteStore):
            self.store.save_wallet(wallet, is_gateway=is_gateway, is_user=is_user, funded_ledger_index=funded_ledger_index)

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

    # =========================================================================
    # In-memory balance tracking for fan-out token distribution
    # =========================================================================

    def _get_balance(self, account: str, currency: str, issuer: str | None = None) -> float:
        """Get tracked balance for an account."""
        if account not in self.balances:
            return 0.0
        if currency == "XRP":
            return self.balances[account].get("XRP", 0.0)
        else:
            key = (currency, issuer) if issuer else currency
            return self.balances[account].get(key, 0.0)

    def _set_balance(self, account: str, currency: str, value: float, issuer: str | None = None):
        """Set tracked balance for an account."""
        if account not in self.balances:
            self.balances[account] = {}
        if currency == "XRP":
            self.balances[account]["XRP"] = value
        else:
            key = (currency, issuer) if issuer else currency
            self.balances[account][key] = value

    def _update_balance(self, account: str, currency: str, delta: float, issuer: str | None = None):
        """Update (credit/debit) tracked balance for an account."""
        current = self._get_balance(account, currency, issuer)
        self._set_balance(account, currency, current + delta, issuer)

    async def _rpc(self, req, *, t=C.RPC_TIMEOUT):
        return await asyncio.wait_for(self.client.request(req), timeout=t)

    async def alloc_seq(self, addr: str) -> int:
        rec = self._record_for(addr)

        async with rec.lock:
            if rec.next_seq is None:
                ai = await self._rpc(AccountInfo(account=addr, ledger_index="current", strict=True))
                rec.next_seq = ai.result["account_data"]["Sequence"]

            s = rec.next_seq
            rec.next_seq += 1
            return s

    async def release_seq(self, addr: str, seq: int) -> None:
        """Release an allocated sequence number back to the pool.

        Used when a transaction gets a tel* (local) error and never submits to the network.
        Only releases if this was the most recently allocated sequence to avoid gaps.
        """
        rec = self._record_for(addr)
        async with rec.lock:
            # Only rollback if this was the most recently allocated sequence
            if rec.next_seq == seq + 1:
                rec.next_seq = seq
                log.debug(f"Released sequence {seq} for {addr} (local error, never submitted)")
            else:
                log.warning(f"Cannot release sequence {seq} for {addr} - next_seq is {rec.next_seq} (gap would be created)")

    async def _open_ledger_fee(self) -> int:
        """Get the fee required to submit a transaction.

        Uses the fee command to get minimum_fee (queue entry) and open_ledger_fee (immediate).
        Caps at MAX_FEE_DROPS to prevent account drainage during extreme escalation.

        Returns:
            Fee in drops. Returns base_fee (10) if queue is empty, minimum_fee if queue has room,
            or raises ValueError if fees exceed MAX_FEE_DROPS.
        """
        MAX_FEE_DROPS = 1000  # Cap to prevent draining accounts (base is 10, this is 100x)

        fee_info = await self.get_fee_info()
        minimum_fee = fee_info.minimum_fee  # Fee to get into queue
        open_ledger_fee = fee_info.open_ledger_fee  # Fee to skip queue and get into open ledger immediately
        base_fee = fee_info.base_fee

        # If queue is not full, use minimum_fee (usually base_fee when queue is empty)
        # If queue is full, minimum_fee will be higher than base_fee
        fee = minimum_fee

        # Always log fee info for monitoring
        log.debug(
            f"💰 Fee: {fee} drops (min={minimum_fee}, open={open_ledger_fee}, base={base_fee}, "
            f"queue={fee_info.current_queue_size}/{fee_info.max_queue_size}, "
            f"ledger={fee_info.current_ledger_size}/{fee_info.expected_ledger_size})"
        )

        # Log if fees are escalated
        if fee > base_fee:
            log.warning(
                f"⚠️  Queue fees escalated: minimum={minimum_fee} (queue), open_ledger={open_ledger_fee} (immediate), base={base_fee}"
            )

        # Cap at max to prevent account drainage
        if fee > MAX_FEE_DROPS:
            raise ValueError(
                f"Fee too high ({fee} drops > {MAX_FEE_DROPS} max) - queue is full, refusing to drain accounts. "
                f"Wait for queue to clear or increase MAX_FEE_DROPS."
            )

        return fee

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

    async def get_fee_info(self) -> "FeeInfo":
        """Get current fee escalation state from rippled using the fee command.

        Returns FeeInfo with:
        - expected_ledger_size: Dynamic limit for base fee transactions
        - current_ledger_size: Number of transactions in open ledger
        - current_queue_size: Number of transactions in queue
        - max_queue_size: Maximum queue capacity
        - base_fee, median_fee, minimum_fee, open_ledger_fee: Fee levels in drops

        Note: current_ledger_size and current_queue_size change rapidly (per transaction).
        Call this method fresh when you need current values, don't cache.

        See reference/FeeEscalation.md for detailed documentation.
        """
        from xrpl.models.requests import Fee
        from workload.fee_info import FeeInfo

        r = await self.client.request(Fee())
        return FeeInfo.from_fee_result(r.result)

    async def _expected_ledger_size(self) -> int:
        """Get the expected number of transactions per ledger from the server.

        Uses the fee command which returns expected_ledger_size as part of fee escalation state.
        Raises RuntimeError if expected_ledger_size is not available.
        We should never submit transactions if we can't determine capacity.
        """
        fee_info = await self.get_fee_info()
        return fee_info.expected_ledger_size

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
        # async def record_submitted(self, p, engine_result, srv_txid):
        if p.state in TERMINAL_STATE:
            pass  # Don't overwrite terminal states. This should probably be an exception.
            return
        old = p.tx_hash
        new_hash = srv_txid or old
        if srv_txid and srv_txid != old:
            self.pending[new_hash] = self.pending.pop(old, p)
            p.tx_hash = new_hash
            await self.store.rekey(old, new_hash)
        p.state = C.TxState.SUBMITTED
        p.engine_result_first = p.engine_result_first or engine_result
        self.pending[new_hash] = p
        await self.store.mark(new_hash, state=C.TxState.SUBMITTED, account=p.account, sequence=p.sequence,
                              transaction_type=p.transaction_type, engine_result_first=p.engine_result_first)

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
            self.save_wallet_to_store(w, is_user=True, funded_ledger_index=rec.seq)  # Persist with funding ledger
            self.update_txn_context()
            log.debug("Adopted new account after validation: %s", w.address)

        # Update in-memory balances for Payment transactions (for fan-out tracking)
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
                            self._update_balance(sender, "XRP", -amount_val)
                            self._update_balance(destination, "XRP", amount_val)
                        elif isinstance(amount, dict):
                            # IOU payment
                            currency = amount.get("currency")
                            issuer = amount.get("issuer")
                            value = float(amount.get("value", 0))

                            if currency and issuer:
                                # Issuers have infinite balance, don't track debits for them
                                if sender != issuer:
                                    self._update_balance(sender, currency, -value, issuer)
                                if destination != issuer:
                                    self._update_balance(destination, currency, value, issuer)
                                log.debug(f"Balance update: {sender[:8]} -> {destination[:8]}: {value} {currency}")
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
                log.warning(f"Failed to extract MPToken issuance ID from {rec.txn}: {e}")

        # BATCH SEQUENCE SYNC: After Batch validates, sync sequence with ledger
        # Batch consumes: 1 (outer) + N (successful inner txns) sequences
        # We can't predict N, so just ask the ledger what it expects next
        if p_live and p_live.transaction_type == C.TxType.BATCH and p_live.account:
            try:
                ai = await self._rpc(AccountInfo(account=p_live.account, ledger_index="current"))
                rec_acct = self._record_for(p_live.account)
                async with rec_acct.lock:
                    old_seq = rec_acct.next_seq
                    rec_acct.next_seq = ai.result["account_data"]["Sequence"]
                    log.info(f"Batch validated: synced {p_live.account[:8]} sequence {old_seq} -> {rec_acct.next_seq}")
            except Exception as e:
                log.warning(f"Failed to sync sequence after Batch validation: {e}")

        log.debug("txn %s validated at ledger %s via %s", rec.txn, rec.seq, rec.src)
        return {"tx_hash": rec.txn, "ledger_index": rec.seq, "source": rec.src, "meta_result": meta_result}

    async def _cascade_expire_account(self, account: str, failed_seq: int, exclude_hash: str | None = None, fetch_seq_from_ledger: bool = False):
        """Cascade-expire all pending txns for account with sequence > failed_seq, and reset next_seq.

        Called when a transaction fails with terPRE_SEQ or expires - all higher-sequence
        txns for the same account are doomed and should be expired immediately.

        Args:
            fetch_seq_from_ledger: If True, fetch next_seq from ledger (for terPRE_SEQ where we
                don't know what ledger expects). If False, set to failed_seq (for expiry where
                we know ledger still expects that sequence).
        """
        cascade_count = 0
        for tx_hash, p in list(self.pending.items()):
            if (tx_hash != exclude_hash and
                p.account == account and
                p.sequence is not None and
                p.sequence > failed_seq and
                p.state not in TERMINAL_STATE):
                p.state = C.TxState.EXPIRED
                # Set engine_result_first to indicate cascade expiry if not already set
                if p.engine_result_first is None:
                    p.engine_result_first = "CASCADE_EXPIRED"
                await self.store.mark(tx_hash, state=C.TxState.EXPIRED, account=p.account,
                                      sequence=p.sequence, transaction_type=p.transaction_type,
                                      engine_result_first=p.engine_result_first)
                cascade_count += 1

        if cascade_count > 0:
            log.warning(f"Cascade expired {cascade_count} txns for {account} with seq > {failed_seq}")

        # Reset account's next_seq
        rec = self._record_for(account)
        async with rec.lock:
            if fetch_seq_from_ledger:
                # terPRE_SEQ case: we don't know what ledger expects, fetch it
                # IMPORTANT: Use "current" (open ledger) not "validated" to include
                # transactions that are queued but not yet validated. Using "validated"
                # could return a stale sequence, causing tefPAST_SEQ when those queued
                # transactions validate.
                try:
                    ai = await self._rpc(AccountInfo(account=account, ledger_index="current"))
                    rec.next_seq = ai.result["account_data"]["Sequence"]
                    log.info(f"Reset next_seq for {account} to {rec.next_seq} (from current ledger)")
                except Exception as e:
                    log.error(f"Failed to fetch sequence for {account}: {e}")
                    rec.next_seq = failed_seq  # Best guess fallback
            else:
                # Expiry case: ledger still expects the failed sequence
                rec.next_seq = failed_seq
                log.info(f"Reset next_seq for {account} to {failed_seq}")

    async def record_expired(self, tx_hash: str):
        if tx_hash not in self.pending:
            return

        p = self.pending[tx_hash]
        p.state = C.TxState.EXPIRED
        await self.store.mark(tx_hash, state=C.TxState.EXPIRED, account=p.account, sequence=p.sequence,
                              transaction_type=p.transaction_type, engine_result_first=p.engine_result_first)

        # Cascade expire higher-sequence txns and reset account sequence
        # IMPORTANT: Always fetch_seq_from_ledger=True because other txns from this
        # account may have validated (advancing ledger seq) before we detected this expiry
        if p.account and p.sequence is not None:
            # For Batch transactions, log explicitly since they pre-allocate multiple sequences
            if p.transaction_type == C.TxType.BATCH:
                log.info(f"Batch expired: {tx_hash[:8]} - will sync sequence from ledger")
            await self._cascade_expire_account(p.account, p.sequence, exclude_hash=tx_hash, fetch_seq_from_ledger=True)

    def find_by_state(self, *states: C.TxState) -> list[PendingTx]:
        return [p for p in self.pending.values() if p.state in set(states)]

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

    def get_pending_txn_counts_by_account(self) -> dict[str, int]:
        """Get count of pending transactions per account.

        Returns:
            Dict mapping account address to count of CREATED/SUBMITTED/RETRYABLE transactions.
            Used to enforce per-account queue limit of 10 (see FeeEscalation.md:260).
        """
        PENDING_STATES = {C.TxState.CREATED, C.TxState.SUBMITTED, C.TxState.RETRYABLE}
        counts = {}
        for p in self.pending.values():
            if p.state in PENDING_STATES and p.account:
                counts[p.account] = counts.get(p.account, 0) + 1
        return counts

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
        base_fee = await self._open_ledger_fee() if need_fee else int(tx["Fee"])

        # Calculate fee based on transaction type
        txn_type = tx.get("TransactionType")
        if txn_type == "Batch":
            # Batch fee = 2 * owner_reserve + base_fee * inner_txn_count
            # owner_reserve is typically 2 XRP = 2,000,000 drops
            OWNER_RESERVE_DROPS = 2_000_000
            inner_txns = tx.get("RawTransactions", [])
            fee = (2 * OWNER_RESERVE_DROPS) + (base_fee * len(inner_txns))
            log.debug(f"Batch fee: 2*{OWNER_RESERVE_DROPS} + {base_fee}*{len(inner_txns)} = {fee} drops")
        elif txn_type == "AMMCreate":
            # AMMCreate fee = owner_reserve (2 XRP)
            OWNER_RESERVE_DROPS = 2_000_000
            fee = OWNER_RESERVE_DROPS
            log.debug(f"AMMCreate fee: {fee} drops (owner_reserve)")
        else:
            fee = base_fee

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
            p.attempts += 1
            log.debug(
                "submit start \n\ttransaction_type=%s\n\tstate=%s\n\tattempts=%s\n\taccount=%s\n\tseq=%s\n\ttx=%s",
                p.transaction_type,
                p.state.name,
                p.attempts,
                p.account,
                p.sequence,
                p.tx_hash,
            )
            if p.transaction_type == "AccountSet":
                pass
            resp = await asyncio.wait_for(self.client.request(SubmitOnly(tx_blob=p.signed_blob_hex)), timeout=timeout)
            if p.transaction_type == "AccountSet":
                log.debug(resp)
            res = resp.result
            er = res.get("engine_result")
            # log.debug("Initial engine result was: %s", er)
            if p.engine_result_first is None:
                p.engine_result_first = er

            if isinstance(er, str) and er.startswith("tel"):
                # tel* = local node rejection. Transaction NOT applied to ledger, BUT:
                # - Server may cache and retry automatically
                # - Could still succeed or fail with different code later
                # - Sequence is still "in use" by this cached transaction
                #
                # telCAN_NOT_QUEUE: Account already has 10 txns queued (per-account limit)
                # telCAN_NOT_QUEUE_FULL: Global queue is full
                # telCAN_NOT_QUEUE_FEE: Fee too low for current queue state
                #
                # DON'T release sequence - let finality check / expiry handle it.
                # Mark as SUBMITTED so we track it until LastLedgerSequence passes.
                p.state = C.TxState.SUBMITTED
                self.pending[p.tx_hash] = p
                await self.store.mark(
                    p.tx_hash,
                    state=p.state,
                    account=p.account,
                    sequence=p.sequence,
                    transaction_type=p.transaction_type,
                    engine_result_first=p.engine_result_first,
                )
                log.warning(f"tel* (may retry): {er} - {p.transaction_type} seq={p.sequence} - tracking until expiry")
                return res

            if er == "tefPAST_SEQ":
                # Sequence already used - ledger's account sequence is higher than our tx's sequence
                # This means we're behind. Reset our sequence counter from ledger.
                log.warning(f"tefPAST_SEQ: {p.transaction_type} from {p.account} seq={p.sequence} - resetting account sequence from ledger")
                p.state = C.TxState.REJECTED
                self.pending[p.tx_hash] = p
                await self.store.mark(
                    p.tx_hash,
                    state=p.state,
                    account=p.account,
                    sequence=p.sequence,
                    transaction_type=p.transaction_type,
                    engine_result_first=p.engine_result_first,
                    engine_result_final=er,
                )
                # Reset sequence from ledger - we're behind
                await self._cascade_expire_account(p.account, p.sequence, exclude_hash=p.tx_hash, fetch_seq_from_ledger=True)
                return res

            if isinstance(er, str) and er.startswith(("tem", "tef")):
                # terminal reject: mark and stop
                p.state = C.TxState.REJECTED
                self.pending[p.tx_hash] = p
                await self.store.mark(
                    p.tx_hash,
                    state=p.state,
                    account=p.account,
                    sequence=p.sequence,
                    transaction_type=p.transaction_type,
                    engine_result_first=p.engine_result_first,
                    engine_result_final=er,
                )
                log.warning(f"REJECTED: {er} - {p.transaction_type} from {p.account} seq={p.sequence} hash={p.tx_hash}")

                # BATCH SEQUENCE SYNC: Batch pre-allocates multiple sequences
                # On rejection, none were consumed - sync with ledger to reset
                if p.transaction_type == C.TxType.BATCH and p.account:
                    try:
                        ai = await self._rpc(AccountInfo(account=p.account, ledger_index="current"))
                        rec_acct = self._record_for(p.account)
                        async with rec_acct.lock:
                            old_seq = rec_acct.next_seq
                            rec_acct.next_seq = ai.result["account_data"]["Sequence"]
                            log.info(f"Batch rejected: synced {p.account[:8]} sequence {old_seq} -> {rec_acct.next_seq}")
                    except Exception as e:
                        log.warning(f"Failed to sync sequence after Batch rejection: {e}")

                return res

            if er == "terPRE_SEQ":
                # Sequence too high - network expects a lower sequence that we don't have
                # This txn is doomed, and so are all higher-sequence txns for this account
                log.warning(f"terPRE_SEQ: {p.transaction_type} from {p.account} seq={p.sequence} - resetting account sequence")
                p.state = C.TxState.EXPIRED
                self.pending[p.tx_hash] = p
                await self.store.mark(
                    p.tx_hash,
                    state=p.state,
                    account=p.account,
                    sequence=p.sequence,
                    transaction_type=p.transaction_type,
                    engine_result_first=er,
                )
                # Cascade expire and fetch correct sequence from ledger
                await self._cascade_expire_account(p.account, p.sequence, fetch_seq_from_ledger=True)
                return res

            # We definitely have not submitted a transaction that isn't retryable if we get here
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
            await self.store.mark(p.tx_hash, state=C.TxState.FAILED_NET, account=p.account, sequence=p.sequence,
                                  transaction_type=p.transaction_type, engine_result_first=p.engine_result_first)
            return {"engine_result": "timeout"}

        except Exception as e:
            p.state = C.TxState.FAILED_NET
            self.pending[p.tx_hash] = p
            log.error("submit error tx=%s: %s", p.tx_hash, e)
            # NEW: persist state transition
            await self.store.mark(p.tx_hash, state=C.TxState.FAILED_NET, account=p.account, sequence=p.sequence,
                                  transaction_type=p.transaction_type, message=str(e))
            return {"engine_result": "error", "message": str(e)}

    def log_validation(self, tx_hash, ledger_index, result, validation_src):
        log.debug(
            "Validated via %s tx=%s li=%s result=%s", validation_src, tx_hash, ledger_index, result
        )  # FIX: DEbug only...

    # TODO: Default constants
    async def check_finality(self, p: PendingTx, grace: int = 2) -> tuple[C.TxState, int | None]:
        try:
            txr = await self.client.request(Tx(transaction=p.tx_hash))
            if txr.is_successful() and txr.result.get("validated"):
                li = int(txr.result["ledger_index"])
                result = txr.result["meta"]["TransactionResult"]

                # Single source of truth: persist via record_validated(), which calls store.mark() once.
                p.state = C.TxState.VALIDATED
                p.validated_ledger = li
                p.meta_txn_result = result
                await self.record_validated(ValidationRecord(p.tx_hash, li, ValidationSrc.POLL), result)
                return p.state, li
        except Exception:
            log.error("Houston, we have a %s", "major problem", exc_info=True)
            pass

        latest_val = await self._latest_validated_ledger()
        if latest_val > (p.last_ledger_seq + grace):
            # Use record_expired to cascade-expire higher-sequence txns for same account
            await self.record_expired(p.tx_hash)
            return p.state, None

        if p.state != C.TxState.SUBMITTED:
            p.state = C.TxState.RETRYABLE
            await self.store.mark(p.tx_hash, state=p.state)
        return p.state, None

    async def submit_signed_tx_blobs(self, items: list):
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
        log.debug(f"Funded {wallet.address} with {int(xrpl.utils.drops_to_xrp(amt_drops))} XRP")
        await debug_last_tx(self.client, p.account)

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
        """Apply per-gateway account flags. One AccountSet per asf flag."""
        flags: list[AccountSetAsfFlag] = []
        if req_auth:
            flags.append(AccountSetAsfFlag.ASF_REQUIRE_AUTH)
        if def_ripple:
            flags.append(AccountSetAsfFlag.ASF_DEFAULT_RIPPLE)

        if not flags or not self.gateways:
            return {"applied": 0, "results": []}

        results: list[dict[str, Any]] = []
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
            log.warning("Validation timeout tx=%s after %.1fs", tx_hash, overall)
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

    async def _submit_batched_by_account(self, all_pending: list[PendingTx], label: str, max_per_account: int = 10) -> dict:
        """Submit transactions in batches, respecting per-account queue limits.

        Groups transactions by account and submits at most max_per_account per account
        at a time. Waits for validations before submitting more from the same account.

        Returns dict of result counts.
        """
        from collections import defaultdict

        # Group by account
        by_account: dict[str, list[PendingTx]] = defaultdict(list)
        for p in all_pending:
            by_account[p.account].append(p)

        total = len(all_pending)
        result_counts: dict[str, int] = {}
        submitted_count = 0

        # Track per-account progress: index of next txn to submit
        account_idx: dict[str, int] = {acc: 0 for acc in by_account}
        # Track per-account in-flight count
        account_inflight: dict[str, int] = {acc: 0 for acc in by_account}

        while submitted_count < total or any(account_inflight[acc] > 0 for acc in by_account):
            # Submit up to max_per_account per account
            to_submit = []
            for acc, txns in by_account.items():
                idx = account_idx[acc]
                inflight = account_inflight[acc]
                available_slots = max_per_account - inflight

                while available_slots > 0 and idx < len(txns):
                    to_submit.append(txns[idx])
                    account_inflight[acc] += 1
                    idx += 1
                    available_slots -= 1
                    submitted_count += 1

                account_idx[acc] = idx

            if to_submit:
                log.info(f"  {label}: Submitting batch of {len(to_submit)} txns ({submitted_count}/{total} total)")

                async with asyncio.TaskGroup() as tg:
                    submit_tasks = [tg.create_task(self.submit_pending(p)) for p in to_submit]

                for task in submit_tasks:
                    try:
                        result = task.result()
                        er = result.get('engine_result') if result else 'None'
                        result_counts[er] = result_counts.get(er, 0) + 1
                    except Exception as e:
                        log.error(f"{label} submission failed: {e}")
                        result_counts['ERROR'] = result_counts.get('ERROR', 0) + 1

            # Wait for a ledger to close, then update inflight counts
            current = await self._current_ledger_index()
            while await self._current_ledger_index() <= current:
                await asyncio.sleep(0.3)

            # Update inflight counts - subtract validated/terminal txns
            for acc, txns in by_account.items():
                terminal = sum(1 for p in txns if p.state in TERMINAL_STATE)
                submitted_for_acc = account_idx[acc]
                account_inflight[acc] = submitted_for_acc - terminal

            # Check if we're done
            all_terminal = all(p.state in TERMINAL_STATE for p in all_pending)
            if all_terminal:
                break

        return result_counts

    async def _establish_trust_lines(self) -> None:
        """Create TrustSet transactions from all users to all configured currencies.

        Submits in batches respecting the 10 txn per-account queue limit.
        """
        if not self.users or not self._currencies:
            log.warning("No users or currencies to establish trust lines")
            return

        trust_limit = str(self.config["transactions"]["trustset"]["limit"])
        total_trustsets = len(self.users) * len(self._currencies)

        log.info(f"Establishing trust lines: {len(self.users)} users × {len(self._currencies)} currencies = {total_trustsets} TrustSets")

        # Build ALL TrustSets for all users and all currencies
        all_pending = []
        for user in self.users:
            for currency in self._currencies:
                trust_tx = TrustSet(
                    account=user.address,
                    limit_amount=IssuedCurrencyAmount(
                        currency=currency.currency,
                        issuer=currency.issuer,
                        value=trust_limit,
                    ),
                )
                pending = await self.build_sign_and_track(trust_tx, user)
                all_pending.append(pending)

        log.info(f"  Built {len(all_pending)} TrustSets")

        # Submit in batches respecting queue limits
        result_counts = await self._submit_batched_by_account(all_pending, "TrustSet")
        log.info(f"  Submission results: {dict(result_counts)}")

        # Final count
        validated_count = sum(1 for p in all_pending if p.state == C.TxState.VALIDATED)
        current = await self._current_ledger_index()
        log.info(f"TrustSet complete in {current}: {validated_count}/{total_trustsets} validated")

    async def _wait_all_validated(
        self,
        pending_list: list[PendingTx],
        label: str,
        timeout: float = 120.0,
    ) -> dict[str, int]:
        """Wait for all transactions to reach terminal state.

        Polls check_finality() for each pending tx until all are terminal
        (VALIDATED, REJECTED, EXPIRED) or timeout.

        Returns dict with counts by final state.
        """
        start = time.time()
        while time.time() - start < timeout:
            # Check finality for each non-terminal tx
            for p in pending_list:
                if p.state not in TERMINAL_STATE:
                    try:
                        await self.check_finality(p)
                    except Exception as e:
                        log.debug(f"check_finality failed for {p.tx_hash}: {e}")

            # Count current states
            counts: dict[str, int] = {}
            for p in pending_list:
                state = p.state.name
                counts[state] = counts.get(state, 0) + 1

            # All terminal?
            if all(p.state in TERMINAL_STATE for p in pending_list):
                validated = counts.get("VALIDATED", 0)
                total = len(pending_list)
                log.info(f"{label} complete: {validated}/{total} validated, {counts}")
                return counts

            await asyncio.sleep(0.5)

        # Timeout - return what we have
        counts = {}
        for p in pending_list:
            state = p.state.name
            counts[state] = counts.get(state, 0) + 1
        log.warning(f"{label} timeout after {timeout}s: {counts}")
        return counts

    async def _submit_and_wait_batched(
        self,
        pending_list: list[PendingTx],
        label: str,
        max_per_account: int = 10,
        max_batch_size: int | None = None,
        timeout: float = 180.0,
    ) -> dict[str, int]:
        """Submit transactions in batches respecting per-account AND global limits.

        Groups by account, submits max_per_account at a time per account,
        BUT also limits total in-flight to max_batch_size to avoid queue overflow.

        Returns dict with counts by final state.
        """
        from collections import defaultdict

        # Group by account
        by_account: dict[str, list[PendingTx]] = defaultdict(list)
        for p in pending_list:
            by_account[p.account].append(p)

        total = len(pending_list)
        submitted_hashes: set[str] = set()

        # Default batch size: expected_ledger_size or 50 (avoid queue overflow)
        if max_batch_size is None:
            try:
                max_batch_size = await self._expected_ledger_size()
            except Exception:
                max_batch_size = 50

        log.info(f"{label}: {total} txns across {len(by_account)} accounts (max {max_batch_size}/batch)")

        start = time.time()
        while time.time() - start < timeout:
            # Count total in-flight across ALL accounts
            total_inflight = sum(
                1 for p in pending_list
                if p.tx_hash in submitted_hashes and p.state not in TERMINAL_STATE
            )
            global_slots = max_batch_size - total_inflight

            if global_slots <= 0:
                # Wait for some to complete before submitting more
                await asyncio.sleep(0.5)
                # Check finality first
                for p in pending_list:
                    if p.tx_hash in submitted_hashes and p.state not in TERMINAL_STATE:
                        try:
                            await self.check_finality(p)
                        except Exception:
                            pass
                continue

            # Submit up to max_per_account per account, but respect global limit
            newly_submitted = []
            for acc, txns in by_account.items():
                if len(newly_submitted) >= global_slots:
                    break  # Hit global limit

                # Count in-flight for this account
                inflight = sum(
                    1 for p in txns
                    if p.tx_hash in submitted_hashes and p.state not in TERMINAL_STATE
                )
                available_slots = min(max_per_account - inflight, global_slots - len(newly_submitted))

                # Find unsubmitted txns for this account
                for p in txns:
                    if available_slots <= 0:
                        break
                    if p.tx_hash not in submitted_hashes:
                        newly_submitted.append(p)
                        submitted_hashes.add(p.tx_hash)
                        available_slots -= 1

            # Submit the batch
            if newly_submitted:
                log.info(f"  {label}: Submitting {len(newly_submitted)} txns ({len(submitted_hashes)}/{total} total)")
                for p in newly_submitted:
                    try:
                        await self.submit_pending(p)
                    except Exception as e:
                        log.error(f"Submit failed for {p.tx_hash}: {e}")

            # Check finality for all submitted txns
            for p in pending_list:
                if p.tx_hash in submitted_hashes and p.state not in TERMINAL_STATE:
                    try:
                        await self.check_finality(p)
                    except Exception:
                        pass

            # Count current states
            counts: dict[str, int] = {}
            for p in pending_list:
                state = p.state.name
                counts[state] = counts.get(state, 0) + 1

            # All terminal?
            if all(p.state in TERMINAL_STATE for p in pending_list):
                validated = counts.get("VALIDATED", 0)
                log.info(f"{label} complete: {validated}/{total} validated, {counts}")
                return counts

            # Wait a bit before next round
            await asyncio.sleep(0.5)

        # Timeout
        counts = {}
        for p in pending_list:
            state = p.state.name
            counts[state] = counts.get(state, 0) + 1
        log.warning(f"{label} timeout after {timeout}s: {counts}")
        return counts

    async def _distribute_initial_tokens(self) -> None:
        """Gateways send initial token balances to all users.

        Submits in batches respecting the 10 txn per-account queue limit.
        """
        if not self.users or not self._currencies:
            log.warning("No users or currencies to distribute tokens")
            return

        initial_amount = str(self.config.get("currencies", {}).get("token_distribution", 1_000_000))

        # Build ALL payments - each gateway sends to each user for each currency it issues
        all_pending = []
        for currency in self._currencies:
            issuer_wallet = self.wallets.get(currency.issuer)
            if not issuer_wallet:
                log.error(f"Cannot find wallet for gateway {currency.issuer}")
                continue

            for user in self.users:
                payment_tx = Payment(
                    account=currency.issuer,
                    destination=user.address,
                    amount=IssuedCurrencyAmount(
                        currency=currency.currency,
                        issuer=currency.issuer,
                        value=initial_amount,
                    ),
                )
                pending = await self.build_sign_and_track(payment_tx, issuer_wallet)
                all_pending.append(pending)

        total_payments = len(all_pending)
        log.info(f"Distributing tokens: {total_payments} payments")

        # Submit in batches respecting queue limits
        result_counts = await self._submit_batched_by_account(all_pending, "TokenDist")
        log.info(f"  Submission results: {dict(result_counts)}")

        # Final count
        validated_count = sum(1 for p in all_pending if p.state == C.TxState.VALIDATED)
        current = await self._current_ledger_index()
        log.info(f"Token distribution complete at {current}: {validated_count}/{total_payments} validated")

    # =========================================================================
    # Simple Init Helpers - Sequential, reliable, separate from continuous workload
    # =========================================================================

    async def _init_batch(
        self,
        txn_wallet_pairs: list[tuple[Transaction, Wallet]],
        label: str,
        batch_size: int | None = None,  # None = use dynamic expected_ledger_size + 1
    ) -> tuple[int, int]:
        """Simple init helper: build, submit, wait in sequential batches.

        Processes transactions in small batches, waiting for each batch to
        fully validate before moving to the next. Ensures fresh LastLedgerSequence
        for each batch.

        Args:
            batch_size: Fixed batch size, or None for dynamic (expected_ledger_size + 1)

        Returns: (validated_count, total_count)
        """
        total = len(txn_wallet_pairs)
        all_pending: list[PendingTx] = []
        submit_failed = 0
        processed = 0
        batch_num = 0

        dynamic = batch_size is None
        if dynamic:
            log.info(f"{label}: {total} transactions (dynamic batch size = expected_ledger_size + 1)")
        else:
            log.info(f"{label}: {total} transactions in batches of {batch_size}")

        while processed < total:
            batch_num += 1

            # Get batch size - dynamic refreshes each iteration to grow with ledger
            if dynamic:
                current_batch_size = (await self._expected_ledger_size()) + 1
            else:
                current_batch_size = batch_size

            batch_end = min(processed + current_batch_size, total)
            batch = txn_wallet_pairs[processed:batch_end]
            batch_pending: list[PendingTx] = []

            # Build, sign, and submit this batch (fresh LLS for each)
            for txn, wallet in batch:
                try:
                    p = await self.build_sign_and_track(txn, wallet)
                    await self.submit_pending(p)
                    batch_pending.append(p)
                except Exception as e:
                    log.error(f"{label}: Failed to submit: {e}")
                    submit_failed += 1

            submitted_count = processed + len(batch_pending)
            log.info(f"  {label}: Submitted batch {batch_num} ({len(batch_pending)} txns) {submitted_count}/{total}")

            # Wait for this batch to complete (all terminal)
            for _ in range(60):  # 30 seconds max per batch
                for p in batch_pending:
                    if p.state not in TERMINAL_STATE:
                        try:
                            await self.check_finality(p)
                        except Exception:
                            pass

                if all(p.state in TERMINAL_STATE for p in batch_pending):
                    break
                await asyncio.sleep(0.5)

            all_pending.extend(batch_pending)
            processed = batch_end

            # Log batch results
            batch_validated = sum(1 for p in batch_pending if p.state == C.TxState.VALIDATED)
            if batch_validated < len(batch_pending):
                batch_counts: dict[str, int] = {}
                for p in batch_pending:
                    batch_counts[p.state.name] = batch_counts.get(p.state.name, 0) + 1
                log.warning(f"  {label}: Batch results: {batch_counts}")

        # Final counts across all batches
        counts: dict[str, int] = {}
        for p in all_pending:
            counts[p.state.name] = counts.get(p.state.name, 0) + 1
        if submit_failed > 0:
            counts["SUBMIT_FAILED"] = submit_failed

        validated = counts.get("VALIDATED", 0)
        log.info(f"{label} complete: {validated}/{total} validated, {counts}")
        return validated, total

    async def init_participants(self, *, gateway_cfg: dict[str, Any], user_cfg: dict[str, Any]) -> dict:
        """Initialize gateways and users with phase-based validation barriers.

        Each phase waits for ALL transactions to validate before proceeding:
        1. Fund gateways -> WAIT -> register gateways
        2. Gateway flags (AccountSet) -> WAIT
        3. Fund users -> WAIT -> register users
        4. TrustSets -> WAIT
        5. Token distribution -> WAIT
        """
        out_gw, out_us = [], []
        req_auth = gateway_cfg["require_auth"]
        def_ripple = gateway_cfg["default_ripple"]
        gateway_names = gateway_cfg.get("names", [])
        currency_codes = self.config["currencies"]["codes"][:4]

        # Timing and ledger tracking
        phase_stats: list[dict] = []  # [{name, time_sec, ledgers, validated, total}]
        init_start_time = time.time()
        init_start_ledger = await self._current_ledger_index()

        # ===== PHASE 1: Fund Gateways =====
        phase1_start = time.time()
        phase1_ledger_start = await self._current_ledger_index()
        gateway_count = gateway_cfg["number"]
        gateway_balance = str(self.config["gateways"]["default_balance"])
        log.info(f"Phase 1: Funding {gateway_count} gateways")

        gateway_pending: list[tuple[Wallet, PendingTx, int]] = []  # (wallet, pending, index)
        for i in range(gateway_count):
            w = Wallet.create()
            fund_tx = Payment(
                account=self.funding_wallet.address,
                destination=w.address,
                amount=gateway_balance,
            )
            p = await self.build_sign_and_track(fund_tx, self.funding_wallet)
            await self.submit_pending(p)
            gateway_pending.append((w, p, i))

        # BARRIER: Wait for all gateway funding to validate
        gw_results = await self._wait_all_validated(
            [p for _, p, _ in gateway_pending],
            "Phase 1: Gateway funding"
        )

        # NOW register validated gateways (after they exist on-chain)
        for w, p, i in gateway_pending:
            if p.state == C.TxState.VALIDATED:
                self.wallets[w.address] = w
                self._record_for(w.address)
                self.gateways.append(w)
                if i < len(gateway_names):
                    self.gateway_names[w.address] = gateway_names[i]
                self.save_wallet_to_store(w, is_gateway=True)
                out_gw.append(w.address)
            else:
                log.error(f"Gateway funding failed: {w.address} state={p.state.name}")

        if not self.gateways:
            raise RuntimeError("No gateways funded successfully, cannot continue")

        # Create currencies issued by each gateway
        for gateway in self.gateways:
            gateway_currencies = issue_currencies(gateway.address, currency_codes)
            self._currencies.extend(gateway_currencies)
            gw_name = self.get_account_display_name(gateway.address)
            log.debug(f"Gateway {gw_name} will issue: {currency_codes}")

        self.update_txn_context()

        # Record Phase 1 stats
        phase1_ledger_end = await self._current_ledger_index()
        phase_stats.append({
            "name": "Phase 1: Gateway funding",
            "time_sec": round(time.time() - phase1_start, 2),
            "ledgers": phase1_ledger_end - phase1_ledger_start,
            "validated": gw_results.get("VALIDATED", 0),
            "total": gateway_count,
        })

        # ===== PHASE 2: Gateway Flags (AccountSet) =====
        phase2_start = time.time()
        phase2_ledger_start = await self._current_ledger_index()
        if req_auth or def_ripple:
            log.info(f"Phase 2: Setting gateway flags (require_auth={req_auth}, default_ripple={def_ripple})")

            flag_pending: list[PendingTx] = []
            flags_to_set: list[AccountSetAsfFlag] = []
            if req_auth:
                flags_to_set.append(AccountSetAsfFlag.ASF_REQUIRE_AUTH)
            if def_ripple:
                flags_to_set.append(AccountSetAsfFlag.ASF_DEFAULT_RIPPLE)

            for w in self.gateways:
                for flag in flags_to_set:
                    tx = AccountSet(account=w.address, set_flag=flag)
                    p = await self.build_sign_and_track(tx, w)
                    await self.submit_pending(p)
                    flag_pending.append(p)

            # BARRIER: Wait for all flags
            p2_results = await self._wait_all_validated(flag_pending, "Phase 2: Gateway flags")
            phase2_validated = p2_results.get("VALIDATED", 0)
            phase2_total = len(flag_pending)
        else:
            log.info("Phase 2: Skipping gateway flags (none configured)")
            phase2_validated = 0
            phase2_total = 0

        # Record Phase 2 stats
        phase2_ledger_end = await self._current_ledger_index()
        phase_stats.append({
            "name": "Phase 2: Gateway flags",
            "time_sec": round(time.time() - phase2_start, 2),
            "ledgers": phase2_ledger_end - phase2_ledger_start,
            "validated": phase2_validated,
            "total": phase2_total,
        })

        # ===== PHASE 3: Fund Users (batched to avoid queue overflow) =====
        phase3_start = time.time()
        phase3_ledger_start = await self._current_ledger_index()
        user_count = user_cfg["number"]
        user_balance = str(self.config["users"]["default_balance"])
        batch_size = 10  # Funding wallet can have max 10 in queue
        log.info(f"Phase 3: Funding {user_count} users in batches of {batch_size}")

        # Create all wallets first
        new_wallets = [Wallet.create() for _ in range(user_count)]
        user_pending: list[tuple[Wallet, PendingTx]] = []

        for batch_start in range(0, user_count, batch_size):
            batch_wallets = new_wallets[batch_start:batch_start + batch_size]
            batch_pending: list[tuple[Wallet, PendingTx]] = []

            # Build, sign, submit this batch
            for w in batch_wallets:
                fund_tx = Payment(
                    account=self.funding_wallet.address,
                    destination=w.address,
                    amount=user_balance,
                )
                p = await self.build_sign_and_track(fund_tx, self.funding_wallet)
                await self.submit_pending(p)
                batch_pending.append((w, p))

            log.info(f"  Phase 3: Submitted batch {batch_start//batch_size + 1} ({len(batch_pending)} users)")

            # Wait for this batch to validate before next batch
            for _ in range(60):
                for w, p in batch_pending:
                    if p.state not in TERMINAL_STATE:
                        try:
                            await self.check_finality(p)
                        except Exception:
                            pass
                if all(p.state in TERMINAL_STATE for _, p in batch_pending):
                    break
                await asyncio.sleep(0.5)

            user_pending.extend(batch_pending)

        # Register validated users
        for w, p in user_pending:
            if p.state == C.TxState.VALIDATED:
                self.wallets[w.address] = w
                self._record_for(w.address)
                self.users.append(w)
                self.save_wallet_to_store(w, is_user=True)
                out_us.append(w.address)
            else:
                log.error(f"User funding failed: {w.address} state={p.state.name}")

        validated_users = len(self.users)
        log.info(f"Phase 3 complete: {validated_users}/{user_count} users funded")

        if not self.users:
            raise RuntimeError("No users funded successfully, cannot continue")

        self.update_txn_context()

        # Record Phase 3 stats
        phase3_ledger_end = await self._current_ledger_index()
        phase_stats.append({
            "name": "Phase 3: User funding",
            "time_sec": round(time.time() - phase3_start, 2),
            "ledgers": phase3_ledger_end - phase3_ledger_start,
            "validated": validated_users,
            "total": user_count,
        })

        # ===== PHASE 4: TrustSets (using simple init batch) =====
        phase4_start = time.time()
        phase4_ledger_start = await self._current_ledger_index()
        total_trustsets = len(self.users) * len(self._currencies)
        log.info(f"Phase 4: Establishing {total_trustsets} trust lines ({len(self.users)} users × {len(self._currencies)} currencies)")

        trust_limit = str(self.config["transactions"]["trustset"]["limit"])

        # Build (Transaction, Wallet) pairs - NOT signed yet
        # IMPORTANT: Interleave accounts (currency-first) to spread load across accounts
        # This way each batch has different accounts rather than hammering one account
        trust_txn_pairs: list[tuple[Transaction, Wallet]] = []
        for currency in self._currencies:
            for user in self.users:
                trust_tx = TrustSet(
                    account=user.address,
                    limit_amount=IssuedCurrencyAmount(
                        currency=currency.currency,
                        issuer=currency.issuer,
                        value=trust_limit,
                    ),
                )
                trust_txn_pairs.append((trust_tx, user))

        # Use dynamic batch size (expected_ledger_size + 1, refreshed each batch)
        phase4_validated, phase4_total = await self._init_batch(trust_txn_pairs, "Phase 4: TrustSets")
        if phase4_validated < phase4_total:
            log.warning(f"Phase 4: Only {phase4_validated}/{phase4_total} TrustSets validated - some tokens may fail")

        # Record Phase 4 stats
        phase4_ledger_end = await self._current_ledger_index()
        phase_stats.append({
            "name": "Phase 4: TrustSets",
            "time_sec": round(time.time() - phase4_start, 2),
            "ledgers": phase4_ledger_end - phase4_ledger_start,
            "validated": phase4_validated,
            "total": phase4_total,
        })

        # ===== PHASE 5: Token Distribution (Fan-out Seeding, using simple init batch) =====
        phase5_start = time.time()
        phase5_ledger_start = await self._current_ledger_index()
        # Only seed a subset of users - they'll redistribute during continuous workload
        # Generic formula: max(sqrt(users), gateways * 2) ensures enough seeds for parallelism
        import math
        num_users = len(self.users)
        num_gateways = len(self.gateways)
        seed_count = min(max(int(math.sqrt(num_users)), num_gateways * 2), num_users)
        seed_users = self.users[:seed_count]

        # Seeds get enough to redistribute to their share of users, with buffer
        base_amount = int(self.config.get("currencies", {}).get("token_distribution", 1_000_000))
        users_per_seed = max(1, num_users // seed_count)
        seed_amount = base_amount * users_per_seed * 2  # 2x buffer for redistribution
        seed_amount_str = str(seed_amount)

        total_payments = seed_count * len(self._currencies)
        log.info(f"Phase 5: Seeding {seed_count} users with tokens ({total_payments} payments)")
        log.info(f"  Formula: max(sqrt({num_users}), {num_gateways}*2) = {seed_count} seeds")
        log.info(f"  Amount per seed: {seed_amount} ({users_per_seed} users/seed × {base_amount} × 2)")
        log.info(f"  Remaining {num_users - seed_count} users receive tokens via fan-out during continuous workload")

        # Build (Transaction, Wallet) pairs - NOT signed yet
        # Also track balance updates to apply after validation
        dist_txn_pairs: list[tuple[Transaction, Wallet]] = []
        balance_updates: list[tuple[str, str, float, str]] = []  # (user, currency, amount, issuer)

        for currency in self._currencies:
            issuer_wallet = self.wallets.get(currency.issuer)
            if not issuer_wallet:
                log.error(f"Cannot find wallet for gateway {currency.issuer}")
                continue

            for user in seed_users:
                payment_tx = Payment(
                    account=currency.issuer,
                    destination=user.address,
                    amount=IssuedCurrencyAmount(
                        currency=currency.currency,
                        issuer=currency.issuer,
                        value=seed_amount_str,
                    ),
                )
                dist_txn_pairs.append((payment_tx, issuer_wallet))
                balance_updates.append((user.address, currency.currency, float(seed_amount), currency.issuer))

        # Use dynamic batch size (expected_ledger_size + 1, refreshed each batch)
        phase5_validated, phase5_total = await self._init_batch(dist_txn_pairs, "Phase 5: Token seeding")

        # Pre-populate balances (will be updated accurately as txns validate via record_validated)
        for user_addr, currency_code, amount, issuer in balance_updates:
            self._set_balance(user_addr, currency_code, amount, issuer)

        if phase5_validated < phase5_total:
            log.warning(f"Phase 5: Only {phase5_validated}/{phase5_total} token distributions validated")

        # Record Phase 5 stats
        phase5_ledger_end = await self._current_ledger_index()
        phase_stats.append({
            "name": "Phase 5: Token seeding",
            "time_sec": round(time.time() - phase5_start, 2),
            "ledgers": phase5_ledger_end - phase5_ledger_start,
            "validated": phase5_validated,
            "total": phase5_total,
        })

        # Persist currencies
        self.save_currencies_to_store()

        # ===== SUMMARY =====
        init_end_time = time.time()
        init_end_ledger = await self._current_ledger_index()
        total_time = round(init_end_time - init_start_time, 2)
        total_ledgers = init_end_ledger - init_start_ledger
        total_validated = sum(p["validated"] for p in phase_stats)
        total_txns = sum(p["total"] for p in phase_stats)

        log.info("=" * 60)
        log.info("INITIALIZATION SUMMARY")
        log.info("=" * 60)
        for p in phase_stats:
            log.info(f"  {p['name']}: {p['validated']}/{p['total']} validated, {p['time_sec']}s, {p['ledgers']} ledgers")
        log.info("-" * 60)
        log.info(f"  TOTAL: {total_validated}/{total_txns} validated, {total_time}s, {total_ledgers} ledgers")
        log.info(f"  Final state: {len(self.gateways)} gateways, {len(self.users)} users at ledger {init_end_ledger}")
        log.info("=" * 60)

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


PER_TX_TIMEOUT = 3


async def periodic_finality_check(w: Workload, stop: asyncio.Event, interval: int = 5):
    while not stop.is_set():
        try:
            for p in w.find_by_state(C.TxState.SUBMITTED):
                try:
                    await w.check_finality(p)
                except Exception:
                    log.exception("[finality] check failed for %s", getattr(p, "tx_hash", p))
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[finality] outer loop error; continuing")
            await asyncio.sleep(0.5)
