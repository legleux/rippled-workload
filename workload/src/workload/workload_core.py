import asyncio
import hashlib
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import xrpl
from xrpl.asyncio.clients import AsyncJsonRpcClient
from xrpl.asyncio.ledger import get_latest_validated_ledger_sequence
from xrpl.core.binarycodec import encode, encode_for_signing
from xrpl.core.keypairs import sign
from xrpl.models import IssuedCurrency, SubmitOnly, Transaction
from xrpl.models.requests import (
    AccountInfo,
    ServerState,
    Tx,
)
from xrpl.models.transactions import Payment
from xrpl.wallet import Wallet

import workload.constants as C
from workload.amm import AMMPoolRegistry, DEXMetrics
from workload.assertions import tx_intentionally_rejected, tx_rejected, tx_submitted, tx_validated
from workload.balances import BalanceTracker
from workload.sqlite_store import SQLiteStore
from workload.txn_factory import TxnContext, generate_txn
from workload.validation import ValidationRecord, ValidationSrc
from workload.validation_hooks import dispatch_validation_hooks

log = logging.getLogger("workload")


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
    engine_result_message: str | None = None
    validated_ledger: int | None = None
    meta_txn_result: str | None = None
    created_at: float = field(default_factory=time.time)
    finalized_at: float | None = None
    account_generation: int = 0  # AccountRecord.generation at build time; used to detect stale pre-signed txns
    # CAUTION: expect_rejection suppresses rejection warnings and assertion
    # severity. A false positive hides real bugs. Only set by workload_runner
    # when intent==INVALID AND taint_txn() was applied. Review any new callers.
    expect_rejection: bool = False

    def __str__(self):
        return f"{self.transaction_type} -- {self.account} -- {self.state}"


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
        """Full recount — called only on startup or after bulk operations."""
        self.count_by_state = Counter(rec.get("state", "UNKNOWN") for rec in self._records.values())
        self.count_by_type = Counter(rec.get("transaction_type", "UNKNOWN") for rec in self._records.values())
        self.validated_by_source = Counter(v.src for v in self.validations)

    def _update_counts(self, prev_state: str | None, new_state: str | None, txn_type: str | None = None) -> None:
        """Incremental count update — O(1) instead of O(n)."""
        if prev_state:
            self.count_by_state[prev_state] = max(0, self.count_by_state.get(prev_state, 0) - 1)
        if new_state:
            self.count_by_state[new_state] = self.count_by_state.get(new_state, 0) + 1
        if txn_type and not prev_state:  # only count type on first insert
            self.count_by_type[txn_type] = self.count_by_type.get(txn_type, 0) + 1

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
        async with self._lock:
            rec = self._records.get(tx_hash, {})
            prev_state = rec.get("state")

            rec.update(fields)
            if source is not None:
                rec["source"] = source

            state = rec.get("state")
            if isinstance(state, C.TxState):
                state = state.name
                rec["state"] = state

            if state in C.TERMINAL_STATE:
                rec.setdefault("finalized_at", time.time())

                if state == "VALIDATED" and prev_state != "VALIDATED":
                    seq = rec.get("validated_ledger") or 0
                    src = source or rec.get("source", "unknown")
                    if not any(v.txn == tx_hash and v.seq == seq for v in self.validations):
                        self.validations.append(ValidationRecord(txn=tx_hash, seq=seq, src=src))
                        self.validated_by_source[src] = self.validated_by_source.get(src, 0) + 1

            self._records[tx_hash] = rec
            self._update_counts(prev_state, state, rec.get("transaction_type"))

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
    generation: int = 0  # incremented on every cascade-expire / sequence reset


def _sha512half(b: bytes) -> bytes:
    return hashlib.sha512(b).digest()[:32]


def _txid_from_signed_blob_hex(signed_blob_hex: str) -> str:
    return _sha512half(bytes.fromhex("54584E00") + bytes.fromhex(signed_blob_hex)).hex().upper()


class Workload:
    def __init__(self, config: dict, client: AsyncJsonRpcClient, *, store: SQLiteStore | None = None):
        self.config = config
        self.client = client

        self.funding_wallet = Wallet.from_seed(
            "snoPBrXtMeMyMHUVTgbuqAfg1SUTb", algorithm=xrpl.CryptoAlgorithm.SECP256K1
        )

        self.accounts: dict[str, AccountRecord] = {}
        self.wallets: dict[str, Wallet] = {}
        self.gateways: list[Wallet] = []
        self.users: list[Wallet] = []
        self.gateway_names: dict[str, str] = {}  # address -> name mapping

        self.pending: dict[str, PendingTx] = {}

        self.store: InMemoryStore = InMemoryStore()  # Hot-path runtime store
        self.persistent_store: SQLiteStore | None = store  # SQLite for durability (flushed periodically)

        self.max_pending_per_account: int = self.config.get("transactions", {}).get("max_pending_per_account", 1)

        self.target_tps: float = 0  # 0 = unlimited (firehose)
        self.submission_set_size: int = self.config.get("transactions", {}).get("submission_set_size", 200)

        self._config_disabled_types: frozenset[str] = frozenset(self.config.get("transactions", {}).get("disabled", []))
        self.disabled_txn_types: set[str] = set(self._config_disabled_types)

        self.user_token_status: dict[str, set[tuple[str, str]]] = {}

        self._currencies: list[IssuedCurrency] = []

        self._mptoken_issuance_ids: dict[str, str] = {}  # {mpt_id: issuer_address}
        self._credentials: list[dict] = []  # [{issuer, subject, credential_type, accepted}]
        self._vaults: list[dict] = []  # [{vault_id, owner, asset}]
        self._domains: list[dict] = []  # [{domain_id, owner}]
        self._nfts: dict[str, str] = {}  # {nft_id: owner_address}
        self._offers: dict[str, dict] = {}  # {offer_key: {type, owner, sequence, ...}}
        self._tickets: dict[str, set[int]] = {}  # {account: {ticket_seq, ...}}
        self._checks: dict[str, dict] = {}  # {check_id: {sender, destination, send_max}}
        self._escrows: dict[str, dict] = {}  # {escrow_id: {owner, sequence, destination, finish_after, cancel_after}}

        self.amm = AMMPoolRegistry()
        self.dex_metrics = DEXMetrics()

        self.balance_tracker = BalanceTracker()

        # Workload lifecycle state (set by app.py orchestrator)
        self.workload_started: bool = False
        self.started_at: float = time.time()  # Wall-clock start time

        # Server status — updated by ws_processor via update_server_status()
        self.latest_server_status: dict = {}
        self.latest_server_status_time: float = 0.0
        self.latest_server_status_computed: dict = {}

        # Ledger close notification — set by ws_processor, awaited by consumer
        self._ledger_closed_event: asyncio.Event = asyncio.Event()
        self.latest_ledger_index: int = 0
        self.first_ledger_index: int = 0  # Set on first ledger close event

        # Cached fee — updated from WS ledgerClosed events, invalidated on telINSUF_FEE_P
        self._cached_fee: int | None = None
        # Reserve info from WS ledgerClosed events
        self._ws_reserve_base: int | None = None
        self._ws_reserve_inc: int | None = None
        # Last closed ledger's transaction count (from WS ledgerClosed)
        self.last_closed_ledger_txn_count: int = 0

        # Cumulative counters (never reset, survive cleanup_terminal)
        self._type_submitted: dict[str, int] = {}
        self._type_validated: dict[str, int] = {}
        self._total_created: int = 0
        self._total_validated: int = 0
        self._total_rejected: int = 0
        self._total_expired: int = 0
        self._failure_codes: dict[str, int] = {}  # engine_result -> count (cumulative)
        self._tem_disabled_types: set[str] = set()  # txn types that got temDISABLED

        self.ctx = self.configure_txn_context(
            wallets=self.wallets,
            funding_wallet=self.funding_wallet,
            config=self.config,
        )

    def configure_txn_context(
        self,
        *,
        funding_wallet: "Wallet",
        wallets: dict[str, "Wallet"] | list["Wallet"],
        currencies: list["IssuedCurrency"] | None = None,
        config: dict | None = None,
    ) -> TxnContext:
        currs = currencies if currencies is not None else self._currencies
        wl = list(wallets.values()) if isinstance(wallets, dict) else list(wallets)
        ctx = TxnContext.build(
            funding_wallet=funding_wallet,
            wallets=wl,
            currencies=currs,
            config=config or self.config,
            base_fee_drops=self._open_ledger_fee,
            next_sequence=self.alloc_seq,
        )
        ctx.mptoken_issuance_ids = self._mptoken_issuance_ids
        ctx.balances = self.balance_tracker.data
        ctx.amm_pools = self.amm.pool_ids
        ctx.amm_pool_registry = self.amm.pools
        ctx.disabled_types = self.disabled_txn_types
        ctx.credentials = self._credentials
        ctx.vaults = self._vaults
        ctx.domains = self._domains
        ctx.nfts = self._nfts
        ctx.offers = self._offers
        ctx.tickets = self._tickets
        ctx.checks = self._checks
        ctx.escrows = self._escrows
        return ctx

    def update_txn_context(self):
        self.ctx = self.configure_txn_context(
            wallets=list(self.wallets.values()),
            funding_wallet=self.funding_wallet,
        )

    def _register_amm_pool(self, asset1: dict, asset2: dict, creator: str) -> None:
        """Register a new AMM pool after successful AMMCreate validation."""
        self.amm.register(asset1, asset2, creator)
        self.dex_metrics.pools_created = len(self.amm)
        self.update_txn_context()
        log.debug("Registered AMM pool: total %d", len(self.amm))

    def get_all_account_addresses(self) -> list[str]:
        """Return all account addresses we're tracking (for WebSocket subscription).

        Returns empty list if no accounts initialized yet (WS will fall back to transaction stream).
        """
        if not self.wallets:
            return []

        addresses = [self.funding_wallet.address]
        addresses.extend(self.wallets.keys())
        return addresses

    def get_account_display_name(self, address: str) -> str:
        """Get display name for an account (gateway name if available, else address)."""
        return self.gateway_names.get(address, address)

    def load_state_from_store(self) -> bool:
        """Load workload state from SQLite store if available.

        On hot-reload, this allows us to skip re-creating accounts and TrustSets.
        We clear any pending transactions (stale from previous session) and
        reset sequence numbers from on-chain state.

        Returns:
            True if state was loaded, False otherwise
        """
        from workload.sqlite_store import SQLiteStore

        if not isinstance(self.persistent_store, SQLiteStore):
            log.warning("Store is not SQLiteStore, cannot load state")
            return False

        if not self.persistent_store.has_state():
            log.debug("No persisted state found in database")
            return False

        log.debug("Loading workload state from database...")

        self.pending.clear()

        gateway_names_from_config = self.config.get("gateways", {}).get("names", [])
        wallet_data = self.persistent_store.load_wallets()
        gateway_idx = 0
        for address, (wallet, is_gateway, is_user) in wallet_data.items():
            self.wallets[address] = wallet
            self._record_for(address)

            if is_gateway:
                self.gateways.append(wallet)
                if gateway_idx < len(gateway_names_from_config):
                    self.gateway_names[address] = gateway_names_from_config[gateway_idx]
                    gateway_idx += 1
            if is_user:
                self.users.append(wallet)

        currencies = self.persistent_store.load_currencies()
        self._currencies = currencies

        log.debug(
            f"Loaded state: {len(self.wallets)} wallets "
            f"({len(self.gateways)} gateways, {len(self.users)} users), "
            f"{len(self._currencies)} currencies"
        )

        if len(self.gateways) > 0 and len(self._currencies) == 0:
            log.warning("Incomplete state detected: gateways exist but no currencies found. Rejecting loaded state.")
            self.wallets.clear()
            self.gateways.clear()
            self.users.clear()
            self.gateway_names.clear()
            self._currencies = []
            return False

        self.update_txn_context()

        return True

    async def load_from_genesis(self, accounts_json_path: str) -> bool:
        """Load accounts from a pre-generated accounts.json (from generate_ledger).

        Skips all init phases — accounts, trust lines, tokens, and AMMs are already
        baked into the genesis ledger.

        Args:
            accounts_json_path: Path to accounts.json with [address, seed] pairs

        Returns:
            True if loaded successfully
        """
        import json as _json
        from pathlib import Path

        path = Path(accounts_json_path)
        if not path.exists():
            log.debug("Genesis accounts file not found: %s", path)
            return False

        with open(path) as f:
            account_data = _json.load(f)

        gateway_count = self.config["gateways"]["number"]
        genesis_cfg = self.config.get("genesis", {})
        currency_codes = genesis_cfg.get("currencies", ["USD", "CNY", "BTC", "ETH"])
        gateway_names = self.config.get("gateways", {}).get("names", [])
        # user_count derived from accounts.json — not config.toml
        user_count = len(account_data) - gateway_count

        log.info(
            "Loading from genesis: %d accounts (%d gateways, %d users)", len(account_data), gateway_count, user_count
        )

        # Build wallets from seeds — try ed25519 first, fall back to secp256k1
        for i, (address, seed) in enumerate(account_data):
            w = Wallet.from_seed(seed, algorithm=xrpl.CryptoAlgorithm.ED25519)
            if w.address != address:
                w = Wallet.from_seed(seed, algorithm=xrpl.CryptoAlgorithm.SECP256K1)
            if w.address != address:
                log.error("Address mismatch for account %d: expected %s, got %s", i, address, w.address)
                continue
            self.wallets[w.address] = w
            self._record_for(w.address)

            if i < gateway_count:
                self.gateways.append(w)
                if i < len(gateway_names):
                    self.gateway_names[w.address] = gateway_names[i]
                self.save_wallet_to_store(w, is_gateway=True)
            else:
                self.users.append(w)
                self.save_wallet_to_store(w, is_user=True)

        # Build currencies (4 per gateway)
        for gw in self.gateways:
            for code in currency_codes:
                self._currencies.append(IssuedCurrency(currency=code, issuer=gw.address))

        self.save_currencies_to_store()
        self.update_txn_context()

        log.info(
            "Genesis loaded: %d gateways, %d users, %d currencies",
            len(self.gateways),
            len(self.users),
            len(self._currencies),
        )

        # Discover AMM pools from the ledger
        await self._discover_amm_pools()

        return True

    async def _discover_amm_pools(self) -> None:
        """Discover existing AMM pools by querying amm_info for known currency pairs."""
        from xrpl.models.currencies import XRP as XRPCurrency
        from xrpl.models.requests import AMMInfo

        discovered = 0
        # Check XRP/IOU pairs (gateway pools)
        for currency in self._currencies:
            try:
                a1 = XRPCurrency()
                a2 = IssuedCurrency(currency=currency.currency, issuer=currency.issuer)
                resp = await self._rpc(AMMInfo(asset=a1, asset2=a2), t=3.0)
                if resp.is_successful() and "amm" in resp.result:
                    self._register_amm_pool(
                        {"currency": "XRP"},
                        {"currency": currency.currency, "issuer": currency.issuer},
                        resp.result["amm"].get("account", "unknown"),
                    )
                    discovered += 1
            except Exception as e:
                log.debug("AMM discovery failed (XRP/IOU pair): %s", e)

        # Check IOU/IOU pairs
        from itertools import combinations

        for c1, c2 in combinations(self._currencies, 2):
            try:
                a1 = IssuedCurrency(currency=c1.currency, issuer=c1.issuer)
                a2 = IssuedCurrency(currency=c2.currency, issuer=c2.issuer)
                resp = await self._rpc(AMMInfo(asset=a1, asset2=a2), t=2.0)
                if resp.is_successful() and "amm" in resp.result:
                    self._register_amm_pool(
                        {"currency": c1.currency, "issuer": c1.issuer},
                        {"currency": c2.currency, "issuer": c2.issuer},
                        resp.result["amm"].get("account", "unknown"),
                    )
                    discovered += 1
            except Exception as e:
                log.debug("AMM discovery failed (IOU/IOU pair %s/%s): %s", c1.currency, c2.currency, e)

        log.info("Discovered %d AMM pools from ledger", discovered)

    def save_wallet_to_store(
        self, wallet: Wallet, is_gateway: bool = False, is_user: bool = False, funded_ledger_index: int | None = None
    ) -> None:
        """Save a wallet to the persistent store."""
        from workload.sqlite_store import SQLiteStore

        if isinstance(self.persistent_store, SQLiteStore):
            self.persistent_store.save_wallet(
                wallet, is_gateway=is_gateway, is_user=is_user, funded_ledger_index=funded_ledger_index
            )

    def save_currencies_to_store(self) -> None:
        """Save all currencies to the persistent store."""
        from workload.sqlite_store import SQLiteStore

        if isinstance(self.persistent_store, SQLiteStore):
            for currency in self._currencies:
                self.persistent_store.save_currency(currency)

    async def _latest_validated_ledger(self) -> int:
        return await get_latest_validated_ledger_sequence(client=self.client)

    def _record_for(self, addr: str) -> AccountRecord:
        rec = self.accounts.get(addr)
        if rec is None:
            log.debug("_record for %s", addr)
            rec = AccountRecord(lock=asyncio.Lock(), next_seq=None)
            self.accounts[addr] = rec
        return rec

    def _set_balance(self, account: str, currency: str, value: float, issuer: str | None = None) -> None:
        self.balance_tracker.set(account, currency, value, issuer)

    def _update_balance(self, account: str, currency: str, delta: float, issuer: str | None = None) -> None:
        self.balance_tracker.update(account, currency, delta, issuer)

    async def _rpc(self, req, *, t=C.RPC_TIMEOUT):
        return await asyncio.wait_for(self.client.request(req), timeout=t)

    async def fetch_ledger_tx_count(self, ledger_index: int) -> int | None:
        """Fetch the number of transactions in a validated ledger. Returns None on failure."""
        from xrpl.models.requests import Ledger as LedgerReq

        resp = await self._rpc(LedgerReq(ledger_index=ledger_index, transactions=True, expand=False))
        if resp.is_successful():
            return len(resp.result.get("ledger", {}).get("transactions", []))
        log.warning("Failed to fetch ledger %s: %s", ledger_index, resp.result)
        return None

    def notify_ledger_closed(self, ledger_index: int) -> None:
        """Called by ws_processor when a ledger closes. Wakes the consumer."""
        self.latest_ledger_index = ledger_index
        self._ledger_closed_event.set()

    async def wait_for_ledger_close(self, timeout: float = 10.0) -> int:
        """Wait for the next ledger close event. Returns the new ledger index."""
        self._ledger_closed_event.clear()
        try:
            await asyncio.wait_for(self._ledger_closed_event.wait(), timeout=timeout)
        except TimeoutError:
            pass
        return self.latest_ledger_index

    def update_server_status(self, msg: dict) -> None:
        """Update server status state from a serverStatus WS message."""
        import time

        self.latest_server_status = msg
        self.latest_server_status_time = time.time()

        load_factor = msg.get("load_factor", 256)
        load_factor_fee_escalation = msg.get("load_factor_fee_escalation")
        load_factor_fee_queue = msg.get("load_factor_fee_queue")
        load_factor_fee_reference = msg.get("load_factor_fee_reference", 256)
        server_status = msg.get("server_status", "unknown")

        queue_multiplier = load_factor_fee_queue / load_factor_fee_reference if load_factor_fee_queue else 1.0
        escalation_multiplier = (
            load_factor_fee_escalation / load_factor_fee_reference if load_factor_fee_escalation else 1.0
        )
        self.latest_server_status_computed = {
            "server_status": server_status,
            "queue_fee_multiplier": queue_multiplier,
            "open_ledger_fee_multiplier": escalation_multiplier,
            "general_load_multiplier": load_factor / 256.0,
        }

    async def alloc_seq(self, addr: str) -> int:
        rec = self._record_for(addr)

        async with rec.lock:
            if rec.next_seq is None:
                ai = await self._rpc(AccountInfo(account=addr, ledger_index="validated", strict=True))
                acct = ai.result.get("account_data")
                if acct is None:
                    raise RuntimeError(f"Account {addr} not found on ledger (unfunded?)")
                rec.next_seq = acct["Sequence"]
                log.debug("First seq for %s: fetched %d from ledger", addr, rec.next_seq)

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
            if rec.next_seq == seq + 1:
                rec.next_seq = seq
                log.debug(f"Released sequence {seq} for {addr} (local error, never submitted)")
            else:
                log.debug(
                    f"Cannot release sequence {seq} for {addr} - next_seq is {rec.next_seq} (gap would be created)"
                )

    async def warm_sequences(self, addresses: list[str], *, batch_size: int = 50) -> int:
        """Pre-fetch and cache sequence numbers for accounts in batches.

        Warms the alloc_seq cache so the build loop doesn't pay RPC latency.
        Processes accounts in chunks of `batch_size` to avoid overwhelming rippled.
        Retries failed accounts once. Returns the number of accounts warmed.
        """
        cold = [addr for addr in addresses if self._record_for(addr).next_seq is None]
        if not cold:
            return 0
        start = time.time()
        warmed = 0
        failed: list[str] = []

        for i in range(0, len(cold), batch_size):
            batch = cold[i : i + batch_size]

            async def _warm(a: str) -> bool:
                try:
                    rec = self._record_for(a)
                    async with rec.lock:
                        if rec.next_seq is not None:
                            return True
                        ai = await self._rpc(AccountInfo(account=a, ledger_index="validated", strict=True))
                        acct = ai.result.get("account_data")
                        if acct:
                            rec.next_seq = acct["Sequence"]
                            return True
                except Exception as e:
                    log.debug("warm_sequences: failed for %s: %s", a, e)
                return False

            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(_warm(a)) for a in batch]
            for a, t in zip(batch, tasks):
                if t.result():
                    warmed += 1
                else:
                    failed.append(a)

            if (i + batch_size) % 200 == 0 or i + batch_size >= len(cold):
                log.info("warm_sequences: %d/%d done, %d failed so far", warmed, len(cold), len(failed))

        # Retry failures once
        if failed:
            log.info("warm_sequences: retrying %d failed accounts", len(failed))
            for i in range(0, len(failed), batch_size):
                batch = failed[i : i + batch_size]
                async with asyncio.TaskGroup() as tg:
                    tasks = [tg.create_task(_warm(a)) for a in batch]
                for t in tasks:
                    if t.result():
                        warmed += 1

        elapsed_ms = (time.time() - start) * 1000
        log.info("Warmed %d/%d account sequences in %.0fms", warmed, len(cold), elapsed_ms)
        return warmed

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

        fee = minimum_fee

        log.debug(
            f"Fee: {fee} drops (min={minimum_fee}, open={open_ledger_fee}, base={base_fee}, "
            f"queue={fee_info.current_queue_size}/{fee_info.max_queue_size}, "
            f"ledger={fee_info.current_ledger_size}/{fee_info.expected_ledger_size})"
        )

        if fee > base_fee:
            log.debug(
                f"Queue fees escalated: minimum={minimum_fee} (queue), open_ledger={open_ledger_fee} (immediate), base={base_fee}"
            )

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
        self.pending[p.tx_hash] = p
        p.state = C.TxState.CREATED
        self._total_created += 1
        if p.transaction_type:
            tt = p.transaction_type
            self._type_submitted[tt] = self._type_submitted.get(tt, 0) + 1
        # Don't persist CREATED to SQLite — it's transient (immediately becomes SUBMITTED).
        # The store write happens in record_submitted() when the txn is on the wire.
        # This avoids a SQLite lock acquisition per transaction during batch building.

    async def record_submitted(self, p: PendingTx, engine_result: str | None, srv_txid: str | None):
        if p.state in C.TERMINAL_STATE:
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
        if p.transaction_type:
            tx_submitted(p.transaction_type, details={"hash": new_hash, "account": p.account, "sequence": p.sequence})
        await self.store.mark(
            new_hash,
            state=C.TxState.SUBMITTED,
            account=p.account,
            sequence=p.sequence,
            transaction_type=p.transaction_type,
            engine_result_first=p.engine_result_first,
        )

    async def record_validated(self, rec: ValidationRecord, meta_result: str | None = None) -> dict:
        # Check if this is a NEW validation (not a duplicate from WS + poll)
        p_before = self.pending.get(rec.txn)
        was_already_validated = p_before is not None and p_before.state == C.TxState.VALIDATED
        p_live = await self._apply_validation_state(rec, meta_result)
        if p_live and p_live.transaction_type and not was_already_validated:
            tt = p_live.transaction_type
            tx_validated(
                tt,
                meta_result or "unknown",
                details={
                    "hash": rec.txn,
                    "ledger_index": rec.seq,
                    "account": p_live.account,
                    "source": rec.src,
                    "tx_json": p_live.tx_json,
                },
            )
            self._type_validated[tt] = self._type_validated.get(tt, 0) + 1
            self._total_validated += 1
        # Keep next_seq in sync: validated txn consumed this sequence, so next must be at least seq+1
        if p_live and p_live.account and p_live.sequence is not None:
            rec_acct = self._record_for(p_live.account)
            expected = p_live.sequence + 1
            if rec_acct.next_seq is None or rec_acct.next_seq < expected:
                rec_acct.next_seq = expected

        await dispatch_validation_hooks(self, p_live, rec, meta_result)
        log.debug("txn %s validated at ledger %s via %s", rec.txn, rec.seq, rec.src)
        return {"tx_hash": rec.txn, "ledger_index": rec.seq, "source": rec.src, "meta_result": meta_result}

    async def _apply_validation_state(self, rec: ValidationRecord, meta_result: str | None) -> PendingTx | None:
        """Update pending tx state and persist to store. Returns the live PendingTx or None."""
        p_live = self.pending.get(rec.txn)
        if p_live:
            p_live.state = C.TxState.VALIDATED
            p_live.validated_ledger = rec.seq
            p_live.meta_txn_result = meta_result
            if meta_result and meta_result.startswith("tec"):
                self._failure_codes[meta_result] = self._failure_codes.get(meta_result, 0) + 1
        else:
            log.debug("record_validated: tx not in pending (race or already finalized): %s", rec.txn)
        await self.store.mark(
            rec.txn,
            state=C.TxState.VALIDATED,
            validated_ledger=rec.seq,
            meta_txn_result=meta_result,
            source=rec.src,
        )
        return self.pending.get(rec.txn)  # re-fetch after await in case pending changed

    # Validation hooks (27 methods) live in validation_hooks.py

    async def _cascade_expire_account(
        self, account: str, failed_seq: int, exclude_hash: str | None = None, fetch_seq_from_ledger: bool = False
    ):
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
            if (
                tx_hash != exclude_hash
                and p.account == account
                and p.sequence is not None
                and p.sequence > failed_seq
                and p.state not in C.TERMINAL_STATE
            ):
                p.state = C.TxState.EXPIRED
                self._total_expired += 1
                if p.engine_result_first is None:
                    p.engine_result_first = "CASCADE_EXPIRED"
                await self.store.mark(
                    tx_hash,
                    state=C.TxState.EXPIRED,
                    account=p.account,
                    sequence=p.sequence,
                    transaction_type=p.transaction_type,
                    engine_result_first=p.engine_result_first,
                )
                cascade_count += 1

        # Count remaining pending txns for this account after cascade
        remaining = sum(1 for p in self.pending.values() if p.account == account and p.state in C.PENDING_STATES)
        log.debug(
            "Cascade check for %s: expired %d txns with seq > %d, %d still pending",
            account,
            cascade_count,
            failed_seq,
            remaining,
        )

        rec = self._record_for(account)
        async with rec.lock:
            rec.generation += 1
            old_seq = rec.next_seq
            if fetch_seq_from_ledger:
                try:
                    ai = await self._rpc(AccountInfo(account=account, ledger_index="validated"))
                    new_seq = ai.result["account_data"]["Sequence"]
                    rec.next_seq = new_seq
                    delta = new_seq - old_seq if old_seq else None
                    if delta is None or delta != 0:
                        log.debug(
                            f"SEQ RESET (ledger): {account} {old_seq} -> {new_seq} (delta: {delta if delta is not None else 'N/A'})"
                        )
                    else:
                        log.debug(f"SEQ RESET (ledger, no-op): {account} {old_seq} -> {new_seq} (delta: 0)")
                except Exception as e:
                    log.error(f"Failed to fetch sequence for {account}: {e}")
                    rec.next_seq = failed_seq  # Best guess fallback
                    delta = failed_seq - old_seq if old_seq else None
                    log.warning(
                        f"SEQ RESET (fallback): {account} {old_seq} -> {failed_seq} (delta: {delta if delta is not None else 'N/A'})"
                    )
            else:
                rec.next_seq = failed_seq
                delta = failed_seq - old_seq if old_seq else None
                if delta is None or delta != 0:
                    log.debug(
                        f"SEQ RESET (expiry): {account} {old_seq} -> {failed_seq} (delta: {delta if delta is not None else 'N/A'})"
                    )
                else:
                    log.debug(f"SEQ RESET (expiry, no-op): {account} {old_seq} -> {failed_seq} (delta: 0)")

    async def record_expired(self, tx_hash: str, *, cascade: bool = True):
        if tx_hash not in self.pending:
            return

        p = self.pending[tx_hash]
        p.state = C.TxState.EXPIRED
        p.finalized_at = time.time()
        self._total_expired += 1
        await self.store.mark(
            tx_hash,
            state=C.TxState.EXPIRED,
            account=p.account,
            sequence=p.sequence,
            transaction_type=p.transaction_type,
            engine_result_first=p.engine_result_first,
        )

        if cascade and p.account and p.sequence is not None:
            if p.transaction_type == C.TxType.BATCH:
                log.debug(
                    f"EXPIRED (Batch): {p.transaction_type} account={p.account} seq={p.sequence} hash={tx_hash} - will cascade and sync from ledger"
                )
            else:
                log.debug(
                    f"EXPIRED: {p.transaction_type} account={p.account} seq={p.sequence} hash={tx_hash} - will cascade and sync from ledger"
                )
            await self._cascade_expire_account(p.account, p.sequence, exclude_hash=tx_hash, fetch_seq_from_ledger=True)

    def expire_past_lls(self, ledger_index: int) -> int:
        """Force-expire pending txns whose LastLedgerSequence has passed.

        This is the producer's self-healing mechanism: when all accounts are blocked
        by stale pending txns (e.g. after tefPAST_SEQ cascades), this scans and expires
        them immediately rather than waiting for the 5s periodic_finality_check.

        Returns count of expired txns.
        """
        expired_count = 0
        for tx_hash, p in list(self.pending.items()):
            if p.state in C.PENDING_STATES and p.last_ledger_seq and p.last_ledger_seq < ledger_index:
                p.state = C.TxState.EXPIRED
                self._total_expired += 1
                if p.engine_result_first is None:
                    p.engine_result_first = "PAST_LLS"
                p.finalized_at = time.time()
                expired_count += 1
        if expired_count:
            log.warning(
                "expire_past_lls: force-expired %d stale txns (ledger=%d)",
                expired_count,
                ledger_index,
            )
        return expired_count

    def diagnostics_snapshot(self) -> dict:
        """Return diagnostic data about pending txn and account health."""
        pending_by_state: dict[str, int] = {}
        oldest_pending_age = 0  # in ledgers
        blocked_accounts: set[str] = set()

        for p in self.pending.values():
            if p.state in C.PENDING_STATES:
                state_name = p.state.value
                pending_by_state[state_name] = pending_by_state.get(state_name, 0) + 1
                blocked_accounts.add(p.account)
                if self.latest_ledger_index and p.created_ledger:
                    age = self.latest_ledger_index - p.created_ledger
                    oldest_pending_age = max(oldest_pending_age, age)

        total_accounts = len(self.wallets)
        free_count = total_accounts - len(blocked_accounts)

        # Sample of blocked accounts with their pending details
        blocked_sample: list[dict] = []
        for addr in list(blocked_accounts)[:10]:
            acct_pending = [
                {
                    "hash": p.tx_hash,
                    "state": p.state.value,
                    "type": p.transaction_type,
                    "seq": p.sequence,
                    "lls": p.last_ledger_seq,
                    "age_ledgers": self.latest_ledger_index - p.created_ledger if p.created_ledger else None,
                    "past_lls": p.last_ledger_seq < self.latest_ledger_index if p.last_ledger_seq else False,
                }
                for p in self.pending.values()
                if p.account == addr and p.state in C.PENDING_STATES
            ]
            blocked_sample.append({"account": addr, "pending": acct_pending})

        return {
            "ledger_index": self.latest_ledger_index,
            "total_accounts": total_accounts,
            "blocked_accounts": len(blocked_accounts),
            "free_accounts": free_count,
            "pending_by_state": pending_by_state,
            "oldest_pending_age_ledgers": oldest_pending_age,
            "blocked_sample": blocked_sample,
        }

    def find_by_state(self, *states: C.TxState) -> list[PendingTx]:
        return [p for p in self.pending.values() if p.state in set(states)]

    def get_pending_txn_counts_by_account(self) -> dict[str, int]:
        """Get count of pending transactions per account.

        Returns:
            Dict mapping account address to count of CREATED/SUBMITTED/RETRYABLE transactions.
            Used to enforce per-account queue limit of 10 (see FeeEscalation.md:260).
        """
        counts = {}
        for p in self.pending.values():
            if p.state in C.PENDING_STATES and p.account:
                counts[p.account] = counts.get(p.account, 0) + 1
        return counts

    async def build_sign_and_track(
        self,
        txn: Transaction,
        wallet: Wallet,
        horizon: int = C.HORIZON,
        *,
        fee_drops: int | None = None,
        created_ledger: int | None = None,
        last_ledger_seq: int | None = None,
        preallocated_seq: int | None = None,
        expect_rejection: bool = False,
    ) -> PendingTx:
        if created_ledger is not None and last_ledger_seq is not None:
            created_li = created_ledger
            lls = last_ledger_seq
        else:
            created_li = (await self._rpc(ServerState(), t=2.0)).result["state"]["validated_ledger"]["seq"]
            lls = created_li + horizon

        tx = txn.to_xrpl()
        if tx.get("Flags") == 0:
            del tx["Flags"]

        need_seq = "TicketSequence" not in tx and not tx.get("Sequence")
        need_fee = not tx.get("Fee")

        if need_seq:
            if preallocated_seq is not None:
                seq = preallocated_seq
            else:
                seq = await self.alloc_seq(wallet.address)
            # In asyncio (cooperative): no await between here and seq assignment,
            # so generation reflects the state at alloc_seq time.
            acct_gen = self._record_for(wallet.address).generation
        else:
            seq = tx.get("Sequence")
            acct_gen = 0  # Ticket-based tx; generation not applicable

        base_fee = (
            fee_drops
            if (fee_drops is not None and need_fee)
            else (await self._open_ledger_fee() if need_fee else int(tx["Fee"]))
        )

        txn_type = tx.get("TransactionType")
        if txn_type == "Batch":
            OWNER_RESERVE_DROPS = 2_000_000
            inner_txns = tx.get("RawTransactions", [])
            fee = (2 * OWNER_RESERVE_DROPS) + (base_fee * len(inner_txns))
            log.debug(f"Batch fee: 2*{OWNER_RESERVE_DROPS} + {base_fee}*{len(inner_txns)} = {fee} drops")
        elif txn_type in ("AMMCreate", "VaultCreate", "PermissionedDomainSet"):
            OWNER_RESERVE_DROPS = 2_000_000
            fee = OWNER_RESERVE_DROPS
            log.debug(f"{txn_type} fee: {fee} drops (owner_reserve)")
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
            account_generation=acct_gen,
            expect_rejection=expect_rejection,
        )
        await self.record_created(p)
        return p

    async def submit_pending(self, p: PendingTx, timeout: float = C.SUBMIT_TIMEOUT) -> dict | None:
        if p.state in C.TERMINAL_STATE:
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
            resp = await asyncio.wait_for(self.client.request(SubmitOnly(tx_blob=p.signed_blob_hex)), timeout=timeout)
            res = resp.result
            er = res.get("engine_result")
            if p.engine_result_first is None:
                p.engine_result_first = er
            if p.engine_result_message is None:
                p.engine_result_message = res.get("engine_result_message")

            if isinstance(er, str) and er.startswith("tel"):
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
                log.debug(f"tel* (may retry): {er} - {p.transaction_type} seq={p.sequence} - tracking until expiry")
                return res

            if er == "tefPAST_SEQ":
                # !! CAUTION: expect_rejection suppresses cascade-expire and downgrades
                # logging/assertions for this rejection. If the flag is ever set
                # incorrectly (e.g., a valid txn tagged as invalid, or a tainting
                # strategy that accidentally produces a valid txn), real sequence
                # desync bugs will be silently swallowed. Any change to how
                # expect_rejection is set MUST be reviewed for false positives.
                # See also the general tem/tef handler below (~20 lines down).
                p.state = C.TxState.REJECTED
                self._total_rejected += 1
                self._failure_codes[er] = self._failure_codes.get(er, 0) + 1
                _reject_fn = tx_intentionally_rejected if p.expect_rejection else tx_rejected
                if p.transaction_type:
                    _reject_fn(
                        p.transaction_type,
                        er,
                        details={"hash": p.tx_hash, "account": p.account, "sequence": p.sequence, "tx_json": p.tx_json},
                    )
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
                # Only cascade + resync if this is the first tefPAST_SEQ for
                # this account (i.e., its next_seq hasn't already been reset
                # below this txn's sequence by a prior cascade).
                # Skip cascade for intentionally bad txns — sequence is still valid
                if p.expect_rejection:
                    log.debug("tefPAST_SEQ (expected): %s %s seq=%s", p.transaction_type, p.account, p.sequence)
                    return res

                rec = self._record_for(p.account)
                try:
                    _ai = await self._rpc(AccountInfo(account=p.account, ledger_index="current"))
                    _actual = _ai.result["account_data"]["Sequence"]
                    log.debug(
                        "tefPAST_SEQ detail: %s %s submitted_seq=%d current_ledger_seq=%d delta=%d next_seq_before=%s",
                        p.transaction_type, p.account, p.sequence, _actual, _actual - p.sequence, rec.next_seq,
                    )
                except Exception as _e:
                    log.debug("tefPAST_SEQ detail: fetch failed: %s", _e)

                await self._cascade_expire_account(
                    p.account, p.sequence, exclude_hash=p.tx_hash, fetch_seq_from_ledger=True
                )
                return res

            if isinstance(er, str) and er.startswith(("tem", "tef")):
                # !! CAUTION: expect_rejection downgrades logging and routes to
                # tx_intentionally_rejected() instead of tx_rejected(). A false
                # positive (valid txn wrongly tagged) hides real builder bugs.
                # Only workload_runner sets this flag, and only when intent==INVALID
                # AND taint_txn() was applied. If you add other callers, audit
                # carefully — a suppressed warning here is invisible in production.
                p.state = C.TxState.REJECTED
                self._total_rejected += 1
                self._failure_codes[er] = self._failure_codes.get(er, 0) + 1
                if er == "temDISABLED" and p.transaction_type:
                    self._tem_disabled_types.add(p.transaction_type)
                _reject_fn = tx_intentionally_rejected if p.expect_rejection else tx_rejected
                if p.transaction_type:
                    _reject_fn(
                        p.transaction_type,
                        er,
                        details={"hash": p.tx_hash, "account": p.account, "sequence": p.sequence, "tx_json": p.tx_json},
                    )
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
                # tem codes are malformed — never submitted to network, sequence NOT consumed.
                # Release the sequence so the account isn't blocked.
                if er.startswith("tem") and p.account and p.sequence is not None:
                    await self.release_seq(p.account, p.sequence)

                if p.expect_rejection:
                    log.debug(f"REJECTED (expected): {er} - {p.transaction_type} from {p.account}")
                elif er == "temDISABLED":
                    log.debug(f"REJECTED: {er} - {p.transaction_type} from {p.account} (amendment not enabled)")
                else:
                    log.warning(
                        f"REJECTED: {er} - {p.transaction_type} from {p.account} seq={p.sequence} hash={p.tx_hash}"
                    )

                if p.transaction_type == C.TxType.BATCH and p.account:
                    try:
                        ai = await self._rpc(AccountInfo(account=p.account, ledger_index="validated"))
                        rec_acct = self._record_for(p.account)
                        async with rec_acct.lock:
                            old_seq = rec_acct.next_seq
                            rec_acct.next_seq = ai.result["account_data"]["Sequence"]
                            log.debug(f"Batch rejected: synced {p.account} sequence {old_seq} -> {rec_acct.next_seq}")
                    except Exception as e:
                        log.warning(f"Failed to sync sequence after Batch rejection: {e}")

                return res

            if er == "terPRE_SEQ":
                log.debug(
                    f"terPRE_SEQ: {p.transaction_type} account={p.account} seq={p.sequence} hash={p.tx_hash} - cascade expire and ledger resync"
                )
                p.state = C.TxState.EXPIRED
                self._total_expired += 1
                self.pending[p.tx_hash] = p
                await self.store.mark(
                    p.tx_hash,
                    state=p.state,
                    account=p.account,
                    sequence=p.sequence,
                    transaction_type=p.transaction_type,
                    engine_result_first=er,
                )
                await self._cascade_expire_account(p.account, p.sequence, fetch_seq_from_ledger=True)
                return res

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
            await self.store.mark(
                p.tx_hash,
                state=C.TxState.FAILED_NET,
                account=p.account,
                sequence=p.sequence,
                transaction_type=p.transaction_type,
                engine_result_first=p.engine_result_first,
            )
            return {"engine_result": "timeout"}

        except Exception as e:
            p.state = C.TxState.FAILED_NET
            self.pending[p.tx_hash] = p
            log.error("submit error tx=%s: %s", p.tx_hash, e)
            await self.store.mark(
                p.tx_hash,
                state=C.TxState.FAILED_NET,
                account=p.account,
                sequence=p.sequence,
                transaction_type=p.transaction_type,
                message=str(e),
            )
            return {"engine_result": "error", "message": str(e)}

    async def check_finality(self, p: PendingTx, grace: int = 2) -> tuple[C.TxState, int | None]:
        try:
            txr = await self.client.request(Tx(transaction=p.tx_hash))
            if txr.is_successful() and txr.result.get("validated"):
                li = int(txr.result["ledger_index"])
                result = txr.result["meta"]["TransactionResult"]

                p.state = C.TxState.VALIDATED
                p.validated_ledger = li
                p.meta_txn_result = result
                await self.record_validated(ValidationRecord(p.tx_hash, li, ValidationSrc.POLL), result)
                return p.state, li
        except Exception:
            log.error("check_finality failed for %s", p.tx_hash, exc_info=True)

        latest_val = await self._latest_validated_ledger()
        if latest_val > (p.last_ledger_seq + grace):
            await self.record_expired(p.tx_hash)
            return p.state, None

        # Don't transition FAILED_NET → RETRYABLE: the tx may still be queued in rippled.
        # Keep it locked (pending) until LLS expires or WS confirms validation.
        if p.state not in (C.TxState.SUBMITTED, C.TxState.FAILED_NET):
            p.state = C.TxState.RETRYABLE
            await self.store.mark(p.tx_hash, state=p.state)
        return p.state, None

    async def submit_random_txn(self):
        txn = await generate_txn(self.ctx)
        if txn is None:
            return None
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
        if txn is None:
            return {"error": f"Cannot build {transaction} — no eligible accounts or missing prerequisites"}
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

        amount = initial_xrp_drops or self.config["users"]["default_balance"]

        w = Wallet.create()

        fund_txn = Payment(
            account=self.funding_wallet.address,
            destination=w.address,
            amount=str(amount),
        )

        pending = await self.build_sign_and_track(fund_txn, self.funding_wallet)
        pending.wallet = w  # stash for *post-validation* adoption in record_validated()
        submit_res = await self.submit_pending(pending)

        return {
            "address": w.address,
            "tx_hash": pending.tx_hash,
            "submitted": True,
            "engine_result": (submit_res or {}).get("engine_result"),
            "funding_drops": int(amount),
        }

    def snapshot_pending(self, *, open_only: bool = True) -> list[dict]:
        out = []
        for txh, p in self.pending.items():
            if open_only and p.state not in C.PENDING_STATES:
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
                    "engine_result_message": p.engine_result_message,
                    "engine_result_final": p.meta_txn_result or p.engine_result_first,
                    "validated_ledger": p.validated_ledger,
                    "meta_txn_result": p.meta_txn_result,
                    "transaction_type": p.transaction_type,
                }
            )
        return out

    def snapshot_failed(self) -> list[dict[str, Any]]:
        """Return transactions that failed — including tec codes (validated on-chain but failed in intent).

        tec transactions are applied to the ledger (sequence consumed, fee burned) but the
        intended action didn't succeed (e.g., tecUNFUNDED_OFFER). They're VALIDATED in state
        but failures from the user's perspective.
        """
        failed_states = {"REJECTED", "EXPIRED", "FAILED_NET"}
        results = []
        for r in self.snapshot_pending(open_only=False):
            if r.get("engine_result_first") == "CASCADE_EXPIRED":
                continue
            if r["state"] in failed_states:
                results.append(r)
            elif r["state"] == "VALIDATED" and r.get("meta_txn_result", "").startswith("tec"):
                results.append(r)
        return results

    def snapshot_failure_codes(self) -> dict[str, int]:
        """Cumulative failure code counts (survives cleanup_terminal)."""
        return dict(self._failure_codes)

    def snapshot_stats(self) -> dict[str, Any]:
        # In-flight counts from pending dict (for "In-Flight" and "Created" cards)
        by_state: dict[str, int] = {}
        for p in self.pending.values():
            state = p.state.name
            by_state[state] = by_state.get(state, 0) + 1

        # Cumulative totals (survive cleanup_terminal)
        by_state["VALIDATED"] = self._total_validated
        by_state["REJECTED"] = self._total_rejected
        by_state["EXPIRED"] = self._total_expired

        uptime = time.time() - self.started_at
        ledgers_elapsed = max(0, self.latest_ledger_index - self.first_ledger_index) if self.first_ledger_index else 0
        result = {
            "total_tracked": self._total_created,
            "by_state": by_state,
            "gateways": len(self.gateways),
            "users": len(self.users),
            "uptime_seconds": round(uptime),
            "started_at": self.started_at,
            "ledger_index": self.latest_ledger_index,
            "first_ledger_index": self.first_ledger_index,
            "ledgers_elapsed": ledgers_elapsed,
        }

        # Merge store-level aggregate stats (submission results, validation sources)
        store_stats = self.store.snapshot_stats()
        for key in ("validated_by_source", "validated_by_result", "submission_results", "recent_validations"):
            if key in store_stats:
                result[key] = store_stats[key]

        # Per-type breakdown: validated / submitted (cumulative counters, survive cleanup)
        result["by_type"] = dict(self.store.count_by_type)
        result["by_type_validated"] = dict(self._type_validated)
        result["by_type_total"] = dict(self._type_submitted)
        result["tem_disabled_types"] = sorted(self._tem_disabled_types)

        result["dex"] = self.snapshot_dex_metrics()

        return result

    def snapshot_dex_metrics(self) -> dict:
        """Return current DEX metrics snapshot for the API."""
        dm = self.dex_metrics
        return {
            "pools_created": dm.pools_created,
            "active_pools": dm.active_pools,
            "total_deposits": dm.total_deposits,
            "total_withdrawals": dm.total_withdrawals,
            "total_offers": dm.total_offers,
            "total_xrp_locked_drops": dm.total_xrp_locked_drops,
            "last_poll_ledger": dm.last_poll_ledger,
            "pool_count": len(self.amm),
            "pool_details": dm.pool_snapshots,
        }

    async def flush_to_persistent_store(self) -> int:
        """Flush in-memory transaction records to the persistent store (SQLite).

        Called on shutdown to ensure durability. Uses a single bulk upsert
        instead of one connection per record.
        """
        from workload.sqlite_store import SQLiteStore

        if not isinstance(self.persistent_store, SQLiteStore):
            return 0

        records = [
            (
                tx_hash,
                {
                    "state": p.state,
                    "account": p.account,
                    "sequence": p.sequence,
                    "transaction_type": p.transaction_type,
                    "engine_result_first": p.engine_result_first,
                    "validated_ledger": p.validated_ledger,
                    "meta_txn_result": p.meta_txn_result,
                    "finalized_at": p.finalized_at,
                },
            )
            for tx_hash, p in self.pending.items()
        ]

        if not records:
            return 0

        try:
            flushed = await self.persistent_store.bulk_upsert(records)
            log.info("Flushed %d records to persistent store", flushed)
            return flushed
        except Exception as e:
            log.warning("Bulk flush failed: %s", e)
            return 0

    async def poll_dex_metrics(self) -> dict:
        """Poll amm_info for all tracked AMM pools and update DEX metrics."""
        from xrpl.models.currencies import XRP as XRPCurrency
        from xrpl.models.requests import AMMInfo

        pool_snapshots = []
        total_xrp_locked = 0

        for pool in self.amm.pools:
            try:
                asset1 = pool["asset1"]
                asset2 = pool["asset2"]

                if asset1.get("currency") == "XRP":
                    a1 = XRPCurrency()
                else:
                    a1 = IssuedCurrency(currency=asset1["currency"], issuer=asset1["issuer"])

                if asset2.get("currency") == "XRP":
                    a2 = XRPCurrency()
                else:
                    a2 = IssuedCurrency(currency=asset2["currency"], issuer=asset2["issuer"])

                resp = await self._rpc(AMMInfo(asset=a1, asset2=a2), t=3.0)
                if resp.is_successful():
                    amm_data = resp.result.get("amm", {})
                    snapshot = {
                        "asset1": asset1,
                        "asset2": asset2,
                        "amount": amm_data.get("amount"),
                        "amount2": amm_data.get("amount2"),
                        "lp_token": amm_data.get("lp_token"),
                        "trading_fee": amm_data.get("trading_fee"),
                        "vote_slots": len(amm_data.get("vote_slots", [])),
                    }
                    pool_snapshots.append(snapshot)

                    amt = amm_data.get("amount")
                    if isinstance(amt, str):
                        total_xrp_locked += int(amt)
                    amt2 = amm_data.get("amount2")
                    if isinstance(amt2, str):
                        total_xrp_locked += int(amt2)

            except Exception as e:
                log.debug(f"Failed to poll amm_info for pool: {e}")

        current_ledger = await self._current_ledger_index()
        self.dex_metrics.pool_snapshots = pool_snapshots
        self.dex_metrics.last_poll_ledger = current_ledger
        self.dex_metrics.total_xrp_locked_drops = total_xrp_locked
        self.dex_metrics.active_pools = len(pool_snapshots)

        return self.dex_metrics

    def snapshot_tx(self, tx_hash: str) -> dict[str, Any]:
        p = self.pending.get(tx_hash)
        ws_port = self.config.get("rippled", {}).get("ws_port", 6006)
        if not p:
            return {}
        return {
            "tx_hash": p.tx_hash,
            "state": p.state.name,
            "transaction_type": p.transaction_type,
            "account": p.account,
            "sequence": p.sequence,
            "last_ledger_seq": p.last_ledger_seq,
            "created_ledger": p.created_ledger,
            "attempts": p.attempts,
            "engine_result_first": p.engine_result_first,
            "validated_ledger": p.validated_ledger,
            "meta_txn_result": p.meta_txn_result,
            "tx_json": p.tx_json,
            "link": f"https://custom.xrpl.org/localhost:{ws_port}/transactions/{tx_hash}",
        }

    def _cleanup_terminal(self, keep_recent: int = 200) -> int:
        """Remove old terminal txns from self.pending (already persisted in store)."""
        terminal = [(txh, p) for txh, p in self.pending.items() if p.state in C.TERMINAL_STATE]
        if len(terminal) <= keep_recent:
            return 0
        terminal.sort(key=lambda x: x[1].finalized_at or x[1].created_at)
        to_remove = terminal[:-keep_recent]
        for txh, _ in to_remove:
            del self.pending[txh]
        return len(to_remove)


async def periodic_dex_metrics(w: Workload, stop: asyncio.Event, poll_interval_ledgers: int = 5):
    """Periodically poll DEX metrics every N ledgers."""
    last_polled_ledger = 0
    while not stop.is_set():
        try:
            if not w.amm:
                await asyncio.sleep(5)
                continue

            current = await w._current_ledger_index()
            if current >= last_polled_ledger + poll_interval_ledgers:
                await w.poll_dex_metrics()
                last_polled_ledger = current
                log.debug("DEX metrics polled at ledger %d (%d pools)", current, len(w.amm))

            await asyncio.sleep(3)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[dex_metrics] poll error; continuing")
            await asyncio.sleep(5)


async def periodic_finality_check(w: Workload, stop: asyncio.Event, interval: int = 5):
    iteration = 0
    while not stop.is_set():
        try:
            for p in w.find_by_state(C.TxState.SUBMITTED, C.TxState.RETRYABLE, C.TxState.FAILED_NET):
                try:
                    await w.check_finality(p)
                except Exception:
                    log.exception("[finality] check failed for %s", getattr(p, "tx_hash", p))

            iteration += 1
            if iteration % 10 == 0:
                removed = w._cleanup_terminal()
                if removed:
                    log.debug("[finality] cleaned up %d terminal txns from pending", removed)

            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[finality] outer loop error; continuing")
            await asyncio.sleep(0.5)
