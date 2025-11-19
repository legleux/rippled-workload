"""SQLite-backed persistent store for workload state."""

import asyncio
import json
import sqlite3
import time
from collections import deque
from pathlib import Path

import xrpl
from xrpl.wallet import Wallet
from xrpl.models import IssuedCurrency

from workload.workload_core import ValidationRecord, TERMINAL_STATE
import workload.constants as C
import logging

log = logging.getLogger("workload.sqlite_store")


class SQLiteStore:
    """Persistent store backed by SQLite."""

    def __init__(self, db_path: str | Path = "workload_state.db") -> None:
        self.db_path = Path(db_path)
        self._lock = asyncio.Lock()
        self.validations: deque[ValidationRecord] = deque(maxlen=5000)
        self.count_by_state: dict[str, int] = {}
        self.validated_by_source: dict[str, int] = {}

        # Initialize database
        self._init_db()
        # Load validations into memory for fast access
        self._load_validations()
        self._recount()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(
                """
                -- Wallets table (for persistent wallet storage)
                CREATE TABLE IF NOT EXISTS wallets (
                    address TEXT PRIMARY KEY,
                    seed TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    is_gateway INTEGER DEFAULT 0,
                    is_user INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                );

                -- Account records (sequence tracking)
                CREATE TABLE IF NOT EXISTS accounts (
                    address TEXT PRIMARY KEY,
                    next_seq INTEGER,
                    created_at REAL NOT NULL
                );

                -- Transaction records
                CREATE TABLE IF NOT EXISTS transactions (
                    tx_hash TEXT PRIMARY KEY,
                    state TEXT,
                    source TEXT,
                    account TEXT,
                    validated_ledger INTEGER,
                    finalized_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    data TEXT NOT NULL  -- JSON blob for all fields
                );
                CREATE INDEX IF NOT EXISTS idx_tx_state ON transactions(state);
                CREATE INDEX IF NOT EXISTS idx_tx_account ON transactions(account);

                -- Validation history
                CREATE TABLE IF NOT EXISTS validations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tx_hash TEXT NOT NULL,
                    ledger_seq INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    validated_at REAL NOT NULL,
                    UNIQUE(tx_hash, ledger_seq)
                );
                CREATE INDEX IF NOT EXISTS idx_val_tx ON validations(tx_hash);
                CREATE INDEX IF NOT EXISTS idx_val_ledger ON validations(ledger_seq);

                -- Issued currencies
                CREATE TABLE IF NOT EXISTS currencies (
                    currency TEXT NOT NULL,
                    issuer TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (currency, issuer)
                );
                """
            )
            conn.commit()
            log.info(f"SQLite database initialized at {self.db_path}")
        finally:
            conn.close()

    def _load_validations(self) -> None:
        """Load recent validations from DB into memory deque."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT tx_hash, ledger_seq, source FROM validations ORDER BY validated_at DESC LIMIT 5000"
            )
            # Reverse to get chronological order
            for tx_hash, ledger_seq, source in reversed(cursor.fetchall()):
                self.validations.append(ValidationRecord(txn=tx_hash, seq=ledger_seq, src=source))
        finally:
            conn.close()

    def _recount(self) -> None:
        """Recompute metrics from database."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Count by state
            cursor = conn.execute("SELECT state, COUNT(*) FROM transactions GROUP BY state")
            self.count_by_state = dict(cursor.fetchall())

            # Count validations by source
            cursor = conn.execute("SELECT source, COUNT(*) FROM validations GROUP BY source")
            self.validated_by_source = dict(cursor.fetchall())
        finally:
            conn.close()

    # =========================================================================
    # Transaction record methods (Store protocol)
    # =========================================================================

    async def update_record(self, tx: dict) -> None:
        """Insert or update a transaction record."""
        tx_hash = tx.get("tx_hash")
        if not tx_hash:
            raise ValueError("update_record() requires 'tx_hash'")

        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                now = time.time()
                conn.execute(
                    """
                    INSERT INTO transactions (tx_hash, state, source, account,
                                             validated_ledger, finalized_at,
                                             created_at, updated_at, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tx_hash) DO UPDATE SET
                        state = excluded.state,
                        source = excluded.source,
                        account = excluded.account,
                        validated_ledger = excluded.validated_ledger,
                        finalized_at = excluded.finalized_at,
                        updated_at = excluded.updated_at,
                        data = excluded.data
                    """,
                    (
                        tx_hash,
                        tx.get("state"),
                        tx.get("source"),
                        tx.get("account"),
                        tx.get("validated_ledger"),
                        tx.get("finalized_at"),
                        tx.get("created_at", now),
                        now,
                        json.dumps(tx),
                    ),
                )
                conn.commit()
                self._recount()
            finally:
                conn.close()

    async def get(self, tx_hash: str) -> dict | None:
        """Retrieve a transaction record by hash."""
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("SELECT data FROM transactions WHERE tx_hash = ?", (tx_hash,))
                row = cursor.fetchone()
                return json.loads(row[0]) if row else None
            finally:
                conn.close()

    async def mark(self, tx_hash: str, *, source: str | None = None, **fields) -> None:
        """Update or insert a transaction record with state transitions."""
        log.info("Mark %s", tx_hash)
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Get existing record
                cursor = conn.execute("SELECT data FROM transactions WHERE tx_hash = ?", (tx_hash,))
                row = cursor.fetchone()
                rec = json.loads(row[0]) if row else {}

                prev_state = rec.get("state")
                rec.update(fields)

                if source is not None:
                    rec["source"] = source

                state = rec.get("state")
                if isinstance(state, C.TxState):
                    state = state.name
                    rec["state"] = state

                # Terminal handling
                if state in TERMINAL_STATE:
                    rec.setdefault("finalized_at", time.time())

                    # Validation history
                    if state == "VALIDATED" and prev_state != "VALIDATED":
                        seq = rec.get("validated_ledger") or 0
                        src = source or rec.get("source", "unknown")

                        # Check if validation already exists
                        cursor = conn.execute(
                            "SELECT 1 FROM validations WHERE tx_hash = ? AND ledger_seq = ?",
                            (tx_hash, seq),
                        )
                        if not cursor.fetchone():
                            conn.execute(
                                "INSERT INTO validations (tx_hash, ledger_seq, source, validated_at) "
                                "VALUES (?, ?, ?, ?)",
                                (tx_hash, seq, src, time.time()),
                            )
                            # Add to in-memory deque
                            if not any(v.txn == tx_hash and v.seq == seq for v in self.validations):
                                log.info("%s ValidationRecord in %s by %s -- %s", state, seq, src, tx_hash)
                                self.validations.append(ValidationRecord(txn=tx_hash, seq=seq, src=src))

                rec["tx_hash"] = tx_hash
                now = time.time()
                conn.execute(
                    """
                    INSERT INTO transactions (tx_hash, state, source, account,
                                             validated_ledger, finalized_at,
                                             created_at, updated_at, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tx_hash) DO UPDATE SET
                        state = excluded.state,
                        source = excluded.source,
                        account = excluded.account,
                        validated_ledger = excluded.validated_ledger,
                        finalized_at = excluded.finalized_at,
                        updated_at = excluded.updated_at,
                        data = excluded.data
                    """,
                    (
                        tx_hash,
                        rec.get("state"),
                        rec.get("source"),
                        rec.get("account"),
                        rec.get("validated_ledger"),
                        rec.get("finalized_at"),
                        rec.get("created_at", now),
                        now,
                        json.dumps(rec),
                    ),
                )
                conn.commit()
                self._recount()
                log.info("%s --> %s  %s", prev_state, state, tx_hash)
            finally:
                conn.close()

    async def rekey(self, old_hash: str, new_hash: str) -> None:
        """Replace a record's key when hash changes."""
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("SELECT data FROM transactions WHERE tx_hash = ?", (old_hash,))
                row = cursor.fetchone()
                if not row:
                    return

                rec = json.loads(row[0])
                rec["tx_hash"] = new_hash

                # Delete old, insert new
                conn.execute("DELETE FROM transactions WHERE tx_hash = ?", (old_hash,))
                now = time.time()
                conn.execute(
                    """
                    INSERT INTO transactions (tx_hash, state, source, account,
                                             validated_ledger, finalized_at,
                                             created_at, updated_at, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_hash,
                        rec.get("state"),
                        rec.get("source"),
                        rec.get("account"),
                        rec.get("validated_ledger"),
                        rec.get("finalized_at"),
                        rec.get("created_at", now),
                        now,
                        json.dumps(rec),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def find_by_state(self, *states: C.TxState | str) -> list[dict]:
        """Return records matching any of the given states."""
        wanted = {s.name if isinstance(s, C.TxState) else s for s in states}
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                placeholders = ",".join("?" * len(wanted))
                cursor = conn.execute(
                    f"SELECT data FROM transactions WHERE state IN ({placeholders})",
                    tuple(wanted),
                )
                return [json.loads(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def all_records(self) -> list[dict]:
        """Return all transaction records."""
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("SELECT data FROM transactions")
                return [json.loads(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    def snapshot_stats(self) -> dict:
        """Return current statistics."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM transactions")
            total = cursor.fetchone()[0]

            return {
                "by_state": dict(self.count_by_state),
                "validated_by_source": dict(self.validated_by_source),
                "total_tracked": total,
                "recent_validations": len(self.validations),
            }
        finally:
            conn.close()

    # =========================================================================
    # Wallet persistence
    # =========================================================================

    def save_wallet(self, wallet: Wallet, is_gateway: bool = False, is_user: bool = False) -> None:
        """Persist a wallet to database."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Get algorithm name
            algo = wallet.algorithm if hasattr(wallet, "algorithm") else "secp256k1"
            if isinstance(algo, xrpl.CryptoAlgorithm):
                algo = algo.value

            conn.execute(
                """
                INSERT INTO wallets (address, seed, algorithm, is_gateway, is_user, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    is_gateway = excluded.is_gateway,
                    is_user = excluded.is_user
                """,
                (wallet.address, wallet.seed, algo, int(is_gateway), int(is_user), time.time()),
            )
            conn.commit()
            log.info(f"Saved wallet {wallet.address} (gateway={is_gateway}, user={is_user})")
        finally:
            conn.close()

    def load_wallets(self) -> dict[str, tuple[Wallet, bool, bool]]:
        """Load all wallets from database. Returns dict[address, (wallet, is_gateway, is_user)]."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT address, seed, algorithm, is_gateway, is_user FROM wallets")
            result = {}
            for address, seed, algo_str, is_gateway, is_user in cursor.fetchall():
                # Convert algorithm string to enum
                try:
                    algo = xrpl.CryptoAlgorithm(algo_str)
                except ValueError:
                    algo = xrpl.CryptoAlgorithm.SECP256K1

                wallet = Wallet.from_seed(seed, algorithm=algo)
                result[address] = (wallet, bool(is_gateway), bool(is_user))

            log.info(f"Loaded {len(result)} wallets from database")
            return result
        finally:
            conn.close()

    # =========================================================================
    # Currency persistence
    # =========================================================================

    def save_currency(self, currency: IssuedCurrency) -> None:
        """Persist an issued currency."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO currencies (currency, issuer, created_at) VALUES (?, ?, ?)",
                (currency.currency, currency.issuer, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def load_currencies(self) -> list[IssuedCurrency]:
        """Load all currencies from database."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT currency, issuer FROM currencies")
            return [IssuedCurrency(currency=curr, issuer=iss) for curr, iss in cursor.fetchall()]
        finally:
            conn.close()

    # =========================================================================
    # Metadata check
    # =========================================================================

    def has_state(self) -> bool:
        """Check if database has any persisted state (wallets or transactions)."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM wallets")
            wallet_count = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM transactions")
            tx_count = cursor.fetchone()[0]

            has_state = wallet_count > 0 or tx_count > 0
            log.info(f"Database state check: {wallet_count} wallets, {tx_count} transactions (has_state={has_state})")
            return has_state
        finally:
            conn.close()
