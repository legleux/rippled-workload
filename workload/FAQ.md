# FAQ

## Sequence Allocation

### How does the workload manage sequence numbers?

Every XRPL transaction needs a sequence number unique to its source account. The workload tracks these locally to avoid round-tripping to the ledger for every transaction.

**The flow:**

```
1. alloc_seq(account)
   ├── First call? Fetch Sequence from ledger via AccountInfo RPC
   └── Subsequent calls? Increment locally: next_seq++

2. build_sign_and_track(txn, wallet)
   └── Signs the txn with the allocated sequence, adds to pending dict

3. submit_pending(pending_tx)
   ├── tesSUCCESS / terQUEUED → tracked, waiting for validation
   ├── tel* (local error) → release_seq() gives the number back
   └── tefPAST_SEQ → sequence is stale, trigger cascade (see below)

4. Validation (via WS or polling)
   └── Transaction confirmed in a ledger → removed from pending
```

With `max_pending_per_account = 10` (config.toml), the workload can have sequences 100-109 in-flight for a single account simultaneously. This is fine as long as they all validate in order.

### What is a "SEQ RESET"?

When something goes wrong — a transaction expires, gets rejected, or returns `tefPAST_SEQ` — the local sequence counter may be out of sync with the ledger. The workload resets it.

**Log format:**
```
SEQ RESET (ledger): rUXwxUD... 488 -> 486 (delta: -2)
```

This means:
- **Account** `rUXwxUD...` had local `next_seq = 488`
- The workload queried the ledger and found the **actual** next expected sequence is `486`
- **delta: -2** means the local counter was 2 ahead of the ledger

**Why does this happen?** The workload allocated sequences 486 and 487 for transactions that never made it into a validated ledger (they expired or were rejected). The ledger still expects sequence 486 because it never saw those transactions validate.

**Three variants:**

| Log message | Trigger | How it resets |
|---|---|---|
| `SEQ RESET (ledger)` | `tefPAST_SEQ` or `terPRE_SEQ` | Fetches actual sequence from ledger via RPC |
| `SEQ RESET (expiry)` | Transaction past LastLedgerSequence | Sets to the expired transaction's sequence |
| `SEQ RESET (fallback)` | Ledger fetch failed | Best guess: uses the failed sequence number |

### What is "CASCADE_EXPIRED"?

When a transaction for an account fails or expires, all higher-sequence transactions for that same account are doomed — the ledger processes sequences strictly in order, so if sequence 486 never validates, sequences 487-489 can never validate either.

The workload immediately marks these doomed transactions as `CASCADE_EXPIRED` rather than waiting for them to individually time out. This is an internal bookkeeping state, not a rippled engine result.

**Example:** Account has sequences 486-489 in-flight. Sequence 486 expires.
```
EXPIRED:          seq 486 (timed out)
CASCADE_EXPIRED:  seq 487 (doomed by 486)
CASCADE_EXPIRED:  seq 488 (doomed by 486)
CASCADE_EXPIRED:  seq 489 (doomed by 486)
SEQ RESET:        next_seq = 486 (start over)
```

The workload then resumes allocating from sequence 486.

`CASCADE_EXPIRED` transactions are filtered out of the dashboard's failure tables — they're a consequence, not a root cause.

### What is "release_seq"?

When a transaction gets a `tel*` error (local errors like `telCAN_NOT_QUEUE`), it was never sent to the network. The allocated sequence number can be safely given back:

```
release_seq(account, 487)  →  next_seq goes from 488 back to 487
```

This only works for the **most recently allocated** sequence. If there's a gap (e.g., 487 was allocated but 488 was already allocated too), the release is skipped to avoid creating holes in the sequence space.

## Startup

### Why does "Loading state from database" take so long?

When the workload restarts with an existing `state.db`, it reconstitutes every wallet from its stored seed via `Wallet.from_seed()`. This involves cryptographic key derivation (secp256k1) for each account. At ~3,000 accounts this takes ~38 seconds; at 10,000+ it will be significantly worse.

This is a known scaling issue. The wallet objects could be cached with pre-derived keys in SQLite, or key derivation could be deferred to first use. For now, expect a slow startup proportional to account count after long runs.

If you don't need to preserve accounts from a previous run, delete `state.db` before starting — the genesis load path is much faster since it only loads the initial account set.

---

## Storage Architecture

### Why are there two stores? What's the difference between InMemoryStore and SQLiteStore?

The workload has two storage layers that serve different purposes:

**SQLiteStore** (`sqlite_store.py`) — the persistent database (`state.db`)

This is the source of truth. It persists across restarts and stores:
- All known wallets (address, seed, gateway/user flag, funded_ledger_index)
- Transaction records (hash, state, engine_result, validated_ledger, etc.)
- Validation records (hash, ledger_index, source)
- Account balances (XRP, IOU, MPToken)
- Currency metadata

When the workload restarts, it checks `state.db` first. If it has state, the workload resumes from it without re-provisioning accounts. This is the "SQLite hot-reload" path.

**InMemoryStore** (`workload_core.py:InMemoryStore`) — runtime metrics and counters

This is a lightweight in-process dict that provides fast aggregate stats for the dashboard:
- `count_by_state` — how many txns in each state (SUBMITTED, VALIDATED, etc.)
- `validated_by_source` — how many validations came via WS vs polling
- `validations` deque — recent validation records (capped at 5000)
- `submission_results` — counts per engine result code

These metrics are computed from the SQLiteStore's data on startup but maintained in memory during runtime for fast reads. The dashboard polls these every second — hitting SQLite that often would be wasteful.

**How they interact:**

```
Transaction submitted
    ├── SQLiteStore.mark(tx_hash, state=SUBMITTED, ...)     # persisted
    └── InMemoryStore.mark(tx_hash, state=SUBMITTED, ...)   # fast counters

Transaction validated
    ├── SQLiteStore.mark(tx_hash, state=VALIDATED, ...)     # persisted
    └── InMemoryStore.mark(tx_hash, state=VALIDATED, ...)   # updates counters
```

Both `mark()` calls happen in `Workload.record_submitted()`, `record_validated()`, etc. The `Store` Protocol (defined in `workload_core.py`) is the shared interface both implement.

**The `pending` dict** (`Workload.pending`) is a third in-memory structure — a dict of `PendingTx` objects for all in-flight transactions. This is what `snapshot_pending()`, `snapshot_failed()`, and `get_pending_txn_counts_by_account()` read from. It's not a "store" — it's the live working set.

### Which store should I look at when debugging?

| Question | Where to look |
|---|---|
| What happened to a specific transaction? | `state.db` (SQLiteStore) or `/state/failed/{error_code}` API |
| How many txns validated via WS vs polling? | InMemoryStore (`/state/summary` → `validated_by_source`) |
| Why did a transaction fail? | `pending` dict via `/state/failed` endpoint |
| Do my accounts survive a restart? | `state.db` — check with `sqlite3 state.db "SELECT count(*) FROM wallets"` |
| What's the current submission rate? | InMemoryStore via `/state/summary` |
