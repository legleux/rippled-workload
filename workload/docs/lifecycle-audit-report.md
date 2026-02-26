# Transaction Submission Lifecycle — Architecture Audit

**Date**: 2026-02-26
**Scope**: The transaction lifecycle subsystem spanning `workload_core.py`, `ws_processor.py`, `app.py`, and `sqlite_store.py`

---

## Executive Summary

The Transaction Submission Lifecycle is scattered across 5 files with 7 state transitions managed through a patchwork of direct state mutations, store interactions, and polling loops. The code has:

- **Dual pending tracking**: `PendingTx` in-memory dict + `InMemoryStore`/`SQLiteStore` flat records operate in parallel
- **Scattered state transitions**: Each transition (CREATED → SUBMITTED → VALIDATED) has multiple code paths
- **Validation paths split**: WS and polling both call `record_validated()` but with different contexts
- **Sequence management fragmented**: Allocation, release, resync, and cascade logic spread across 6 functions
- **No single transaction state machine**: Logic distributed across `build_sign_and_track()`, `submit_pending()`, `record_*()`, `check_finality()`, and `ws_processor`

---

## 1. Lifecycle Map: State Transitions by Location

### CREATED
- **Entered**: `workload_core.py:record_created()` — creates `PendingTx`, adds to `self.pending`, calls `store.update_record()`
- **Source**: Called from `build_sign_and_track()` after signing

### SUBMITTED
- **Path 1**: Normal submission → `record_submitted()`
- **Path 2**: `tel*` error → inline state set in `submit_pending()` + `store.mark()`
- Handles hash rekey if server returns different hash

### RETRYABLE
- **Entered**: `check_finality()` — if pending but not yet expired
- **Periodic**: `periodic_finality_check()` polls every 5s

### VALIDATED
- **Path 1 (WS)**: `ws_processor._handle_tx_validated()` → `workload.record_validated(src=WS)`
- **Path 2 (Polling)**: `check_finality()` → `record_validated(src=POLL)`
- **Path 3 (Init)**: `_wait_all_validated()`, `_submit_and_wait_batched()`, `_init_batch()`
- **Side effects**: Account adoption, balance updates, MPToken/AMM registration

### REJECTED
- **Path 1**: `tefPAST_SEQ` → cascade expire + sequence resync from ledger
- **Path 2**: `tem/tef` codes → terminal rejection, Batch-specific sequence sync
- **Path 3**: Engine result dispatch in `submit_pending()`

### EXPIRED
- **Path 1**: `record_expired()` via `check_finality()` when past LastLedgerSequence + grace
- **Path 2**: `terPRE_SEQ` in `submit_pending()` — treated as expiry + cascade
- **Cascade**: `_cascade_expire_account()` marks all higher-sequence txns as `CASCADE_EXPIRED`

### FAILED_NET
- **Path 1**: `asyncio.TimeoutError` in `submit_pending()`
- **Path 2**: Generic exception in `submit_pending()`

### store.mark() Call Sites (12 total)

| Location | Transition |
|----------|-----------|
| `record_created()` | → CREATED (uses `update_record`, not `mark`) |
| `record_submitted()` | → SUBMITTED |
| `submit_pending()` tel* | → SUBMITTED |
| `submit_pending()` tefPAST_SEQ | → REJECTED |
| `submit_pending()` tem/tef | → REJECTED |
| `submit_pending()` terPRE_SEQ | → EXPIRED |
| `submit_pending()` timeout | → FAILED_NET |
| `submit_pending()` exception | → FAILED_NET |
| `record_validated()` | → VALIDATED |
| `record_expired()` | → EXPIRED |
| `check_finality()` | → RETRYABLE |
| `_cascade_expire_account()` | → EXPIRED |

---

## 2. Logic Duplication

### Sequence Reset Logic (3 implementations)

1. **`_cascade_expire_account()`** — fetches from ledger or sets to `failed_seq`, with account lock
2. **`submit_pending()` after Batch rejection** — fetches from ledger inline, separate lock
3. **`record_validated()` after Batch validation** — fetches from ledger inline, yet another implementation

Same operation (sync sequence from ledger) implemented 3 times with different error handling.

### Store Interactions (Dual-Path Pattern)

Every state transition manually:
1. Mutates `p.state` on the `PendingTx` object
2. Writes `self.pending[p.tx_hash] = p` (redundant — `p` is already there by reference)
3. Calls `store.mark()` with a hand-selected set of fields

This 3-step pattern repeats 10+ times with no abstraction.

### Validation Deduplication

`InMemoryStore.mark()` and `SQLiteStore.mark()` both implement deduplication by `(tx_hash, ledger_index)` with different mechanisms (in-memory list scan vs SQL query). Same logic, two implementations.

---

## 3. Coupling Issues

### app.py Reaches into Workload Internals

`continuous_workload()` directly accesses: `wl.max_pending_per_account`, `wl.target_txns_per_ledger`, `wl.wallets`, `wl.ctx.wallets` (mutates it!), `wl.pending` via `get_pending_txn_counts_by_account()`. This is business logic in the web layer.

### submit_pending() Mixes 3 Concerns

150 lines mixing: network submission protocol, engine result classification, and state transitions with side effects. Should be:
- `_submit_signed_blob()` — just the network call
- `_handle_submission_result()` — classify engine result
- State machine methods — per-transition side effects

### record_validated() is a Side-Effect Switchboard

108 lines doing 6 unrelated things: state update, store persistence, wallet adoption, balance tracking, MPToken registration, AMM pool registration. These side effects should be in a post-validation dispatcher.

### check_finality() Does 3 Things

1. Polls RPC for finality
2. Records validation if found
3. Marks as RETRYABLE or EXPIRED

These are three separate operations compressed into one method.

---

## 4. Recommendations (Ordered by Impact)

### 1. Introduce a Transaction State Machine (HIGH)

Single class owning all state transitions. Every `record_*()` method moves here. Each transition updates `PendingTx`, calls `store.mark()`, and triggers side effects — in one place per transition type.

### 2. Extract SequenceManager (HIGH)

`alloc_seq`, `release_seq`, sync-from-ledger, `AccountRecord` — one class instead of 6 scattered functions. Single implementation of the sequence-fetch-from-ledger pattern.

### 3. Split submit_pending() into 3 Methods (HIGH)

- `_submit_signed_blob()` — network call only
- `_handle_submission_result()` — engine result dispatch
- Per-error handlers delegating to the state machine

### 4. Create ValidationDispatcher (MEDIUM)

Single entry point for WS and polling validation. Deduplicates at the application level. Runs post-validation side effects once.

### 5. Move continuous_workload into Workload (MEDIUM)

app.py becomes a thin HTTP layer. Workload owns its own orchestration loop.

### 6. Extract CascadeExpiry (MEDIUM)

One implementation handling all cascade scenarios. Clear parameters vs magic boolean flags.

### 7. Consolidate store.mark() Calls (MEDIUM)

Move all 12 `store.mark()` calls inside the state machine. Callers never construct field sets manually.

---

## 5. Store Architecture Issues

### Three Parallel Representations

1. **`self.pending: dict[str, PendingTx]`** — strongly-typed, mutated directly
2. **`InMemoryStore._records: dict[str, dict]`** — weakly-typed flat dicts
3. **`SQLiteStore`** — persisted flat dicts

No invariant that they stay in sync. After `_cleanup_terminal` removes from `pending`, data only lives in the store. But `snapshot_pending` only reads from `pending`.

### InMemoryStore Doesn't Satisfy Store Protocol

The `Store` protocol declares `get() -> PendingTx | None`. InMemoryStore's `get()` returns `dict | None`. The protocol is not enforced.

### update_record vs mark

Two overlapping store methods. `record_created` uses `update_record`; everything else uses `mark`. No clear reason for the split.

### SQLite-Specific Type Checks

`save_wallet_to_store`, `save_currencies_to_store` do `isinstance(self.store, SQLiteStore)` — breaks the protocol abstraction.
