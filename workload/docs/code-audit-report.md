# Code Audit Report: Workload Subsystem

**Date**: 2026-02-26
**Scope**: `app.py` (1,969 lines), `workload_core.py` (2,693 lines), `txn_factory/builder.py` (988 lines)
**Total**: ~5,650 lines across 3 files

---

## Executive Summary

The workload is functional but has accumulated significant technical debt across its three main files. The core issues are:

1. **God class/function problem**: `Workload` has 53 methods spanning 2,385 lines. `app.py` embeds ~700 lines of HTML/CSS/JS. `continuous_workload()` is a 120-line orchestration function in the web layer.
2. **Dead code**: 9 dead methods in `workload_core.py`, 14 dead imports in `app.py`, 5 dead builder wrappers and 5 effectively dead builders in `builder.py`.
3. **Scattered state machine**: Transaction lifecycle transitions happen in 10+ places across 3 files with no central state machine.
4. **Broken builders**: MPToken Set/Authorize/Destroy pick random accounts instead of the issuer. NFT secondary market builders always raise at runtime. Batch inner types ignore disabled_types.
5. **Store confusion**: Three parallel representations of transaction state (`pending` dict, InMemoryStore, SQLiteStore) with inconsistent APIs.

---

## Action Items by Priority

### P0 — Critical (bugs that produce wrong results or crashes)

| # | File | Issue | Lines |
|---|------|-------|-------|
| 1 | builder.py | **MPToken Set/Authorize/Destroy use random account instead of issuer** — always produces `tecNO_PERMISSION` | 523, 535, 547 |
| 2 | builder.py | **`from random import random` inside 4 builders bypasses SystemRandom** — breaks Antithesis deterministic replay | 211, 283, 401, 559 |
| 3 | builder.py | **Batch inner type list ignores `disabled_types`** — generates disabled types inside Batch txns | 570 |
| 4 | builder.py | **Negative weight crash when config percentages sum > 1.0** — `random.choices` raises `ValueError` | 884-891 |
| 5 | workload_core.py | **Sequence leak in `build_sign_and_track`** — if `_open_ledger_fee()` raises after `alloc_seq()`, the sequence is never released | 1032-1033 |
| 6 | workload_core.py | **`check_finality` swallows all exceptions** — logs error then falls through to expired-ledger check, may incorrectly expire transactions | 1241-1243 |
| 7 | app.py | **Race condition in workload start/stop** — two concurrent POST `/workload/start` can create two orphaned tasks | 1631-1667 |
| 8 | app.py | **`wl.ctx.wallets` mutation is shared state** — concurrent API calls see filtered wallet list during continuous_workload | 1575 |
| 9 | app.py | **`network_reset` deletes `state.db` while SQLiteStore connection is open** — orphaned inode, data integrity issue | 1861-1864 |

### P1 — High (dead code, broken abstractions, major smells)

| # | File | Issue | Lines |
|---|------|-------|-------|
| 10 | workload_core.py | **God class: Workload has 53 methods** — should extract SequenceManager, TransactionStateMachine, InitOrchestrator, AMMRegistry, BalanceTracker | 247-2693 |
| 11 | workload_core.py | **9 dead methods** — `debug_last_tx`, `_ensure_funded`, `_acctset_flags`, `_apply_gateway_flags`, `bootstrap_gateway`, `_establish_trust_lines`, `_distribute_initial_tokens`, `_update_account_balances`, `submit_signed_tx_blobs`, `log_validation`, `wait_for_validation` | various |
| 12 | workload_core.py | **`InMemoryStore` does not satisfy `Store` protocol** — `get()` returns `dict`, not `PendingTx`; missing `upsert`, `all` methods | 85-200 |
| 13 | workload_core.py | **Double `log` definition + `logging.basicConfig` in non-main module** — hijacks root logger | 43-48 |
| 14 | workload_core.py | **`_fee_cache` and `_fee_lock` initialized but never used** — caching infrastructure abandoned | 266-267 |
| 15 | app.py | **Duplicate import block** (lines 1-27 duplicate lines 30-51) | 1-51 |
| 16 | app.py | **~700 lines of embedded HTML/CSS/JS** — should be Jinja2 templates (`Jinja2Templates` is imported but unused) | 558-1247 |
| 17 | app.py | **4 mutable module-level globals** (`workload_running`, `workload_stop_event`, `workload_task`, `workload_stats`) — should be a class on `app.state` | 1502-1505 |
| 18 | app.py | **`debug=True` on production FastAPI app** — leaks tracebacks in HTTP 500s | 294 |
| 19 | app.py | **`state_dashboard` crashes if `generate_ledger` not installed** — no try/except around `from generate_ledger.config import ComposeConfig` | 543-556 |
| 20 | app.py | **`network_reset` uses synchronous `subprocess.run`** — blocks the event loop for up to 60s | 1846, 1869, 1882 |
| 21 | builder.py | **5 NFT/Offer builders are effectively dead** — `ctx.nfts` and `ctx.offers` are never populated by `workload_core.py` | 334-492 |
| 22 | builder.py | **`create_*` wrapper functions never called externally** — 7 dead exported functions | 923-981 |
| 23 | builder.py | **`TxnContext.base_fee_drops` never called by any builder** — dead required field | 63 |
| 24 | builder.py | **`_build_offer_create` degenerate case** — when only 1 currency, creates same-currency offer (rejected by rippled) | 311-313 |

### P2 — Medium (duplication, inconsistency, maintainability)

| # | File | Issue | Lines |
|---|------|-------|-------|
| 25 | workload_core.py | **`submit_pending` is 150 lines mixing submission, error handling, and state transitions** — should split into `_submit_signed_blob`, `_handle_submission_result`, per-error handlers | 1074-1224 |
| 26 | workload_core.py | **`init_participants` is 600 lines with 8 inlined phases** — should be an `InitOrchestrator` class | 1881-2480 |
| 27 | workload_core.py | **AMM pool registration logic duplicated 3 times** — `record_validated`, Phase 7, Phase 8 | 863, 2342, 2434 |
| 28 | workload_core.py | **Phase 5/6 wait loops are copy-pasted** (16 lines each) — extract `_wait_for_ledger_resolution()` | 2150, 2246 |
| 29 | workload_core.py | **`PENDING_STATES` defined 3 times** as local constants — should be module-level alongside `TERMINAL_STATE` | 999, 1013, 2488 |
| 30 | workload_core.py | **RPC call inside asyncio.Lock** in `_cascade_expire_account` — blocks other coroutines allocating sequences for the same account | 937 |
| 31 | workload_core.py | **`record_validated` is a side-effect switchboard** — 108 lines doing 6 different things (state update, store, wallet adoption, balance tracking, MPToken registration, AMM registration) | 786-893 |
| 32 | app.py | **`lifespan()` is 134 lines** — should extract `_init_store()`, `_provision_accounts()`, `_spawn_background_tasks()` | 156-289 |
| 33 | app.py | **`setup_complete()` called twice** in succession during lifespan | 262, 276 |
| 34 | app.py | **5 `asyncio.sleep` calls violating ledger-tick principle** — 3 with explicit TODO comments | 277, 286, 1561, 1596, 1622 |
| 35 | builder.py | **`_build_amm_deposit` and `_build_amm_withdraw` are ~90% duplicated** — extract shared AMM builder helper | 696-800 |
| 36 | builder.py | **6 copy-pasted prerequisite filter blocks in `generate_txn`** — should be a data-driven prerequisite map | 851-879 |
| 37 | builder.py | **`_build_amm_create` fallthrough submits duplicate pool** after max_attempts exhausted | 687-693 |
| 38 | builder.py | **`_build_accountset` produces no-op transactions** — burns fee/sequence with no state change, undocumented intent | 355-361 |

### P3 — Low (style, minor cleanup, modernization)

| # | File | Issue | Lines |
|---|------|-------|-------|
| 39 | workload_core.py | **72 log calls use f-strings** (eager formatting) vs 42 using %-style (lazy) — standardize on %-style | various |
| 40 | workload_core.py | **Emoji in production log messages** | 641, 648 |
| 41 | workload_core.py | **12 deferred imports that should be top-level** — `defaultdict`, `math`, `shuffle`, `combinations`, `Path`, `json as _json` | various |
| 42 | workload_core.py | **`num_cpus = multiprocessing.cpu_count()`** — dead import and variable | 41 |
| 43 | workload_core.py | **Magic number `max_batch = 22`** with no explanation | 1813 |
| 44 | workload_core.py | **Dead parameters**: `wait` on `create_account`, `n` on `submit_random_txn` | various |
| 45 | workload_core.py | **Redundant `self.pending[p.tx_hash] = p`** on in-place mutations (6 occurrences) | various |
| 46 | app.py | **8 unused imports** — `dataclass`, `StaticFiles`, `Jinja2Templates`, `AnyUrl`, `CORSMiddleware`, etc. | 30-43 |
| 47 | app.py | **Dead function `_dump_tasks`** — debug helper never called | 146-153 |
| 48 | app.py | **Dead model `PaymentReq`** — creates a wallet at import time as default, never used by any route | 335-338 |
| 49 | app.py | **`app.state.tg` and `app.state.ws_stop_event`** — stored but never read | 184, 186 |
| 50 | app.py | **`Path(__file__).resolve().parents[3]`** — fragile ancestor traversal in `network_reset` | 1839 |
| 51 | builder.py | **`sample_omit` defined but never called** | 188-189 |
| 52 | builder.py | **`choice_omit` duplicated from `utils.py`** | 46-50 |
| 53 | builder.py | **`token_metadata` is a single-element list** — `choice()` always returns the same dict | 176-185 |
| 54 | builder.py | **`TxnContext.tickets` field never populated or read** | 70 |
| 55 | builder.py | **`TxnContext.derive()` never called** | 152-153 |
| 56 | builder.py | **Line 60 has a comment-embedded duplicate field declaration** (cosmetic merge artifact) | 60 |
| 57 | builder.py | **`import inspect` inside hot-path `generate_txn()`** — should pre-compute `is_async` flag at module load | 841 |
| 58 | builder.py | **`deep_update` mutates in-place AND returns** — confusing API | 192-199 |

### P4 — Python 3.13+ Modernization

| # | File | Issue | Lines |
|---|------|-------|-------|
| 59 | workload_core.py | **Use `Counter` instead of manual dict counting** — 7 copy-pasted counting loops | various |
| 60 | workload_core.py | **`AccountRecord` and `ValidationRecord` should use `slots=True`** | 203, 214 |
| 61 | workload_core.py | **Use `match`/`case` for engine result dispatch** in `submit_pending` | 1074-1224 |
| 62 | app.py | **Use `asyncio.to_thread` for subprocess calls** in `network_reset` | 1846-1888 |
| 63 | app.py | **Replace mutable globals with `WorkloadController` dataclass** on `app.state` | 1502-1505 |
| 64 | builder.py | **`T = TypeVar("T")` should use PEP 695 `[T]` syntax** | 43 |
| 65 | builder.py | **`AwaitInt`/`AwaitSeq` should use `type` statement** | 53-54 |
| 66 | builder.py | **Stale string quotes on forward refs** — types are defined before the class, quotes unnecessary | 59-64 |

---

## Modularity Recommendations

### workload_core.py (2,693 lines) should become:

| Module | Extracts | Est. lines |
|--------|----------|------------|
| `sequence_manager.py` | `alloc_seq`, `release_seq`, `_record_for`, `AccountRecord`, sequence sync logic | ~150 |
| `transaction_state_machine.py` | All `record_*` methods, `_cascade_expire_account`, state transition logic, `TERMINAL_STATE`/`PENDING_STATES` | ~300 |
| `init_orchestrator.py` | `init_participants` (8 phases), `load_from_genesis`, `load_state_from_store` | ~800 |
| `amm_registry.py` | `_register_amm_pool`, `_discover_amm_pools`, `poll_dex_metrics`, `snapshot_dex_metrics` | ~200 |
| `balance_tracker.py` | `_get_balance`, `_set_balance`, `_update_balance`, `balances` dict | ~100 |
| `workload_core.py` | Workload class (submission, context management, snapshots) | ~800 |

### app.py (1,969 lines) should become:

| Module | Extracts | Est. lines |
|--------|----------|------------|
| `templates/dashboard.html` | Dashboard HTML/CSS/JS (Jinja2 template) | ~700 |
| `startup.py` | `_probe_rippled`, `wait_for_ledgers`, lifespan helpers | ~100 |
| `routes/accounts.py` | `r_accounts` router | ~100 |
| `routes/transactions.py` | `r_transaction` router | ~100 |
| `routes/state.py` | `r_state` router | ~200 |
| `routes/workload.py` | `r_workload` router, `WorkloadController`, `continuous_workload` | ~200 |
| `routes/dex.py` | `r_dex` router | ~80 |
| `app.py` | FastAPI app creation, middleware, router includes | ~100 |

### builder.py (988 lines) should become:

| Change | Description |
|--------|-------------|
| Remove dead builders | NFT secondary market (5 builders), `create_*` wrappers, `sample_omit`, `derive`, `update_transaction` |
| Extract AMM helper | Shared logic between `_build_amm_deposit` and `_build_amm_withdraw` |
| Data-driven prerequisites | Replace 6 copy-pasted filter blocks with a prerequisite map |
| Fix broken builders | MPToken Set/Authorize/Destroy need issuer tracking; OfferCreate single-currency guard |

---

## Quick Wins (can be done immediately, < 30 min each)

1. **Delete 9 dead methods** from `workload_core.py` — zero risk, immediate clarity
2. **Fix duplicate import block** in `app.py` — consolidate lines 1-51 into one clean block
3. **Add `PENDING_STATES` as module-level constant** — replace 3 local definitions
4. **Fix `from random import random`** in 4 builders — use `SystemRandom` from `randoms.py`
5. **Remove `debug=True`** from FastAPI app constructor
6. **Guard `generate_ledger` import** in `state_dashboard` with try/except
7. **Delete `PaymentReq` model** — creates wallet at import time, never used
8. **Delete `_dump_tasks`, `app.state.tg`, `app.state.ws_stop_event`** — dead code
