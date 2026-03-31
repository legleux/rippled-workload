# Sequence Management & Submission Loop Fix

**Created:** 2026-03-31
**Status:** Pending review & execution

## Context

The workload has a persistently high failure rate (tefPAST_SEQ, terPRE_SEQ) caused by sequence management bugs — not by missing features or complexity gaps. The round-robin / multi-group ideas were a wrong turn; the real issues are:

1. The self-healing path (`expire_past_lls`) frees accounts without fixing their stale sequences
2. Stale pre-signed transactions are never detected before submission
3. `max_pending_per_account > 1` creates cascading failures by design
4. The clean/partial account pool split adds complexity for a mode (max_pending>1) that doesn't work

**Goals:** Continuously fill ledgers. Give user control over load (TPS knob). Low failure rate. Simple.

---

## Bug Analysis

### Bug 1 (Critical): `expire_past_lls()` doesn't reset sequences
**File:** `workload_core.py:967-991`

When all accounts are blocked (`consecutive_empty >= 3`), `expire_past_lls` marks pending txns as EXPIRED but never resets `AccountRecord.next_seq`. So freed accounts immediately get tefPAST_SEQ on next submission — then cascade_expire fires N RPC calls to fix it. 100% failure rate on the recovery iteration.

Compare with `record_expired()` (line 939) which correctly calls `_cascade_expire_account(..., fetch_seq_from_ledger=True)`.

### Bug 2 (Medium): `account_generation` captured but never checked
**File:** `workload_core.py:1088,1134`

`build_sign_and_track()` captures generation at line 1088, stores it in PendingTx at line 1134, but `submit_pending()` never checks it. If a cascade-expire bumps generation between build and submit (possible during parallel TaskGroup), the stale txn is submitted anyway → tefPAST_SEQ.

### Bug 3 (Design): `max_pending_per_account > 1` is broken by design

With multiple in-flight txns per account, if seq N fails for ANY reason, seqs N+1..N+k are all doomed. The cascade handles recovery but creates thundering-herd failures. With 1000 accounts at max_pending=1, theoretical throughput is ~250-333 TPS (1000 / 3-4s ledger close) — more than a single rippled node can process. The knob adds complexity for negative value.

### Bug 4 (Minor): `record_validated()` updates `next_seq` without lock
**File:** `workload_core.py:828-832`

Safe today (asyncio single-threaded, no await between read/write), but fragile if code is ever refactored to add an await in between.

---

## Implementation Plan

### Phase 1: Fix `expire_past_lls` — reset sequences on forced expiry

**Why first:** Highest-impact bug. Every self-heal cycle currently guarantees a full round of tefPAST_SEQ failures.

**File:** `workload_core.py`

Change `expire_past_lls` (line 967):
- After marking txns EXPIRED, collect affected account addresses
- For each affected account: set `rec.next_seq = None` and `rec.generation += 1`
- Setting `next_seq = None` forces the next `alloc_seq()` call to fetch from the validated ledger (existing cold-start path, line 600-606)
- The generation bump invalidates any pre-signed txns built before the reset (needed for Phase 2)
- Method stays synchronous — no RPC cost. The RPC cost is deferred to `alloc_seq()` on next use, naturally batched by the TaskGroup.

**File:** `workload_runner.py`

No change needed — `expire_past_lls` is already called synchronously and stays synchronous. After it resets sequences to None, the next loop iteration's `_alloc_and_sign` TaskGroup will do parallel `alloc_seq` calls that each fetch from ledger (because `next_seq is None`). This naturally batches the RPC cost.

### Phase 2: Add generation guard before submission

**Why:** Prevents wasted submissions of stale txns. Cheap check, no RPC.

**File:** `workload_core.py`

In `submit_pending()` (line 1140), add after the terminal-state guard:
```python
# Generation guard: skip txns whose sequence was invalidated by a cascade-expire
if p.account and p.account_generation != self._record_for(p.account).generation:
    p.state = C.TxState.EXPIRED
    p.engine_result_first = "STALE_GENERATION"
    self._total_expired += 1
    self.pending.pop(p.tx_hash, None)
    return {"engine_result": "STALE_GENERATION"}
```

This is safe because the generation read is synchronous (no await between read and comparison in asyncio).

### Phase 3: Remove `max_pending_per_account` knob, simplify pools

**Why:** Removes the broken mode and the complexity it requires.

**Files to change:**

1. **`workload_core.py:205`** — Hardcode `self.max_pending_per_account = 1`, remove config read
2. **`workload_runner.py:108-114`** — Simplify free account discovery:
   ```python
   # With max_pending=1, every free account has 0 pending (i.e. is "clean")
   free_accounts = [addr for addr in wl.wallets if pending_counts.get(addr, 0) == 0]
   ```
   Remove `clean_accounts`, `partial_accounts` variables. Pass `free_accounts` as both `free_accounts` and `clean_accounts` args to `compose_submission_set`.
3. **`workload_runner.py:136-138`** — Same simplification in the self-heal recovery block
4. **`routers/workload.py`** — Change POST `/workload/max-pending` to return 400 ("locked to 1"). Keep GET for observability.
5. **`routers/state_pages.py`** — Remove max-pending slider from dashboard HTML
6. **`config.toml`** — Remove or comment out `max_pending_per_account` key

### Phase 4: Lock consistency in `record_validated`

**Why:** Defensive correctness. Prevents future bugs if code adds awaits nearby.

**File:** `workload_core.py:828-832`

Wrap `next_seq` update in `async with rec_acct.lock:`.

### Phase 5: Fee strategy — cache-and-react

**Why:** Reduce telINSUF_FEE_P churn without over-engineering fee tracking.

**⚠️ NOTE: This phase needs re-review before execution.** The approach below was drafted late and needs fresh-eyes validation. Key question: should the workload _try_ to pay escalated fees, or just accept base-fee and let rippled queue/reject?

**Candidate approach (cache + react):**
- Cache the fee we use for submissions (already done: `_cached_fee`)
- On `telINSUF_FEE_P` response, decide per-txn:
  - If we've decided this txn _should_ pay higher fees to get into the open ledger: resubmit with higher fee (using the fee info from the rejection or from `latest_server_status_computed.open_ledger_fee_multiplier`)
  - If we've decided txns should only pay base fee: accept the rejection, release the sequence, and wait for the fee to drop naturally
- This avoids the current pattern of: telINSUF_FEE_P → invalidate cache → re-fetch → get minimum_fee again → telINSUF_FEE_P again (loop)

**Alternative approach (already partially implemented):**
- The WS `serverStatus` handler already computes `open_ledger_fee_multiplier` and `queue_fee_multiplier`
- Could compute `effective_fee = base * max(1, queue_multiplier)` to always meet queue entry
- Or `effective_fee = base * max(1, open_ledger_multiplier)` to always get into open ledger
- Tradeoff: higher fees drain accounts faster but reduce rejection noise

**Decision needed:** What's the workload's fee philosophy? "Always get in ASAP" (pay escalated) vs "only pay base, accept queuing/rejection" (patient). This affects both throughput AND account balance longevity. Re-evaluate with fresh eyes.

---

## Verification

1. **Unit check:** After Phase 1, add a log line in `alloc_seq()` when `next_seq is None` that says "cold fetch after expire_past_lls". Run the workload and confirm these appear INSTEAD of tefPAST_SEQ storms after self-heal.
2. **Integration:** Run the workload against a 4-node testnet, watch the dashboard:
   - Failure rate should drop to near-zero for valid txns (only intentional INVALID should reject)
   - Self-heal cycles should recover in 1 iteration, not 2+
   - No tefPAST_SEQ from valid-intent txns
3. **Fee escalation:** After Phase 5, manually spike load, confirm fee handling matches chosen strategy
4. **Dashboard:** Confirm max-pending slider is gone, rate controls still work

## Critical Files
- `workload/src/workload/workload_core.py` — Phases 1, 2, 4, 5
- `workload/src/workload/workload_runner.py` — Phases 1, 3, 5
- `workload/src/workload/routers/workload.py` — Phase 3
- `workload/src/workload/routers/state_pages.py` — Phase 3
- `workload/src/workload/config.toml` — Phase 3

## Reference Material Read
- `docs/FeeEscalation.md` — Fee escalation formula, TxQ sizing, queue limits
- `docs/Finality_of_Results.md` — Result code finality rules
- `docs/LocalTxs.cpp` — How rippled re-applies local txns after consensus
- `docs/error_codes.md` — ter/tel/tem/tef/tec taxonomy
- `docs/NetworkOPs.cpp` — Transaction batching, sync/async submission
- `docs/Transactor.cpp` — Preflight validation (NetworkID, flags, fees)
- `docs/TxQ.cpp` — Queue internals: per-account limit (10), fee escalation formula, canBeHeld checks
