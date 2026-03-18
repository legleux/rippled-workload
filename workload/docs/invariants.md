# Workload Invariants

Rules that must hold at all times. Violating any of these causes tefPAST_SEQ cascades,
double-spends, stalls, or silent data corruption. When adding features, check every
invariant in this list.

---

## 1. One In-Flight Transaction Per Account

```
max_pending_per_account = 1
```

An account's next sequence is only allocated after the previous transaction reaches a
terminal state (VALIDATED, REJECTED, EXPIRED). The producer checks `pending_counts`
and skips any account with `count > 0`.

**Why**: Raising this causes tefPAST_SEQ. If any in-flight txn fails without advancing
the ledger sequence (tem*/tel*/FAILED_NET), cascade-expire resets next_seq, conflicting
with txns already in rippled's queue. To raise throughput, grow the account pool.

**Enforced in**: `_txn_producer` (free_accounts filter), `alloc_seq` (per-account lock).

---

## 2. One Submission Per Account Per Batch

The consumer deduplicates the queue by account before submitting. If the producer
enqueued multiple sequential txns for the same account between ledger closes (each
individually valid at build time), only the latest is submitted. Earlier ones are expired.

**Why**: The producer runs continuously. Between ledger closes (~4s), a tx can validate
via WS, freeing the account, and the producer immediately builds the next tx. The queue
accumulates multiple txns for the same account. Submitting them all in parallel causes
tefPAST_SEQ because they compete for the same sequence slot in rippled.

**Enforced in**: `continuous_workload` consumer drain loop (`by_account` dict keeps latest per account).

---

## 3. Sequences Come From Validated Ledger

```python
AccountInfo(account=addr, ledger_index="validated")
```

Never use `"current"` for sequence allocation. The current/open ledger includes
tentative state from queued transactions, which can revert.

**Why**: `"current"` returned stale sequences when rippled's tx queue hadn't applied yet,
causing tefPAST_SEQ. Resolved 2026-03-09.

**Enforced in**: `alloc_seq` (first fetch), `_cascade_expire_account` (seq reset).

---

## 4. Generation Counter Detects Stale Pre-Signed Transactions

```
AccountRecord.generation  — incremented on every cascade-expire / sequence reset
PendingTx.account_generation — snapshot of generation at build time
```

The consumer checks `rec.generation != p.account_generation` before submitting. If a
cascade-expire happened between build and submit, the pre-signed tx has a stale sequence
and is discarded.

**Why**: The producer pre-signs transactions before the consumer submits them. If a
cascade-expire resets an account's sequence between those two points, the pre-signed tx
would use the old (now invalid) sequence.

**Enforced in**: `continuous_workload` consumer drain loop, `_cascade_expire_account`
(increments generation under lock).

---

## 5. Terminal States Are Final

```python
TERMINAL_STATE = frozenset({VALIDATED, REJECTED, EXPIRED})
```

Once a transaction reaches a terminal state, its state is never overwritten.
`record_submitted` and `submit_pending` check `p.state in TERMINAL_STATE` and return
early.

**Why**: Race conditions between WS validation and RPC polling could otherwise flip a
VALIDATED tx back to SUBMITTED.

**Enforced in**: `record_submitted`, `submit_pending`, `record_validated`.

---

## 6. FAILED_NET Is NOT Terminal

```python
PENDING_STATES = frozenset({CREATED, SUBMITTED, RETRYABLE, FAILED_NET})
```

A network timeout means the tx blob *may* have reached rippled. The account stays locked
until the tx either validates on-chain (WS/poll) or its LastLedgerSequence expires.

**Why**: Treating FAILED_NET as terminal would release the sequence, but the tx might
still be in rippled's queue. Submitting a new tx with the same sequence → tefPAST_SEQ.

**Enforced in**: `PENDING_STATES` definition in `constants.py`, `periodic_finality_check`
polls FAILED_NET txns.

---

## 7. LastLedgerSequence Horizon

```python
HORIZON = 15  # ledgers (~45-60 seconds)
```

Every transaction gets `LastLedgerSequence = created_ledger + HORIZON`. If not validated
within 15 ledgers, the tx is definitively expired — rippled will never apply it.

**Why**: Without LLS, a transaction could be applied at any future ledger, making sequence
management impossible. The horizon bounds the maximum time an account can be locked.

**Enforced in**: `build_sign_and_track` (sets LLS), `periodic_finality_check` (expires
past-LLS txns).

---

## 8. Cascade-Expire on Sequence Failure

When a transaction fails with tefPAST_SEQ or expires, ALL pending txns for that account
with higher sequences are expired, and `next_seq` is reset from the validated ledger.

**Why**: Higher-sequence txns depend on the failed one. If seq=5 fails, seq=6 and seq=7
are doomed. Without cascade, those txns would sit in pending until LLS expiry, locking
the account for ~60 seconds unnecessarily.

**Enforced in**: `_cascade_expire_account` (called from `submit_pending` tefPAST_SEQ
handler and `periodic_finality_check` expiry path).

---

## 9. Ledger Is The Tick, Not The Clock

The consumer fires once per new ledger close, not on wall-clock intervals. Submissions
are batched per-ledger. Time-based delays are only used for:
- Timeouts (network connectivity, RPC calls)
- Metrics (measuring operation duration)

**Why**: XRPL consensus operates on discrete ledger closes (~3-4 seconds). Transaction
validation, sequence numbers, and queue behaviour are all tied to ledger boundaries.
Time-based logic creates race conditions.

**Enforced in**: `continuous_workload` consumer loop (`batch_ledger == last_batch_ledger`
check), `_txn_producer` (uses `fee_info.ledger_current_index` for LLS calculation).

---

## 10. Owner Reserve Fee For Ledger Object Creation

Transactions that create new ledger objects must pay the owner reserve (2,000,000 drops)
as the fee, not the base fee.

```python
if txn_type in ("AMMCreate", "VaultCreate", "PermissionedDomainSet"):
    fee = OWNER_RESERVE_DROPS  # 2_000_000
```

**Why**: rippled rejects these transactions with `telINSUF_FEE_P` if the fee is too low.
`submit_and_wait` handles this automatically, but our manual signing path does not.

**Enforced in**: `build_sign_and_track` fee calculation.

---

## 11. Capability-Aware Type Selection

Transaction types are only eligible for selection if the required state exists:

| Requirement | Types gated |
|---|---|
| No MPToken IDs | MPTokenAuthorize, Set, Destroy |
| No NFTs | NFTokenBurn, CreateOffer |
| No offers | OfferCancel, NFTokenCancelOffer, AcceptOffer |
| No AMM pools | AMMDeposit, AMMWithdraw |
| No credentials | CredentialAccept, CredentialDelete |
| No domains | PermissionedDomainDelete |
| No vaults | VaultSet, Delete, Deposit, Withdraw, Clawback |

Per-account filters additionally check wallet ownership (vault owner for VaultSet,
LP holder for AMMWithdraw, etc.).

**Why**: Submitting a type without its prerequisites wastes a sequence and an account lock
cycle (~4-60 seconds). Builders return None as a second line of defense, but the filter
prevents the attempt entirely.

**Enforced in**: `pick_eligible_txn_type` (global + per-account filters), `generate_txn`
(legacy path), individual builders (return None).

---

## 12. Producer Must Not Crash

Every operation in the producer loop (`build_txn_dict`, `from_xrpl`, `alloc_seq`,
`build_sign_and_track`) is wrapped in try/except. An unhandled exception kills the
producer task silently, and the consumer spins on an empty queue forever (the self-stop
bug).

**Why**: The producer is a background `asyncio.Task`. Its exception is only logged via
`_log_task_exception` done-callback — the consumer has no mechanism to detect or restart
it.

**Enforced in**: `_txn_producer` (three try/except blocks: build, alloc_seq, sign).
