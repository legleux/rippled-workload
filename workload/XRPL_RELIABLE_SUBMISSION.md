# XRPL Reliable Transaction Submission - Analysis

## Overview

This document compares our workload implementation against XRPL's official best practices for reliable transaction submission.

**Reference:** https://xrpl.org/docs/concepts/transactions/reliable-transaction-submission

---

## XRPL Recommended Workflow

### 1. Construction Phase
1. Find latest validated ledger index (call it "A")
2. Construct transaction with:
   - Sequence number from account
   - LastLedgerSequence = A + 4 (or small offset)
3. Sign transaction
4. **Save to persistent storage** (hash, Sequence, LastLedgerSequence, validated ledger "A")

### 2. Submission Phase
5. Submit signed transaction blob

### 3. Verification Loop
6. Wait ~4s for next validated ledger
7. Query transaction by hash
8. Decision tree:
   - **Found in validated ledger?** → Check result code → Save final outcome → Done
   - **Not found + current ledger > LastLedgerSequence?** → Expired, will never be included
   - **Not found + account Sequence > tx Sequence?** → Different tx with same sequence included
   - **Ledger gaps?** → Wait for gaps to fill or query different server
   - Otherwise → Continue monitoring

### 4. Failure Cases

**Failure Case 1 (tec codes):**
- Transaction included in ledger and cost burned, but failed to achieve intended effect
- Examples: tecUNFUNDED_PAYMENT, tecPATH_DRY, tecNO_DST_INSUF_XRP
- **Status:** Transaction IS in a validated ledger
- **Result:** Cost destroyed, sequence consumed, but operation failed

**Failure Case 2 (not included):**
- Transaction never made it into any ledger
- Causes: Fee too low, LastLedgerSequence too soon, network issues
- **Status:** Transaction will never be in a ledger
- **Result:** No cost, sequence not consumed

**Failure Case 3 (sequence collision):**
- Different transaction with same sequence number was included
- Causes: Malleability, concurrent submission systems, resubmission with modifications
- **Status:** Our transaction will never be included
- **Result:** Sequence consumed by different transaction

---

## Our Implementation - What We Do Correctly ✅

### Construction Phase
- ✅ **Get validated ledger index** (`build_sign_and_track` line 743-745)
- ✅ **Allocate sequence number** with per-account locks (line 754, `alloc_seq()`)
- ✅ **Set LastLedgerSequence** = current + HORIZON (line 746, 771)
- ✅ **Sign transaction** (line 773-776)
- ✅ **Save to persistent storage** (line 789 `record_created()`, line 565-576)
  - Stores: tx_hash, state, created_ledger, sequence, last_ledger_seq

### Submission Phase
- ✅ **Submit signed blob** (`submit_pending()` line 811)
- ✅ **Handle hash mismatch/rekey** (line 584-588 in `record_submitted()`)
  - If server returns different hash, we rekey our tracking
- ✅ **Record submission** with engine_result (line 578-593)
- ✅ **Handle terminal rejections** tem/tef codes (line 820-833)

### Verification Phase
- ✅ **Query by hash** (`check_finality()` line 866)
- ✅ **Check if validated** (line 869-879)
- ✅ **Extract meta TransactionResult** (line 871)
- ✅ **Check for expiry** (line 888-893)
- ✅ **Periodic finality checking** with concurrency limits (semaphore max=20)

### Failure Handling
- ✅ **Reset sequence on expiry** with cascading (line 713-736)
  - When txn expires, reset sequence tracking and cascade-expire dependent txns
- ✅ **Track transaction states** (CREATED → SUBMITTED → VALIDATED/REJECTED/EXPIRED)
- ✅ **Per-account locks** prevent sequence conflicts during generation

---

## Gaps & Improvements ⚠️

### 1. Sequence Collision Detection ❌ MISSING (High Priority)

**XRPL Best Practice:**
> "Check if account Sequence in latest ledger > transaction Sequence → Different transaction with same sequence was included"

**Our Gap:**
When a transaction expires, we don't check if a DIFFERENT transaction consumed that sequence number.

**Scenario:**
1. We submit tx with seq=100, hash=ABC123
2. Transaction never validates, expires
3. We reset sequence tracking and cascade-expire dependent txns
4. **But** account sequence on ledger is now 101 - meaning seq=100 WAS consumed by a different tx!

**Impact:**
- We incorrectly reset sequence tracking when sequence is already filled
- We might try to resubmit with sequence=100, causing terPRE_SEQ
- We lose track of what actually happened on-ledger

**Fix Needed:**
Before calling `record_expired()`, check account sequence on ledger. If > transaction sequence, a different tx consumed it - don't reset.

---

### 2. Ledger Gap Detection ❌ MISSING (Low Priority - Custom Test Network)

**XRPL Best Practice:**
> "If server lacks continuous history from A to LastLedgerSequence, wait for gaps to fill or query different server"

**Our Gap:**
We don't check if rippled has continuous ledger history.

**Impact:**
- On production networks, could incorrectly expire txns when node lacks history
- **Not relevant for our custom test network** where we control all nodes

**Decision:** Skip this - not needed for controlled test environment.

---

### 3. tec Code Distinction ❌ MISSING (Medium Priority - Metrics)

**XRPL Best Practice:**
> "Transaction with tec code is in a validated ledger but failed to achieve intended effect"

**Our Gap:**
We treat all validated transactions the same. Don't distinguish:
- `tesSUCCESS` - operation succeeded
- `tec*` codes - operation failed, but cost still burned (tecUNFUNDED_PAYMENT, tecPATH_DRY, etc.)

**Impact:**
- Metrics conflate success with failure
- Can't distinguish "transaction was included" from "transaction succeeded"
- Harder to diagnose why operations are failing

**Fix Needed:**
Keep state as `VALIDATED` (semantically correct - it IS in a ledger), but:
- Log tec codes prominently with warnings
- Track separate metrics: `validated_success` vs `validated_tec` vs `validated_other`
- Include `meta_result` in all validation logging

---

### 4. Grace Period ⚠️ UNDERSIZED (Low Priority - Working Fine)

**XRPL Best Practice:**
> "Wait ~4s for next validated ledger before checking"

**Our Implementation:**
- `grace=2` ledgers in `check_finality()` (~6-10 seconds)
- We check frequently via periodic_finality_check every 5 seconds

**Impact:**
- Slightly aggressive expiry checking
- Not a problem in practice since we check continuously

**Decision:** Current grace period is fine for our use case.

---

## Summary Table

| Best Practice | Status | Priority | Action |
|---------------|--------|----------|--------|
| Construction with Sequence + LastLedgerSequence | ✅ Complete | - | None |
| Sign and persist to storage | ✅ Complete | - | None |
| Submit signed blob | ✅ Complete | - | None |
| Query by hash for validation | ✅ Complete | - | None |
| Handle hash mismatch (rekey) | ✅ Complete | - | None |
| Terminal rejection handling (tem/tef) | ✅ Complete | - | None |
| Expiry detection | ✅ Complete | - | None |
| Sequence collision detection | ❌ Missing | **HIGH** | **Implement** |
| Ledger gap detection | ❌ Missing | LOW | Skip (test network) |
| Distinguish tec vs tesSUCCESS | ❌ Missing | **MEDIUM** | **Implement** |
| Grace period (4s recommendation) | ⚠️ Undersized | LOW | Keep current |

---

## Implementation Plan

### Priority 1: Distinguish tec Codes
**Goal:** Better metrics and logging for transaction outcomes

**Changes:**
1. Update `record_validated()` to log tec codes with warnings
2. Add metrics tracking:
   - `validated_success` (tesSUCCESS count)
   - `validated_tec` (tec* code count)
   - `validated_other` (other result codes)
3. Include `meta_result` in validation logging

**Impact:** Better observability, clearer success/failure metrics

---

### Priority 2: Sequence Collision Detection
**Goal:** Prevent incorrect sequence resets when different tx consumed sequence

**Changes:**
1. In `check_finality()` before marking expired:
   - Query account info for current sequence
   - If current > transaction sequence: sequence was consumed by different tx
   - Log warning and mark REJECTED (not EXPIRED)
   - Do NOT reset sequence tracking or cascade-expire
2. Add new rejection reason: "sequence consumed by different transaction"

**Impact:** Correct sequence tracking, prevent terPRE_SEQ errors, better auditability

---

## Notes

- All analysis based on workload/src/workload/workload_core.py
- HORIZON currently set to 15 ledgers (~45-60 seconds)
- RPC_TIMEOUT = 2.0s, SUBMIT_TIMEOUT = 20s
- Concurrent finality checking limited to 20 parallel RPC calls
- Custom test network eliminates need for ledger gap detection
