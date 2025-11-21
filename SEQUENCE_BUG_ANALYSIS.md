# Sequence Number Synchronization Bug - Critical Fix Needed

**Date**: 2025-11-21
**Status**: BLOCKING - 187 transactions stuck with terPRE_SEQ errors

## The Problem

Transactions are getting `terPRE_SEQ` errors (sequence too high) because our cached sequence numbers diverge from on-chain reality when transactions fail to validate.

### Current State (from /state/summary):
```json
{
  "total_tracked": 878,
  "by_state": {
    "VALIDATED": 553,
    "REJECTED": 5,
    "EXPIRED": 160,
    "SUBMITTED": 160
  }
}
```

### Pending Transactions (from /state/pending):
- 187 transactions with `terPRE_SEQ` errors
- 2 transactions with no error (newly submitted)
- **Nothing is making it on-ledger**

### Fee State (from /state/fees):
```json
{
  "expected_ledger_size": 124,
  "current_queue_size": 0,
  "queue_utilization": "0/2480",
  "minimum_fee": 10,
  "base_fee": 10
}
```
Queue is empty, fees are normal, but transactions aren't validating.

## Root Cause

**File**: `workload/src/workload/workload_core.py:490-495`

```python
async def alloc_seq(self, addr: str) -> int:
    rec = self._record_for(addr)
    if rec.next_seq is None:
        ai = await self._rpc(AccountInfo(account=addr, ...))
        rec.next_seq = ai.result["account_data"]["Sequence"]

    async with rec.lock:
        s = rec.next_seq
        rec.next_seq += 1
        return s
```

**The Issue**:
1. We query AccountInfo ONCE (when `next_seq is None`)
2. We allocate sequences optimistically: 10, 11, 12, 13...
3. If transaction with seq 10 fails to validate (fee error, queue full, etc.)
4. On-chain sequence stays at 10
5. Our cached `next_seq` is now 14
6. Next transaction uses seq 14 → `terPRE_SEQ` (expected 10, got 14)
7. ALL subsequent transactions for that account fail with terPRE_SEQ

**The lock prevents concurrent allocation races, but does NOT prevent cache divergence from on-chain reality.**

## Why This Happens

### Scenario 1: Fee escalation during init
- Submit 33 TrustSets in batch
- First 20 succeed, queue fills up
- Last 13 get `telINSUF_FEE_P` (insufficient fee)
- Sequences for those 13 are "consumed" in our cache but not on-chain
- Account now out of sync by 13 sequences

### Scenario 2: Parallel submission failures
- Build 71 transactions with allocated sequences
- Submit in parallel via TaskGroup
- Some fail immediately with `tel*` errors (local errors)
- `release_seq()` tries to release them, but can't if there's a gap
- Account diverges from on-chain

## Attempted Fixes (Didn't Work)

### 1. `release_seq()` method (workload_core.py:495-509)
- Tries to rollback `next_seq` when we get `tel*` errors
- Only works if the failed sequence was the MOST RECENTLY allocated
- If seq 10 fails but we've already allocated 11, 12, 13, we can't release 10
- **Result**: Minimal impact, sequences still diverge

### 2. Dynamic fee adjustment (workload_core.py:510-551)
- Query `fee` command and pay `minimum_fee` instead of base fee
- Prevents `telINSUF_FEE_P` errors
- **Result**: Helps, but doesn't fix the fundamental sequence tracking issue

## The Solution (To Implement Tomorrow)

### Option A: Global Re-sync on terPRE_SEQ (RECOMMENDED)

When ANY account gets `terPRE_SEQ`:
1. **Pause workload immediately** (set a flag)
2. **Re-sync ALL accounts**:
   ```python
   from xrpl.asyncio.account import get_next_valid_seq_number

   async def resync_all_accounts(self):
       log.warning("Detected terPRE_SEQ - re-syncing all account sequences")
       for addr in self.wallets.keys():
           rec = self._record_for(addr)
           async with rec.lock:
               # Query on-chain sequence
               on_chain_seq = await get_next_valid_seq_number(addr, self.client)
               rec.next_seq = on_chain_seq
               log.info(f"Re-synced {addr[:8]}: seq={on_chain_seq}")
   ```
3. **Resume workload** after re-sync completes
4. **Retry failed terPRE_SEQ transactions** with correct sequences

**Why this works**:
- Treats terPRE_SEQ as "network sanity lost" signal
- Re-establishes ground truth from blockchain
- Simple, robust, no per-account tracking needed

**Implementation Steps**:
1. Add `workload_paused` flag to Workload class
2. Modify `submit_pending()` to detect terPRE_SEQ and call `resync_all_accounts()`
3. Modify `continuous_workload()` to check `workload_paused` flag
4. After re-sync, mark terPRE_SEQ transactions as RETRYABLE and retry

### Option B: Validate on Every Allocation (TOO SLOW)
```python
async def alloc_seq(self, addr: str) -> int:
    rec = self._record_for(addr)
    async with rec.lock:
        # Query on-chain EVERY TIME
        on_chain_seq = await get_next_valid_seq_number(addr, self.client)
        rec.next_seq = on_chain_seq + 1
        return on_chain_seq
```
**Rejected**: Too many network calls, kills throughput

### Option C: Tickets (NOT ALLOWED)
- Use XRPL Tickets feature (sequences you can use in any order)
- User explicitly said "We cannot rely on Tickets"
- **Rejected**

## Code Locations

### Files to Modify:
1. **workload_core.py:490-509** - `alloc_seq()` and `release_seq()`
2. **workload_core.py:780-850** - `submit_pending()` - detect terPRE_SEQ
3. **app.py:908-1000** - `continuous_workload()` - check paused flag
4. **workload_core.py** - Add new methods:
   - `resync_all_accounts()`
   - `pause_workload()` / `resume_workload()`

### Key xrpl-py Method:
```python
from xrpl.asyncio.account import get_next_valid_seq_number

# Returns the next valid sequence number from on-chain AccountInfo
seq = await get_next_valid_seq_number(address, client, ledger_index="current")
```

## Testing Plan

1. **Stop current workload** (187 stuck transactions)
2. **Implement re-sync logic**
3. **Restart network** (fresh state)
4. **Test scenario**:
   - Start workload
   - Manually trigger sequence divergence (submit invalid txn)
   - Verify re-sync triggers and fixes it
   - Verify workload resumes normally
5. **Monitor for terPRE_SEQ errors** - should be ZERO after re-sync

## Success Criteria

- ✅ No terPRE_SEQ errors after re-sync
- ✅ Transactions validate consistently
- ✅ Ledgers fill to expected_ledger_size
- ✅ SUBMITTED count stays low (< 50)
- ✅ VALIDATED count grows steadily

## Related Files

- `SESSION_STATE.md` - Session history
- `reference/FeeEscalation.md` - Queue and fee mechanics
- `workload/src/workload/constants.py` - Transaction states
- `workload/src/workload/fee_info.py` - FeeInfo dataclass

## Notes

- The lock in `alloc_seq()` IS working correctly - it prevents race conditions
- The problem is NOT the lock, it's the **lack of re-sync when on-chain diverges**
- Dynamic fees help but don't solve the core issue
- This is the #1 blocker for continuous workload operation
