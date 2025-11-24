# Workload Development Progress

## Session 2025-11-23 (continued): Assessment Fixes

### Completed from current_assessment.md

1. **Renamed database** `workload_state.db` → `state.db`
   - Updated sqlite_store.py, app.py, .gitignore

2. **Updated CLAUDE.md** with network management commands
   - Added `dcom down && dcom up -d` for restart
   - Added `rm state.db` for clearing state

3. **Capped batch size to 200** (app.py:1017-1019)
   - Prevents unbounded ledger growth attempts

4. **Changed tel* log to DEBUG level** (workload_core.py:851)
   - No longer looks like an error during init
   - Changed message from "Local error (tel*)" to "tel* from API node"

5. **Show submission results in dashboard** (sqlite_store.py:375-383, app.py:551-554, 763)
   - Added `submission_results` to snapshot_stats() - counts by engine_result_first
   - Added "Submission Results" table to dashboard (terPRE_SEQ, tesSUCCESS, etc.)
   - Fixed json_extract query for meta_txn_result

6. **Sped up dashboard refresh** from 3s to 1s (app.py:561)

7. **Fixed abbreviated account_id in logs** (workload_core.py:506, 508, 1340)
   - Now shows full address instead of `[:8]...`

8. **Associated gateway names from config** (workload_core.py:319, 390-392, 1316-1326, 1339-1340)
   - Added `gateway_names: dict[str, str]` mapping
   - Added `get_account_display_name()` helper
   - Logs now show gateway name with address

9. **Re-enabled state loading** (app.py:220-222, workload_core.py:398-464)
   - Allows hot-reload to skip re-creating accounts/TrustSets
   - Clears pending transactions on load (stale from previous session)
   - Restores gateway name associations
   - Sequence numbers are lazy-loaded from network on first use

10. **Improved rejection logging** (workload_core.py:886)
    - Changed from DEBUG to WARNING level
    - Shows full context: error, txn type, account, sequence, hash

11. **Log ledger index at init completion** (app.py:240, 251)
    - Shows which ledger init finished at for debugging timing

12. **Simplified batch logging** (app.py:1033-1034)
    - Combined into single line with @ ledger notation

### Still TODO

- [ ] Investigate 2-ledger gap between TrustSet batches during init
- [ ] Add dashboard tab for txns by state
- [ ] Re-enable other transaction types after XRP payments stable
- [ ] TRY LATER: Option B - continuous submission without waiting for next ledger
- [ ] tecPATH_DRY during init (TrustSet failing causes token payment to fail)

13. **Re-enabled all transaction types** (app.py:1036-1081)
    - Continuous workload now uses `generate_txn()` for random transaction types
    - Excluded: AMMCreate (expensive), MPTokenAuthorize/Set/Destroy (need state)
    - Included: Payment, TrustSet, AccountSet, NFTokenMint, NFTokenBurn, NFTokenCreateOffer,
      NFTokenCancelOffer, NFTokenAcceptOffer, OfferCreate, OfferCancel, MPTokenIssuanceCreate, Batch
    - Uses retry loop to handle excluded types and per-account limits

14. **Added Batch fee calculation** (workload_core.py:802-810)
    - Batch fee = 2 * owner_reserve + base_fee * inner_txn_count
    - owner_reserve = 2,000,000 drops (2 XRP)
    - Also added AMMCreate fee calculation (owner_reserve)

15. **Added 80% Payment bias + disabled Batch** (app.py:1049, 1062-1068)
    - 80% simple XRP Payments for reliable throughput
    - 20% other transaction types (TrustSet, AccountSet, NFToken*, Offer*, MPTokenIssuanceCreate)
    - **Batch disabled** - allocates N+1 sequences which all burn on expiry, causing cascading terPRE_SEQ
    - Checked stash@{1} (lots_o_changes) - no Batch sequence fix there, was TrustSet init batching

### Notes from run with all txn types (ledger 384)

- 17.7% success rate - BAD
- 4243 EXPIRED + terPRE_SEQ errors
- Root cause: Batch transactions allocating multiple sequences that burn on expiry
- Fix: Disabled Batch, added 80% Payment bias

### Notes from earlier run (ledger 289, ~898s)

- Successfully hitting 200 txn/ledger max with XRP-only payments
- Only 8 tefPAST_SEQ rejections (acceptable edge case)

---

## Session 2025-11-23: tel* Handling, XRP-Only Payments, Queue Utilization

### Summary

Fixed critical issues with transaction error handling and throughput:

1. **tel* Error Handling Fix** (workload_core.py:1265-1290)
   - Problem: `telCAN_NOT_QUEUE` errors caused sequence burn and tracking loss
   - Discovery: In multi-node network (1 API + 5 validators), `tel*` from API doesn't mean rejection - txn propagates to validators
   - Fix: Changed `tel*` handling to keep tracking as SUBMITTED instead of releasing sequence
   - Result: Transactions now validate via validators even when API rejects

2. **XRP-Only Continuous Workload** (app.py:843-970)
   - Problem: `telINSUF_FEE_P` errors on complex txns (Batch, AMMCreate)
   - Fix: Constrained continuous workload to simple XRP payments only
   - Result: No more fee-related rejections

3. **10 Pending Per Account** (app.py:875-920)
   - Problem: 1-pending-per-account bottleneck limited throughput to ~20 txns/ledger
   - Fix: Allow up to 10 pending per account (MAX_PENDING_PER_ACCOUNT = 10)
   - Result: Can now submit up to 410 txns in-flight vs 41

4. **Account Pool Growth** (app.py:858-870)
   - Increased create_account probability from 10% to 50%
   - Users grew from 16 to 41+ during testing

### Key Files Modified
- `workload/src/workload/workload_core.py`: tel* handling
- `workload/src/workload/app.py`: continuous_workload rewrite
- `workload/src/workload/txn_factory/builder.py`: create_xrp_payment helper

### Verified Working
- Init phase: 538 transactions validated (4 gateways + 16 users + 256 TrustSets + 256 token distributions + 6 other)
- Continuous workload: 1174+ validations with no errors or expired txns

---

## Session 2025-11-24: Dynamic Batch Sizing, Phase 4 Interleaving, Batch Sequence Handling

### Summary

Fixed initialization throughput issues and implemented proper Batch transaction sequence management:

1. **Dynamic Batch Sizing in Init Phases** (workload_core.py:1690-1779)
   - Problem: Phase 4 used initial expected_ledger_size (33) for all batches, never growing
   - Fix: Changed `_init_batch` to support dynamic sizing via `batch_size: int | None`
   - None = dynamic mode, refreshes `expected_ledger_size + 1` each iteration
   - Result: Batches grow with ledger (33 → 64 → 128...) to push size without fee escalation

2. **Phase 4 Account Interleaving** (workload_core.py:1976-1997)
   - Problem: User-first loop order hammered same account with many txns per batch
   - Fix: Changed to currency-first loop order: `for currency in currencies: for user in users`
   - Result: Each batch spreads across different accounts, better load distribution

3. **M/N Progress Logging** (workload_core.py:1750-1751)
   - Added batch progress: `"Submitted batch 9 (33 txns) 297/1024"`
   - Shows batch number, size, and cumulative progress

4. **Batch Transaction Sequence Handling** (workload_core.py:808-820, 1066-1077, 889-891)
   - Problem: Batch pre-allocates 1 (outer) + K (inner) sequences at build time
   - Actual consumption is 1 + N (successful inner) where N ≤ K
   - Can't predict N until validation
   - Fix: After Batch validates/rejects/expires, sync sequence from ledger via AccountInfo
   - Added sync in `record_validated` (after validation)
   - Added sync in `submit_pending` (after tem/tef rejection)
   - Added logging in `record_expired` (cascade_expire handles sync)

5. **Init Phase Timing & Summary** (workload_core.py:1777-1780, 2087-2106)
   - Added per-phase tracking: time_sec, ledgers, validated/total counts
   - Final summary table shows all phases with metrics

6. **Enabled Batch Transactions** (config.toml:49-52)
   - Re-enabled Batch (removed from disabled list)
   - Now safe with sequence sync after completion

### Technical Details

**Batch Transaction Mechanics:**
- Fee = 2 × owner_reserve + (inner_count × base_fee)
- Sequence consumption = 1 (outer) + N (successful inner), where 0 ≤ N ≤ K
- Must pre-allocate all K+1 sequences at build time
- Sync actual consumption from ledger after terminal state using AccountInfo with ledger_index="current"

**Dynamic Batching:**
- Phase 4 & 5 now use `await self._init_batch(pairs, label)` with no batch_size parameter
- Each batch iteration calls `await self._expected_ledger_size()` to get fresh limit
- Batch size = expected_ledger_size + 1 (just over limit to push growth without triggering escalation)

### Key Files Modified
- `workload/src/workload/workload_core.py`: _init_batch, Phase 4/5, record_validated, submit_pending, record_expired
- `workload/src/workload/config.toml`: Enabled Batch

### Objectives

This work addresses issues from latest_workload_run_info.md:
- Mitigate tefPAST_SEQ (148) and terPRE_SEQ (299) sequence errors
- Improve transaction distribution across ledgers more consistently
- Fix Batch sequence handling to prevent cascade sequence burn on expiry
- Increase validated rate from 81.5% toward 90%+ target

### Status

**Completed:**
- Dynamic batch sizing implemented and tested in code
- Phase 4 interleaving from user-first to currency-first
- Batch sequence sync after all terminal states
- M/N progress logging
- Per-phase timing and summary table

**Pending:**
- Full workload run to verify dynamic batching grows correctly
- Monitor tefPAST_SEQ/terPRE_SEQ counts with new Batch handling
- Verify ledgers consistently hit expected_ledger_size + 1
- Confirm validated rate improvement

---

## Testing Commands

```bash
# Restart network
dcom down && dcom up -d

# Remove state db
rm state.db

# Full reset (network + state)
dcom down && rm state.db && dcom up -d

# Check state
curl -s http://localhost:8000/state/summary | jq

# View dashboard
open http://localhost:8000/state/dashboard
```
