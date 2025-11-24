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
