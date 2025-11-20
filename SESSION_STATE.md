# Workload Development Session State
**Last Updated**: 2025-11-20

## Current Status
Working on rippled-workload testing framework for Antithesis. The workload generates XRPL transactions and tracks their lifecycle through consensus.

**Branch**: `huge-refactor`

## Recent Work Completed

### 1. Heartbeat Transaction System
- **Purpose**: Submit exactly 1 transaction per ledger as a "canary" to detect network issues
- **Implementation**: Separate from normal workload metrics to avoid pollution
- **Location**: `workload/src/workload/workload_core.py:2188-2300` (`submit_heartbeat`)
- **Key Fix**: Heartbeats now bypass `build_sign_and_track()` and submit directly without tracking in pending/store
- **Bug Fixed**: Removed references to undefined variable `p` in Antithesis assertions (lines 2282-2301)

### 2. Transaction State Machine Fixes
**Problem**: `terPRE_SEQ` errors (sequence conflicts) were blocking accounts and preventing transaction submission

**Root Causes Identified**:
1. `terPRE_SEQ` transactions were marked as SUBMITTED instead of RETRYABLE
2. `check_finality()` had broken logic for RETRYABLE state transitions
3. Finality checker was skipping terPRE_SEQ transactions entirely, letting them block accounts forever
4. Batching didn't check for pending transactions across batches, causing account reuse

**Fixes Applied**:

#### Fix 1: Handle `ter*` codes at submission time
- **File**: `workload/src/workload/workload_core.py:1265-1277`
- **Change**: Added handler for `ter*` codes in `submit_pending()` to mark as RETRYABLE immediately
- **Before**: All non-terminal codes → SUBMITTED
- **After**: `ter*` → RETRYABLE, `tes*` → SUBMITTED, `tem/tef` → REJECTED

#### Fix 2: Simplified check_finality logic
- **File**: `workload/src/workload/workload_core.py:1368-1371`
- **Change**: Removed broken RETRYABLE transition logic
- **Before**: Tried to convert non-SUBMITTED to RETRYABLE (backwards)
- **After**: Keep current state (RETRYABLE stays RETRYABLE, SUBMITTED stays SUBMITTED)

#### Fix 3: Check terPRE_SEQ for expiry
- **File**: `workload/src/workload/workload_core.py:2451-2481`
- **Change**: Finality checker now checks terPRE_SEQ transactions for expiry
- **Before**: Skipped all terPRE_SEQ transactions (never expired, blocked accounts forever)
- **After**: Checks terPRE_SEQ for expiry, allowing accounts to be freed when txns expire

#### Fix 4: Improved batching to avoid account reuse
- **File**: `workload/src/workload/app.py:889-920`
- **Change**: Check for pending transactions GLOBALLY before building batch
- **New Method**: `workload_core.py:2161-2172` - `get_accounts_with_pending_txns()`
- **Before**: Only prevented duplicates within single batch
- **After**: Checks `get_accounts_with_pending_txns()` and excludes any account with CREATED/SUBMITTED/RETRYABLE txns

### 3. File Logging
- **File**: `workload/src/workload/logging_config.py`
- **Change**: Added file handler to log to `/tmp/workload.log`
- **Status**: Not working in devcontainer (file not created), logs only visible in debug console

### 4. Known Issues & TODOs

#### TODO: Account uniqueness in transaction generation
- **File**: `workload/src/workload/txn_factory/builder.py:73-77`
- **Issue**: Same account can be selected for src/dest, causing "Must not be equal to the account" errors
- **Solution**: Use `random.sample()` for multi-account selection + transaction-specific viability checks
- **Status**: Documented, not implemented (accepting errors for now)

## Architecture Overview

### Transaction States
```
CREATED → SUBMITTED → VALIDATED (success path)
       ↘ RETRYABLE → EXPIRED/VALIDATED (retry path)
       ↘ REJECTED (terminal error: tem/tef)
       ↘ FAILED_NET (network timeout)
```

### Key Components

**Workload Core** (`workload/src/workload/workload_core.py`)
- `build_sign_and_track()`: Create, sign, track transaction (state: CREATED)
- `submit_pending()`: Submit transaction via RPC (state: SUBMITTED or RETRYABLE)
- `check_finality()`: Poll for validation/expiry
- `record_validated()`: Mark as VALIDATED when confirmed
- `alloc_seq()`: Allocate sequence numbers with per-account locking

**Background Tasks** (`workload/src/workload/app.py:166-190`)
1. `ws_listener`: WebSocket listener for ledger close + tx validation events
2. `ws_processor`: Process WS events, submit heartbeat on ledger close
3. `periodic_finality_check`: Poll RPC for tx validation (backup to WS)

**Continuous Workload** (`workload/src/workload/app.py:840-980`)
- Runs in background when `/workload/start` is called
- Fetches expected ledger size from rippled
- Builds batch of transactions (one per account, no reuse across batches)
- Submits batch in parallel
- Waits for next ledger before submitting next batch

## Configuration

**Accounts** (`workload/src/workload/config.toml`)
- Gateways: 4
- Users: 16
- Total: ~20-40 (additional accounts created dynamically)

**Issue**: Not enough accounts! With 34/37 accounts having pending txns, batching stalls waiting for accounts to free up.

## Current Problems Being Monitored

1. **Low ledger throughput**: Ledgers have 0-35 txns (should be consistent ~40)
2. **Account exhaustion**: Almost all accounts have pending txns, blocking new batches
3. **terPRE_SEQ accumulation**: 20+ RETRYABLE txns waiting (should expire and free accounts with latest fix)

## Testing Workflow

1. User restarts network (docker compose down/up)
2. User starts workload app in debugger (devcontainer)
3. User watches debug console for "STARTUP COMPLETE"
4. User tells Claude to hit `/workload/start`
5. Monitor state: `curl -s http://localhost:8000/state/summary | jq`
6. Monitor heartbeat: `curl -s http://localhost:8000/state/heartbeat | jq`

## Key Endpoints

- `POST /workload/start` - Start continuous workload
- `POST /workload/stop` - Stop workload
- `GET /state/summary` - Transaction counts by state
- `GET /state/heartbeat` - Heartbeat status
- `GET /state/pending` - Pending transactions
- `GET /state/dashboard` - HTML dashboard (auto-refresh every 3s)
- `POST /transaction/random` - Submit single random transaction
- `POST /transaction/payment` - Submit specific transaction type

## Files Modified (Unstaged)
```
M prepare-workload/prepare_workload/templates/rippled.cfg.mako
M workload/src/workload/app.py
M workload/src/workload/config.toml
M workload/src/workload/constants.py
M workload/src/workload/logging_config.py
M workload/src/workload/sqlite_store.py
M workload/src/workload/txn_factory/builder.py
M workload/src/workload/workload_core.py
M workload/src/workload/ws_processor.py
```

## Next Steps

1. **Monitor latest fixes**: Check if terPRE_SEQ expiry fix frees up accounts
2. **Increase account count**: If still bottlenecked, increase users in config.toml
3. **Implement TODO**: Fix account uniqueness in transaction generation
4. **Optimize batching**: Consider not waiting for full ledger close between batches

## Quick Debug Commands

```bash
# Check state
curl -s http://localhost:8000/state/summary | jq

# Check pending by state
curl -s http://localhost:8000/state/pending | jq '[.pending[]] | group_by(.state) | map({state: .[0].state, count: length})'

# Check account exhaustion
curl -s http://localhost:8000/state/pending | jq '[.pending[]] | length'
curl -s http://localhost:8000/state/summary | jq '.users + .gateways'

# Check heartbeat health
curl -s http://localhost:8000/state/heartbeat | jq '{last: .last_heartbeat_ledger, total: .total_heartbeats, missed: .missed_count}'

# View dashboard
open http://localhost:8000/state/dashboard
```

## Important Notes

- Logs don't write to `/tmp/workload.log` in devcontainer - check debug console instead
- Don't make changes while app is running (hot reload can cause issues)
- User will say "hit it!" when ready to start workload
- Always check if fixes are working before making more changes
