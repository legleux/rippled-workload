# Workload Development Session State
**Last Updated**: 2025-11-20

## Current Status
Working on rippled-workload testing framework for Antithesis. The workload generates XRPL transactions and tracks their lifecycle through consensus.

**Branch**: `huge-refactor`

**CRITICAL BUG** (2025-11-21): Sequence number synchronization issue causing 187 transactions stuck with terPRE_SEQ errors. See `SEQUENCE_BUG_ANALYSIS.md` for detailed analysis and fix plan.

## Recent Work Completed

### Session 2025-11-21: Fee Escalation and Queue Management

**Implemented**:
1. **Fee Command Integration** (workload_core.py:530-559)
   - Created `FeeInfo` dataclass (fee_info.py)
   - Added `get_fee_info()` method using xrpl Fee command
   - Replaced `_expected_ledger_size()` to use fee command instead of server_info
   - Fixed startup crash (expected_ledger_size wasn't in server_info)

2. **Dynamic Fee Adjustment** (workload_core.py:510-551)
   - `_open_ledger_fee()` now queries fee command for minimum_fee
   - Pays escalated fees when queue fills up
   - Caps at MAX_FEE_DROPS=1000 to prevent account drainage
   - Logs warnings when fees escalate above base

3. **TrustSet Round-Robin Interleaving** (workload_core.py:1138-1165)
   - Fixed telCAN_NOT_QUEUE errors during init
   - Interleaves TrustSets across accounts (max 3 per account per batch)
   - Prevents hitting 10 txn per-account queue limit
   - Clean 256/256 TrustSet validation during init

4. **Per-Account Queue Limit Enforcement** (app.py:913-948)
   - Added `get_pending_txn_counts_by_account()` method
   - Continuous workload respects max 10 pending txns per account
   - Prevents telCAN_NOT_QUEUE errors during workload

5. **Batch Size Fix** (app.py:908)
   - Changed from `max(20, ledger_size)` to `ledger_size + 1`
   - Encourages ledger growth without overwhelming queue

6. **WebSocket Server Stream** (ws.py:76, 85)
   - Added "server" stream subscription for fee monitoring
   - Added server_status handler (ws_processor.py:285-335)
   - Computes human-readable fee multipliers

7. **Dashboard Improvements** (app.py:513-745)
   - Added fee info card (min/open/base drops)
   - Added queue utilization card with progress bar
   - Added ledger utilization card with progress bar
   - Added ledger index + hostname to subtitle
   - Added Start/Stop workload buttons
   - Added `/state/fees` endpoint

8. **RPC Probing with Retry** (app.py:79-104)
   - Added retry logic for `_probe_rippled()` (max 30 attempts)
   - Gracefully handles starting workload before rippled is ready

**Issues Identified But Not Fixed**:
- **Sequence synchronization bug** - terPRE_SEQ errors accumulate when transactions fail
- See `SEQUENCE_BUG_ANALYSIS.md` for full analysis and fix plan

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

1. **Fix batch_size during init**: Change from `ledger_size * 2` to `ledger_size` to avoid queue overflow
2. **Test tel* sequence release**: Verify sequences are properly released on local errors
3. **Verify ledger growth**: Confirm 20% growth rate experimentally or in rippled source
4. **Fix account uniqueness in transaction generation**: Use random.sample() for multi-account selection

## High Priority TODOs

- **State restoration on startup**: Currently DISABLED (app.py:220) due to sequence conflicts
  - Problem: Loading wallets from DB but re-querying AccountInfo doesn't account for pending txns
  - Fix needed:
    1. Clear all pending transactions from previous session (stale)
    2. Re-query AccountInfo from network for all loaded wallets
    3. Reset next_seq in AccountRecord to match on-chain state
    4. Then proceed with loaded wallets and currencies
  - Benefit: Avoid recreating 256 TrustSets on every restart, significantly speed up startup
  - Current: We write to DB but never load from it, reinitialize from scratch every time

- **Multiple WebSocket listeners**: Connect to multiple rippled nodes, not just one
- **Verify queue limits**: Confirm "max 10 txns per account" from FeeEscalation.md
- **Optimize batch sizing**: Experiment with expected_ledger_size+1, expected_ledger_size*1.2 to maximize throughput

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
