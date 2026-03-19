# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **rippled-workload** repository for Antithesis testing of the XRPL (XRP Ledger) rippled node. The main component is a **FastAPI-based workload generator** that creates and submits XRPL transactions against a local testnet, tracking their lifecycle through the consensus process.

A separate **sidecar** container provides passive monitoring and Antithesis assertions.

Network setup (configs, genesis ledger, docker-compose) is handled by the external `generate_ledger` package (`gen auto` CLI), not by code in this repo.

## Repository Structure

```
rippled-workload/
├── workload/                # Main workload application
│   └── src/workload/
│       ├── app.py                 # FastAPI application, endpoints, lifespan, dashboard
│       ├── workload_core.py       # Core workload logic (Workload class, stores, validation)
│       ├── txn_factory/           # Transaction generation
│       │   └── builder.py         # Transaction builders and registry
│       ├── ws.py                  # WebSocket listener for ledger/tx events
│       ├── ws_processor.py        # WS event dispatcher (validation, ledger close, server status)
│       ├── sqlite_store.py        # SQLite persistence (primary store)
│       ├── constants.py           # Transaction types, states, timeouts
│       ├── fee_info.py            # Fee escalation data
│       ├── randoms.py             # SystemRandom for Antithesis determinism
│       └── config.toml            # Configuration (accounts, currencies, tx weights, etc.)
├── sidecar/                 # Antithesis monitoring sidecar (separate container)
├── test_composer/           # Curl-based load test scripts
├── specs/                   # Feature specs (001-priority-improvements)
└── prepare-workload/        # LEGACY — superseded by generate_ledger package
```

## Architecture

### Core Design Principles

**LEDGER-BASED TIMING: The ledger is the tick, not the clock**

The ledger close is the tick for **validation tracking and lifecycle management**, not for submission:

- DO: Use ledger close events for tracking validation, expiry (LastLedgerSequence), and metrics
- DO: Use ledger index as the canonical time unit for transaction lifecycle
- DO: Submit transactions as fast as possible — real users don't wait for ledger closes
- DON'T: Gate submission on ledger close events
- DON'T: Use wall-clock time for submission pacing

**Rationale**: XRPL consensus operates on discrete ledger closes (~3-4 seconds). Validation, sequence numbers, and queue behavior are tied to ledger boundaries. But submission should be immediate — txns sit in rippled's internal queue until applied. The workload submits continuously (build → sign → submit → repeat) like a real-world client.

### Domain Model

The workload follows a layered architecture:

- **domain**: Pure data types (`WalletModel`, `IssuedCurrencyModel`)
- **infrastructure**: External interactions (xrpl client, storage)
- **application/logic**: Coordination (`txn_factory`, account generation)
- **interface/API**: Entry points (FastAPI app)

### Transaction Lifecycle

Transactions move through these states (constants.py):

1. `CREATED` - Transaction built and signed locally
2. `SUBMITTED` - Sent to rippled node
3. `RETRYABLE` - Temporary failure, can retry
4. `VALIDATED` - Confirmed in a validated ledger
5. `REJECTED` - Terminal rejection (tem/tef codes)
6. `EXPIRED` - Past LastLedgerSequence without validation
7. `FAILED_NET` - Network/timeout error

### Transaction Generation

The `txn_factory` uses a registry pattern (builder.py):

- `_BUILDERS` dict maps type name → (builder_fn, model_class) pairs
- `TxnContext` provides wallets, currencies, AMM pools, credentials, vaults, domains, and defaults
- `generate_txn()` selects a random or specified transaction type, weighted by config percentages
- `pick_eligible_txn_type(wallet, ctx)` applies global + per-account capability filters before weighted sampling
- Builders return dicts (or None if ineligible) that are converted to xrpl-py Transaction models
- Capability-aware: skips types requiring MPT IDs, NFTs, AMM pools, credentials, vaults, or domains when those don't exist

Supported transaction types (31): Payment, OfferCreate, OfferCancel, TrustSet, AccountSet, AMMCreate, AMMDeposit, AMMWithdraw, NFTokenMint, NFTokenBurn, NFTokenCreateOffer/CancelOffer/AcceptOffer, MPTokenIssuanceCreate/Set/Authorize/Destroy, TicketCreate, Batch, DelegateSet, CredentialCreate/Accept/Delete, PermissionedDomainSet/Delete, VaultCreate/Set/Delete/Deposit/Withdraw/Clawback

### Validation Tracking

Two concurrent paths to validation (see workload/ws-architecture.md):

1. **WebSocket** (primary): `ws_listener` → event queue → `ws_processor` → `record_validated(src=WS)`
   - Subscribes to `accounts_proposed` (early engine_result feedback) + `transactions` stream (catches newly-created accounts)
   - `ledgerClosed` events provide `txn_count`, `fee_base`, `reserve_base`, `reserve_inc` — eliminates RPC calls
2. **RPC Polling** (fallback): `periodic_finality_check` every 5s → `record_validated(src=POLL)`

Both paths deduplicate by `(tx_hash, ledger_index)`. See `workload/ws-architecture.md` and `workload/ws-architecture.excalidraw` for the full architecture diagram.

### Submission Architecture

Single unified loop in `continuous_workload()` (app.py): build → sign → submit → repeat. No queue, no producer-consumer split. Submissions are not gated on ledger close — txns go to rippled's internal queue immediately. `target_txns_per_ledger` controls batch size per iteration. Self-healing via `expire_past_lls()` when all accounts are blocked.

### Assertions Framework

`assertions.py` centralises all Antithesis SDK interaction:
- SDK available → delegates to `antithesis.assertions.always/sometimes` and `antithesis.lifecycle`
- SDK unavailable → logs + tracks stats locally via `get_stats()`
- Transaction helpers: `tx_submitted(type)`, `tx_validated(type, result)`, `tx_rejected(type, code)`
- Replaces inline try/except SDK detection in `ws_processor.py` and `app.py`

### Store Architecture

- **SQLiteStore** (opt-in via `WORKLOAD_PERSIST=1`): Persistent storage for accounts, transactions, validations, balances. Survives restarts.
- **InMemoryStore**: In-process metrics, recent validations deque, validation-by-source counters.

### Startup Modes

Two-tier initialization cascade in `app.py:lifespan()` (SQLite only when `WORKLOAD_PERSIST=1`):

1. **SQLite hot-reload** (opt-in): If `WORKLOAD_PERSIST=1` and `state.db` exists with state, load accounts/balances from it (fastest)
2. **Genesis load**: If `accounts.json` exists from `generate_ledger`, import pre-provisioned accounts and discover AMM pools from ledger
3. **Full init**: Fund gateways, set flags, fund users, establish trust lines, create AMM pools from scratch (slowest)

## Development Commands

### Setup and Installation

```bash
cd workload
uv sync
```

### Linting and Formatting

Pre-commit hooks run ruff automatically on commit (ruff-check with `--fix` + ruff-format). To run manually:

```bash
cd workload

# Via pre-commit (recommended — matches CI)
uv run --group dev pre-commit run --all-files

# Or directly
uv run --group lint ruff check --fix
uv run --group lint ruff format
```

Configuration in `pyproject.toml`: Methods must have return types (ANN201), line-length 120, Python 3.13+ target

### Running the Workload

```bash
cd workload

# Start (defaults to localhost:5005 RPC, localhost:6006 WS)
uv run workload

# Or with explicit endpoints
RPC_URL="http://localhost:5005" WS_URL="ws://localhost:6006" uv run workload
```

On startup the workload will:
1. Probe the RPC endpoint until it responds
2. Wait for ledger closes to confirm the network is progressing
3. Load state (SQLite → genesis → full init, whichever is available)
4. Start continuous transaction submission

### Network Setup (via generate_ledger)

```bash
# Generate everything: ledger.json, rippled configs, docker-compose.yml
# Defaults: 1000 accounts, 4 gateways, USD/CNY/BTC/ETH, full trust line coverage
# --amendment-source automatically uses develop profile
gen auto --amendment-source /path/to/rippled/include/xrpl/protocol/detail/features.macro

# Start network
cd testnet
docker compose up -d

# Verify nodes are synced
docker exec val0 rippled --silent server_info | python3 -c "
import sys,json; i=json.load(sys.stdin)['result']['info']
print(f\"state: {i['server_state']}, ledgers: {i['complete_ledgers']}, peers: {i['peers']}\")"
```

### Testing

No formal test suite. Test via API endpoints:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/transaction/random
curl http://localhost:8000/state/summary
curl http://localhost:8000/state/dashboard   # HTML dashboard
curl http://localhost:8000/docs              # Swagger UI
```

Curl-based load scripts in `test_composer/all_transactions/`.

## Configuration

### Workload Configuration (workload/src/workload/config.toml)

Key settings:

- **funding_account**: Genesis account (funds all new accounts)
- **gateways**: Number (6), names, balance, flags (DefaultRipple)
- **users**: Number (1000) and balance
- **amm**: Trading fee, pool counts (12 gateway + 100 user), deposit/withdraw amounts
- **currencies**: 20 currency codes with rates
- **transactions.disabled**: Transaction types to skip (currently: Batch, DelegateSet)
- **transactions.percentages**: Weight distribution (Payment=0.25, OfferCreate=0.20, AMMDeposit=0.15, AMMWithdraw=0.10)
- **genesis**: Path to accounts.json, gateway/user counts, currencies per gateway
- **rippled**: Connection settings (docker hostname, local IP, ports)
- **timeout**: Startup timeout (600s), RPC timeout, initial ledger wait

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RPC_URL` | `http://{rippled_ip}:5005` | RPC endpoint |
| `WS_URL` | `ws://{rippled_ip}:6006` | WebSocket endpoint |
| `RIPPLED_IP` | auto-detected | Override rippled host |

## Key Implementation Details

### Sequence Number Management

Per-account sequence allocation uses asyncio locks to prevent double-spending. `alloc_seq()` fetches from ledger once, then increments locally. `release_seq()` rolls back on local errors (tel* codes).

### Transaction Hash Handling

Locally computed hashes may differ from server hashes. The workload handles rekey operations when the server returns a different hash.

### LastLedgerSequence Horizon

Transactions expire if not validated within `HORIZON = 15` ledgers (~45-60 seconds). Configurable in `constants.py`.

### AMM Pool Registry

AMM pools are tracked in `_amm_pool_registry` (list of pool dicts) with deduplication via `frozenset`. New pools registered on AMMCreate validation. Used by AMMDeposit/AMMWithdraw builders.

### Error Handling

- `tem`/`tef` codes: Terminal rejection, mark as REJECTED
- `ter` codes: Retryable, submit again
- `tel` codes: Local error, release sequence
- Network timeouts: Mark as FAILED_NET
- Expired (past LastLedgerSequence): Mark as EXPIRED, cascade-expire dependent txns

## Common Patterns

### Adding a New Transaction Type

1. Add to `TxType` enum in constants.py
2. Add xrpl-py model import and `_build_*` function in txn_factory/builder.py (sync, returns dict or None)
3. Add entry to `_BUILDERS` dict: `"MyNewTxn": (_build_mynew_txn, MyNewTxn)`
4. Add capability filters in `pick_eligible_txn_type()` (global and per-account) and `generate_txn()`
5. If stateful: add tracking list in workload_core.py (`self._things`), wire into `configure_txn_context()`, add `TxnContext` field, add validation hook(s) in `record_validated()`
6. Add tunable params in config.toml under `[transactions.mynew_txn]`, read via `ctx.config`
7. Optionally add percentage weight in `[transactions.percentages]`
8. Add POST endpoint in app.py and test_composer script
9. If creates a ledger object: add to owner_reserve fee path in `build_sign_and_track()`

### Working with Accounts

The workload maintains three account pools:
- `self.funding_wallet`: Genesis account (funds new accounts)
- `self.gateways`: Issuer accounts (with DefaultRipple flag)
- `self.users`: Regular user accounts

New accounts are adopted into `self.users` after validation of their funding Payment.

## Docker Images

- **workload:latest**: Built from `workload/Dockerfile` (uvicorn FastAPI app). For local development, run natively with `uv run workload` instead.
- **workload (Antithesis)**: Built from root `Dockerfile` (includes test_composer scripts at `/opt/antithesis/test/`)
- **rippleci/xrpld:develop**: Default rippled image for testnet (from Docker Hub). Override with `gen auto --image`
- **config:latest**: Built from `Dockerfile.config` (network configs)
- **sidecar:latest**: Built from `sidecar/Dockerfile` (monitoring service)

## Important Notes

- The workload uses Python 3.14 (requires 3.13+) and the `uv` package manager
- `generate-ledger` is a local editable dependency (`../../generate_ledger`). Importable as both `generate_ledger` and `gl` (alias package).
- SQLite persistence is **opt-in** via `WORKLOAD_PERSIST=1`. Default: fresh genesis load every restart.
- All timestamps are in seconds since epoch (time.time())
- Account initialization happens at startup in `lifespan()` (app.py)
- WebSocket listener, WS processor, finality checker, and DEX metrics poller run as concurrent tasks in an asyncio.TaskGroup
- `prepare-workload/` is legacy — superseded by the `generate_ledger` package (`gen auto` CLI). Do not add new code there.
- DelegateSet requires `PermissionDelegationV1_1` which is `Supported::no` in rippled develop — disabled in config.toml until rippled enables it.
- Vaults require `SingleAssetVault` (`Supported::yes` in develop) — works if testnet is generated with `--amendment-source` pointing at current features.macro.
- `--amendment-source` on `gen auto` now automatically uses the develop profile (no need for `--amendment-profile develop`).
- Default rippled image is now `rippleci/xrpld:develop` (override with `--image`).

## Current Priorities

See `workload/docs/todo/TODO.md` for the full list. The three P0 items are:

1. **Code health**: Dead code cleanup, modularization, modern Python 3.13+ conventions. No backwards compatibility — use StrEnum, match, type parameter syntax, TaskGroup, etc.
2. **Public network support**: The workload must easily target the public XRPL devnet or testnet (faucet-funded), not just local docker networks.
3. **XRP accounting / fund recovery**: On shutdown or Ctrl-C, sweep all XRP back to the funding source. The only permanently consumed XRP should be transaction fees (burned) and account reserves. Stretch: AccountDelete to reclaim reserves.

## Active Technologies
- Python 3.14 (3.13+ required) + FastAPI, xrpl-py 4.5.0, uvicorn, asyncio.TaskGroup
- SQLite3 (via sqlite_store.py, opt-in) for persistence, InMemoryStore for metrics
- WebSocket (ws.py + ws_processor.py) for real-time validation tracking
- Antithesis SDK 0.2.0 (assertions.py) for coverage-guided assertions
- generate_ledger / gl package (external, importable as library or CLI) for network setup
- pynacl for fast ed25519 key generation
