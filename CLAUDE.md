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

This workload operates on **ledger close events** as the fundamental unit of time, not wall-clock time:

- DO: Wait for ledger closes, count ledgers, use ledger index as the tick
- DO: Use time for timeouts (network connectivity issues) and measuring operation duration (metrics)
- DON'T: Use time-based delays for submission logic (no `await asyncio.sleep()` between batches)
- DON'T: Spread submissions over time intervals
- DON'T: Use time to control submission rate

**Rationale**: XRPL consensus operates on discrete ledger closes (~3-4 seconds). Transaction validation, sequence numbers, and queue behavior are all tied to ledger boundaries. Time-based logic creates race conditions and unpredictable behavior.

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

- `@register_txn(Payment)` decorators register builder functions
- `TxnContext` provides wallets, currencies, AMM pools, and defaults
- `generate_txn()` selects a random or specified transaction type, weighted by config percentages
- Builders return dicts that are converted to xrpl-py Transaction models
- Capability-aware: skips types requiring MPT IDs, NFTs, or AMM pools when those don't exist

Supported transaction types: Payment, OfferCreate, OfferCancel, TrustSet, AccountSet, AMMCreate, AMMDeposit, AMMWithdraw, NFTokenMint, NFTokenBurn, NFTokenCreateOffer/CancelOffer/AcceptOffer, MPTokenIssuanceCreate/Set/Authorize/Destroy, TicketCreate, Batch

### Validation Tracking

Two concurrent paths to validation (see workload/ws-architecture.md):

1. **WebSocket** (primary): `ws_listener` → event queue → `ws_processor` → `record_validated(src=WS)`
2. **RPC Polling** (fallback): `periodic_finality_check` every 5s → `record_validated(src=POLL)`

Both paths deduplicate by `(tx_hash, ledger_index)`. See `workload/ws-architecture.md` and `workload/ws-architecture.excalidraw` for the full architecture diagram.

### Store Architecture

- **SQLiteStore** (primary): Persistent storage for accounts, transactions, validations, balances. Survives restarts.
- **InMemoryStore**: In-process metrics, recent validations deque, validation-by-source counters.
- **Store Protocol**: Clean interface both stores implement.

### Startup Modes

Three-tier initialization cascade in `app.py:lifespan()`:

1. **SQLite hot-reload**: If `state.db` exists with state, load accounts/balances from it (fastest)
2. **Genesis load**: If `accounts.json` exists from `generate_ledger`, import pre-provisioned accounts and discover AMM pools from ledger
3. **Full init**: Fund gateways, set flags, fund users, establish trust lines, create AMM pools from scratch (slowest)

## Development Commands

### Setup and Installation

```bash
cd workload
uv sync
```

### Linting and Formatting

```bash
# Format and fix imports
ruff check --select I --fix
ruff format

# Check for issues
ruff check
```

Configuration in `pyproject.toml`: Methods must have return types (ANN201), line-length 120, Python 3.13+

### Running the Workload

```bash
cd workload

# Start (pointing at the local network)
RPC_URL="http://localhost:5005" WS_URL="ws://localhost:6006" \
  uv run uvicorn workload.app:app --host 0.0.0.0 --port 8000
```

On startup the workload will:
1. Probe the RPC endpoint until it responds
2. Wait for ledger closes to confirm the network is progressing
3. Load state (SQLite → genesis → full init, whichever is available)
4. Start continuous transaction submission

### Network Setup (via generate_ledger)

```bash
# Generate everything: ledger.json, rippled configs, docker-compose.yml
gen auto -o /path/to/testnet -v 5 -n 40 -t "0:1:USD:1000000000"

# Start network
cd /path/to/testnet
docker compose up -d

# Verify nodes are synced
docker exec rippled rippled --silent server_info | python3 -c "
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
- **transactions.disabled**: Transaction types to skip
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
2. Add model import and builder in txn_factory/builder.py:
```python
@register_txn(MyNewTxn)
def build_mynew_txn(ctx: TxnContext) -> dict:
    src = ctx.rand_account()
    return {"TransactionType": "MyNewTxn", "Account": src.address, ...}
```
3. Optionally add to `disabled` list in config.toml to exclude from random selection
4. Optionally add percentage weight in `[transactions.percentages]`
5. Add defaults if needed to config.toml `[transactions.mynew_txn]`

### Working with Accounts

The workload maintains three account pools:
- `self.funding_wallet`: Genesis account (funds new accounts)
- `self.gateways`: Issuer accounts (with DefaultRipple flag)
- `self.users`: Regular user accounts

New accounts are adopted into `self.users` after validation of their funding Payment.

## Docker Images

- **workload:latest**: Built from `workload/Dockerfile` (uvicorn FastAPI app). For local development, run natively with `uv run uvicorn` instead.
- **workload (Antithesis)**: Built from root `Dockerfile` (includes test_composer scripts at `/opt/antithesis/test/`)
- **rippled:latest**: Built from `Dockerfile.rippled` (clones and builds rippled with Antithesis instrumentation)
- **config:latest**: Built from `Dockerfile.config` (network configs)
- **sidecar:latest**: Built from `sidecar/Dockerfile` (monitoring service)

## Important Notes

- The workload uses Python 3.13+ and the `uv` package manager
- All timestamps are in seconds since epoch (time.time())
- Account initialization happens at startup in `lifespan()` (app.py)
- WebSocket listener, WS processor, finality checker, and DEX metrics poller run as concurrent tasks in an asyncio.TaskGroup
- `prepare-workload/` is legacy — superseded by the `generate_ledger` package (`gen auto` CLI). Do not add new code there.

## Current Priorities

See `workload/docs/todo/TODO.md` for the full list. The three P0 items are:

1. **Code health**: Dead code cleanup, modularization, modern Python 3.13+ conventions. No backwards compatibility — use StrEnum, match, type parameter syntax, TaskGroup, etc.
2. **Public network support**: The workload must easily target the public XRPL devnet or testnet (faucet-funded), not just local docker networks.
3. **XRP accounting / fund recovery**: On shutdown or Ctrl-C, sweep all XRP back to the funding source. The only permanently consumed XRP should be transaction fees (burned) and account reserves. Stretch: AccountDelete to reclaim reserves.

## Active Technologies
- Python 3.13+ + FastAPI, xrpl-py (minimal usage), uvicorn, asyncio.TaskGroup
- SQLite3 (via sqlite_store.py) for persistence, InMemoryStore for metrics
- WebSocket (ws.py + ws_processor.py) for real-time validation tracking
- generate_ledger package (external) for network setup
