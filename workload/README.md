# XRPL Workload Generator

A FastAPI application that generates realistic transaction traffic against an XRPL network. It provisions accounts, establishes trust lines, creates AMM pools, and continuously submits transactions (Payments, OfferCreates, AMMDeposit/Withdraw, TrustSets, NFTokenMints, MPTokens, Batch, etc.) at a rate driven by ledger closes.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- Docker + Docker Compose (for running a local rippled network)

## Quick Start

### 1. Install the workload

```bash
cd workload
uv sync
```

This installs all Python dependencies into a local `.venv`.

### 2. Set up a local XRPL testnet

You need a running rippled network. The fastest way is with the [`generate-ledger`](https://github.com/legleux/generate-ledger) package, which generates configs, a pre-baked genesis ledger (with accounts, trust lines, and AMM pools), and a docker-compose.yml in one command.

#### With generate-ledger (recommended)

```bash
# Install generate-ledger (once)
uv tool install generate-ledger
# Or from a local clone:
# uv tool install -e /path/to/generate_ledger

# Generate a 5-validator testnet with 40 accounts and pre-baked trust lines
gen auto -o testnet -v 5 -n 40 -t "0:1:USD:1000000000"

# Start the network
cd testnet
docker compose up -d
```

The generated `testnet/accounts.json` is picked up by the workload automatically, skipping the slow account-provisioning phase.

#### Without generate-ledger

You can point the workload at any rippled network (local or remote). Without a pre-baked genesis, the workload will provision everything from scratch using the genesis account — funding 6 gateways, 1000 users, establishing trust lines, and creating AMM pools. This works but takes several minutes.

```bash
# If you already have a rippled node running on a different host:
RPC_URL="http://<rippled-host>:5005" WS_URL="ws://<rippled-host>:6006" uv run workload
```

### 3. Run the workload

```bash
cd workload
uv run workload
```

By default this connects to `localhost:5005` (RPC) and `localhost:6006` (WS) on `0.0.0.0:8000`. Override with env vars if needed:

```bash
RPC_URL="http://other-host:5005" WS_URL="ws://other-host:6006" uv run workload
```

### 4. Verify it's working

```bash
# Check the dashboard
open http://localhost:8000/state/dashboard

# Or via API
curl -s http://localhost:8000/state/summary | python3 -m json.tool
```

### What happens on startup

1. Probe the RPC endpoint until it responds
2. Wait for ledger closes to confirm the network is progressing
3. Load state via the fastest available path:
   - **SQLite** (`state.db`): If a previous run's state exists, resume from it
   - **Genesis** (`accounts.json`): If a pre-baked genesis was generated, import accounts and discover AMM pools from the ledger
   - **Full init**: Fund 6 gateways, set flags, fund 1000 users, establish trust lines, create 112 AMM pools from scratch
4. Start continuous transaction submission

> **Note**: If you're starting fresh against a new network but `state.db` exists from a previous run, delete it first: `rm -f state.db`

## Web UI & API

Once running, open in your browser:

| URL | Description |
|-----|-------------|
| http://localhost:8000/state/dashboard | Live dashboard with stats, progress bars, and auto-refresh |
| http://localhost:8000/docs | Swagger UI (interactive API explorer) |

### API Endpoints

#### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |

#### Accounts

| Method | Path | Description |
|--------|------|-------------|
| GET | `/accounts/create` | Create and fund a new account |
| GET | `/accounts/create/random` | Create a random funded account |
| POST | `/accounts/create` | Create account with options (seed, address, drops) |
| GET | `/accounts/{id}` | Get account_info from the ledger |
| GET | `/accounts/{id}/balances` | Get balances from the local database |
| GET | `/accounts/{id}/lines` | Get trust lines from the ledger |

#### Transactions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/random` | Submit a random transaction |
| GET | `/create/{type}` | Create a specific transaction type |
| POST | `/payment` | Submit a Payment |
| POST | `/trustset` | Submit a TrustSet |
| POST | `/accountset` | Submit an AccountSet |
| POST | `/ammcreate` | Submit an AMMCreate |
| POST | `/ammdeposit` | Submit an AMMDeposit |
| POST | `/ammwithdraw` | Submit an AMMWithdraw |
| POST | `/nftokenmint` | Submit an NFTokenMint |
| POST | `/mptokenissuancecreate` | Submit an MPTokenIssuanceCreate |
| POST | `/batch` | Submit a Batch transaction |

#### State & Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | `/state/summary` | JSON summary: counts by state, validation sources, DEX stats |
| GET | `/state/dashboard` | HTML dashboard (auto-refreshes every second) |
| GET | `/state/pending` | List all pending (in-flight) transactions |
| GET | `/state/validations` | Recent validation records |
| GET | `/state/failed` | Failed/rejected transactions |
| GET | `/state/ws/stats` | WebSocket event processing stats |

#### DEX

| Method | Path | Description |
|--------|------|-------------|
| GET | `/dex/metrics` | Pool counts, deposit/withdraw totals, XRP locked |
| GET | `/dex/pools` | List all tracked AMM pools |
| GET | `/dex/pools/{index}` | Live amm_info for a specific pool |
| POST | `/dex/poll` | Trigger a manual DEX metrics poll |

#### Workload Control

| Method | Path | Description |
|--------|------|-------------|
| POST | `/workload/start` | Start continuous transaction generation |
| POST | `/workload/stop` | Stop continuous generation |
| GET | `/workload/status` | Running state and submission stats |
| GET | `/workload/fill-fraction` | Current ledger fill fraction |
| POST | `/workload/fill-fraction` | Adjust fill fraction (0.0-1.0) to control throughput |

## Configuration

Edit `src/workload/config.toml`:

- **funding_account**: Genesis account (funds all new accounts)
- **gateways**: Number (6), names, balance, and flags for gateway accounts
- **users**: Number (1000) and balance for user accounts
- **amm**: Trading fee, pool counts (12 gateway + 100 user), deposit/withdraw amounts
- **currencies**: 20 available currency codes and rates
- **transactions.disabled**: Transaction types to skip
- **transactions.percentages**: Weight distribution (Payment=0.25, OfferCreate=0.20, AMMDeposit=0.15, AMMWithdraw=0.10)
- **genesis**: Path to accounts.json, gateway/user counts, currencies per gateway
- **rippled**: Connection settings (docker hostname, local IP, ports)
- **timeout**: Startup timeout (600s), RPC timeout, initial ledger wait count

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RPC_URL` | `http://{xrpld_ip}:5005` | RPC endpoint |
| `WS_URL` | `ws://{xrpld_ip}:6006` | WebSocket endpoint |
| `XRPLD_IP` | auto-detected | Override xrpld host (`RIPPLED_IP` also accepted) |

## Architecture

| Layer | Responsibility | Examples |
|-------|---------------|----------|
| **domain** | Pure data types, value objects | `WalletModel`, `IssuedCurrencyModel` |
| **infrastructure** | External interactions (xrpl client, storage) | `Workload`, `SQLiteStore`, `xrpl` bindings |
| **application/logic** | Coordination and orchestration | `txn_factory`, account generation |
| **interface/API** | Entry points | FastAPI app, CLI |

### Key Design Principle

**The ledger is the tick, not the clock.** Transaction submission is driven by ledger close events (~3-4s), not wall-clock timers. This aligns with how rippled consensus actually works.

### Transaction Lifecycle

```
CREATED -> SUBMITTED -> VALIDATED   (success)
                     -> REJECTED    (tem/tef codes, terminal)
                     -> RETRYABLE   (ter codes, will retry)
                     -> EXPIRED     (past LastLedgerSequence)
                     -> FAILED_NET  (network/timeout error)
```

### Validation Tracking

Transactions are tracked via two concurrent paths:
- **WebSocket**: Real-time ledger events (primary, fast)
- **Polling**: Periodic RPC checks (backup, catches anything WS misses)

See `ws-architecture.md` for the full architecture diagram.

State is persisted in SQLite (`state.db`) so the workload can restart without re-provisioning accounts.

## Development

```bash
cd workload

# Run locally (no Docker needed)
uv sync
uv run workload

# Format and lint
uv run ruff check --select I --fix && uv run ruff format

# Check for issues
uv run ruff check
```
