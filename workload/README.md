# XRPL Workload Generator

A FastAPI application that generates realistic transaction traffic against a local XRPL testnet. It provisions accounts, establishes trust lines, and continuously submits transactions (Payments, OfferCreates, TrustSets, NFTokenMints, MPTokens, Batch, etc.) at a rate driven by ledger closes.

## Quick Start

### 1. Generate the testnet

Requires the `generate-ledger` package (`gen` CLI):

```bash
cd /path/to/generate_ledger

# Generate everything in one shot: ledger.json, rippled configs, docker-compose.yml
gen auto -o /path/to/testnet -v 5 -n 40 -t "0:1:USD:1000000000"
```

### 2. Start the network

```bash
cd /path/to/testnet
docker compose up -d

# Verify nodes are synced
docker exec rippled rippled --silent server_info | python3 -c "
import sys,json; i=json.load(sys.stdin)['result']['info']
print(f\"state: {i['server_state']}, ledgers: {i['complete_ledgers']}, peers: {i['peers']}\")"
```

### 3. Run the workload

```bash
cd workload

# Install dependencies
uv sync

# Start (pointing at the local network)
RPC_URL="http://localhost:5005" WS_URL="ws://localhost:6006" \
  uv run uvicorn workload.app:app --host 0.0.0.0 --port 8000
```

On startup the workload will:
1. Probe the RPC endpoint until it responds
2. Wait for ledger closes to confirm the network is progressing
3. Fund gateway accounts (4 by default) and set AccountSet flags
4. Fund user accounts (96 by default) in batches
5. Establish trust lines (users x currencies)
6. Seed token balances (fan-out payments from gateways to users)
7. Start continuous transaction submission

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
| POST | `/nftokenmint` | Submit an NFTokenMint |
| POST | `/mptokenissuancecreate` | Submit an MPTokenIssuanceCreate |
| POST | `/batch` | Submit a Batch transaction |

#### Payments

| Method | Path | Description |
|--------|------|-------------|
| POST | `/payment` | Send a payment (XRP or IOU) with source, destination, amount |

#### State & Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | `/state/summary` | JSON summary: counts by state, validation sources, submission results |
| GET | `/state/dashboard` | HTML dashboard (auto-refreshes every second) |
| GET | `/state/pending` | List all pending (in-flight) transactions |
| GET | `/state/validations` | Recent validation records |
| GET | `/state/failed` | Failed/rejected transactions |
| GET | `/state/ws/stats` | WebSocket event processing stats |

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
- **gateways**: Number, names, balance, and flags for gateway accounts
- **users**: Number and balance for user accounts
- **currencies**: Available currency codes and rates
- **transactions.disabled**: Transaction types to skip
- **transactions.percentages**: Weight distribution (e.g. Payment=0.4, OfferCreate=0.3)
- **rippled**: Connection settings (docker hostname, local IP, ports)
- **timeout**: Startup timeout, RPC timeout, initial ledger wait count

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RPC_URL` | `http://{rippled_ip}:5005` | RPC endpoint |
| `WS_URL` | `ws://{rippled_ip}:6006` | WebSocket endpoint |
| `RIPPLED_IP` | auto-detected | Override rippled host |

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

State is persisted in SQLite (`state.db`) so the workload can restart without re-provisioning accounts.

## Development

```bash
cd workload

# Format and lint
uv run ruff check --select I --fix && uv run ruff format

# Check for issues
uv run ruff check
```
