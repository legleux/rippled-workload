# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **rippled-workload** repository for Antithesis testing of the XRPL (XRP Ledger) rippled node. It consists of three main components:

1. **prepare-workload**: Network configuration generator that creates testnet topologies
2. **workload**: FastAPI-based workload generator that creates and submits XRPL transactions
3. **sidecar**: Auxiliary service for monitoring

The project tests rippled nodes under load by generating realistic transaction patterns and tracking their lifecycle through the consensus process.

## Repository Structure

```
rippled-workload/
├── prepare-workload/     # Network configuration generation
│   ├── prepare_workload/ # Config templates, UNL generation, compose rendering
│   └── main.py          # Entry point for network setup
├── workload/            # Main workload application
│   └── src/workload/
│       ├── app.py              # FastAPI application & endpoints
│       ├── workload_core.py    # Core workload logic (Workload class)
│       ├── txn_factory/        # Transaction generation
│       │   └── builder.py      # Transaction builders and registry
│       ├── ws.py               # WebSocket listener for ledger events
│       ├── constants.py        # Transaction types and states
│       └── config.toml         # Configuration (gateways, users, currencies, etc.)
├── sidecar/             # Monitoring sidecar
└── test_composer/       # Test composition scripts
```

## Architecture

### Domain Model

The workload follows a layered architecture (see workload/README.md:7-13):

- **domain**: Pure data types (`WalletModel`, `IssuedCurrencyModel`)
- **infrastructure**: External interactions (XRPL client, network I/O, storage)
- **application/logic**: Coordination (`txn_factory`, account generation)
- **interface/API**: Entry points (FastAPI, CLI)

### Transaction Lifecycle

Transactions move through these states (workload/src/workload/constants.py:21-28):

1. `CREATED` - Transaction built and signed locally
2. `SUBMITTED` - Sent to rippled node
3. `RETRYABLE` - Temporary failure, can retry
4. `VALIDATED` - Confirmed in a validated ledger
5. `REJECTED` - Terminal rejection (tem/tef codes)
6. `EXPIRED` - Past LastLedgerSequence without validation
7. `FAILED_NET` - Network/timeout error

The `Workload` class (workload/src/workload/workload_core.py:248) manages:
- **Pending transactions**: In-flight txns tracked in `self.pending` dict
- **Store**: Historical record in `InMemoryStore` for metrics
- **Account management**: Sequence number allocation with per-account locks
- **Validation tracking**: Both polling and WebSocket-based confirmation

### Transaction Generation

The `txn_factory` uses a registry pattern (workload/src/workload/txn_factory/builder.py:122-128):

- `@register_txn(Payment)` decorators register builder functions
- `TxnContext` provides wallets, currencies, and defaults
- `generate_txn()` selects a random or specified transaction type
- Builders return dicts that are converted to xrpl-py Transaction models

Supported transaction types: Payment, TrustSet, AccountSet, NFTokenMint, MPTokenIssuanceCreate, Batch (experimental)

## Development Commands

### Setup and Installation

```bash
# Install prepare-workload dependencies
cd prepare-workload
uv sync

# Install workload dependencies
cd workload
uv sync
```

### Linting and Formatting

Both projects use `ruff` for linting and formatting:

```bash
# Format and fix imports
ruff check --select I --fix
ruff format

# Check for issues
ruff check
```

Configuration in `pyproject.toml`:
- prepare-workload: Google docstring style, line-length 120, Python 3.13+
- workload: Methods must have return types (ANN201), line-length 120, Python 3.13+

### Running the Workload

The workload runs as a FastAPI application. There are two primary ways to run it:

#### Local Development (via docker-compose.yml)

```bash
# Start workload container with hot-reload
docker compose up workload

# The workload will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

#### Full Network Setup

Follow the test flow documented in test_start_experiment_flow.md:

1. **Generate network configuration**:
```bash
cd prepare-workload
uvx --from legleux-generate-ledger gen \
    --config_only False \
    --include_services sidecar-compose.yml \
    --include_services workload-compose.yml
```

2. **Build images**:
```bash
# Build sidecar
docker build sidecar --file sidecar/Dockerfile --tag sidecar:latest

# Build workload
docker build . --file Dockerfile --tag workload:latest

# Build config
docker build . --file Dockerfile.config \
    --build-arg TEST_NETWORK_DIR=testnet \
    --tag config:latest

# Build rippled
docker build . --file Dockerfile.rippled \
    --tag rippled:latest \
    --build-arg RIPPLED_REPO=https://github.com/XRPLF/rippled.git \
    --build-arg RIPPLED_COMMIT=develop
```

3. **Start the network**:
```bash
cd prepare-workload/testnet
docker compose up -d

# Check network status
docker exec -it rippled rippled --silent server_info | tail -n+4 | jq .result.info.complete_ledgers

# Access workload API
workload_ip=$(docker inspect workload | jq -r '.[0].NetworkSettings.Networks[].IPAddress')
curl -s "${workload_ip}:8000/accounts" | jq
```

### Testing

The workload does not currently have a formal test suite, but you can test endpoints via:

```bash
# Health check
curl http://localhost:8000/health

# Create random account
curl http://localhost:8000/accounts/create/random

# Submit random transaction
curl http://localhost:8000/transaction/random

# Create specific transaction type
curl http://localhost:8000/transaction/create/Payment

# Check state
curl http://localhost:8000/state/summary
curl http://localhost:8000/state/pending
curl http://localhost:8000/state/validations
```

## Configuration

### Workload Configuration (workload/src/workload/config.toml)

Key settings:

- **funding_account**: Genesis account used to fund new accounts
- **gateways**: Number, balance, and flags for gateway accounts
- **users**: Number and balance for user accounts
- **currencies**: Available currency codes and rates
- **transactions.available**: Enabled transaction types
- **timeout**: Network startup and RPC timeouts
- **rippled**: Connection settings (docker/local, ports)

### Network Configuration (prepare-workload)

Settings in `prepare-workload/settings.toml`:

- Network topology (validators, peers, edges)
- UNL configuration (use_unl, validator_list_sites)
- Node config templates (ports, features, voting)

Generate with `prepare-workload/main.py`:

```bash
cd prepare-workload
./main.py -n network.toml -t testnet -v 5
```

## Key Implementation Details

### Sequence Number Management

Per-account sequence allocation uses locks to prevent double-spending (workload/src/workload/workload_core.py:329-339):

```python
async def alloc_seq(self, addr: str) -> int:
    rec = self._record_for(addr)
    if rec.next_seq is None:
        # Fetch from ledger once
        ai = await self._rpc(AccountInfo(account=addr, ...))
        rec.next_seq = ai.result["account_data"]["Sequence"]

    async with rec.lock:
        s = rec.next_seq
        rec.next_seq += 1
        return s
```

### Transaction Hash Handling

Locally computed hashes may differ from server hashes. The workload handles rekey operations when the server returns a different hash (workload/src/workload/workload_core.py:365-379).

### Validation Tracking

Two paths to validation (workload/src/workload/workload_core.py:381-409):

1. **Polling**: `periodic_finality_check()` polls RPC for submitted txns
2. **WebSocket**: `ws_listener()` receives real-time ledger events

Both call `record_validated()` which deduplicates and records to the store exactly once per (tx_hash, ledger_index).

### Store Architecture

- **InMemoryStore**: Flat dict of transaction records with metrics
- **ValidationRecord**: Tracks which txns validated via which path (poll vs ws)
- **Metrics**: Counts by state, validated by source, recent validations deque

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
3. Add to `available` list in config.toml
4. Add defaults if needed to config.toml `[transactions.mynew_txn]`

### Working with Accounts

The workload maintains three account pools:
- `self.funding_wallet`: Genesis account (funds new accounts)
- `self.gateways`: Issuer accounts (with AccountSet flags)
- `self.users`: Regular user accounts

New accounts are adopted into `self.users` after validation (workload/src/workload/workload_core.py:399-406).

### Error Handling

- `tem`/`tef` codes: Terminal rejection, mark as REJECTED
- `ter` codes: Retryable, submit again
- Network timeouts: Mark as FAILED_NET
- Expired (past LastLedgerSequence): Mark as EXPIRED

## Docker Images

- **workload:latest**: Built from Dockerfile (uvicorn FastAPI app)
- **workload:dev**: Development target with sleep infinity
- **rippled:latest**: Built from Dockerfile.rippled (clones and builds rippled)
- **config:latest**: Built from Dockerfile.config (network configs)
- **sidecar:latest**: Built from sidecar/Dockerfile (monitoring service)

## Important Notes

- The workload uses Python 3.13+ and the `uv` package manager
- All timestamps are in seconds since epoch (time.time())
- The LastLedgerSequence horizon is configurable (default: 3 ledgers, see constants.py:35)
- Account initialization happens at startup in `lifespan()` (app.py:92-133)
- WebSocket listeners and finality checks run as concurrent tasks in an asyncio.TaskGroup
- Retry logic is not yet implemented (marked as TODO in several places)
