# rippled-workload

Transaction traffic generator for XRPL networks. Submits realistic transaction patterns (Payments, OfferCreates, AMM operations, NFTokens, MPTokens, etc.) driven by ledger closes, and tracks every transaction through to validation.

Built for [Antithesis](https://antithesis.com/) testing of [rippled](https://github.com/XRPLF/rippled).

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- Docker + Docker Compose (for running a local rippled network)

### 1. Install

```bash
cd workload
uv sync
```

### 2. Set up a testnet

The workload can target any rippled node, but we **strongly recommend** using [`generate-ledger`](https://github.com/legleux/generate-ledger) to create a local testnet. It pre-bakes accounts, trust lines, and AMM pools into the genesis ledger so the workload starts submitting transactions immediately instead of spending minutes provisioning from scratch.

```bash
# Install generate-ledger (once)
uv tool install generate-ledger

# Generate a 5-validator testnet
gen auto -o testnet -v 5 -n 40 -t "0:1:USD:1000000000"

# Start it
cd testnet && docker compose up -d
```

<details>
<summary>Don't have generate-ledger?</summary>

You can point the workload at any running rippled node. It will provision accounts, trust lines, and AMM pools from scratch using the genesis account. This works but takes several minutes.

```bash
RPC_URL="http://<rippled-host>:5005" WS_URL="ws://<rippled-host>:6006" \
  uv run uvicorn workload.app:app --port 8000
```

</details>

### 3. Run

```bash
cd workload

RPC_URL="http://localhost:5005" WS_URL="ws://localhost:6006" \
  uv run uvicorn workload.app:app --host 0.0.0.0 --port 8000
```

### 4. Open the dashboard

http://localhost:8000/state/dashboard — live stats, transaction stream, fill-fraction control

http://localhost:8000/docs — Swagger UI for all API endpoints

## What It Does

On startup, the workload loads state via the fastest available path:

| Path | When | Speed |
|------|------|-------|
| **SQLite resume** | `state.db` exists from a previous run | Instant |
| **Genesis import** | `accounts.json` from generate-ledger | Seconds |
| **Full provisioning** | No prior state, bare network | Minutes |

Then it continuously submits transactions at a configurable rate, tracking each one through validation (via WebSocket) or expiry. Transaction types are weighted by config (default: Payment 25%, OfferCreate 20%, AMMDeposit 15%, AMMWithdraw 10%, rest shared evenly).

> **Note**: Starting fresh against a new network? Delete stale state first: `rm -f state.db`

## Project Structure

```
rippled-workload/
├── workload/          # The main application (see workload/README.md for full docs)
│   └── src/workload/
│       ├── app.py              # FastAPI app, endpoints, dashboard
│       ├── workload_core.py    # Core logic, stores, validation tracking
│       ├── txn_factory/        # Transaction builders (registry pattern)
│       ├── ws.py               # WebSocket listener
│       ├── ws_processor.py     # WS event dispatcher
│       ├── sqlite_store.py     # Persistent state
│       └── config.toml         # Configuration
├── sidecar/           # Antithesis monitoring sidecar
├── test_composer/     # Load test scripts
├── scripts/           # Standalone utilities
└── specs/             # Feature specs
```

## Documentation

- **[workload/README.md](workload/README.md)** — Full API reference, configuration, architecture
- **[workload/ws-architecture.md](workload/ws-architecture.md)** — WebSocket validation architecture
- **[workload/TODO.md](workload/TODO.md)** — Prioritized roadmap
