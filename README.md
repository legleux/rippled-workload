# rippled-workload

Transaction traffic generator for XRPL networks. Submits realistic transaction patterns (Payments, OfferCreates, AMM operations, NFTokens, MPTokens, etc.) driven by ledger closes, and tracks every transaction through to validation.

Built for [Antithesis](https://antithesis.com/) testing of [rippled](https://github.com/XRPLF/rippled).

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- Docker + Docker Compose (for running a local rippled network)

### 1. Start a testnet

The workload can target any rippled node, but we **strongly recommend** using [`generate-ledger`](https://github.com/legleux/generate-ledger) to create a local testnet. It pre-bakes accounts, trust lines, and AMM pools into the genesis ledger so the workload starts submitting immediately.

```bash
# Install generate-ledger (once)
uv tool install generate-ledger

# Generate a 5-validator testnet
gen auto -o testnet -v 5 -n 40 -t "0:1:USD:1000000000"

# Start it (leave this running)
cd testnet && docker compose up -d
```

This exposes rippled on `localhost:5005` (RPC) and `localhost:6006` (WebSocket).

<details>
<summary>Don't have generate-ledger?</summary>

You can point the workload at any running rippled node. It will provision accounts, trust lines, and AMM pools from scratch using the genesis account. This works but takes several minutes.

</details>

### 2. Run the workload

In a new terminal:

```bash
cd workload
uv sync                       # first time only

uv run workload
```

If the network isn't reachable, the workload will tell you:

```
Cannot reach rippled at http://localhost:5005
Is the network running? Start it with:
  cd /path/to/testnet && docker compose up -d
```

### 3. Verify it's working

| URL | What you'll see |
|-----|----------------|
| http://localhost:8000/state/dashboard | Live dashboard — stats, transaction stream, type controls |
| http://localhost:8000/docs | Swagger UI for all API endpoints |
| https://custom.xrpl.org/localhost:6006 | XRPL Explorer showing your network's ledger progression |

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
- **[workload/docs/todo/TODO.md](workload/docs/todo/TODO.md)** — Prioritized roadmap
