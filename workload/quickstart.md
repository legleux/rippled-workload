# Quickstart

All commands run from the `workload/` directory.

## Setup

```bash
cd workload
uv sync
```

## Option A: Generate a new testnet and start everything

```bash
# 1. Generate testnet configs + docker-compose.yml
#    (requires generate_ledger: uv pip install -e ../../generate_ledger)
uv run workload gen --amendment-profile develop

# 2. Start the network and workload together
docker compose up -d --build
```

`workload gen` produces:
- `testnet/` — ledger.json, accounts.json, validator configs, testnet docker-compose.yml
- `docker-compose.yml` — includes testnet + workload service in the same Docker network

## Option B: Use an existing testnet

If you already have a `testnet/` directory (e.g. copied from another machine):

```bash
# 1. Write docker-compose.yml for the existing testnet
uv run workload compose

# 2. Start everything
docker compose up -d --build
```

## Option C: Run the workload natively

If you prefer running the workload outside Docker (e.g. for development):

```bash
# Start just the testnet in Docker
cd testnet && docker compose up -d && cd ..

# Run workload natively
uv run workload
```

## Verify

```bash
# Dashboard
open http://localhost:8000

# Health check
curl http://localhost:8000/health

# Exercise all 31 transaction types
./test_composer/all_transactions/exercise_all_types.sh localhost:8000
```

## Tear down

```bash
docker compose down
```

## CLI reference

| Command | Description |
|---------|-------------|
| `uv run workload gen` | Generate testnet + docker-compose.yml (needs generate_ledger) |
| `uv run workload compose` | Write docker-compose.yml for existing testnet/ |
| `uv run workload run` | Run workload server natively (default if no subcommand) |

## gen defaults

| Setting | Default |
|---------|---------|
| output directory | `testnet` |
| validators | 5 |
| accounts | 1000 (4 gateways + 996 users) |
| assets per gateway | 4 |
| gateway currencies | USD, CNY, BTC, ETH |
| amendment profile | `mainnet` |
