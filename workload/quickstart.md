## 1. Tear down old network (if any)

```bash
docker compose down
```

## 2. Generate testnet

Generates the testnet configs (`testnet/`) and a `docker-compose.yml` that includes both the rippled network and the workload container.

Amendment profiles: `mainnet` (default), `develop` (auto-fetch from GitHub), or provide a local `--amendment-source`.

The defaults are:
**output directory:** testnet
**validators:** 5
**accounts:** 1000
**gateways:** 4
**assets-per-gateway:** 4
**gateway-currencies:** USD, CNY, BTC, ETH
**gateway-coverage:** 1.0 # TODO: Explain
**gateway-connectivity:** 1.0 # TODO: Explain

```bash
# Option A: develop profile (auto-fetches features.macro from GitHub)
uv run workload gen --amendment-profile develop

# Option B: local features.macro
uv run workload gen --amendment-source ../../rippled/rippled/develop/include/xrpl/protocol/detail/features.macro
```

## 3. Start everything (network + workload)

### Docker (builds and runs workload in same network as testnet)

```bash
docker compose up -d --build
```

### Or run workload natively (if testnet is already running)

```bash
cd testnet && docker compose up -d && cd ..
uv run workload
```

To enable SQLite persistence across restarts (not needed for dev):
```bash
WORKLOAD_PERSIST=1 uv run workload
```

Once it's submitting, from another terminal:

## Exercise all 31 types

```bash
./test_composer/all_transactions/exercise_all_types.sh localhost:8000
```

## Or check the dashboard

```bash
open http://localhost:8000/state/dashboard
```
