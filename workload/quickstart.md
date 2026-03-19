## 1. Tear down old network (if any)

```bash
cd testnet && docker compose down && cd ..
```

## 2. Generate testnet

Using 4 gateways and 1000 users. Point `--amendment-source` to the xrpld `features.macro` file
(${REPO_PATH}/include/xrpl/protocol/detail/features.macro).
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
uv run gen auto --amendment-source ../../rippled/rippled/develop/include/xrpl/protocol/detail/features.macro
```

## 3. Start network

```bash
cd testnet && docker compose up -d && cd ..
```

## 4. Start workload

```bash
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
