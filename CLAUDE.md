# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **rippled-workload** repository for Antithesis testing of the XRPL (XRP Ledger) xrpld node. The main component is a **FastAPI-based workload generator** that creates and submits XRPL transactions against a local testnet, tracking their lifecycle through the consensus process.

A separate **sidecar** container provides passive monitoring and Antithesis assertions.

Network setup (configs, genesis ledger, docker-compose) is handled by the external `generate_ledger` package (`gen auto` CLI), not by code in this repo.

## Repository Structure

```
rippled-workload/
├── workload/                # Main workload application
│   └── src/workload/
│       ├── app.py                 # FastAPI application, endpoints, lifespan, dashboard
│       ├── workload_core.py       # Core workload logic (Workload class, stores, validation)
│       ├── txn_factory/           # Transaction generation (modular)
│       │   ├── context.py         # TxnContext dataclass + utilities
│       │   ├── registry.py        # Builder registry, eligibility, compose_submission_set()
│       │   ├── taint.py           # Semantic tainting for invalid txns
│       │   └── builders/          # Per-group builder modules (payment, dex, nft, etc.)
│       ├── ledger_objects.py      # Deterministic XRPL object ID computation
│       ├── ws.py                  # WebSocket listener for ledger/tx events
│       ├── ws_processor.py        # WS event dispatcher (validation, ledger close, server status)
│       ├── sqlite_store.py        # SQLite persistence (primary store)
│       ├── constants.py           # Transaction types, states, timeouts
│       ├── fee_info.py            # Fee escalation data
│       ├── randoms.py             # SystemRandom for Antithesis determinism
│       ├── test_cmd.py            # Lifecycle test CLI (clean → gen → up → monitor → report)
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

**Rationale**: XRPL consensus operates on discrete ledger closes (~3-4 seconds). Validation, sequence numbers, and queue behavior are tied to ledger boundaries. But submission should be immediate — txns sit in xrpld's internal queue until applied. The workload submits continuously (build → sign → submit → repeat) like a real-world client.

### Domain Model

The workload follows a layered architecture:

- **domain**: Pure data types (`WalletModel`, `IssuedCurrencyModel`)
- **infrastructure**: External interactions (xrpl client, storage)
- **application/logic**: Coordination (`txn_factory`, account generation)
- **interface/API**: Entry points (FastAPI app)

### Transaction Lifecycle

Transactions move through these states (constants.py):

1. `CREATED` - Transaction built and signed locally
2. `SUBMITTED` - Sent to xrpld node
3. `RETRYABLE` - Temporary failure, can retry
4. `VALIDATED` - Confirmed in a validated ledger
5. `REJECTED` - Terminal rejection (tem/tef codes)
6. `EXPIRED` - Past LastLedgerSequence without validation
7. `FAILED_NET` - Network/timeout error

### Transaction Generation

The `txn_factory` package uses a modular registry pattern:

```
txn_factory/
├── context.py      # TxnContext dataclass + shared utilities
├── registry.py     # Builder registry, eligibility, compose_submission_set()
├── taint.py        # Semantic tainting for intentionally invalid txns
├── builders/       # Per-group builder modules
│   ├── payment.py  # Payment, TrustSet, AccountSet
│   ├── dex.py      # OfferCreate/Cancel, AMM*
│   ├── nft.py      # NFToken*
│   ├── mptoken.py  # MPToken*
│   ├── vault.py    # Vault*
│   ├── credential.py # Credential*, DelegateSet
│   ├── domain.py   # PermissionedDomain*
│   ├── batch.py    # Batch, TicketCreate, TicketUse
│   ├── check.py    # CheckCreate, CheckCash, CheckCancel
│   └── escrow.py   # EscrowCreate, EscrowFinish, EscrowCancel
```

Each builder module exports:
- `BUILDERS`: dict mapping type name → (builder_fn, model_class) pairs
- `ELIGIBILITY` (optional): per-account eligibility predicates
- `TAINTERS` (optional): semantic tainting strategies for invalid txns

**Submission set composition** (`compose_submission_set()` in registry.py):
1. Type-first: rolls N types from weighted config distribution
2. Per-type intent: rolls VALID/INVALID per txn using configurable ratio
3. Account matching: INVALID intent → clean accounts only (0 pending); VALID → any free account
4. Tainting: applies `taint_txn()` to INVALID-intent txns before model construction

**Note**: Groups of transactions submitted together are called **submission sets**, not "batches" (overloaded with the Batch transaction type).

Supported transaction types (38): Payment, OfferCreate, OfferCancel, TrustSet, AccountSet, AMMCreate, AMMDeposit, AMMWithdraw, NFTokenMint, NFTokenBurn, NFTokenCreateOffer/CancelOffer/AcceptOffer, MPTokenIssuanceCreate/Set/Authorize/Destroy, TicketCreate, TicketUse, Batch, DelegateSet, CredentialCreate/Accept/Delete, PermissionedDomainSet/Delete, VaultCreate/Set/Delete/Deposit/Withdraw/Clawback, CheckCreate/Cash/Cancel, EscrowCreate/Finish/Cancel

### Validation Tracking

Two concurrent paths to validation (see workload/ws-architecture.md):

1. **WebSocket** (primary): `ws_listener` → event queue → `ws_processor` → `record_validated(src=WS)`
   - Subscribes to `accounts` (validated txns per-account) + `accounts_proposed` (early engine_result feedback)
   - Account subscription is deferred until genesis accounts are loaded (`accounts_ready` event in bootstrap.py)
   - `ledgerClosed` events provide `txn_count`, `fee_base`, `reserve_base`, `reserve_inc` — eliminates RPC calls
   - `ledgerClosed` also triggers `expire_past_lls()` to expire txns past their LastLedgerSequence
   - Dedicated logger: `workload.ws.accounts` logs all subscription and validation events
2. **RPC Polling** (fallback): `periodic_finality_check` every 5s — only checks txns past their LLS (overdue), not all pending

Both paths deduplicate by `(tx_hash, ledger_index)`. See `workload/ws-architecture.md` and `workload/ws-architecture.excalidraw` for the full architecture diagram.

**Known issue (2026-03-31):** `_subscribe_accounts()` reads the next WS message as the subscription ack. On reconnect, a `ledgerClosed` event can arrive first, causing the subscription to report failure. Needs message-id filtering. See TODO.

**Known issue (2026-04-01):** WS connection drops every 1-2 minutes under 1005-account subscription load. Unknown whether the drop originates from our WS module (websockets lib, asyncio backpressure, read loop stalling) or xrpld. Each reconnect gap causes mass LLS expiry (150-515 txns). This is now the dominant failure mode (55% success rate). See P0 in TODO.

### Submission Architecture

Single unified loop in `continuous_workload()` (workload_runner.py). Two-phase build: sync compose, then parallel alloc_seq + sign via TaskGroup. Token-bucket rate limiter (`target_tps`, 0=unlimited). No queue, no producer-consumer split. Submissions are not gated on ledger close — txns go to xrpld's internal queue immediately. `submission_set_size` caps txns built per iteration; `max_pending_per_account` is locked to 1 (multi-pending causes cascading tefPAST_SEQ). Self-healing via `expire_past_lls()` when all accounts are blocked — resets `next_seq=None` and bumps `generation` to invalidate pre-signed txns. Generation guard in `submit_pending()` catches stale txns. Account sequences are pre-warmed in parallel on startup via `warm_sequences()`.

### Intentionally Invalid Transactions

Configurable ratio of valid/invalid transactions via `[transactions.intent]` in config.toml or dynamically via `POST /workload/intent`. Invalid txns are semantically tainted (valid structure, wrong values) so they pass xrpl-py model validation and binary codec encoding but get rejected by xrpld (tem/tef/tec codes). Signing uses manual `encode_for_signing` + `sign` + `encode` + `SubmitOnly` — no xrpl-py convenience methods that could interfere.

### Ledger Object ID Computation

`ledger_objects.py` computes XRPL object IDs deterministically from transaction fields (no RPC needed): NFTokenID, Offer/NFTOffer/Check/Escrow/Ticket/Vault/Domain/Credential indices. Self-contained module designed for eventual xrpl-py contribution. Used by validation hooks to track NFTs and offers without metadata RPC calls.

### Assertions Framework

`assertions.py` centralises all Antithesis SDK interaction:
- SDK available → delegates to `antithesis.assertions.always/sometimes` and `antithesis.lifecycle`
- SDK unavailable → logs + tracks stats locally via `get_stats()`
- Transaction helpers: `tx_submitted(type)`, `tx_validated(type, result)`, `tx_rejected(type, code)`
- Replaces inline try/except SDK detection in `ws_processor.py` and `app.py`

### Store Architecture

- **SQLiteStore** (opt-in via `WORKLOAD_PERSIST=1`): Persistent storage for accounts, transactions, validations, balances. Survives restarts.
- **InMemoryStore**: In-process metrics, recent validations deque, validation-by-source counters.
- **Cumulative counters** on `Workload`: `_failure_codes` (engine_result → count), `_tem_disabled_types` (set of types that got temDISABLED). Survive `_cleanup_terminal()`.

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

### Lifecycle Test (`workload test`)

```bash
cd workload

# Full lifecycle: clean → gen → up → monitor (5min) → report
uv run workload test --up

# Full lifecycle, shorter monitoring window
uv run workload test --up --duration 60

# Monitor-only (network already running)
uv run workload test --duration 120

# Include extra endpoints in the report
uv run workload test --focus dex/amm-pools

# Rebuild without wiping testnet
uv run workload test --up --no-clean
```

Reports are written to `docs/todo/{YYYY-MM-DD-HHMM}-test-results.md`.

### Network Setup (via generate_ledger)

```bash
# Generate everything: ledger.json, xrpld configs, docker-compose.yml
# Defaults: 1000 accounts, 4 gateways, USD/CNY/BTC/ETH, full trust line coverage
# Amendment profiles: release (mainnet amendments, default), develop (auto-fetch from GitHub), custom (JSON file)
gen auto --amendment-profile develop
# Or with a local features.macro (implies develop profile):
gen auto --amendment-source /path/to/rippled/include/xrpl/protocol/detail/features.macro

# Start network
cd testnet
docker compose up -d

# Verify nodes are synced
docker exec val0 xrpld --silent server_info | python3 -c "
import sys,json; i=json.load(sys.stdin)['result']['info']
print(f\"state: {i['server_state']}, ledgers: {i['complete_ledgers']}, peers: {i['peers']}\")"
```

### Testing

```bash
cd workload

# Default: GET endpoints only (~45 tests, requires running network)
uv run --group test pytest

# Include mutating tests (submits txns, toggles workload state)
uv run --group test pytest -m "not reset"

# Everything including network reset (destructive!)
uv run --group test pytest -m ""
```

Tests hit the live API via httpx. Markers: `mutating` (deselected by default), `reset` (deselected by default, tears down network).

Curl-based load scripts in `test_composer/all_transactions/`.

## Configuration

### Workload Configuration (workload/src/workload/config.toml)

Key settings:

- **funding_account**: Genesis account (funds all new accounts)
- **gateways**: Number (6), names, balance, flags (DefaultRipple)
- **users**: Number (1000) and balance
- **amm**: Trading fee, pool counts (12 gateway + 100 user), deposit/withdraw amounts
- **currencies**: 20 currency codes with rates
- **transactions.disabled**: Transaction types to skip (currently: Batch)
- **transactions.max_pending_per_account**: Max in-flight txns per account (default 1, runtime-adjustable)
- **transactions.submission_set_size**: Max txns built per loop iteration (default 200)
- **transactions.percentages**: Weight distribution (Payment=0.25, OfferCreate=0.20, AMMDeposit=0.15, AMMWithdraw=0.10)
- **genesis**: Path to accounts.json, gateway/user counts, currencies per gateway
- **xrpld**: Connection settings (docker hostname, local IP, ports)
- **timeout**: Startup timeout (600s), RPC timeout, initial ledger wait

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RPC_URL` | `http://{xrpld_ip}:5005` | RPC endpoint |
| `WS_URL` | `ws://{xrpld_ip}:6006` | WebSocket endpoint |
| `XRPLD_IP` | auto-detected | Override xrpld host (`RIPPLED_IP` also accepted) |

## Key Implementation Details

### Sequence Number Management

Per-account sequence allocation uses asyncio locks to prevent double-spending. `alloc_seq()` fetches from ledger once, then increments locally. `release_seq()` rolls back on local errors (tel* codes). `warm_sequences()` pre-fetches all account sequences in parallel on startup so the build loop doesn't pay RPC latency on first use.

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

### `expect_rejection` — Handle With Extreme Care

`PendingTx.expect_rejection` suppresses rejection warnings and routes assertions to `tx_intentionally_rejected()` instead of `tx_rejected()`. For `tefPAST_SEQ`, it also skips the cascade-expire sequence resync. This means a false positive (a valid transaction incorrectly tagged as intentionally invalid) will **silently swallow real bugs** — sequence desyncs, builder errors, and rejection spikes will not be logged or alerted.

**The flag is only set in one place**: `workload_runner.py`, when `intent == TxIntent.INVALID` AND `taint_txn()` was applied. Any new code path that sets this flag must be reviewed with the same scrutiny as disabling a safety check. See the CAUTION comments in `submit_pending()` (~line 1506 and ~1572 in `workload_core.py`).

## Common Patterns

### Adding a New Transaction Type

1. Add to `TxType` enum in constants.py
2. Create builder function in the appropriate `txn_factory/builders/*.py` module (sync, returns dict or None). Respect `ctx.forced_account`.
3. Add entry to the module's `BUILDERS` dict: `"MyNewTxn": (build_mynew_txn, MyNewTxn)`
4. If per-account eligibility needed: add predicate to module's `ELIGIBILITY` dict
5. Add tainting strategies to module's `TAINTERS` dict
6. Register the builder module in both `registry.py` (`_MODULES` list) AND `taint.py` (module list)
7. If stateful: add `TxnContext` field, tracking dict in `Workload.__init__`, wire into `configure_txn_context()`, add validation hooks in `record_validated()`. Use `ledger_objects.py` for deterministic ID computation.
8. Add global capability filter in `registry.py:global_eligible_types()` if needed
9. Add tunable params in config.toml under `[transactions.mynew_txn]`, read via `ctx.config`
10. Optionally add percentage weight in `[transactions.percentages]`
11. Add POST endpoint in app.py and test_composer script
12. Add to dashboard: `TXN_TYPE_GROUPS` and `TXN_COLORS` in app.py
13. If creates a ledger object: add to owner_reserve fee path in `build_sign_and_track()`

### Working with Accounts

The workload maintains three account pools:
- `self.funding_wallet`: Genesis account (funds new accounts)
- `self.gateways`: Issuer accounts (with DefaultRipple flag)
- `self.users`: Regular user accounts

New accounts are adopted into `self.users` after validation of their funding Payment.

## Docker Images

- **workload:latest**: Built from `workload/Dockerfile` (uvicorn FastAPI app). For local development, run natively with `uv run workload` instead.
- **workload (Antithesis)**: Built from root `Dockerfile` (includes test_composer scripts at `/opt/antithesis/test/`)
- **rippleci/xrpld:develop**: Default xrpld image for testnet (from Docker Hub). Override with `gen auto --image`
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
- DelegateSet requires `PermissionDelegationV1_1` which is `Supported::no` in xrpld develop — disabled in config.toml until xrpld enables it.
- Vaults require `SingleAssetVault` (`Supported::yes` in develop) — works if testnet is generated with `--amendment-source` pointing at current features.macro.
- Amendment profiles: `release` (default, fetches enabled amendments from mainnet RPC), `develop` (auto-fetches features.macro from GitHub), `custom` (local JSON). Use `--amendment-source` to provide a local features.macro instead of GitHub fetch. Per-amendment overrides via `--enable`/`--disable` flags.
- Default xrpld image is now `rippleci/xrpld:develop` (override with `--image`).

## Current Priorities

See `workload/docs/todo/TODO.md` for the full list. The P0 items are:

1. **WS connection stability**: WS drops every 1-2 minutes under 1005-account subscription load, causing mass expiry and 55% success rate. Unknown if it's our WS module or xrpld — first step is determining which side drops.
2. **Code health**: Dead code cleanup, modularization, modern Python 3.13+ conventions. No backwards compatibility — use StrEnum, match, type parameter syntax, TaskGroup, etc.
3. **Public network support**: The workload must easily target the public XRPL devnet or testnet (faucet-funded), not just local docker networks.
4. **XRP accounting / fund recovery**: On shutdown or Ctrl-C, sweep all XRP back to the funding source. The only permanently consumed XRP should be transaction fees (burned) and account reserves. Stretch: AccountDelete to reclaim reserves.

## Active Technologies
- Python 3.14 (3.13+ required) + FastAPI, xrpl-py 4.5.0, uvicorn, asyncio.TaskGroup
- SQLite3 (via sqlite_store.py, opt-in) for persistence, InMemoryStore for metrics
- WebSocket (ws.py + ws_processor.py) for real-time validation tracking
- Antithesis SDK 0.2.0 (assertions.py) for coverage-guided assertions
- generate_ledger / gl package (external, importable as library or CLI) for network setup
- pynacl for fast ed25519 key generation
