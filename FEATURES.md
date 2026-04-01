# rippled-workload — Feature List

## Workload Application

### Transaction Types (38)

| Group | Types |
|-------|-------|
| **Payment & Trust** | Payment, TrustSet, AccountSet |
| **DEX & AMM** | OfferCreate, OfferCancel, AMMCreate, AMMDeposit, AMMWithdraw |
| **NFTokens** | NFTokenMint, NFTokenBurn, NFTokenCreateOffer, NFTokenCancelOffer, NFTokenAcceptOffer |
| **MPTokens** | MPTokenIssuanceCreate, MPTokenIssuanceSet, MPTokenAuthorize, MPTokenIssuanceDestroy |
| **Checks** | CheckCreate, CheckCash, CheckCancel |
| **Escrows** | EscrowCreate, EscrowFinish, EscrowCancel |
| **Vaults** | VaultCreate, VaultSet, VaultDelete, VaultDeposit, VaultWithdraw, VaultClawback |
| **Credentials & Domains** | CredentialCreate, CredentialAccept, CredentialDelete, PermissionedDomainSet, PermissionedDomainDelete, DelegateSet |
| **Batch & Tickets** | Batch, TicketCreate, TicketUse |

### Submission Engine

- **Continuous submission loop** — build → sign → submit → repeat, not gated on ledger closes
- **Type-first composition** — rolls N types from weighted distribution, then matches accounts
- **Two-phase parallel build** — sync compose, then concurrent `alloc_seq` + sign via `TaskGroup`
- **Token-bucket TPS limiter** — configurable 0–10,000 TPS (0 = unlimited)
- **Per-account pending cap** — locked to 1 to prevent cascading `tefPAST_SEQ`
- **Self-healing** — `expire_past_lls()` force-expires stale txns and resets sequences when all accounts are blocked
- **Generation guard** — bumps generation counter on expiry to invalidate pre-signed txns
- **Sequence pre-warming** — parallel `warm_sequences()` on startup and account adoption

### Intentionally Invalid Transactions

- **Configurable valid/invalid ratio** — global default + per-type overrides via API
- **Semantic tainting** — structurally valid dicts with wrong values (self-sends, zero amounts, overspends, bad sequences, expired timestamps, etc.)
- **14 type-specific tainter groups** — Payment, TrustSet, OfferCreate/Cancel, AMMDeposit, NFToken (mint/burn/offer), MPToken, Vault, Check, Credential, Escrow + generic fallback
- **Manual signing pipeline** — `encode_for_signing` + `sign` + `encode` + `SubmitOnly` bypasses xrpl-py convenience methods
- **`expect_rejection` flag** — routes assertions to `tx_intentionally_rejected()`, suppresses rejection warnings

### Validation Tracking

- **Dual-path validation** — WebSocket (primary) + RPC polling (fallback)
- **WebSocket streams** — `accounts` (validated txns per-account) + `accounts_proposed` (early engine_result feedback)
- **Deferred subscription** — WS subscribes to accounts only after genesis load completes
- **Ledger-close-driven expiry** — `ledgerClosed` events trigger `expire_past_lls()` for txns past their `LastLedgerSequence`
- **Fee/reserve from WS** — `fee_base`, `txn_count`, `reserve_base`, `reserve_inc` extracted from `ledgerClosed` (eliminates RPC calls)
- **Overdue-only RPC fallback** — `periodic_finality_check` every 5s, only checks txns past their LLS
- **Deduplication** — both paths deduplicate by `(tx_hash, ledger_index)`

### Assertions Framework

- **Antithesis SDK integration** — delegates to `antithesis.assertions.always/sometimes` when SDK is available
- **Standalone fallback** — logs + tracks stats locally when SDK is unavailable
- **Transaction lifecycle helpers** — `tx_submitted()`, `tx_validated()`, `tx_rejected()`, `tx_intentionally_rejected()`
- **Invariant assertions** — `always()`, `sometimes()`, `unreachable()`, `reachable()`
- **Lifecycle signals** — `setup_complete()`, `send_event()`

### Ledger Object ID Computation

- **Deterministic index functions** — compute XRPL object IDs from transaction fields without RPC
- **Supported objects** — Offer, NFTokenOffer, Check, Escrow, Ticket, Vault, PermissionedDomain, Credential, MPTokenIssuance, NFTokenID
- **Self-contained module** — designed for eventual xrpl-py contribution

### API Endpoints (75+)

| Category | Examples |
|----------|---------|
| **Health** | `GET /health`, `GET /version` |
| **Accounts** | `GET /accounts`, `POST /accounts/create`, `GET /accounts/{id}/balances` |
| **Transaction submission** | `POST /payment`, `POST /offercreate`, `POST /vaultcreate`, `GET /random`, `GET /create/{type}` |
| **Workload control** | `POST /workload/start`, `POST /workload/stop`, `GET /workload/status` |
| **Rate control** | `GET/POST /workload/target-tps`, `GET/POST /workload/intent`, `POST /workload/toggle-type` |
| **State & monitoring** | `GET /state/summary`, `GET /state/pending`, `GET /state/failed/{code}`, `GET /state/failure-codes` |
| **DEX** | `GET /dex/metrics`, `GET /dex/pools`, `GET /dex/pools/{index}` |
| **Network** | `POST /network/reset` |

### Dashboard & HTML Pages

- **Live dashboard** (`/state/dashboard`) — auto-refreshing metrics, progress bars, transaction stats, fee info, WS event counters
- **Failure pages** — `/state/failures`, `/state/failures/{code}` with sortable tables
- **Type pages** — `/state/types/{txn_type}` with transaction lists
- **DEX page** — `/dex/amm-pools` with sortable AMM pool table
- **MPToken page** — `/state/mpt-issuances`
- **Log viewer** — `/logs/page`

### Persistence

- **SQLite store** (opt-in via `WORKLOAD_PERSIST=1`) — accounts, transactions, validations, balances. Survives restarts.
- **In-memory store** — metrics, recent validations deque, validation-by-source counters
- **Cumulative counters** — `_failure_codes`, `_tem_disabled_types` survive pending cleanup
- **Three startup modes** — SQLite hot-reload → genesis load → full init (fastest to slowest)

### CLI Commands

| Command | Description |
|---------|-------------|
| `workload run` | Start FastAPI server |
| `workload gen` | Generate testnet via generate_ledger |
| `workload compose` | Write docker-compose.yml for existing testnet |
| `workload test` | Full lifecycle: clean → gen → up → monitor → report |

### Configuration

- **`config.toml`** — accounts, gateways, currencies, transaction weights, disabled types, intent ratios, per-type params, connection settings, timeouts
- **Environment variables** — `RPC_URL`, `WS_URL`, `XRPLD_IP`, `LOG_LEVEL`, `WORKLOAD_PERSIST`
- **Runtime API** — target TPS, intent ratio, disabled types all adjustable without restart

---

## generate_ledger

### Genesis Ledger Generation

- **Accounts** — thousands to millions, with configurable balance and cryptographic algorithm (ed25519/secp256k1)
- **Gateway topology** — N gateways issuing M assets each, with configurable coverage (fraction of users with trustlines) and connectivity (fraction of gateways each user connects to)
- **Trustlines** — explicit (`acct1:acct2:currency:limit`) or random generation with currency pool
- **AMM pools** — full LP token management with auction slots, vote slots, creator directories, and proper `lsfAMMNode` flags
- **MPToken issuances** — with holders, transfer fees, metadata, and automatic `OutstandingAmount` calculation
- **Amendments** — three profiles (release/develop/custom) with per-amendment overrides
- **Fee settings** — configurable base fee, account reserve, owner reserve
- **Directory node consolidation** — merges overlapping ownership directories from trustlines and AMMs

### Cryptographic Backends

| Backend | Dependencies | Rate | Notes |
|---------|-------------|------|-------|
| PyNaCl + coincurve | libsodium, libsecp256k1 | ~25–88k/sec | Default, auto-scales to parallel |
| CuPy/CUDA | CUDA toolkit | ~280k/sec | GPU-accelerated ed25519 |
| xrpl-py | xrpl-py only | ~60–80/sec | Last-resort fallback |

- **Automatic fallback chain** — best available backend selected at runtime
- **Parallel CPU generation** — `ProcessPoolExecutor` for >50k accounts
- **GPU batch pipelining** — 50k chunks with async decoding

### xrpld Configuration Generation

- **Multi-node configs** — validators + hub node, each with UNL, voting stanzas, and peer discovery
- **Config layer system** — base TOML → environment → role (validator/node) → host overrides
- **Key generation** — in-process (xrpl-py) or Docker-based (`legleux/vkt`)
- **Amendment list** — injected from profile into `[features]` section
- **Voting settings** — fee/reserve voting for validators
- **All standard sections** — `[server]`, `[port_*]`, `[node_db]`, `[ips_fixed]`, `[validators]`, `[ssl_verify]`, `[compression]`, `[debug_logfile]`, etc.

### Docker Compose Generation

- **Multi-validator network** — `val0..valN` + `xrpld` hub node
- **Health checks** — `xrpld --silent ping` on first validator
- **Service dependencies** — validators depend on first, hub depends on first
- **Port mapping** — per-validator ports if `--expose-all-ports`
- **Volume mounts** — config dirs + genesis ledger.json
- **Custom bridge network** — `xrpl_net`

### Amendment System

| Profile | Source | Behavior |
|---------|--------|----------|
| **release** | Mainnet RPC → bundled JSON fallback | Matches mainnet-enabled amendments |
| **develop** | GitHub features.macro → env var fallback | Enables all `Supported::yes` |
| **custom** | User-provided JSON | Full manual control |

- **Per-amendment overrides** — `--enable-amendment` / `--disable-amendment` (repeatable)
- **features.macro parser** — handles `XRPL_FEATURE`, `XRPL_FIX`, `XRPL_RETIRE` macros
- **Hash computation** — `SHA512Half(name)` matching xrpld's implementation

### CLI (`gen`)

| Subcommand | Description |
|------------|-------------|
| `gen` (root) | Full pipeline: ledger + xrpld configs + docker-compose |
| `gen ledger` | Genesis ledger only (accounts, trustlines, AMMs, MPTs, amendments) |
| `gen xrpld` | Validator config files only |

Key flags: `--accounts`, `--gateways`, `--gpu`, `--amendment-profile`, `--amendment-source`, `--image`, `--expose-all-ports`, `--log-level`, `--base-fee`, `--reserve-base`, `--reserve-inc`

### Ledger Object Index Computation

- **Account indices** — `account_root_index()`, `owner_dir()`
- **Trustline indices** — `ripple_state_index()` with low/high address ordering
- **AMM indices** — `amm_index()`, `amm_account_id()`, `amm_lpt_currency()`
- **MPToken indices** — `mpt_issuance_index()`, `mptoken_index()`, `mpt_id_to_hex()`
- **30+ namespace prefixes** — matching xrpld's `LedgerNameSpace` enum

### Output Files

```
testnet/
├── ledger.json          # Genesis ledger (accountState array)
├── accounts.json        # Address/seed pairs
├── docker-compose.yml   # Multi-node docker-compose
└── volumes/
    ├── val0/xrpld.cfg   # Validator configs
    ├── val1/xrpld.cfg
    ├── ...
    └── xrpld/xrpld.cfg  # Hub node config
```
