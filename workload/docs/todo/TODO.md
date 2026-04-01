# TODO

## Completed: Producer Stall Fix + Dashboard + WS Improvements (2026-03-18 evening)
- [x] **Producer stall bug resolved** — root cause: consumer queue drain called `record_expired()` with cascade RPC for every stale txn. Fix: `cascade=False` for queue drain expiry.
- [x] **Unified build-submit loop** — eliminated producer-consumer split. Single loop: build → sign → submit → repeat. No queue, no ledger-close gating.
- [x] **Self-healing** — `expire_past_lls()` force-expires stale pending txns when all accounts blocked.
- [x] **WS improvements** — `accounts_proposed` for early feedback, `transactions` stream for new accounts, `fee_base`/`txn_count`/`reserve` from WS ledgerClosed (eliminated 2+ RPC/ledger).
- [x] **`send_event()` for Antithesis** — structured tx lifecycle events (submitted/validated/rejected) with details.
- [x] **`unreachable()` assertions** — tefINTERNAL and tecINTERNAL trigger unreachable assertions.
- [x] **`rand_owner()`** — builders for ownership-dependent types (Vault*, Credential*, Domain*, AMMWithdraw) pick from known owners instead of random-then-filter.
- [x] **`record_expired(cascade=False)`** — skip cascade RPC for stale-gen and dedup expiry in consumer.
- [x] **Cumulative counters** — `_total_created/validated/rejected/expired` and `_type_submitted/validated` survive `_cleanup_terminal()`.
- [x] **Log rotation on startup** — `workload.log` rotated before each run.
- [x] **Log level audit** — per-txn noise to DEBUG, temDISABLED to DEBUG, full addresses everywhere.
- [x] **Dashboard** — target slider 0-1000, success rate table sorted by %, side-by-side layout, sortable error/type pages, diagnostics endpoint, ledger utilization from WS.
- [x] **test_composer scripts** — fixed endpoints, deleted broken scripts, `create_transaction` handles None gracefully.

## Completed: Branch Integration (2026-03-18)
- [x] **Assertions framework** (`assertions.py`) — centralised Antithesis SDK delegation with standalone fallback
- [x] **AntithesisRandom** in `randoms.py` — try AntithesisRandom, fallback SystemRandom
- [x] **12 new transaction types** ported from upstream branch (DelegateSet, Credentials x3, PermissionedDomains x2, Vaults x6)
- [x] **Assertion calls** wired into `record_submitted`, `record_validated`, `submit_pending` rejection paths
- [x] **Import migration** `generate_ledger.*` → `gl.*`
- [x] **SQLite persistence opt-in** via `WORKLOAD_PERSIST=1` (default: off)
- [x] **`/network/reset`** uses library API instead of shelling out to `gen auto`
- [x] **Shutdown fixes** — 3s timeout on workload_task, skip flush when no store, 30s timeout on WS wait
- [x] **Genesis user count** derived from accounts.json, not config.toml
- [x] **Fee handling** — VaultCreate/PermissionedDomainSet added to owner_reserve path
- [x] **12 POST endpoints** for new transaction types
- [x] **12 test_composer scripts** + `exercise_all_types.sh` (all 31 types)
- [x] **Deps updated** — xrpl-py 4.5.0, antithesis 0.2.0, pynacl 1.6.2
- [x] **Port parameter deviations** documented in `workload/docs/port-parameter-deviations.md`
- [x] **Consumer dedup** — one submission per account per batch (prevents parallel seq submission)
- [x] **Producer crash protection** — top-level try/except with `PRODUCER CRASH (recovering)` log
- [x] **Producer build protection** — `build_txn_dict` + `from_xrpl` wrapped in try/except
- [x] **File logging** — `workload.log` at DEBUG, console at INFO, rotating 50MB × 5
- [x] **Invariants doc** — `workload/docs/invariants.md` with 12 invariants

## P0: WS Connection Stability — Drops Every 1-2 Minutes Under Load
**Discovered:** 2026-04-01 lifecycle test run (55% success rate, 1,818 expired in 120s)

The WS connection drops every 1-2 minutes when subscribed to 1005 accounts. Each reconnect creates a gap where validated txns aren't seen, causing mass LLS expiry (150-515 txns per event) and account blocking. The seq fixes are working; this is now the dominant failure mode.

**Unknown: is this our WS module (websockets library, asyncio backpressure, read loop stalling) or xrpld dropping the connection?** Hopefully it's us — xrpld's default `websocket_max_connections` is 500 (we only use 1 connection with 1005 subscribed accounts), so xrpld should handle this fine.

- [ ] **Determine which side drops the connection** — check xrpld logs for WS close reason. Add close code + reason logging to our `ws.py` reconnect handler. Is our read loop falling behind? Is the websockets library buffering and then disconnecting?
- [ ] **Add dedicated WS accounts log file** — `workload.ws.accounts` logger exists but writes to shared `workload.log`. Add a file handler in `logging_config.py` for isolated analysis.
- [ ] **Reduce reconnect gap impact** — faster resubscription, or buffer/replay missed events during reconnect window.
- [ ] **If it's us**: profile the event loop during high throughput — are we blocking the WS read task? Is `ws_processor` keeping up with the event queue?
- [ ] **If it's xrpld**: consider splitting into multiple WS connections with smaller account sets, or file an xrpld issue.

**Evidence from 2026-04-01 run:**
- 7 WS disconnects in 16 minutes
- `expire_past_lls` force-expiring 150-515 txns per event
- 876/1004 accounts blocked at end of 120s monitoring window
- WS validation working when connected: 8,448 by WS, 0 by poll

---

## P0: Sequence Management & Submission Fix
**Full plan:** `workload/docs/todo/seq-management-fix-plan.md` (2026-03-31)

Four root causes identified for persistent tefPAST_SEQ failures:
- [x] **Phase 1 (Critical):** `expire_past_lls()` resets `next_seq=None` + bumps `generation` on affected accounts (2026-03-31)
- [x] **Phase 2:** Generation guard in `submit_pending()` — skips stale pre-signed txns (2026-03-31)
- [x] **Phase 3:** `max_pending_per_account` locked to 1, dashboard slider removed, POST returns 400 (2026-03-31)
- [x] **Phase 4:** `record_validated()` wraps `next_seq` update in `async with rec.lock` (2026-03-31)
- [ ] **Phase 5 (needs re-review):** Fee strategy — cache-and-react vs. WS-derived escalated fee

Previously fixed (still valid):
- [x] ~~`next_seq` sync in `record_validated`~~ DONE (2026-03-27)
- [x] ~~`tem` sequence leak~~ DONE (2026-03-27)
- [x] ~~Aggressive `expire_past_lls`~~ DONE (2026-03-27)



## P0: Code Health — Dead Code Cleanup, Modularization, Python 3.13+

Top priority. The codebase works but has accumulated dead code, debug artifacts, and rough edges. Clean it up before adding features.

**Target**: Modern Python 3.13+ only. No backwards compatibility. Use StrEnum, match statements, type parameter syntax (`type Foo = ...`), `asyncio.TaskGroup`, etc. wherever appropriate.

**Flesh out test framework**
- [ ] Backport tests for `test_cmd.py` — written before TDD was adopted, needs unit tests for step functions, report generation, snapshot logic, and implication chain (`--up` → `--gen` → `--clean`)
- [ ] Test that auto-generated txns can survive fee escalation. Initially it'll be ok to mark as `xfail` until the feature is actually implemented.
- [ ] Test the txn lifecycle. How to actually do that?


## Features
- [ ] Query endpoints for tracked objects (all have internal tracking, none have GET endpoints):
  - `GET /state/nfts` — NFTs (id → owner)
  - `GET /state/offers` — DEX offers
  - `GET /state/tickets` — tickets (account → set of seqs)
  - `GET /state/checks` — checks
  - `GET /state/escrows` — escrows
  - `GET /state/credentials` — credentials
  - `GET /state/vaults` — vaults
  - `GET /state/domains` — permissioned domains
- [ ] Text box/field/separate page that allows us to just submit arbitrary txn JSON data.
- [ ] Ability to send a txn to a _specific_ host — for when we have more than one p2p node defined or just to submit txns directly to the validators.
- [ ] Standalone mode functionality
- [ ] Implement assertions for our project that can *optionally* be overidden as assertions from the
   Antithesis SDK if this project optionally uses it.
### Dead Code Removal
- [ ] Redefine the way we aggregate groups of txns to be submitted to not use the term "batch" in the source (or docs) to avoid confusion with the new Batch txn type and the xrpld batch submission feature.
- [x] ~~16 dead methods removed from workload_core.py, app.py, sqlite_store.py, utils.py~~

### Bug Fixes
- [ ] `config.toml`: `funding_seed = false` is never read (dead config key)
- [x] ~~workload_started, ws.py typos, dashboard import guard, logging config~~

### Modularization
- [x] ~~Split `app.py` (2,220 lines) into 14 files: routers/, bootstrap.py, schemas.py, dependencies.py~~ (2026-03-25)
- [x] ~~Extract 27 validation hooks from `workload_core.py` to `validation_hooks.py`~~ (2026-03-26)
- [x] ~~Remove dead `init_participants` + 4 helpers (~930 lines) from `workload_core.py`~~ (2026-03-26)
- [x] ~~Add `/version` endpoint using `importlib.metadata` (matches generate_ledger pattern)~~ (2026-03-26)
- [ ] **P0: Reimplement `init_participants`** — organic genesis init for non-pre-genesis environments. See `workload/docs/todo/reimplement_init_participants.md`
- [ ] **Next refactor target: submission pipeline** — `submit_pending` (CC D/28, cognitive 51) + `build_sign_and_track` (CC C/15, cognitive 19). Seq fix plan phases 1-2 will simplify these.
- [ ] Move `workload_running`, `workload_stop_event`, `workload_task`, `workload_stats` from module-level globals onto `app.state`
- [ ] Extract constants: hardcoded WS port (6006)
- [ ] `sqlite_store.by_type` always returns `{}` — implement or remove
- [ ] Delete old `builder.py` (orphaned, no imports reference it)

### New Transaction Type Follow-ups
- [ ] Enable DelegateSet when `PermissionDelegationV1_1` is marked `Supported::yes` in xrpld (currently `no`)
- [ ] Consider baking MPToken issuances into genesis via `gl.ledger.LedgerConfig.mpt_issuances` for faster cold start
- [ ] gen auto metadata sidecar file — write gateway count alongside accounts.json so workload doesn't need config.toml for genesis loading
- [ ] Intentionally bad txns — configurable knob to submit invalid/malformed transactions for testing rejection paths
- [x] ~~Enable SingleAssetVault, fix alloc_seq log~~

### Python 3.13+ Modernization
- [ ] Audit for opportunities: type parameter syntax, match statements, StrEnum patterns
- [x] ~~Package & CI pipeline~~ DONE (2026-04-01) — tests.yml (lint → complexity → test matrix → build → docker), publish.yml (v* tags)
- [x] ~~Pre-commit linting and formatting with ruff~~
- [x] ~~Conventional commits + pre-push tests in pre-commit~~ DONE (2026-04-01)
- [x] ~~`--version` CLI flag~~ DONE (2026-04-01)
- [x] ~~Complexity reporting (complexipy + radon) in CI~~ DONE (2026-04-01)
- [ ] Configure OIDC trusted publishing on TestPyPI/PyPI and uncomment publish job

---

## generate_ledger
- [ ] Redesign amendment profiles: `mainnet`/`testnet`/`devnet` (copy from real networks) + `custom` (any file path). Current "release"/"develop" distinction is meaningless.
- [x] ~~`--amendment-source` auto-infers profile, default image `rippleci/xrpld:develop`, default accounts/gateways/coverage~~

---

## P2: Public Network Support — Run on Devnet/Testnet

The workload must easily target the public XRPL devnet or testnet, not just local docker networks.

### Requirements
- [ ] Configurable target network (local, devnet, testnet) — single config switch or env var
- [ ] Faucet integration for account funding on public networks (no genesis account)
- [ ] No hardcoded docker hostnames or local IPs in the hot path
- [ ] Handle public network constraints: rate limits, no `--ledgerfile`, existing ledger state
- [ ] Document the public network workflow (connect, fund, run, recover)

---

## P0: XRP Accounting — Zero-Loss Fund Recovery

See also: [shutdown_procedure.md](shutdown_procedure.md) — current shutdown behavior and open questions.

When workload completes (or on Ctrl-C / crash), all XRP should be returned to the funding source. The only XRP permanently consumed should be:
- Transaction fees (burned)
- Account reserves (if accounts not deleted)

### Requirements
- [ ] Track every XRP outflow from funding account with ledger-level precision
- [ ] Track reserves locked in created accounts
- [ ] On shutdown: sweep all user account balances back to funding account
- [ ] On shutdown: sweep all gateway account balances back to funding account
- [ ] On shutdown: close trust lines, cancel offers, withdraw AMM LP tokens before sweeping
- [ ] Report: total XRP disbursed, total recovered, total burned (fees), total locked (reserves), delta
- [ ] Handle Ctrl-C gracefully (signal handler triggers sweep before exit)
- [ ] Handle crash recovery (on next startup, detect orphaned accounts from state.db and sweep them)

### Stretch Goal
- [ ] Delete created accounts (`AccountDelete`) to reclaim reserves back to funding source
- [ ] Requires waiting 256 ledgers after last transaction (XRPL rule)

---

## P1: Architectural Improvements

### Transaction Finality Assurance — When to Stop Waiting

Currently we stop waiting for a tx when either (a) the WS stream reports it validated, or (b) its `LastLedgerSequence` expires and the finality checker marks it `EXPIRED`. `FAILED_NET` txns (submission timeout / connection drop) stay locked until LLS for safety — the tx may have reached xrpld's queue even though we got no response.

Options to investigate for faster/more reliable finality signaling:

- **WebSocket tx stream** (already active): `ws_processor` fires `record_validated()` on hash match. Latency = one ledger close after the tx validates. Should be sufficient for most cases.
- **xrpld gRPC stream**: Connect directly to validator nodes for sub-ledger event delivery. Faster than WS for high-throughput scenarios. User has an existing script to connect to the gRPC stream — evaluate whether the latency improvement justifies the added infrastructure.
- **Validator log parsing**: Fragile — couples to log format, hard to maintain. Not recommended.
- **Periodic RPC poll** (`Tx` lookup): Current fallback via `periodic_finality_check`. Works but is ~5s delayed and doesn't scale well at high txn volume.

Long-term goal: a single, reliable event source that tells us definitively "tx X is terminal" so accounts can be freed immediately without waiting for LLS expiry as a safety margin.

#### Root Cause Analysis (2026-03-31 verification run)

With intent=0% (all valid), 4,834 tefPAST_SEQ and 14,310 expired txns in ~850 ledgers. The self-heal `expire_past_lls` path never triggered (0 times), so the seq-fix Phases 1+2 are defense-in-depth for a scenario that doesn't arise under normal load. The tefPAST_SEQ errors come from the normal submission path.

**Root cause: `periodic_finality_check` bottleneck.** Every 5s the poller does sequential RPC `Tx` lookups for ~1000 pending txns. This takes much longer than 5s. Meanwhile `check_finality()` expires txns past LLS+2 — but some of those txns DID validate on-chain. The poller just hasn't reached them yet. When we submit the next txn for that account, the ledger has already advanced the sequence → tefPAST_SEQ.

WS `transactions` stream is working (121K `validated_matched`) but misses ~32% of validations (found only by poll). Possible causes: xrpld drops WS events under high throughput, or our asyncio event loop can't process them fast enough.

**Candidate fixes (ordered by impact):**

1. **Parallelize `periodic_finality_check`** — use `asyncio.TaskGroup` to batch RPC `Tx` lookups instead of sequential iteration. ~1000 parallel lookups should complete in <1s vs current ~30s+.
2. **Subscribe to `accounts` stream** (not just `accounts_proposed`) — the `accounts` WS subscription delivers validated txns for specific accounts. Currently we rely on the global `transactions` stream for validated events + `accounts_proposed` for early engine_result feedback. Using `accounts` would give us a per-account validated feed.  Downside: 1000 subscribed accounts is fine — xrpld handles it. The subscription persists, so no need to resub per-txn.
3. **Don't expire in `check_finality` after RPC error** — currently if the `Tx` lookup throws (line 1390), execution falls through to the LLS expiry check (line 1393). A transient RPC error → false expiry. Guard: only expire if the `Tx` lookup succeeded but returned not-validated.
4. **Increase `HORIZON`** — currently 15 ledgers (~45-60s). A larger horizon gives more time for WS/poll to catch validations, at the cost of slower account recovery on genuine failures.

### Submission Throughput
- [x] ~~Parallelize build loop~~ DONE (2026-03-24)
- [ ] Simplify `submit_pending` (CC D/28, cognitive 51) — worst remaining function in workload_core.py
- [ ] Re-implement heartbeat — periodic signal that the workload is alive and submitting
- [x] ~~Unified build-submit loop~~

### Object Tracking After Validation
- [ ] Track minted NFTs (NFTokenID) per account after NFTokenMint validation
- [ ] Track created offers (OfferSequence) per account after OfferCreate validation
- [ ] This unblocks: NFTokenBurn, NFTokenCreateOffer/CancelOffer/AcceptOffer, OfferCancel in continuous mode
- [x] ~~Track MPToken issuance IDs after MPTokenIssuanceCreate validation~~

### State Reload Performance
- [ ] `load_state_from_store()` takes ~38s for ~3K wallets — will not scale to longer runs with 10K+ accounts
- [ ] Profile: is it SQLite reads, `Wallet.from_seed()` deserialization, or `_record_for()` lock creation?
- [ ] Consider bulk loading wallets without per-wallet crypto key derivation on startup (defer to first use)

### AMM Improvements
- [ ] Bump genesis AMM pool count — only ~4 pools with 4 gateways × 4 currencies. Need more currency pairs in generate_ledger.
- [ ] Prevent tecDUPLICATE AMMCreate — builder should check registry and only create new asset pairs
- [ ] AMM metrics dashboard — separate page with pool details, LP holder counts, deposit/withdraw ratios, TVL per pool
- [ ] Persist AMM pool registry to SQLite (currently lost on hot-reload)
- [ ] Parallelize `poll_dex_metrics` with `asyncio.gather` (112 sequential RPC calls)
- [ ] Fix pool discovery: reduce IOU/IOU search space or read from generation output
- [x] ~~LP token holder tracking, AMMWithdraw rand_owner~~

### Shutdown / Flush Performance
- [x] ~~bulk_upsert, stop before flush, second Ctrl-C skips flush~~

---

## P2: Dashboard & UI

- [ ] Pie chart of txn types by volume
- [ ] Match filter button colors to stream colors
- [ ] Interactive txn buttons — click to submit, tag and track the specific txn
- [ ] Book depth visualization for asset pairs
- [x] ~~Target slider, success rate table, side-by-side layout, clickable types/errors, disabled labels, cumulative counters, diagnostics endpoint, ledger utilization from WS, explorer embed~~
- [x] ~~WS dropdown fallback, TPS slider cooldown, cumulative failure codes, Ledger Stream reorder, temDISABLED accuracy~~ (2026-03-26)

---

## P2: Observability & Metrics Export

- [ ] Add Prometheus-compatible metrics export to `scripts/ledger_monitor.py` — the `LedgerClose` dataclass and `CadenceStats` are already structured for it. Expose a `/metrics` endpoint or write to a Prometheus pushgateway.
- [ ] Consider `prometheus_client` Python package for the workload itself — expose txn rates, pending counts, validation latency as Prometheus gauges/counters

---

## P2: Test Composer

- [ ] Flesh out infra — complex interactions need setup before fuzzing starts
- [ ] Sentinel variable each module sets before fuzzing can begin

### Scenarios
- [ ] **Memecoin Drop** — MPToken release scenario with pre-lifecycle setup
- [ ] **Out of Sync Update** — some nodes update version much sooner than the rest
- [ ] **UNL validator add/delete** — add and remove validators from UNL during a run
- [ ] **Rolling upgrade** — upgrade N validators during run, verify consensus continues

---

## P3: Network Stress Testing

- [ ] Overlapping UNLs (mess with consensus)
- [ ] Integrate sidecar monitoring into the workload process

---

## P3: Documentation

- [ ] Document the transaction lifecycle end-to-end (context → pre-flight → in-flight → terminal state)
- [ ] Explain zero-sum property: no txn can be lost
- [ ] Document running on a pre-production feature branch (e.g. smart-escrow)
- [x] ~~Deprecate init_participants~~ — removed entirely (2026-03-26), design doc at `reimplement_init_participants.md`

---

## Open Questions

- What exactly happens at a flag ledger? (examine ledgers 256 & 257)
- Is the per-account xrpld queue limit still 10 (as defined in xrpld source)? (`config.toml` TODO)

---

## Historical Reference Docs

- **`workload/XRPL_RELIABLE_SUBMISSION.md`** — Audit of our implementation against the XRPL reliable submission best practices. Gaps 1 (sequence collision) and 3 (tec code distinction) are now resolved; line numbers are stale. Keep for reference.

## Bugs

- [ ] Logs page doesn't work
- [ ] Batch page doesn't work

1. Flesh out a way to implement assertions for our project that can *optionally* be overidden as assertions from the
   Antithesis SDK if this project optionally uses it.
