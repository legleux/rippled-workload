# TODO

## P00: Absolutely do this first thing next session
Flesh out test framework
- [ ] Test that auto-generated txns can survive fee escalation. Initially it'll be ok to mark as `xfail` until the feature is actually implemented.
- [ ] Test the txn lifecycle. How to actually do that?
- [ ]
## P0: Code Health — Dead Code Cleanup, Modularization, Python 3.13+

Top priority. The codebase works but has accumulated dead code, debug artifacts, and rough edges. Clean it up before adding features.

**Target**: Modern Python 3.13+ only. No backwards compatibility. Use StrEnum, match statements, type parameter syntax (`type Foo = ...`), `asyncio.TaskGroup`, etc. wherever appropriate.

## Features
- [ ] Dashboard page links to "DEX" data. Start off with just a lis of open offers on IOUs from/to/price. very basic
- [ ] Text box/field/separate page that allows us to  just submit arbitrary txn JSON data.
- [ ] Ability to send a txn to a _specific_ host .for when we have more than one p2p node defined or just to submit txns directory to the validators. Should be able to translate that payload in such a wway that it can put the txn on the wire via JSON-RPC or WS with the user only needing to specify which API to use.
- [ ] Standalone mode functionality
- [ ] Dev/testnet connection

### Dead Code Removal
- [x] `workload_core.py`: Remove dead `_post()` method — DONE (had SyntaxWarning: return in finally)
- [x] `workload_core.py`: Remove dead `validator_state()` method — DONE (malformed docker inspect)
- [ ] `workload_core.py`: Remove 9 additional dead methods (see [code-audit-report.md](../code-audit-report.md) P1 #11)
- [ ] `workload_core.py`: Remove duplicate `logging.basicConfig()` at module level
- [ ] `workload_core.py`: Fix duplicate `log` variable assignment (workload vs workload.core)
- [ ] `workload_core.py`: Remove `AccountSet: pass` dead branch in `submit_pending()`
- [ ] `app.py`: Remove duplicated import block at top (lines 1-18)
- [ ] `app.py`: Remove `print("Submit result:", res)` debug artifact in `debug_fund()`
- [ ] `utils.py`: Delete or gut — sync-era leftover, `check_validator_proposing()` not called anywhere
- [ ] Redefine the way we aggregate groups of txns to be submitted to not use the term "batch" in the source (or docs) to avoid confusion with the new Batch txn type and the rippled batch submission feature.

### Bug Fixes
- [ ] `_workload_started` is checked in `ws_processor.py` but never set → Antithesis assertion silently dead
- [ ] `config.toml`: `[logging.handlers.file]` has no `filename` key → would crash if activated
- [ ] `config.toml`: `funding_seed = false` is never read (dead config key)
- [ ] `ws.py`: `callable` → `Callable` (capital C) type hint
- [ ] `ws.py`: `steams_string` typo

### Modularization
- [ ] Move `workload_running`, `workload_stop_event`, `workload_task` from module-level globals onto `app.state`
- [ ] Extract constants: queue maxsize (1000), hardcoded WS port (6006), OPEN_STATES set
- [ ] `sqlite_store.by_type` always returns `{}` — implement or remove

### Python 3.13+ Modernization
- [ ] Audit for opportunities: type parameter syntax, match statements, StrEnum patterns
- [ ] Pre-commit linting and formatting with ruff
- [ ] Package & CI pipeline

---

## P0: Public Network Support — Run on Devnet/Testnet

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

### Ledger-Close Event Bridge
- [ ] Bridge WS processor ledger_closed events to the workload submission loop
- [ ] Eliminate the three `asyncio.sleep(0.5)` polling loops in `continuous_workload()`
- [ ] Workload loop should `await` a ledger-close signal, not poll

### Object Tracking After Validation
- [ ] Track minted NFTs (NFTokenID) per account after NFTokenMint validation
- [ ] Track created offers (OfferSequence) per account after OfferCreate validation
- [ ] Track MPToken issuance IDs after MPTokenIssuanceCreate validation
- [ ] This unblocks: NFTokenBurn, NFTokenCreateOffer/CancelOffer/AcceptOffer, OfferCancel in continuous mode

### State Reload Performance
- [ ] `load_state_from_store()` takes ~38s for ~3K wallets — will not scale to longer runs with 10K+ accounts
- [ ] Profile: is it SQLite reads, `Wallet.from_seed()` deserialization, or `_record_for()` lock creation?
- [ ] Consider bulk loading wallets without per-wallet crypto key derivation on startup (defer to first use)
- [ ] Consider caching derived wallet objects in SQLite (store public/private key bytes, not just seed)

### AMM Improvements
- [ ] Persist AMM pool registry to SQLite (currently lost on hot-reload)
- [ ] Parallelize `poll_dex_metrics` with `asyncio.gather` (112 sequential RPC calls)
- [ ] Track LP token holders for smarter AMMWithdraw targeting
- [ ] Fix pool discovery: reduce IOU/IOU search space or read from generation output

---

## P2: Dashboard & UI

- [ ] Pie chart of txn types by volume
- [x] Color-code MPT and NFT txn types separately in the WS terminal — DONE (grouped by family)
- [x] Group related txn types in columns — DONE (Core, DEX, AMM, NFT, MPT, Other)
- [x] Transaction Control pane with runtime enable/disable — DONE (group toggles + config-disabled indicators)
- [x] Fill slider and target-txns control moved into Transaction Control pane — DONE
- [x] Clickable error codes in Top Failures → detail page — DONE (`/state/failed/{code}/page`)
- [x] Explorer embed cropped to ledger list only — DONE (see [explorer-embed-proposal.md](../explorer-embed-proposal.md) for upstream fix)
- [ ] Don't abbreviate addresses in the terminal validation stream
- [ ] Match filter button colors to stream colors
- [ ] Interactive txn buttons — click to submit, tag and track the specific txn
- [ ] Book depth visualization for asset pairs

---

## P2: Observability & Metrics Export

- [ ] Add Prometheus-compatible metrics export to `scripts/ledger_monitor.py` — the `LedgerClose` dataclass and `CadenceStats` are already structured for it. Expose a `/metrics` endpoint or write to a Prometheus pushgateway.
- [ ] Consider `prometheus_client` Python package for the workload itself — expose txn rates, pending counts, validation latency as Prometheus gauges/counters

---

## P2: Test Composer

- [ ] Flesh out infra — complex interactions need setup before fuzzing starts
- [ ] Sentinel variable each module sets before fuzzing can begin

### Scenarios
- [ ] **Out of Sync Update** — some nodes update version much sooner than the rest
- [ ] **Memecoin Drop** — MPToken release scenario with pre-lifecycle setup

---

## P3: Network Stress Testing

- [ ] Allow specifying which node receives a transaction
- [ ] Overlapping UNLs (mess with consensus)
- [ ] Start with M-of-N validators running, bring up more during the run
- [ ] Integrate sidecar monitoring into the workload process

---

## P3: Documentation

- [ ] Document the transaction lifecycle end-to-end (context → pre-flight → in-flight → terminal state)
- [ ] Explain zero-sum property: no txn can be lost
- [ ] Document running on a pre-production feature branch (e.g. smart-escrow)
- [ ] Deprecate (don't remove) the init_participants setup stages — genesis load is the default path now

---

## Open Questions

- What exactly happens at a flag ledger? (examine ledgers 256 & 257)
- Is the per-account rippled queue limit still 10 (as defined in xrpld source)? (`config.toml` TODO)
