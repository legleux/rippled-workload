# TODO

## P00: Absolutely do this first thing next session
Flesh out test framework
- [ ] Test that auto-generated txns can survive fee escalation. Initially it'll be ok to mark as `xfail` until the feature is actually implemented.
- [ ] Test the txn lifecycle. How to actually do that?

## P0: Code Health — Dead Code Cleanup, Modularization, Python 3.13+

Top priority. The codebase works but has accumulated dead code, debug artifacts, and rough edges. Clean it up before adding features.

**Target**: Modern Python 3.13+ only. No backwards compatibility. Use StrEnum, match statements, type parameter syntax (`type Foo = ...`), `asyncio.TaskGroup`, etc. wherever appropriate.

## Features
- [ ] Dashboard page links to "DEX" data. Start off with just a list of open offers on IOUs from/to/price. very basic
- [ ] Text box/field/separate page that allows us to just submit arbitrary txn JSON data.
- [ ] Ability to send a txn to a _specific_ host — for when we have more than one p2p node defined or just to submit txns directly to the validators. Should be able to translate that payload in such a way that it can put the txn on the wire via JSON-RPC or WS with the user only needing to specify which API to use.
- [ ] Standalone mode functionality
- [ ] Dev/testnet connection

### Dead Code Removal
- [x] `workload_core.py`: Remove dead `_post()` method — DONE
- [x] `workload_core.py`: Remove dead `validator_state()` method — DONE
- [x] `workload_core.py`: Remove 11 dead methods: `debug_last_tx`, `_update_account_balances`, `log_validation`, `submit_signed_tx_blobs`, `_is_account_active`, `_ensure_funded`, `_acctset_flags`, `wait_for_validation`, `bootstrap_gateway`, `_apply_gateway_flags`, `_establish_trust_lines`, `_distribute_initial_tokens` — DONE
- [x] `workload_core.py`: Remove additional dead methods: `_get_balance`, `get_accounts_with_pending_txns`, `wait_until_validated`, `snapshot_finalized` + constant `PER_TX_TIMEOUT` — DONE
- [x] `workload_core.py`: Remove duplicate `logging.basicConfig()` at module level — DONE
- [x] `workload_core.py`: Remove `import multiprocessing` / `num_cpus` dead import — DONE
- [x] `workload_core.py`: Remove `_fee_cache` and `_fee_lock` (abandoned caching infrastructure) — DONE
- [x] `workload_core.py`: Remove `Store(Protocol)` — dead and broken, never used as an interface — DONE
- [x] `workload_core.py`: Remove `InMemoryStore.update_record` dead method — DONE
- [x] `app.py`: Remove duplicated import block — DONE
- [x] `app.py`: Remove `print("Submit result:", res)` debug artifact in `debug_fund()` — DONE
- [x] `app.py`: Remove `debug=True` on FastAPI app — DONE
- [x] `app.py`: Remove `_dump_tasks`, `app.state.tg`, `app.state.ws_stop_event` — DONE
- [x] `app.py`: Remove dead `PaymentReq` model (created wallet at import time) — DONE
- [x] `sqlite_store.py`: Remove 6 dead methods mirroring InMemoryStore (`mark`, `rekey`, `find_by_state`, `get`, `update_record`, `all_records`) — DONE
- [x] `utils.py`: Deleted — sync-era leftover, nothing imported it — DONE
- [ ] Redefine the way we aggregate groups of txns to be submitted to not use the term "batch" in the source (or docs) to avoid confusion with the new Batch txn type and the rippled batch submission feature.

### Bug Fixes
- [x] `_workload_started` / `workload_started` is checked in `ws_processor.py` but never set → Antithesis assertion silently dead — DONE (now set in `start_workload`/`stop_workload`)
- [x] `ws.py`: `callable` → `Callable` (capital C) type hint — DONE
- [x] `ws.py`: `steams_string` typo → `streams_string` — DONE
- [x] `state_dashboard`: `generate_ledger` import crashes if package not installed — DONE (guarded with try/except)
- [ ] `config.toml`: `[logging.handlers.file]` has no `filename` key → would crash if activated
- [ ] `config.toml`: `funding_seed = false` is never read (dead config key)

### Type / Import Hygiene
- [x] `TERMINAL_STATE` moved from `workload_core.py` to `constants.py` — DONE
- [x] `PENDING_STATES` / `OPEN_STATES` deduplicated into `C.PENDING_STATES` in `constants.py` — DONE
- [x] `ValidationSrc` and `ValidationRecord` extracted into `validation.py` — breaks circular `sqlite_store` → `workload_core` import — DONE
- [x] `SQLiteStore` import in `workload_core.py` promoted to top-level (was deferred in 3 places) — DONE
- [x] `persistent_store` retyped from `Store | None` to `SQLiteStore | None` — DONE

### Modularization
- [ ] Move `workload_running`, `workload_stop_event`, `workload_task`, `workload_stats` from module-level globals onto `app.state`
- [ ] Extract constants: queue maxsize (1000), hardcoded WS port (6006)
- [ ] `sqlite_store.by_type` always returns `{}` — implement or remove

### Python 3.13+ Modernization
- [ ] Audit for opportunities: type parameter syntax, match statements, StrEnum patterns
- [x] Pre-commit linting and formatting with ruff — DONE (pre-commit hooks configured)
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

### Transaction Finality Assurance — When to Stop Waiting

Currently we stop waiting for a tx when either (a) the WS stream reports it validated, or (b) its `LastLedgerSequence` expires and the finality checker marks it `EXPIRED`. `FAILED_NET` txns (submission timeout / connection drop) stay locked until LLS for safety — the tx may have reached rippled's queue even though we got no response.

Options to investigate for faster/more reliable finality signaling:

- **WebSocket tx stream** (already active): `ws_processor` fires `record_validated()` on hash match. Latency = one ledger close after the tx validates. Should be sufficient for most cases.
- **rippled gRPC stream**: Connect directly to validator nodes for sub-ledger event delivery. Faster than WS for high-throughput scenarios. User has an existing script to connect to the gRPC stream — evaluate whether the latency improvement justifies the added infrastructure.
- **Validator log parsing**: Fragile — couples to log format, hard to maintain. Not recommended.
- **Periodic RPC poll** (`Tx` lookup): Current fallback via `periodic_finality_check`. Works but is ~5s delayed and doesn't scale well at high txn volume.

Long-term goal: a single, reliable event source that tells us definitively "tx X is terminal" so accounts can be freed immediately without waiting for LLS expiry as a safety margin.

### Ledger-Close Event Bridge
- [ ] Bridge WS processor ledger_closed events to the workload submission loop
- [ ] Eliminate the `asyncio.sleep` polling loops in `continuous_workload()`
- [ ] Workload loop should `await` a ledger-close signal, not poll

### Object Tracking After Validation
- [ ] Track minted NFTs (NFTokenID) per account after NFTokenMint validation
- [ ] Track created offers (OfferSequence) per account after OfferCreate validation
- [ ] This unblocks: NFTokenBurn, NFTokenCreateOffer/CancelOffer/AcceptOffer, OfferCancel in continuous mode
- [x] Track MPToken issuance IDs after MPTokenIssuanceCreate validation — DONE

### State Reload Performance
- [ ] `load_state_from_store()` takes ~38s for ~3K wallets — will not scale to longer runs with 10K+ accounts
- [ ] Profile: is it SQLite reads, `Wallet.from_seed()` deserialization, or `_record_for()` lock creation?
- [ ] Consider bulk loading wallets without per-wallet crypto key derivation on startup (defer to first use)

### AMM Improvements
- [ ] Persist AMM pool registry to SQLite (currently lost on hot-reload)
- [ ] Parallelize `poll_dex_metrics` with `asyncio.gather` (112 sequential RPC calls)
- [x] Track LP token holders for smarter AMMWithdraw/AMMDeposit targeting — DONE (`lp_holders` per pool)
- [ ] Fix pool discovery: reduce IOU/IOU search space or read from generation output

### Shutdown / Flush Performance
- [x] `flush_to_persistent_store` now uses `SQLiteStore.bulk_upsert()` — one connection, one commit — DONE (was 2+ minutes for 5k records, now near-instant)
- [x] Shutdown now stops workload before flushing — DONE
- [x] Second Ctrl-C skips flush — DONE

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

---

## Historical Reference Docs

- **`workload/XRPL_RELIABLE_SUBMISSION.md`** — Audit of our implementation against the XRPL reliable submission best practices. Gaps 1 (sequence collision) and 3 (tec code distinction) are now resolved; line numbers are stale. Keep for reference.
