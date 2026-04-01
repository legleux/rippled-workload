# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-01

### Added
- 38 transaction builders across 10 builder modules (payment, dex, nft, mptoken, vault, credential, domain, batch, check, escrow)
- Modular `txn_factory` with registry pattern, eligibility predicates, and type-first composition
- Intentionally invalid transaction system with 14 type-specific tainter groups
- Dual-path validation tracking: WebSocket `accounts` + `accounts_proposed` streams, RPC polling fallback
- Ledger-close-driven `LastLedgerSequence` expiry
- Token-bucket TPS rate limiter with runtime API control
- Sequence pre-warming (`warm_sequences()`) on startup
- Self-healing via `expire_past_lls()` with generation counter to invalidate stale txns
- Deterministic ledger object ID computation (`ledger_objects.py`) for 10 object types
- Assertions framework with Antithesis SDK integration and standalone fallback
- FastAPI app with 75+ endpoints: submission, monitoring, rate control, state inspection
- Live HTML dashboard with auto-refresh, sortable tables, failure/type detail pages
- `workload test` CLI for lifecycle orchestration (clean -> gen -> up -> monitor -> report)
- `workload gen` CLI bridging config.toml to generate_ledger library
- SQLite persistence (opt-in via `WORKLOAD_PERSIST=1`)
- Pre-commit hooks (ruff, trailing whitespace, TOML/YAML checks, debug statements)
- pytest suite with GET endpoint coverage
- Testnet generation via generate_ledger integration

### Changed
- Renamed all `rippled` references to `xrpld` (config keys, variable names, env vars, comments, docs)
- `XRPLD_IP` is now the primary env var (`RIPPLED_IP` accepted as fallback)
- Config section `[rippled]` renamed to `[xrpld]`
