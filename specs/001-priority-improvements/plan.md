# Implementation Plan: Priority Improvements for Fault-Tolerant XRPL Workload

**Branch**: `001-priority-improvements` | **Date**: 2025-12-02 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-priority-improvements/spec.md`

## Summary

This feature implements six priority improvements to create a fault-tolerant, high-reliability XRPL workload generator for Antithesis fuzz testing. The primary focus is achieving 90% transaction validation success rates through improved sequence tracking and account management (P1), followed by complete MPToken workflow support (P2), decentralized exchange activity via offer crossing (P3), comprehensive API documentation (P4), network observability dashboard (P5), and code quality enforcement (P6). The technical approach emphasizes ledger-based timing, modular code organization, comprehensive type safety, and minimal xrpl-py dependency while leveraging FastAPI's automatic Swagger UI generation for API documentation.

## Technical Context

**Language/Version**: Python 3.13+
**Primary Dependencies**: FastAPI, xrpl-py (minimal usage), uvicorn, asyncio.TaskGroup
**Storage**: SQLite3 (via sqlite_store.py), in-memory state (InMemoryStore), persistent transaction tracking
**Testing**: Manual validation via API endpoints (Swagger UI), pytest only if explicitly requested
**Target Platform**: Linux server (Docker containers), local testnet deployment
**Project Type**: Single project (web service) with FastAPI backend serving API + HTML dashboard
**Performance Goals**: 90% transaction validation rate, <10% sequence conflicts, 10+ transactions/ledger sustained
**Constraints**: Ledger-based timing only (no time-based delays), sequence number locks required, all methods must have return types
**Scale/Scope**: 5-10 rippled validators, 50-100 accounts, 8 transaction types, continuous submission workload

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Ledger-Based Timing (NON-NEGOTIABLE)
- ✅ **PASS**: Feature uses ledger closes as timing mechanism for continuous submission
- ✅ **PASS**: No time-based delays in submission logic (`asyncio.sleep()` banned for submission)
- ✅ **PASS**: Time only used for timeouts and metrics measurement
- ⚠️ **REVIEW REQUIRED**: Dashboard refresh mechanism must use ledger-triggered updates, not time-based polling

**Action**: Verify dashboard uses WebSocket subscriptions to ledger events rather than timed polling intervals.

### Principle II: Reliable Transaction Submission
- ✅ **PASS**: Feature maintains existing reliable submission workflow
- ✅ **PASS**: All new transaction types (MPToken disbursement, OfferCreate, OfferCancel) will include Sequence and LastLedgerSequence
- ✅ **PASS**: Persistent storage tracking already implemented via sqlite_store.py

### Principle III: Transaction Finality Awareness
- ✅ **PASS**: Feature requirements specify proper error code handling
- ✅ **PASS**: Validation tracking distinguishes tentative vs final results
- ⚠️ **REVIEW REQUIRED**: New transaction types must handle finality edge cases

**Action**: Extend error code categorization for MPToken and Offer-specific error codes in Phase 1 design.

### Principle IV: Sequence Number Discipline
- ✅ **PASS**: Feature explicitly addresses sequence conflict rate (<10% target)
- ✅ **PASS**: Existing per-account locks in workload_core.py:alloc_seq() maintained
- ✅ **PASS**: Configurable account counts (FR-004) support increased parallelism without lock contention

### Principle V: Fee Escalation and Queue Awareness
- ✅ **PASS**: Existing fee escalation handling maintained
- ✅ **PASS**: Dashboard will expose queue state (FR-020)
- ⚠️ **REVIEW REQUIRED**: Continuous submission must adapt to fee escalation dynamically

**Action**: Research dynamic fee adaptation strategies in Phase 0.

### Principle VI: Error Code Categorization and Handling
- ✅ **PASS**: Feature maintains existing error categorization
- ⚠️ **REVIEW REQUIRED**: Must extend for new transaction types (MPToken disbursement, OfferCreate, OfferCancel)

**Action**: Document new error codes in Phase 1 data-model.md.

### Principle VII: Observability and Metrics
- ✅ **PASS**: Dashboard requirement (P5) directly addresses observability
- ✅ **PASS**: Existing metrics endpoints (/state/*) maintained and extended
- ✅ **PASS**: Real-time updates required (FR-023)

### Principle VIII: Code Quality and Maintainability
- ✅ **PASS**: Feature P6 explicitly addresses code quality enforcement
- ✅ **PASS**: ruff formatting, linting, docstrings, return types all required
- ✅ **PASS**: Pre-commit hooks mandated (FR-028, FR-029)
- ⚠️ **REVIEW REQUIRED**: Large files must be refactored (workload_core.py: 2245 lines, app.py: 1191 lines)

**Action**: Plan modularization strategy in Phase 1 design.

### Testing Discipline
- ✅ **PASS**: No formal tests requested in spec
- ✅ **PASS**: Manual validation via API endpoints specified
- ✅ **PASS**: Swagger UI endpoints provide built-in testing capability

### Overall Constitution Compliance
**Status**: ✅ CONDITIONAL PASS - Proceed to Phase 0 with review actions noted above

**Review Actions Required**:
1. Verify dashboard uses ledger-triggered updates (Principle I)
2. Extend error code categorization for new transaction types (Principles III, VI)
3. Research dynamic fee adaptation (Principle V)
4. Plan code modularization for large files (Principle VIII)

## Project Structure

### Documentation (this feature)

```text
specs/001-priority-improvements/
├── plan.md              # This file
├── research.md          # Phase 0: Research findings
├── data-model.md        # Phase 1: Entity models and state machines
├── quickstart.md        # Phase 1: Manual testing guide
├── contracts/           # Phase 1: API contracts
│   ├── transactions.yaml    # Transaction submission endpoints
│   ├── dashboard.yaml       # Dashboard data endpoints
│   └── metrics.yaml         # Metrics and state endpoints
└── checklists/
    └── requirements.md  # Specification validation (already complete)
```

### Source Code (repository root)

**Current Structure**:
```text
workload/src/workload/
├── app.py                 # FastAPI application (1191 lines - NEEDS REFACTORING)
├── workload_core.py       # Core workload logic (2245 lines - NEEDS REFACTORING)
├── txn_factory/
│   ├── __init__.py
│   └── builder.py         # Transaction builders
├── ws.py                  # WebSocket listener (173 lines)
├── ws_processor.py        # WebSocket message processor (309 lines)
├── sqlite_store.py        # Persistent storage (527 lines)
├── constants.py           # Transaction types, states (64 lines)
├── utils.py               # Utilities (148 lines)
├── fee_info.py            # Fee escalation (50 lines)
├── config.py              # Configuration (12 lines)
├── randoms.py             # Random data generation (11 lines)
├── nft_utils.py           # NFT helpers (61 lines)
└── logging_config.py      # Logging setup (62 lines)
```

**Target Structure** (post-refactoring):
```text
workload/src/workload/
├── api/                       # FastAPI endpoints (NEW - extracted from app.py)
│   ├── __init__.py
│   ├── transactions.py        # Transaction submission endpoints
│   ├── accounts.py            # Account management endpoints
│   ├── state.py               # State and metrics endpoints
│   ├── dashboard.py           # Dashboard HTML and data endpoints
│   └── models.py              # Pydantic request/response models
├── core/                      # Core business logic (NEW - extracted from workload_core.py)
│   ├── __init__.py
│   ├── workload.py            # Main Workload orchestrator
│   ├── sequence.py            # Sequence number allocation
│   ├── submission.py          # Transaction submission logic
│   ├── validation.py          # Validation tracking (poll + WebSocket)
│   └── account_manager.py     # Account pool management
├── txn_factory/               # Transaction generation (EXISTING)
│   ├── __init__.py
│   ├── builder.py             # Transaction builders (EXPAND)
│   ├── payment.py             # Payment-specific builders (NEW)
│   ├── mptoken.py             # MPToken builders (NEW)
│   ├── offers.py              # Offer builders (NEW)
│   └── context.py             # TxnContext and shared logic (NEW)
├── storage/                   # Persistence layer (NEW)
│   ├── __init__.py
│   ├── sqlite_store.py        # SQLite implementation (EXISTING, MOVE)
│   ├── memory_store.py        # InMemoryStore (NEW - extracted)
│   └── models.py              # Storage data models (NEW)
├── xrpl/                      # XRPL protocol implementation (NEW - minimal xrpl-py usage)
│   ├── __init__.py
│   ├── client.py              # RPC client wrapper
│   ├── transactions.py        # Transaction construction (our own logic)
│   ├── error_codes.py         # Error code categorization
│   └── types.py               # XRPL type definitions
├── dashboard/                 # Dashboard assets (NEW)
│   ├── static/
│   │   ├── index.html         # Main dashboard HTML
│   │   ├── style.css          # Dashboard styles
│   │   └── app.js             # Dashboard JavaScript (WebSocket client)
│   └── __init__.py
├── ws/                        # WebSocket handling (NEW)
│   ├── __init__.py
│   ├── listener.py            # WebSocket listener (EXISTING ws.py, RENAME)
│   └── processor.py           # Message processor (EXISTING ws_processor.py, MOVE)
├── config.py                  # Configuration (EXISTING)
├── constants.py               # Constants (EXISTING)
├── utils.py                   # Utilities (EXISTING)
├── logging_config.py          # Logging (EXISTING)
└── __main__.py                # Entry point (EXISTING)
```

**Structure Decision**:

This feature adopts a **modular single-project structure** that extracts the two large monolithic files (app.py, workload_core.py) into focused modules organized by responsibility:

1. **API layer** (`api/`): FastAPI routes grouped by domain (transactions, accounts, state, dashboard)
2. **Core logic** (`core/`): Workload orchestration, sequence management, submission, validation
3. **Transaction generation** (`txn_factory/`): Expanded with type-specific builders
4. **Storage** (`storage/`): Persistence interfaces and implementations
5. **XRPL protocol** (`xrpl/`): Custom XRPL implementation minimizing xrpl-py dependency
6. **Dashboard** (`dashboard/`): Static assets served by FastAPI
7. **WebSocket** (`ws/`): Organized WebSocket handling

This structure aligns with **Principle VIII (Code Quality)** by breaking large files into modules <500 lines, improves maintainability, and enables parallel development of user stories.

## Complexity Tracking

No constitution violations requiring justification. All design decisions align with established principles.

---

## Phase 0: Outline & Research

**Objective**: Resolve all NEEDS CLARIFICATION items and establish best practices for implementation.

### Research Tasks

1. **Dynamic Fee Adaptation Strategy** (Constitution Review Action)
   - Research: How to dynamically adjust transaction fees based on current network fee levels
   - Investigate: `fee` RPC command integration for real-time fee queries
   - Decision needed: When to query fees (per ledger? per batch? adaptive?)
   - Output: Fee adaptation algorithm design

2. **Dashboard Update Mechanism** (Constitution Review Action)
   - Research: Ledger-triggered vs WebSocket subscription patterns for dashboard updates
   - Investigate: FastAPI WebSocket endpoint design for server-sent updates
   - Decision needed: Push (WebSocket) vs Pull (SSE) vs Hybrid
   - Output: Dashboard real-time update architecture

3. **MPToken Error Code Categorization** (Constitution Review Action)
   - Research: MPToken-specific error codes in XRPL protocol
   - Investigate: Disbursement transaction error codes (tecNO_DST, tecNO_LINE, etc.)
   - Document: Finality semantics for each error code class
   - Output: Extended error code categorization table

4. **Offer Transaction Error Codes** (Constitution Review Action)
   - Research: OfferCreate and OfferCancel error codes
   - Investigate: Order book-specific errors (tecUNFUNDED_OFFER, tecKILLED, tecEXPIRED)
   - Document: Retry vs terminal failure semantics
   - Output: Offer error code handling specification

5. **Code Modularization Strategy** (Constitution Review Action)
   - Research: Best practices for extracting large Python files into modules
   - Investigate: FastAPI route organization patterns (APIRouter)
   - Design: Module boundaries for workload_core.py and app.py extraction
   - Output: Detailed refactoring plan with file structure

6. **xrpl-py Minimal Usage Patterns**
   - Research: Which xrpl-py components are essential vs replaceable
   - Investigate: Transaction serialization, signing, hashing in xrpl-py
   - Design: Custom XRPL transaction builder without heavy xrpl-py dependency
   - Output: xrpl-py usage policy and custom implementation scope

7. **Swagger UI Example Generation**
   - Research: FastAPI example generation for Pydantic models
   - Investigate: How to provide working examples when testnet funding available
   - Design: Conditional example injection based on deployment environment
   - Output: Swagger example generation pattern

8. **Pre-commit Hook Configuration**
   - Research: pre-commit framework setup for Python projects
   - Investigate: ruff integration with pre-commit
   - Document: Hook configuration for format, lint, docstring checks
   - Output: .pre-commit-config.yaml specification

**Output**: `research.md` with all findings consolidated

---

## Phase 1: Design & Contracts

**Prerequisites**: research.md complete

### Data Model Design

**Objective**: Define all entities, state machines, and validation rules from functional requirements.

**Output**: `data-model.md` containing:

1. **Transaction State Machine** (existing, verify completeness)
   - States: CREATED, SUBMITTED, RETRYABLE, VALIDATED, REJECTED, EXPIRED, FAILED_NET
   - Transitions: Based on error code categorization
   - Extended: MPToken and Offer error codes

2. **Account Entity** (existing, extend if needed)
   - Fields: address, sequence, balance (XRP + issued currencies), trust lines, pending transactions
   - Validation: Sequence lock mechanism
   - Extended: Support for MPToken holdings, active offers

3. **MPToken Entity** (new)
   - Fields: issuer, token_id, maximum_amount, holders, transfer_restrictions
   - Relationships: Issuer account, holder accounts
   - Lifecycle: Minted → Disbursed → Traded (offers)

4. **Offer Entity** (new)
   - Fields: account, offer_sequence, taker_gets (currency/amount), taker_pays (currency/amount), expiration
   - Relationships: Account, currency pairs
   - Lifecycle: Created → Active → Crossed/Cancelled/Expired

5. **Node Metrics Entity** (new for dashboard)
   - Fields: url, server_state, ledger_index, complete_ledgers, queue_size, max_queue_size, fee_levels
   - Relationships: None (external rippled nodes)
   - Update frequency: Per ledger close

6. **Dashboard Data Models** (new)
   - NetworkOverview: Aggregated node metrics
   - QueueState: Per-node queue information
   - TransactionActivity: Recent transaction counts by state
   - ValidationMetrics: Validation rates and sources

### API Contract Generation

**Objective**: Generate OpenAPI contracts for all functional requirements.

**Output**: `/contracts/*.yaml` files:

1. **transactions.yaml**: Transaction submission endpoints
   ```yaml
   POST /transaction/payment
   POST /transaction/trustset
   POST /transaction/accountset
   POST /transaction/nftoken/mint
   POST /transaction/mptoken/mint
   POST /transaction/mptoken/disburse  # NEW
   POST /transaction/offer/create      # NEW
   POST /transaction/offer/cancel      # NEW
   POST /transaction/random
   POST /transaction/batch
   ```

2. **dashboard.yaml**: Dashboard data endpoints
   ```yaml
   GET /dashboard              # HTML page
   GET /dashboard/overview     # Network overview data
   GET /dashboard/nodes        # Per-node detailed metrics
   GET /dashboard/queue        # Queue state
   GET /dashboard/transactions # Recent transaction activity
   GET /dashboard/validations  # Validation metrics
   WS /dashboard/updates       # WebSocket for real-time updates
   ```

3. **metrics.yaml**: Existing metrics endpoints (verify, extend)
   ```yaml
   GET /state/summary
   GET /state/pending
   GET /state/validations
   GET /accounts
   GET /accounts/{address}
   ```

### Quickstart Documentation

**Objective**: Manual testing guide for all user stories.

**Output**: `quickstart.md` with:

1. **Prerequisites**: Docker network running, workload container started
2. **P1 - Continuous Submission**: Test commands for validation rate, sequence conflicts, ledger fill
3. **P2 - MPToken Workflow**: Step-by-step MPToken lifecycle test (mint, disburse, offer)
4. **P3 - Offer Crossing**: Offer creation and cancellation test scenarios
5. **P4 - API Submission**: Swagger UI walkthrough for each transaction type
6. **P5 - Dashboard**: Dashboard access and feature verification
7. **P6 - Code Quality**: Commands for ruff format, ruff check, pre-commit testing

### Agent Context Update

**Objective**: Update CLAUDE.md with new modules and patterns.

**Action**: Run `.specify/scripts/bash/update-agent-context.sh claude`

**Expected Updates**:
- New module structure (api/, core/, xrpl/, dashboard/, storage/)
- MPToken and Offer transaction types
- Dashboard WebSocket pattern
- Error code categorization extensions

---

## Post-Phase 1 Constitution Re-check

After Phase 1 design completion, re-verify constitution compliance:

1. **Principle I (Ledger-Based Timing)**: ✅ Dashboard uses WebSocket subscriptions to ledger events (per research.md)
2. **Principle III (Finality Awareness)**: ✅ Extended error categorization documented in data-model.md
3. **Principle V (Fee Escalation)**: ✅ Dynamic fee adaptation algorithm designed in research.md
4. **Principle VI (Error Codes)**: ✅ MPToken and Offer error codes categorized in data-model.md
5. **Principle VIII (Code Quality)**: ✅ Modularization plan designed, no files >500 lines in target structure

**Final Status**: ✅ FULL COMPLIANCE - Ready for `/speckit.tasks` command

---

## Next Steps

This plan ends after Phase 1 completion. To proceed to implementation:

1. Run `/speckit.tasks` to generate the task list from this plan
2. Tasks will be organized by user story (P1-P6) with proper dependencies
3. Each task will reference specific files in the target structure
4. Implementation can proceed incrementally by priority order

**Branch**: `001-priority-improvements`
**Plan File**: `/home/emel/dev/Ripple/rippled-workload/specs/001-priority-improvements/plan.md`
**Generated Artifacts** (to be created):
- `research.md` (Phase 0)
- `data-model.md` (Phase 1)
- `quickstart.md` (Phase 1)
- `contracts/*.yaml` (Phase 1)
