# Tasks: Priority Improvements for Fault-Tolerant XRPL Workload

**Input**: Design documents from `/specs/001-priority-improvements/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Not requested in feature specification - manual validation via API endpoints only

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**Status note (2026-03-17)**: Implementation diverged from the spec's big-bang restructure approach.
Features were built in-place in the existing files rather than extracted into new module directories.
Tasks are checked off based on *functional equivalence* — the capability exists, even if the file
path differs from what was originally planned. Markings:
- `[x]` — done (possibly at a different path than planned; note says where)
- `[N/A]` — pure file-move / directory-creation task that no longer applies
- `[ ]` — not yet done
See "Reassessment" section at the bottom.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `workload/src/workload/` at repository root
- ~~Target modular structure per plan.md~~ (see Reassessment)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and module structure creation

**Status**: Directory restructure was not pursued. Code remains in original flat layout. See Reassessment.

- [x] T007 [P] Create txn_factory/ subdirectories — txn_factory/ exists with builder.py
- [N/A] T001 Create api/ module directory — endpoints remain in app.py
- [N/A] T002 Create core/ module directory — logic remains in workload_core.py
- [N/A] T003 [P] Create storage/ module directory — sqlite_store.py stays in place
- [N/A] T004 [P] Create xrpl/ module directory — not needed
- [N/A] T005 [P] Create dashboard/ module directory — dashboard is inline in app.py + templates/
- [N/A] T006 [P] Create ws/ module directory — ws.py and ws_processor.py stay in place
- [N/A] T008 [P] Create __init__.py files for all new modules — no new modules created

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

### Code Quality Foundation (P6 - Must be first)

- [x] T009 Configure pre-commit framework in .pre-commit-config.yaml with ruff hooks
- [ ] T010 Create custom docstring checker script in scripts/check_docstrings.py
- [x] T011 [P] Run ruff format on entire codebase in workload/src/workload/
- [x] T012 [P] Run ruff check --fix on entire codebase in workload/src/workload/
- [ ] T013 Add docstrings and return types to existing public methods in workload/src/workload/workload_core.py — partial (~65-70% coverage)
- [ ] T014 [P] Add docstrings and return types to existing public methods in workload/src/workload/app.py — partial (~8% return types, partial docstrings)
- [x] T015 Install pre-commit hooks via pre-commit install
- [x] T016 Test pre-commit hooks by attempting commit with bad formatting

### XRPL Protocol Layer (Shared by all transaction types)

**Status**: No separate xrpl/ module. Functionality exists in original files.

- [x] T017 Create XRPL type definitions — TxType/TxState in constants.py, ValidationSrc/ValidationRecord in validation.py, FeeInfo in fee_info.py
- [x] T018 Implement error code categorization — error handling in workload_core.py (tem/tef/ter/tel classification)
- [x] T019 [P] Implement custom RPC client — RPC calls in workload_core.py (submit, account_info, server_info, etc.)
- [x] T020 [P] Implement custom transaction builders — txn_factory/builder.py with registry pattern

### Storage Layer (Shared state management)

**Status**: No separate storage/ module. Functionality exists in original files.

- [N/A] T021 Move sqlite_store.py to storage/ — stays at workload/src/workload/sqlite_store.py
- [x] T022 Extract InMemoryStore — InMemoryStore exists in workload_core.py (was cleaned up: dead methods removed, circular imports fixed)
- [x] T023 [P] Create storage data models — PendingTx, AccountRecord dataclasses in workload_core.py; ValidationRecord in validation.py

### Core Business Logic Extraction (Enables all user stories)

**Status**: No core/ module. Logic remains in workload_core.py, with targeted extractions into
validation.py, constants.py, fee_info.py to break circular imports and deduplicate.

- [x] T024 Extract sequence allocation logic — alloc_seq/release_seq in workload_core.py (generation-based safety added)
- [x] T025 Extract transaction submission logic — submit_pending/build_sign_and_track in workload_core.py, producer in app.py
- [x] T026 Extract validation tracking logic — validation.py created with ValidationSrc, ValidationRecord
- [x] T027 Extract account management logic — AccountRecord, gateway/user pools in workload_core.py (dead methods removed)
- [x] T028 Create main Workload orchestrator — Workload class in workload_core.py (cleaned up, functional)
- [N/A] T029 Update workload_core.py to re-export — no extraction into submodules, so no re-exports needed

### API Layer Foundation (Enables all API endpoints)

**Status**: No api/ module. app.py uses APIRouter pattern (r_state, r_txn, r_acct) with all routes in one file.

- [x] T030 Create Pydantic request/response models — basic models exist inline in app.py (not in separate file)
- [x] T031 Create FastAPI app with lifespan — app.py lifespan() with full startup/shutdown lifecycle
- [x] T032 Extract existing account endpoints — account routes on r_acct router in app.py
- [x] T033 Extract existing state/metrics endpoints — state routes on r_state router in app.py
- [x] T034 Update app.py to use APIRouter pattern — done (r_state, r_txn, r_acct, r_debug routers)

### WebSocket Infrastructure (Shared by dashboard and validation)

**Status**: No ws/ module. ws.py and ws_processor.py stay in original locations, fully functional.

- [N/A] T035 Move ws.py to ws/listener.py — stays at workload/src/workload/ws.py
- [N/A] T036 Move ws_processor.py to ws/processor.py — stays at workload/src/workload/ws_processor.py
- [x] T037 Extend WebSocket processor to support dashboard broadcasts — ws_processor.py dispatches to dashboard via WS

**Checkpoint**: Foundation ready (in-place) — user story features were implemented without the extraction step.

---

## Phase 3: User Story 1 - Continuous Submission Reliability (Priority: P1) 🎯 MVP

**Goal**: Achieve 90% validation rate, <10% sequence conflicts, zero ledger gaps

**Status**: LARGELY COMPLETE — implemented in-place in workload_core.py, app.py, fee_info.py, builder.py

### Implementation for User Story 1

- [x] T038 [P] [US1] Implement dynamic fee adaptation — fee_info.py FeeInfo dataclass, _open_ledger_fee() in workload_core.py
- [x] T039 [P] [US1] Add fee escalation metrics tracking — FeeInfo tracked per ledger, exposed in dashboard
- [x] T040 [US1] Extend sequence allocation with contention handling — generation-based stale tx detection, max_pending_per_account=1 invariant
- [x] T041 [US1] Implement adaptive batch sizing based on fee state — producer-consumer pipeline in app.py, queue drain based on target
- [x] T042 [US1] Add configurable account count support — config.toml has [users] number and [gateways] number
- [ ] T043 [US1] Implement ledger fill rate tracking — not yet implemented as a dedicated metric
- [x] T044 [US1] Add sequence conflict detection and metrics — AccountRecord.generation, cascade_expire increments generation
- [x] T045 [US1] Implement validation rate calculation endpoint — /state/summary exposes validation counts and rates
- [x] T046 [US1] Add continuous submission mode improvements — producer-consumer split (_txn_producer + consumer in continuous_workload)

**Checkpoint**: US1 is functional. Validation rate, sequence safety, and continuous submission all work.

---

## Phase 4: User Story 2 - Complete MPToken Transaction Workflow (Priority: ~~P2~~ PUNTED)

**Goal**: Support full MPToken lifecycle (mint, disburse, offer)

**Status**: Core functionality complete. Dedicated endpoints and entity models punted — `/transaction/{type}` suffices for now.

### Implementation for User Story 2

- [x] T048 [P] [US2] Create MPToken builder in workload/src/workload/txn_factory/builder.py — _build_mptoken_issuance_create, _build_mptoken_issuance_set, _build_mptoken_authorize, _build_mptoken_issuance_destroy
- [x] T053 [US2] Implement MPToken operation probability distribution — weighted in config.toml [transactions.percentages]
- [x] T054 [US2] Extend random transaction selection to include MPToken operations — pick_eligible_txn_type includes MPToken types with capability checks

### Punted (dedicated endpoints & richer tracking — revisit later)

- [ ] T047 [P] [US2] Create MPToken entity model — tracked via mpt_issuance_ids list, no dedicated model
- [ ] T049 [US2] Implement MPToken disbursement transaction builder
- [ ] T050 [US2] Add MPToken disbursement endpoint
- [ ] T051 [US2] Implement MPToken holder tracking — partial (mpt_issuance_ids tracked, no per-holder)
- [ ] T052 [US2] Add trust line verification before disbursement
- [ ] T055 [US2] Add MPToken metrics tracking

**Checkpoint**: MPToken creation/authorization/destruction works in continuous mode via `/transaction/{type}`.

---

## Phase 5: User Story 3 - Offer Crossing and Order Book Activity (Priority: ~~P3~~ PUNTED)

**Goal**: Support OfferCreate and OfferCancel for DEX activity

**Status**: Core functionality complete. Dedicated endpoints and entity models punted — `/transaction/{type}` suffices for now.

### Implementation for User Story 3

- [x] T057 [P] [US3] Create Offer builder in workload/src/workload/txn_factory/builder.py — _build_offer_create
- [x] T058 [US3] Implement OfferCreate transaction builder — in builder.py
- [x] T059 [US3] Implement OfferCancel transaction builder — _build_offer_cancel in builder.py
- [x] T064 [US3] Implement realistic offer parameter generation (prices, amounts) — rate-based pricing from config currencies
- [x] T065 [US3] Add offer operation probability distribution — weighted in config.toml [transactions.percentages]
- [x] T066 [US3] Extend random transaction selection to include Offers — pick_eligible_txn_type includes OfferCreate/OfferCancel

### Punted (dedicated endpoints & richer tracking — revisit later)

- [ ] T056 [P] [US3] Create Offer entity model
- [ ] T060 [US3] Add OfferCreate endpoint
- [ ] T061 [US3] Add OfferCancel endpoint
- [ ] T062 [US3] Implement active offer tracking
- [ ] T063 [US3] Add self-trade prevention logic
- [ ] T067 [US3] Add offer metrics tracking (created, cancelled, crossed)

**Checkpoint**: Offer creation and cancellation work in continuous mode via `/transaction/{type}`.

---

## Phase 6: User Story 4 - Arbitrary Transaction Submission via API (Priority: ~~P4~~ LOW-PRIORITY STRETCH)

**Goal**: Swagger UI endpoints with proper parameters for all transaction types

**Status**: Low-priority stretch goal. `/transaction/{type}` works. Rich Pydantic models and per-type validation to be figured out later if needed.

### Punted (all — revisit if/when needed)

- [ ] T068-T079 — Pydantic request models, per-type validation, Swagger UI polish

**Checkpoint**: Swagger UI accessible at /docs. Current endpoints functional without structured request models.

---

## Phase 7: User Story 5 - Network Node Observability Dashboard (Priority: P5)

**Goal**: Dashboard showing node metrics with real-time updates

**Status**: SUBSTANTIALLY COMPLETE — dashboard exists in app.py (state_dashboard endpoint) with HTML template, CSS, JS, and WebSocket updates. Implemented in-place rather than in dashboard/ subdirectory.

### Implementation for User Story 5

- [x] T086 [US5] Create dashboard HTML page — inline in app.py state_dashboard() + templates/
- [x] T087 [P] [US5] Create dashboard CSS styles — inline in dashboard HTML
- [x] T088 [P] [US5] Create dashboard JavaScript WebSocket client — inline JS with WS connection
- [x] T089 [US5] Implement dashboard HTML endpoint — /state/dashboard in app.py
- [x] T093 [P] [US5] Implement transaction activity endpoint — /state/summary, /state/dashboard data
- [x] T094 [P] [US5] Implement validation metrics endpoint — validation counts/rates in dashboard
- [x] T095 [US5] Implement dashboard WebSocket endpoint — WS broadcast on ledger close
- [x] T096 [US5] Integrate ledger-triggered dashboard broadcasts — ws_processor triggers dashboard updates
- [ ] T080-T084 [P] [US5] Create typed data models (NodeMetrics, NetworkOverview, QueueState, etc.) — data is ad-hoc dicts, not typed models
- [ ] T085 [US5] Implement node metrics collection (trigger on ledger close) — partial (server_info polled, not per-node)
- [ ] T090 [US5] Implement network overview endpoint — no dedicated endpoint
- [ ] T091 [P] [US5] Implement per-node metrics endpoint — single node only
- [ ] T092 [P] [US5] Implement queue state endpoint — no dedicated endpoint
- [ ] T097 [US5] Add node unreachability detection and handling — not implemented

**Checkpoint**: Dashboard works with real-time WS updates. Multi-node and typed data models not done.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T098 [P] Verify all modules have proper __init__.py exports
- [ ] T099 [P] Run final ruff format on all modified files
- [ ] T100 [P] Run final ruff check --fix on all modified files
- [ ] T101 Verify all new public methods have docstrings and return types
- [ ] T102 Audit file sizes - ensure no files exceed 500 lines
- [x] T103 [P] Update CLAUDE.md with new module structure and patterns — CLAUDE.md is up to date
- [ ] T104 [P] Test all user stories via quickstart.md procedures
- [N/A] T105 Remove old workload_core.py re-exports — no extraction happened, nothing to remove
- [N/A] T106 Update imports across codebase to use new module paths — no new module paths
- [ ] T107 Verify pre-commit hooks pass on all changes
- [ ] T108 Run continuous submission for 100 ledgers - validate all success criteria (SC-001 through SC-019)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3 → P4 → P5)
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May use MPToken from US2 but independently testable with regular currencies
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) - Documents endpoints created in US1-3
- **User Story 5 (P5)**: Can start after Foundational (Phase 2) - Displays metrics from US1-3
- **User Story 6 (P6)**: Embedded in Foundational phase (T009-T016) - Must complete first

### Within Each User Story

- Tasks marked [P] can run in parallel (different files, no dependencies)
- Tasks without [P] must run sequentially (depend on previous tasks in story)
- Code Quality (P6) tasks in Phase 2 block everything else
- Extraction tasks (T024-T029) block API layer tasks (T030-T034)

### Parallel Opportunities

- **Phase 1 (Setup)**: T003-T008 can all run in parallel
- **Phase 2 (Foundational)**:
  - T011-T012 can run in parallel
  - T013-T014 can run in parallel
  - T019-T020 can run in parallel
  - T023 standalone
- **Phase 3 (US1)**: T038-T039 can run in parallel
- **Phase 4 (US2)**: T047-T048 can run in parallel
- **Phase 5 (US3)**: T056-T057 can run in parallel
- **Phase 6 (US4)**: T068-T075 can run in parallel (all Pydantic models)
- **Phase 7 (US5)**: T080-T084 can run in parallel, T087-T088 can run in parallel, T091-T094 can run in parallel
- **Phase 8 (Polish)**: T098-T100, T103-T104 can run in parallel

---

## Parallel Example: User Story 1

```bash
# After Foundational phase completes, launch US1 tasks:

# Parallel group 1 (different files):
Task T038: "Implement dynamic fee adaptation in core/submission.py"
Task T039: "Add fee escalation metrics in core/submission.py"

# Then sequential:
Task T040: "Extend sequence allocation in core/sequence.py"
Task T041: "Implement adaptive batch sizing in core/submission.py"
# ... continue with remaining US1 tasks
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T008)
2. Complete Phase 2: Foundational (T009-T037) - CRITICAL
3. Complete Phase 3: User Story 1 (T038-T046)
4. **STOP and VALIDATE**: Test US1 via quickstart.md P1 procedures
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational (Phases 1-2) → Foundation ready
2. Add User Story 1 (Phase 3) → Test independently → Deploy/Demo (MVP! - 90% validation rate)
3. Add User Story 2 (Phase 4) → Test independently → Deploy/Demo (MPToken support)
4. Add User Story 3 (Phase 5) → Test independently → Deploy/Demo (Offer crossing)
5. Add User Story 4 (Phase 6) → Test independently → Deploy/Demo (API documentation)
6. Add User Story 5 (Phase 7) → Test independently → Deploy/Demo (Dashboard)
7. Complete Phase 8 (Polish) → Final validation
8. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (CRITICAL - everyone needed)
2. Once Foundational is done:
   - Developer A: User Story 1 (T038-T046) - 9 tasks
   - Developer B: User Story 2 (T047-T055) - 9 tasks
   - Developer C: User Story 3 (T056-T067) - 12 tasks
   - Developer D: User Story 4 (T068-T079) - 12 tasks
   - Developer E: User Story 5 (T080-T097) - 18 tasks
3. Stories complete and integrate independently
4. Team reconvenes for Phase 8 (Polish)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Stop at any checkpoint to validate story independently via quickstart.md
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- All tasks follow strict checkbox format: `- [ ] [TaskID] [P?] [Story?] Description with file path`
- No tests requested in spec - manual validation via API endpoints only
- Code Quality (P6) embedded in Foundational phase as blocking prerequisite

---

## Reassessment (2026-03-17)

### What happened vs what was planned

The spec envisioned a **big-bang restructure**: extract everything into `core/`, `api/`, `storage/`,
`xrpl/`, `ws/`, `dashboard/` subdirectories (Phase 1-2), *then* build features on top. In practice,
features were built **in-place** — fixing real bugs, adding the producer-consumer pipeline, building
MPToken/Offer/AMM builders, and shipping the dashboard — all without the extraction step.

### Scorecard

| Area | Planned | Actual | Gap |
|------|---------|--------|-----|
| Directory restructure | 8 new modules | 0 new modules (N/A) | Intentionally skipped |
| Pre-commit / ruff | Configure + enforce | Done | None |
| Dead code cleanup | Implicit | ~30 items removed | None — exceeded expectations |
| Type extraction | Into xrpl/types.py, storage/models.py | validation.py, constants.py, fee_info.py | Partial — extracted what was needed |
| US1: Continuous submission | 9 tasks | 7/9 done in-place | Ledger fill rate metric missing |
| US2: MPToken lifecycle | 9 tasks | 4/9 done | No disbursement flow or dedicated endpoint |
| US3: Offers/DEX | 12 tasks | 6/12 done | No dedicated endpoints, entity model, or crossing metrics |
| US4: Swagger models | 12 tasks | 0/12 done | Not started |
| US5: Dashboard | 18 tasks | 8/18 done | Multi-node, typed models, queue state not done |
| Docstrings/return types | Full coverage | ~65% workload_core, ~8% app.py | Significant gap |

### Should we still do the restructure?

**Arguments for**:
- workload_core.py is a god class (~1200+ lines) — hard to navigate
- app.py is similarly large with all routes + dashboard inline
- Extraction would make the codebase more approachable for new contributors

**Arguments against**:
- Everything works — restructuring is high-risk churn with zero feature value
- The original plan assumed extraction *before* features; features are already built
- Import graph is already clean (circular import was fixed via validation.py)
- Single-developer project — discoverability is less of a concern

### Recommended path forward

Rather than the original Phase 1-2 extraction, consider a **lighter modularization**:

1. **Split app.py routes** into separate router files (already uses APIRouter pattern) — low risk
2. **Split workload_core.py** along natural seams: Workload class, InMemoryStore, AccountRecord/PendingTx models — medium risk
3. **Skip** xrpl/, dashboard/, ws/ subdirectories — not worth the churn
4. **Focus remaining effort** on the three P0 items from CLAUDE.md:
   - Finish docstrings/return types (T013, T014)
   - Public network support (not in this spec)
   - XRP accounting / fund recovery (not in this spec)
