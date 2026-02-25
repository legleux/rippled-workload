# Tasks: Priority Improvements for Fault-Tolerant XRPL Workload

**Input**: Design documents from `/specs/001-priority-improvements/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Not requested in feature specification - manual validation via API endpoints only

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `workload/src/workload/` at repository root
- Target modular structure per plan.md

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and module structure creation

- [ ] T001 Create api/ module directory structure in workload/src/workload/api/
- [ ] T002 Create core/ module directory structure in workload/src/workload/core/
- [ ] T003 [P] Create storage/ module directory structure in workload/src/workload/storage/
- [ ] T004 [P] Create xrpl/ module directory structure in workload/src/workload/xrpl/
- [ ] T005 [P] Create dashboard/ module directory structure in workload/src/workload/dashboard/static/
- [ ] T006 [P] Create ws/ module directory structure in workload/src/workload/ws/
- [ ] T007 [P] Create txn_factory/ subdirectories for new builders in workload/src/workload/txn_factory/
- [ ] T008 [P] Create __init__.py files for all new modules (api, core, storage, xrpl, dashboard, ws)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Code Quality Foundation (P6 - Must be first)

- [ ] T009 Configure pre-commit framework in .pre-commit-config.yaml with ruff hooks
- [ ] T010 Create custom docstring checker script in scripts/check_docstrings.py
- [ ] T011 [P] Run ruff format on entire codebase in workload/src/workload/
- [ ] T012 [P] Run ruff check --fix on entire codebase in workload/src/workload/
- [ ] T013 Add docstrings and return types to existing public methods in workload/src/workload/workload_core.py
- [ ] T014 [P] Add docstrings and return types to existing public methods in workload/src/workload/app.py
- [ ] T015 Install pre-commit hooks via pre-commit install
- [ ] T016 Test pre-commit hooks by attempting commit with bad formatting

### XRPL Protocol Layer (Shared by all transaction types)

- [ ] T017 Create XRPL type definitions in workload/src/workload/xrpl/types.py
- [ ] T018 Implement error code categorization in workload/src/workload/xrpl/error_codes.py (extend for MPToken and Offer codes per research.md)
- [ ] T019 [P] Implement custom RPC client in workload/src/workload/xrpl/client.py (minimal xrpl-py usage)
- [ ] T020 [P] Implement custom transaction builders in workload/src/workload/xrpl/transactions.py

### Storage Layer (Shared state management)

- [ ] T021 Move sqlite_store.py to workload/src/workload/storage/sqlite_store.py
- [ ] T022 Extract InMemoryStore from workload_core.py to workload/src/workload/storage/memory_store.py
- [ ] T023 [P] Create storage data models in workload/src/workload/storage/models.py

### Core Business Logic Extraction (Enables all user stories)

- [ ] T024 Extract sequence allocation logic from workload_core.py to workload/src/workload/core/sequence.py
- [ ] T025 Extract transaction submission logic from workload_core.py to workload/src/workload/core/submission.py
- [ ] T026 Extract validation tracking logic from workload_core.py to workload/src/workload/core/validation.py
- [ ] T027 Extract account management logic from workload_core.py to workload/src/workload/core/account_manager.py
- [ ] T028 Create main Workload orchestrator in workload/src/workload/core/workload.py
- [ ] T029 Update workload_core.py to re-export from new core/ modules (backward compatibility)

### API Layer Foundation (Enables all API endpoints)

- [ ] T030 Create Pydantic request/response models in workload/src/workload/api/models.py
- [ ] T031 Create FastAPI app with lifespan in workload/src/workload/api/__init__.py
- [ ] T032 Extract existing account endpoints from app.py to workload/src/workload/api/accounts.py
- [ ] T033 Extract existing state/metrics endpoints from app.py to workload/src/workload/api/state.py
- [ ] T034 Update app.py to use APIRouter pattern and import from new api/ modules

### WebSocket Infrastructure (Shared by dashboard and validation)

- [ ] T035 Move ws.py to workload/src/workload/ws/listener.py
- [ ] T036 Move ws_processor.py to workload/src/workload/ws/processor.py
- [ ] T037 Extend WebSocket processor to support dashboard broadcasts in workload/src/workload/ws/processor.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Continuous Submission Reliability (Priority: P1) 🎯 MVP

**Goal**: Achieve 90% validation rate, <10% sequence conflicts, zero ledger gaps

**Independent Test**: Run continuous submission for 100 ledgers, measure validation rate, sequence conflict rate, ledger fill rate

### Implementation for User Story 1

- [ ] T038 [P] [US1] Implement dynamic fee adaptation in workload/src/workload/core/submission.py (per-ledger fee querying per research.md)
- [ ] T039 [P] [US1] Add fee escalation metrics tracking in workload/src/workload/core/submission.py
- [ ] T040 [US1] Extend sequence allocation with contention handling in workload/src/workload/core/sequence.py
- [ ] T041 [US1] Implement adaptive batch sizing based on fee state in workload/src/workload/core/submission.py
- [ ] T042 [US1] Add configurable account count support: extend config.toml with `[users.count]` and `[gateways.count]` settings, update workload/src/workload/config.py to read these values
- [ ] T043 [US1] Implement ledger fill rate tracking in workload/src/workload/core/validation.py
- [ ] T044 [US1] Add sequence conflict detection and metrics in workload/src/workload/core/sequence.py
- [ ] T045 [US1] Implement validation rate calculation endpoint in workload/src/workload/api/state.py
- [ ] T046 [US1] Add continuous submission mode improvements to workload/src/workload/core/workload.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently via quickstart.md P1 tests

---

## Phase 4: User Story 2 - Complete MPToken Transaction Workflow (Priority: P2)

**Goal**: Support full MPToken lifecycle (mint, disburse, offer)

**Independent Test**: Execute complete MPToken lifecycle: mint → disburse to 5 accounts → create offers → verify

### Implementation for User Story 2

- [ ] T047 [P] [US2] Create MPToken entity model in workload/src/workload/storage/models.py
- [ ] T048 [P] [US2] Create MPToken builder in workload/src/workload/txn_factory/mptoken.py
- [ ] T049 [US2] Implement MPToken disbursement transaction builder in workload/src/workload/txn_factory/mptoken.py
- [ ] T050 [US2] Add MPToken disbursement endpoint in workload/src/workload/api/transactions.py
- [ ] T051 [US2] Implement MPToken holder tracking in workload/src/workload/core/account_manager.py
- [ ] T052 [US2] Add trust line verification before disbursement in workload/src/workload/txn_factory/mptoken.py
- [ ] T053 [US2] Implement MPToken operation probability distribution in workload/src/workload/txn_factory/builder.py
- [ ] T054 [US2] Extend random transaction selection to include MPToken operations in workload/src/workload/txn_factory/builder.py
- [ ] T055 [US2] Add MPToken metrics tracking in workload/src/workload/storage/memory_store.py

**Checkpoint**: At this point, User Story 2 should be fully functional - can mint, disburse, and track MPTokens independently

---

## Phase 5: User Story 3 - Offer Crossing and Order Book Activity (Priority: P3)

**Goal**: Support OfferCreate and OfferCancel for DEX activity

**Independent Test**: Create 100 offers, cancel 25%, verify 25% cross, check order book consistency

### Implementation for User Story 3

- [ ] T056 [P] [US3] Create Offer entity model in workload/src/workload/storage/models.py
- [ ] T057 [P] [US3] Create Offer builder in workload/src/workload/txn_factory/offers.py
- [ ] T058 [US3] Implement OfferCreate transaction builder in workload/src/workload/txn_factory/offers.py
- [ ] T059 [US3] Implement OfferCancel transaction builder in workload/src/workload/txn_factory/offers.py
- [ ] T060 [US3] Add OfferCreate endpoint in workload/src/workload/api/transactions.py
- [ ] T061 [US3] Add OfferCancel endpoint in workload/src/workload/api/transactions.py
- [ ] T062 [US3] Implement active offer tracking in workload/src/workload/core/account_manager.py
- [ ] T063 [US3] Add self-trade prevention logic in workload/src/workload/txn_factory/offers.py
- [ ] T064 [US3] Implement realistic offer parameter generation (prices, amounts) in workload/src/workload/txn_factory/offers.py
- [ ] T065 [US3] Add offer operation probability distribution in workload/src/workload/txn_factory/builder.py
- [ ] T066 [US3] Extend random transaction selection to include Offers in workload/src/workload/txn_factory/builder.py
- [ ] T067 [US3] Add offer metrics tracking (created, cancelled, crossed) in workload/src/workload/storage/memory_store.py

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should all work independently

---

## Phase 6: User Story 4 - Arbitrary Transaction Submission via API (Priority: P4)

**Goal**: Swagger UI endpoints with proper parameters for all transaction types

**Independent Test**: Access Swagger UI, test all 8 transaction type endpoints with valid/invalid parameters

### Implementation for User Story 4

- [ ] T068 [P] [US4] Create PaymentRequest Pydantic model with examples in workload/src/workload/api/models.py
- [ ] T069 [P] [US4] Create TrustSetRequest Pydantic model with examples in workload/src/workload/api/models.py
- [ ] T070 [P] [US4] Create AccountSetRequest Pydantic model with examples in workload/src/workload/api/models.py
- [ ] T071 [P] [US4] Create NFTokenMintRequest Pydantic model with examples in workload/src/workload/api/models.py
- [ ] T072 [P] [US4] Create MPTokenMintRequest Pydantic model with examples in workload/src/workload/api/models.py
- [ ] T073 [P] [US4] Create MPTokenDisburseRequest Pydantic model with examples in workload/src/workload/api/models.py
- [ ] T074 [P] [US4] Create OfferCreateRequest Pydantic model with examples in workload/src/workload/api/models.py
- [ ] T075 [P] [US4] Create OfferCancelRequest Pydantic model with examples in workload/src/workload/api/models.py
- [ ] T076 [US4] Implement environment-based example generation helper in workload/src/workload/api/models.py
- [ ] T077 [US4] Add parameter validation for all transaction endpoints in workload/src/workload/api/transactions.py
- [ ] T078 [US4] Implement clear error messages for invalid parameters in workload/src/workload/api/transactions.py
- [ ] T079 [US4] Verify Swagger UI displays all parameters correctly via /docs endpoint

**Checkpoint**: All 8 transaction types should have fully documented Swagger UI endpoints

---

## Phase 7: User Story 5 - Network Node Observability Dashboard (Priority: P5)

**Goal**: Dashboard showing node metrics with real-time updates

**Independent Test**: Open dashboard, verify metrics display, confirm WebSocket updates within 5 seconds

### Implementation for User Story 5

- [ ] T080 [P] [US5] Create NodeMetrics data model in workload/src/workload/storage/models.py
- [ ] T081 [P] [US5] Create NetworkOverview data model in workload/src/workload/storage/models.py
- [ ] T082 [P] [US5] Create QueueState data model in workload/src/workload/storage/models.py
- [ ] T083 [P] [US5] Create TransactionActivity data model in workload/src/workload/storage/models.py
- [ ] T084 [P] [US5] Create ValidationMetrics data model in workload/src/workload/storage/models.py
- [ ] T085 [US5] Implement node metrics collection in workload/src/workload/core/workload.py (trigger on ledger close)
- [ ] T086 [US5] Create dashboard HTML page in workload/src/workload/dashboard/static/index.html
- [ ] T087 [P] [US5] Create dashboard CSS styles in workload/src/workload/dashboard/static/style.css
- [ ] T088 [P] [US5] Create dashboard JavaScript WebSocket client in workload/src/workload/dashboard/static/app.js
- [ ] T089 [US5] Implement dashboard HTML endpoint in workload/src/workload/api/dashboard.py
- [ ] T090 [US5] Implement network overview endpoint in workload/src/workload/api/dashboard.py
- [ ] T091 [P] [US5] Implement per-node metrics endpoint in workload/src/workload/api/dashboard.py
- [ ] T092 [P] [US5] Implement queue state endpoint in workload/src/workload/api/dashboard.py
- [ ] T093 [P] [US5] Implement transaction activity endpoint in workload/src/workload/api/dashboard.py
- [ ] T094 [P] [US5] Implement validation metrics endpoint in workload/src/workload/api/dashboard.py
- [ ] T095 [US5] Implement dashboard WebSocket endpoint in workload/src/workload/api/dashboard.py
- [ ] T096 [US5] Integrate ledger-triggered dashboard broadcasts in workload/src/workload/ws/processor.py
- [ ] T097 [US5] Add node unreachability detection and handling in workload/src/workload/core/workload.py

**Checkpoint**: Dashboard should display all metrics with real-time updates via WebSocket

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T098 [P] Verify all modules have proper __init__.py exports
- [ ] T099 [P] Run final ruff format on all modified files
- [ ] T100 [P] Run final ruff check --fix on all modified files
- [ ] T101 Verify all new public methods have docstrings and return types
- [ ] T102 Audit file sizes - ensure no files exceed 500 lines
- [ ] T103 [P] Update CLAUDE.md with new module structure and patterns
- [ ] T104 [P] Test all user stories via quickstart.md procedures
- [ ] T105 Remove old workload_core.py re-exports (breaking change - coordinate)
- [ ] T106 Update imports across codebase to use new module paths
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
