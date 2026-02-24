<!--
SYNC IMPACT REPORT:
Version change: Initial creation → 1.0.0
Modified principles: None (new constitution)
Added sections: All sections newly created
Removed sections: None
Templates requiring updates:
  ✅ plan-template.md - Reviewed, Constitution Check section aligns with principles
  ✅ spec-template.md - Reviewed, user story requirements align with testing discipline
  ✅ tasks-template.md - Reviewed, task categorization reflects principle-driven workflow
Templates not requiring updates:
  - No command files exist in .specify/templates/commands/
Runtime guidance:
  ✅ CLAUDE.md - Existing file contains complementary technical guidance
  ⚠️ README.md - Does not exist; will need creation if project documentation is formalized
Follow-up TODOs: None
-->

# rippled-workload Constitution

## Core Principles

### I. Ledger-Based Timing (NON-NEGOTIABLE)

The XRPL ledger close event is the fundamental unit of time and synchronization, not wall-clock time.

**Rules**:
- MUST wait for ledger closes, count ledgers, and use ledger index as the primary tick mechanism
- MUST use time only for timeouts (network connectivity issues) and measuring operation duration (metrics)
- MUST NOT use time-based delays for submission logic (no `await asyncio.sleep()` between transaction batches)
- MUST NOT spread submissions over time intervals or control submission rate with wall-clock time

**Rationale**: XRPL consensus operates on discrete ledger closes (~3-4 seconds). Transaction validation, sequence numbers, queue behavior, and fee escalation are all tied to ledger boundaries. Time-based logic creates race conditions and unpredictable behavior. Ledger-based logic is deterministic and aligns with how rippled actually processes transactions.

**Examples**:
- ✅ "Wait for 5 ledger closes before re-checking transaction state"
- ✅ "Submit batch when ledger closes and accounts have available sequence slots"
- ❌ "Wait 2 seconds between submissions"
- ❌ "Submit 100 transactions/second"

### II. Reliable Transaction Submission

All transaction submission MUST follow the reliable submission workflow documented in `docs/reliable-tx-submission.svg`.

**Rules**:
- MUST construct transactions with both `Sequence` and `LastLedgerSequence` fields
- MUST save transaction metadata (hash, Sequence, LastLedgerSequence, ledger index) to persistent storage before submission
- MUST track transaction outcomes until finality is reached
- MUST check persistent storage on recovery to resume tracking transactions without recorded final outcomes
- MUST handle unknown validated transactions (possible malleability, concurrent submission, or lost records)

**Rationale**: The XRPL's asynchronous consensus process means transactions can succeed, fail, or remain pending in non-obvious ways. The reliable submission workflow ensures transactions are tracked from creation through final validation or rejection, handling edge cases like transaction malleability, sequence conflicts, and ledger gaps.

### III. Transaction Finality Awareness

Transaction results are NOT final until the transaction appears in a validated ledger or is provably expired.

**Rules**:
- MUST distinguish between tentative results (from initial submission) and final results (from validated ledgers)
- MUST understand that tentative success can become final failure, and tentative failure can become final success
- MUST mark transactions as final ONLY when:
  - `tesSUCCESS` or any `tec` code appears in a validated ledger
  - `tem` code is received (final unless protocol changes)
  - `tefPAST_SEQ` is received AND another transaction with the same sequence is validated
  - `tefMAX_LEDGER` is received AND a validated ledger exists beyond `LastLedgerSequence` without including the transaction
- MUST treat all other codes (`ter`, `tel`) as potentially non-final

**Rationale**: XRPL transaction ordering and consensus can change tentative results. A transaction that succeeded in the open ledger may fail in the validated ledger, or vice versa, depending on canonical ordering, offer book changes, or account balance updates from other transactions.

### IV. Sequence Number Discipline

Sequence number allocation MUST be strictly controlled to prevent double-spending and sequence conflicts.

**Rules**:
- MUST use per-account locks when allocating sequence numbers
- MUST fetch account sequence from ledger exactly once per account session
- MUST increment the local sequence counter only after successful allocation under lock
- MUST track pending transactions by sequence number to detect conflicts
- MUST handle `terPRE_SEQ` (sequence too high) and queue management correctly

**Rationale**: XRPL accounts have a monotonically increasing sequence number. Submitting two transactions with the same sequence creates a conflict where at most one can validate. Concurrent sequence allocation without locking leads to double-spending attempts and validation failures.

### V. Fee Escalation and Queue Awareness

Transaction submission MUST account for dynamic fee escalation and transaction queue behavior.

**Rules**:
- MUST understand that base fees (256 fee level, 10 drops) apply only up to a limit of transactions per ledger
- MUST recognize that fees escalate exponentially once the open ledger exceeds this limit
- MUST handle transaction queue rejection codes (`telCAN_NOT_QUEUE`, `telCAN_NOT_QUEUE_FULL`, `telCAN_NOT_QUEUE_BALANCE`, etc.)
- MUST use the `fee` RPC command to query current fee levels and queue state before batch submission
- MUST NOT assume transactions will enter the queue if they meet base fee requirements

**Rationale**: Fee escalation protects the network from spam during high traffic. Understanding fee levels and queue limits allows the workload to adapt submission rates and fees to current network conditions, preventing systematic rejection and ensuring realistic traffic patterns.

### VI. Error Code Categorization and Handling

All XRPL transaction error codes MUST be categorized and handled according to their finality and retry semantics.

**Rules**:
- `tes` (success): Final when in validated ledger → mark VALIDATED
- `tec` (claimed fee failure): Final when in validated ledger → mark VALIDATED (with failure semantics)
- `tem` (malformed): Terminal rejection, do not retry → mark REJECTED
- `tef` (failure): Check specific code for finality (e.g., `tefPAST_SEQ`, `tefMAX_LEDGER`) → mark REJECTED or EXPIRED
- `ter` (retry): Temporary failure, can retry → mark RETRYABLE
- `tel` (local): Local rejection (queue full, wrong network), can retry or resubmit → mark RETRYABLE or FAILED_NET
- MUST assert/raise exceptions on `terLAST`, `tecINTERNAL`, `tecINVARIANT_FAILED` (should never occur)

**Rationale**: Different error code classes have different finality and retry semantics. Treating all failures uniformly leads to incorrect retry logic, wasted resources retrying terminal failures, or incorrectly marking retryable failures as final.

### VII. Observability and Metrics

All workload operations MUST be observable through structured logging and real-time metrics.

**Rules**:
- MUST log all transaction state transitions (CREATED → SUBMITTED → VALIDATED/REJECTED/EXPIRED/FAILED_NET/RETRYABLE)
- MUST track metrics: transactions by state, validation rates, validation source (polling vs WebSocket), ledger close rates
- MUST expose metrics via API endpoints (`/state/summary`, `/state/pending`, `/state/validations`)
- MUST maintain recent validation history (deque) for debugging
- MUST record timestamps for all state transitions to measure latency

**Rationale**: Antithesis fuzz testing requires detailed observability to diagnose failures, understand system behavior under load, and correlate workload actions with rippled node behavior. Metrics and logs are the primary interface for understanding workload effectiveness.

### VIII. Code Quality and Maintainability

All code MUST meet Python 3.13+ quality standards with strict linting and type checking.

**Rules**:
- MUST use `ruff` for linting and formatting with project configuration
- MUST format imports (`ruff check --select I --fix`)
- MUST provide return type annotations for all methods (ANN201)
- MUST use Google docstring style for all public methods and classes
- MUST run `ruff format` before committing
- MUST fix all `ruff check` errors before submission
- MUST configure pre-commit hooks to gate future changes

**Rationale**: High code quality reduces bugs, improves maintainability, and ensures consistency across contributors. Type annotations catch errors early and improve IDE support. Automated formatting eliminates style debates and simplifies reviews.

## Testing and Validation

### Testing Discipline

Tests are OPTIONAL unless explicitly requested in the feature specification.

**Rules**:
- IF tests are requested: MUST write tests FIRST, ensure they FAIL, then implement
- IF tests are requested: MUST follow Red-Green-Refactor TDD cycle strictly
- MUST validate all features manually via API endpoints (`/transaction/*`, `/accounts/*`, `/state/*`)
- MUST document testing procedures in feature quickstart documentation
- MUST NOT create tests proactively without explicit user request

**Rationale**: This workload is a testing tool itself, not production software. Manual validation via API endpoints is sufficient for most changes. Formal tests add overhead without proportional value unless specifically needed for regression or complex logic.

### Integration Testing Priorities

Focus integration tests on areas where failures are costly or difficult to debug.

**Priority areas** (if tests are requested):
- Sequence number allocation under concurrent load
- Transaction state transition correctness
- Fee escalation response and queue handling
- WebSocket vs polling validation reconciliation
- Persistent storage recovery after crash

## Development Workflow

### Feature Implementation Process

1. **Specification**: Create feature spec in `/specs/[###-feature-name]/spec.md` using spec template
2. **Planning**: Generate implementation plan in `/specs/[###-feature-name]/plan.md` using plan template
3. **Constitution Check**: Verify compliance with all principles (particularly Ledger-Based Timing and Reliable Submission)
4. **Task Generation**: Generate task list in `/specs/[###-feature-name]/tasks.md` using tasks template
5. **Implementation**: Execute tasks in dependency order, marking progress
6. **Validation**: Test via API endpoints, verify metrics, check logs
7. **Documentation**: Update CLAUDE.md and feature documentation as needed

### Commit Discipline

- Commit after each logical task completion
- Use descriptive commit messages: `feat: add X`, `fix: correct Y`, `refactor: simplify Z`, `docs: update W`
- Run `ruff format` and `ruff check` before committing
- Use pre-commit hooks to enforce formatting (once configured per priority item #6)

### Code Review Requirements

- All changes MUST comply with Core Principles I-VIII
- Reviewers MUST verify Ledger-Based Timing is preserved (no new `asyncio.sleep()` for submission logic)
- Reviewers MUST verify error codes are categorized correctly
- Reviewers MUST verify sequence number allocation uses locks
- Reviewers MUST verify all public methods have return types and docstrings

## Constraints and Standards

### Technology Stack

- **Language**: Python 3.13+
- **Package Manager**: `uv` (NOT pip)
- **Web Framework**: FastAPI
- **XRPL Library**: `xrpl-py`
- **Concurrency**: `asyncio` with `TaskGroup`
- **Linting/Formatting**: `ruff`
- **Deployment**: Docker containers

### Performance Standards

- **Transaction Throughput**: Target 90% successful validation during continuous submission phase
- **Sequence Conflict Rate**: <10% of transactions should experience `terPRE_SEQ` or sequence conflicts
- **Ledger Fill Rate**: Must achieve contiguous ledgers being filled without gaps
- **Acceptable Failures**: `tecPATH_DRY`, unfunded offers, and `ter` codes are acceptable as realistic traffic patterns

### Scope and Scale

- **Network Size**: 5-10 rippled validators in local testnet
- **Account Pool**: Configurable (currently: 1 funding account, 5 gateways, 50 users via config.toml)
- **Transaction Types**: Payment, TrustSet, AccountSet, NFTokenMint, MPTokenIssuanceCreate, OfferCreate, OfferCancel, MPToken disbursement
- **Execution Environment**: Docker Compose network with workload, sidecar, and rippled nodes

## Governance

### Amendment Process

1. Propose amendment with rationale and affected sections
2. Assess version bump type:
   - **MAJOR**: Backward incompatible governance/principle removals or redefinitions
   - **MINOR**: New principle/section added or materially expanded guidance
   - **PATCH**: Clarifications, wording, typo fixes, non-semantic refinements
3. Update constitution with new version and amendment date
4. Update Sync Impact Report comment at top of file
5. Verify template consistency (plan.md, spec.md, tasks.md)
6. Commit with message: `docs: amend constitution to vX.Y.Z (description)`

### Compliance Review

- Constitution supersedes all other practices
- All feature specs MUST include Constitution Check section verifying compliance with principles
- All pull requests MUST verify compliance with Core Principles I-VIII
- Complexity violations MUST be justified in plan.md Complexity Tracking table

### Runtime Guidance

- Use `CLAUDE.md` for technical implementation details, patterns, and command references
- Constitution defines WHAT (principles, rules) while CLAUDE.md defines HOW (commands, examples, architecture)
- In case of conflict, Constitution takes precedence

**Version**: 1.0.0 | **Ratified**: 2025-12-02 | **Last Amended**: 2025-12-02
