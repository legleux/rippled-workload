# Feature Specification: Priority Improvements for Fault-Tolerant XRPL Workload

**Feature Branch**: `001-priority-improvements`
**Created**: 2025-12-02
**Status**: Draft
**Input**: User description: "We need to address the 6 items that were delineated in the pre-amble first to work toward our fault-tolerant traffic-generating workload for the XRPL."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Continuous Submission Reliability (Priority: P1)

As an Antithesis tester, I need the workload to achieve 90% transaction success rates during continuous submission so that I can generate realistic, sustained traffic patterns that fill contiguous ledgers and expose consensus edge cases under load.

**Why this priority**: This is the foundation of effective fuzz testing. Without reliable continuous submission, the workload cannot generate the sustained traffic patterns needed to expose rippled node failures, consensus issues, or performance bottlenecks. All other improvements depend on having a stable baseline submission capability.

**Independent Test**: Can be fully tested by running continuous submission for 100 ledgers and measuring validation success rate, sequence conflict rate, and ledger fill rate against target thresholds.

**Acceptance Scenarios**:

1. **Given** the workload is in continuous submission mode, **When** 1000 transactions are submitted across 50 ledgers, **Then** at least 900 transactions validate successfully (excluding legitimate failures like tecPATH_DRY and unfunded offers)
2. **Given** the workload is tracking sequence numbers, **When** transactions are submitted concurrently from multiple accounts, **Then** sequence conflicts occur in less than 10% of submissions
3. **Given** the network has sufficient capacity, **When** the workload submits transactions continuously, **Then** each ledger contains at least one workload transaction (no gaps in ledger fill)
4. **Given** the workload encounters sequence tracking issues, **When** more accounts are available at initialization, **Then** the system uses additional accounts to maintain throughput without fixing all sequence tracking issues

---

### User Story 2 - Complete MPToken Transaction Workflow (Priority: P2)

As an Antithesis tester, I need the workload to support full MPToken lifecycle operations (minting, disbursement, and offers) so that I can test multi-party token scenarios and token exchange functionality under various network conditions.

**Why this priority**: MPTokens are a critical XRPL feature. Currently, only minting is implemented, which prevents testing of token distribution patterns, secondary market behavior, and cross-currency exchange scenarios involving MPTokens. This limits the workload's ability to expose MPToken-related bugs.

**Independent Test**: Can be fully tested by executing a complete MPToken lifecycle: mint tokens to an issuer, disburse tokens to multiple holders, create offers for tokens, and verify all transactions validate successfully.

**Acceptance Scenarios**:

1. **Given** an MPToken has been minted, **When** the issuer disburses tokens to 5 different accounts, **Then** all 5 disbursement transactions validate and recipients hold the correct token balances
2. **Given** accounts hold MPTokens, **When** they create offers to trade MPTokens for XRP or other currencies, **Then** the offers are placed successfully in the order books
3. **Given** matching MPToken offers exist, **When** the workload submits transactions to cross offers, **Then** the trades execute and token ownership transfers correctly
4. **Given** the workload is in continuous mode, **When** MPToken operations are enabled, **Then** minting, disbursement, and offer operations are randomly selected and executed in realistic proportions

---

### User Story 3 - Offer Crossing and Order Book Activity (Priority: P3)

As an Antithesis tester, I need the workload to create and cancel offers so that I can generate realistic decentralized exchange activity and test order book management, offer matching logic, and cross-currency payment paths.

**Why this priority**: Decentralized exchange activity is a major XRPL use case. Without offer creation and cancellation, the workload cannot test order book edge cases, partial fills, fee escalation effects on trading, or cross-currency payment pathfinding under load.

**Independent Test**: Can be fully tested by creating 100 offers across multiple currency pairs, canceling 25% of them, executing 25% through matching, and verifying order book state consistency.

**Acceptance Scenarios**:

1. **Given** accounts have established trust lines, **When** the workload creates OfferCreate transactions for various currency pairs, **Then** offers are placed in the correct order books and visible via RPC queries
2. **Given** offers exist in the order books, **When** the workload submits OfferCancel transactions, **Then** the specified offers are removed and no longer executable
3. **Given** matching offers exist (buy/sell price overlap), **When** new offers are submitted that cross existing offers, **Then** trades execute automatically and both sides receive the expected currencies
4. **Given** the workload is in continuous mode, **When** offer operations are enabled, **Then** OfferCreate and OfferCancel transactions are randomly generated with realistic spreads and volumes

---

### User Story 4 - Arbitrary Transaction Submission via API (Priority: P4)

As an Antithesis operator, I need Swagger UI endpoints for each transaction type with correct parameters so that I can manually submit specific transactions for targeted testing scenarios without writing custom code.

**Why this priority**: Manual transaction submission is essential for reproducing specific bugs, testing edge cases discovered during fuzzing, and validating fixes. Currently, the API lacks proper parameter documentation and validation for individual transaction types.

**Independent Test**: Can be fully tested by accessing Swagger UI, selecting each transaction type endpoint, filling in parameters, submitting, and verifying the transaction is constructed and submitted correctly.

**Acceptance Scenarios**:

1. **Given** the Swagger UI is open, **When** I select the Payment transaction endpoint, **Then** I see all required and optional Payment fields (destination, amount, currency, issuer, etc.) with type validation and descriptions
2. **Given** I fill in valid parameters for a TrustSet transaction, **When** I click Execute, **Then** the transaction is constructed with my parameters, signed, submitted, and I receive the transaction hash and submission result
3. **Given** I enter invalid parameters (e.g., negative amount), **When** I attempt to submit, **Then** I receive a clear validation error before submission
4. **Given** each transaction type has an endpoint, **When** I test all endpoints with valid parameters, **Then** each transaction type can be successfully submitted via the API

---

### User Story 5 - Network Node Observability Dashboard (Priority: P5)

As an Antithesis operator, I need a dashboard showing network node status (server_info, queue state, transactions, validations) so that I can monitor network health, diagnose submission failures, and correlate workload behavior with node state.

**Why this priority**: Observability is critical for debugging fuzzing results. Without visibility into rippled node state, it's difficult to determine whether workload failures are due to network issues, node bugs, or workload logic errors.

**Independent Test**: Can be fully tested by opening the dashboard, verifying real-time display of node metrics, and confirming data updates as the network state changes.

**Acceptance Scenarios**:

1. **Given** the dashboard is open, **When** I view the network overview, **Then** I see key metrics from server_info for each node (ledger index, server state, complete_ledgers range)
2. **Given** transactions are in the queue, **When** I view the queue tab, **Then** I see current queue size, max queue size, fee levels, and pending transaction counts per node
3. **Given** the network is processing transactions, **When** I view the transactions tab, **Then** I see recent transaction activity, validation rates, and state transitions in real-time
4. **Given** validators are running, **When** I view the validations tab, **Then** I see validation messages, validator participation, and consensus agreement metrics
5. **Given** the network state changes, **When** I keep the dashboard open, **Then** the displayed data updates automatically without page refresh

---

### User Story 6 - Code Quality Enforcement (Priority: P6)

As a developer, I need all code to be linted, formatted, documented, and gated by pre-commit hooks so that the codebase remains maintainable, consistent, and high-quality as it grows.

**Why this priority**: Code quality issues accumulate technical debt, slow development, and introduce bugs. While important, this is lower priority than functional improvements because it doesn't directly affect workload effectiveness for Antithesis testing.

**Independent Test**: Can be fully tested by running ruff format, ruff check, verifying all public methods have docstrings and return types, and confirming pre-commit hooks reject improperly formatted code.

**Acceptance Scenarios**:

1. **Given** the codebase exists, **When** I run ruff format, **Then** all code is formatted consistently with project style (no changes after first run)
2. **Given** the codebase exists, **When** I run ruff check, **Then** no linting errors are reported
3. **Given** the codebase contains public methods, **When** I audit all public methods and classes, **Then** each has a Google-style docstring and return type annotation
4. **Given** pre-commit hooks are configured, **When** I attempt to commit improperly formatted code, **Then** the commit is rejected with a clear error message indicating which checks failed
5. **Given** I fix the formatting issues, **When** I re-attempt the commit, **Then** the commit succeeds
6. **Given** the code is modular, **When** I review the file structure, **Then** large files are split into focused modules with clear responsibilities

---

### Edge Cases

- **Sequence tracking under high concurrency**: What happens when 50+ accounts submit transactions simultaneously and sequence allocation locks create bottlenecks? System should gracefully handle contention without deadlocks.

- **MPToken disbursement to non-existent accounts**: What happens when disbursement targets an account that doesn't exist or lacks the required trust line? Transaction should fail with appropriate error code (tecNO_DST, tecNO_LINE).

- **Offer creation with insufficient balance**: What happens when an account creates an offer but lacks sufficient balance to fulfill it? Transaction should fail with tecUNFUNDED_OFFER.

- **API parameter validation edge cases**: What happens when API receives malformed JSON, missing required fields, or out-of-range values? System should return clear error messages without crashing.

- **Dashboard during network partition**: What happens when some nodes are unreachable? Dashboard should show partial data for reachable nodes and indicate unreachable nodes clearly.

- **Pre-commit hooks with partial staging**: What happens when only some files are staged for commit? Hooks should only check staged files, not the entire codebase.

## Requirements *(mandatory)*

### Functional Requirements

**Continuous Submission (P1)**:

- **FR-001**: System MUST achieve 90% transaction validation success rate during continuous submission (excluding legitimate failures: tecPATH_DRY, unfunded offers, ter codes)
- **FR-002**: System MUST maintain sequence conflict rate below 10% of submitted transactions
- **FR-003**: System MUST fill every ledger with at least one transaction during continuous submission (no ledger gaps)
- **FR-004**: System MUST support configurable initial account counts to increase available accounts for parallel submission
- **FR-005**: System MUST track and report submission success metrics (validated, rejected, expired, failed) in real-time

**MPToken Workflow (P2)**:

- **FR-006**: System MUST support MPTokenIssuanceCreate transactions (already implemented, verify functionality)
- **FR-007**: System MUST support MPToken disbursement transactions to transfer tokens from issuer to holders (Note: "disbursement" uses a standard Payment transaction with an MPToken amount in the Amount field)
- **FR-008**: System MUST support OfferCreate transactions for MPToken trading (MPToken/XRP and MPToken/IOU pairs)
- **FR-009**: System MUST support OfferCancel transactions to remove MPToken offers
- **FR-010**: System MUST randomly select MPToken operations (mint, disburse, offer) during continuous submission based on the following probability distribution: 40% Payment, 20% TrustSet, 15% MPToken operations (mint/disburse), 20% Offer operations (create/cancel), 5% other (AccountSet, NFTokenMint)

**Offer Crossing (P3)**:

- **FR-011**: System MUST support OfferCreate transactions for all supported currency pairs (XRP/IOU, IOU/IOU, MPToken/XRP, MPToken/IOU)
- **FR-012**: System MUST support OfferCancel transactions for all offer types
- **FR-013**: System MUST generate realistic offer parameters (prices, amounts) that can result in crossing
- **FR-014**: System MUST track offer creation and cancellation in transaction metrics

**API Transaction Submission (P4)**:

- **FR-015**: System MUST expose Swagger UI endpoint for each supported transaction type (Payment, TrustSet, AccountSet, NFTokenMint, MPTokenIssuanceCreate, OfferCreate, OfferCancel, MPToken disbursement)
- **FR-016**: Each transaction endpoint MUST include all transaction-specific parameters with type validation, descriptions, and examples
- **FR-017**: System MUST validate API parameters before transaction construction and return clear error messages for invalid inputs
- **FR-018**: System MUST return transaction hash, submission result, and any error details from API submissions

**Network Observability (P5)**:

- **FR-019**: System MUST provide a dashboard displaying server_info metrics for all configured nodes (ledger index, server state, complete_ledgers)
- **FR-020**: System MUST display current queue state (queue size, max queue size, fee levels, open ledger fee) per node
- **FR-021**: System MUST display recent transaction activity (submitted, validated, rejected counts) and validation source (polling vs WebSocket)
- **FR-022**: System MUST display validator participation and consensus metrics
- **FR-023**: Dashboard MUST update automatically without page refresh (WebSocket or polling)

**Code Quality (P6)**:

- **FR-024**: All code MUST pass ruff format checks with project configuration
- **FR-025**: All code MUST pass ruff check with zero errors
- **FR-026**: All public methods and classes MUST have Google-style docstrings
- **FR-027**: All methods MUST have return type annotations
- **FR-028**: Pre-commit hooks MUST be configured to run ruff format, ruff check, and enforce docstring requirements
- **FR-029**: Pre-commit hooks MUST reject commits that fail any quality checks
- **FR-030**: Large files (>500 lines) SHOULD be refactored into smaller, focused modules

### Key Entities

- **Transaction**: Represents an XRPL transaction with state tracking (CREATED, SUBMITTED, VALIDATED, REJECTED, EXPIRED, FAILED_NET, RETRYABLE), hash, sequence number, LastLedgerSequence, submission timestamp, validation timestamp

- **Account**: Represents an XRPL account with address, sequence number, balance (XRP and issued currencies), trust lines, and pending transaction tracking

- **MPToken**: Represents a multi-party token with issuer, token ID, maximum amount, holders, and transfer restrictions

- **Offer**: Represents a decentralized exchange offer with account, offer sequence, taker gets (currency/amount), taker pays (currency/amount), and expiration

- **Node**: Represents a rippled node with URL, server_info metrics, queue state, transaction counts, and reachability status

- **Metrics**: Aggregates transaction counts by state, validation rates, sequence conflict rates, ledger fill rates, and validation source distribution

## Success Criteria *(mandatory)*

### Measurable Outcomes

**Continuous Submission (P1)**:

- **SC-001**: Transaction validation success rate reaches 90% or higher during 100 consecutive ledgers of continuous submission
- **SC-002**: Sequence conflict rate stays below 10% of all submitted transactions during continuous submission
- **SC-003**: Zero ledger gaps occur during 100 consecutive ledgers of continuous submission (every ledger contains at least one workload transaction)
- **SC-004**: System sustains submission rate of at least 10 transactions per ledger for 100 consecutive ledgers

**MPToken Workflow (P2)**:

- **SC-005**: Complete MPToken lifecycle (mint, disburse to 5 accounts, create 10 offers, cancel 3 offers) completes with 100% success rate
- **SC-006**: MPToken operations represent at least 15% of total transaction mix during continuous submission

**Offer Crossing (P3)**:

- **SC-007**: OfferCreate and OfferCancel operations complete with 95% success rate (excluding legitimate failures like tecUNFUNDED_OFFER)
- **SC-008**: At least 10% of created offers result in automatic crossing and execution
- **SC-009**: Offer operations represent at least 20% of total transaction mix during continuous submission

**API Transaction Submission (P4)**:

- **SC-010**: All supported transaction types have functional Swagger UI endpoints with complete parameter documentation
- **SC-011**: 100% of valid API submissions result in transactions being constructed and submitted to the network
- **SC-012**: 100% of invalid API submissions return clear error messages without server errors

**Network Observability (P5)**:

- **SC-013**: Dashboard displays server_info, queue state, transaction metrics, and validation metrics for all configured nodes
- **SC-014**: Dashboard updates within 5 seconds of network state changes
- **SC-015**: Dashboard remains functional when up to 50% of nodes are unreachable

**Code Quality (P6)**:

- **SC-016**: Zero ruff format or ruff check errors exist in the codebase
- **SC-017**: 100% of public methods have docstrings and return type annotations
- **SC-018**: Pre-commit hooks successfully prevent commits with quality violations
- **SC-019**: No files exceed 500 lines (excluding generated code and external dependencies)

## Assumptions

1. **Network capacity**: Assumes the local rippled testnet has sufficient capacity to process at least 10-20 transactions per ledger consistently
2. **Account initialization**: Assumes initial account funding and trust line setup complete successfully before continuous submission begins
3. **Configuration flexibility**: Assumes config.toml can be extended to support additional MPToken and offer configuration parameters
4. **Dashboard technology**: Assumes dashboard will be HTML/JavaScript served by the FastAPI app, with real-time updates via WebSocket or SSE
5. **Pre-commit hook framework**: Assumes pre-commit framework (https://pre-commit.com/) or similar will be used for git hook management
6. **Realistic transaction mix**: Assumes realistic probability distribution is 40% Payment, 20% TrustSet, 15% MPToken operations, 20% Offers, 5% other (AccountSet, NFTokenMint)

## Dependencies

1. **External**: xrpl-py library must support all required transaction types (MPToken disbursement, OfferCreate, OfferCancel)
2. **Internal**: Reliable transaction submission workflow (Principle II) must be functioning correctly
3. **Internal**: Sequence number discipline (Principle IV) must be maintained for all new transaction types
4. **Internal**: Error code categorization (Principle VI) must be extended for new transaction type error codes
5. **Internal**: Dashboard depends on existing metrics collection and WebSocket infrastructure
