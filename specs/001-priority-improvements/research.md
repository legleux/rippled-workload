# Research Findings: Priority Improvements for Fault-Tolerant XRPL Workload

**Feature**: Priority Improvements (001-priority-improvements)
**Date**: 2025-12-02
**Status**: Complete

This document consolidates research findings for all technical decisions and clarifications required for implementation.

---

## 1. Dynamic Fee Adaptation Strategy

###

 Decision

**Adopt per-ledger fee querying with adaptive batch sizing based on current fee escalation state.**

### Rationale

The XRPL fee escalation mechanism changes rapidly based on transaction volume. Querying fees per ledger ensures the workload adapts to current network conditions without excessive overhead. Fee queries are lightweight RPC calls that return instantly.

**Algorithm**:
```
On each ledger close:
  1. Call `fee` RPC to get current open_ledger_fee and minimum_fee
  2. If open_ledger_fee > base_fee * 2:
     - High escalation mode: Reduce batch size, increase per-txn fee
  3. If open_ledger_fee == base_fee:
     - Normal mode: Standard batch size, base fee
  4. Track fee history (rolling window of 10 ledgers) for trend detection
```

**Implementation location**: `workload/src/workload/core/submission.py`

### Alternatives Considered

- **Per-batch fee query**: Too granular, adds latency without benefit since fees change per ledger
  - Rejected: Violates ledger-based timing principle (Principle I)

- **Static fee configuration**: Cannot adapt to network conditions
  - Rejected: Fails to meet performance goals under load (SC-001: 90% validation rate)

- **Reactive fee adjustment only after rejections**: Too slow, causes systematic failures
  - Rejected: Increases sequence conflicts (violates FR-002: <10% conflict rate)

### References

- Constitution Principle V: Fee Escalation and Queue Awareness
- `docs/FeeEscalation.md`: Fee escalation algorithm details
- Existing implementation: `workload/src/workload/fee_info.py`

---

## 2. Dashboard Update Mechanism

### Decision

**Use FastAPI WebSocket endpoint with server-push updates triggered by ledger close events (hybrid push model).**

### Rationale

Dashboard updates must align with Principle I (Ledger-Based Timing). WebSocket connections allow the server to push updates when ledger events occur, avoiding time-based polling. This approach:
- Ensures dashboard reflects ledger state changes immediately
- Eliminates polling overhead and race conditions
- Maintains <5 second update latency (SC-014)

**Architecture**:
```
┌─────────────┐     Ledger Close Event      ┌──────────────┐
│   rippled   │ ─────WebSocket────────────> │   Workload   │
│   Nodes     │                              │   (ws.py)    │
└─────────────┘                              └──────────────┘
                                                     │
                                                     │ Trigger
                                                     ▼
                                             ┌──────────────┐
                                             │  Dashboard   │
                                             │  WebSocket   │
                                             │  (server)    │
                                             └──────────────┘
                                                     │
                                                     │ Push Update
                                                     ▼
                                             ┌──────────────┐
                                             │  Browser     │
                                             │  Dashboard   │
                                             └──────────────┘
```

**Implementation**:
- Server: `workload/src/workload/api/dashboard.py` - WebSocket endpoint `/dashboard/updates`
- Client: `workload/src/workload/dashboard/static/app.js` - WebSocket client
- Trigger: Existing `ws.py` ledger listener emits events to dashboard WebSocket broadcaster

### Alternatives Considered

- **Server-Sent Events (SSE)**: Simpler than WebSocket, but less flexible
  - Rejected: WebSocket already in use for rippled connections, reuse pattern

- **Time-based polling (HTTP GET every 2 seconds)**: Violates Principle I
  - Rejected: Introduces time-based delays, creates race conditions

- **Long polling**: Adds server complexity without WebSocket benefits
  - Rejected: WebSocket is more efficient and already used in codebase

### References

- Constitution Principle I: Ledger-Based Timing (NON-NEGOTIABLE)
- FastAPI WebSocket documentation: https://fastapi.tiangolo.com/advanced/websockets/
- Existing WebSocket usage: `workload/src/workload/ws.py`, `workload/src/workload/ws_processor.py`

---

## 3. MPToken Error Code Categorization

### Decision

**MPToken transactions follow standard XRPL error code finality semantics with specific extensions for disbursement scenarios.**

### Error Code Table

| Error Code | Category | Finality | Retry | Description |
|------------|----------|----------|-------|-------------|
| tesSUCCESS | tes | Final | No | MPToken operation succeeded |
| tecNO_DST | tec | Final | No | Disbursement destination account does not exist |
| tecNO_LINE | tec | Final | No | Recipient lacks trust line for MPToken |
| tecNO_PERMISSION | tec | Final | No | Account not authorized for MPToken operation |
| tecUNFUNDED | tec | Final | No | Issuer lacks sufficient MPToken balance for disbursement |
| tecINSUFFICIENT_FUNDS | tec | Final | No | Account lacks XRP for fee |
| temINVALID | tem | Final | No | Malformed MPToken transaction (invalid token ID, amount) |
| temDISABLED | tem | Final | No | MPToken amendment not enabled |
| terPRE_SEQ | ter | Retry | Yes | Sequence number too high (queue or retry) |
| terQUEUED | ter | Retry | Yes | Transaction queued for future ledger |
| telINSUF_FEE_P | tel | Retry | Yes | Fee too low for current load |

### MPToken-Specific Considerations

1. **tecNO_DST vs tecNO_LINE**: Distinguish between account existence and trust line setup
   - Both are final failures but indicate different remediation paths
   - Workload should log differently for diagnostics

2. **tecNO_PERMISSION**: MPToken transfer restrictions may prevent disbursement
   - Final failure, indicates issuer configuration issue
   - Log as configuration error, not transient failure

3. **Disbursement Prerequisites**: Account must exist AND have trust line
   - Workload must ensure trust lines exist before disbursement (P2 implementation)
   - Order: Mint → TrustSet (recipient) → Disburse

### Implementation Location

- Error categorization: `workload/src/workload/xrpl/error_codes.py` (new file)
- MPToken-specific handling: `workload/src/workload/txn_factory/mptoken.py` (new file)

### References

- Constitution Principle III: Transaction Finality Awareness
- Constitution Principle VI: Error Code Categorization and Handling
- `docs/error_codes.md`: Base error code documentation
- XRPL MPToken amendment documentation

---

## 4. Offer Transaction Error Codes

### Decision

**Offer transactions have unique error codes for order book states; categorize by finality with special handling for partial fills.**

### Error Code Table

| Error Code | Category | Finality | Retry | Description |
|------------|----------|----------|-------|-------------|
| tesSUCCESS | tes | Final | No | Offer created or cancelled successfully |
| tecUNFUNDED_OFFER | tec | Final | No | Account lacks balance to fulfill offer |
| tecKILLED | tec | Final | No | Offer crossed partially, remainder killed (tfImmediateOrCancel) |
| tecEXPIRED | tec | Final | No | Offer already expired |
| tecNO_PERMISSION | tec | Final | No | Account not authorized for offer (frozen trust line) |
| tecINSUFFICIENT_RESERVE | tec | Final | No | Account lacks reserve for offer object |
| temINVALID | tem | Final | No | Malformed offer (invalid amounts, currencies) |
| temBAD_OFFER | tem | Final | No | Offer crosses own order (self-trade) |
| terPRE_SEQ | ter | Retry | Yes | Sequence number too high |
| terQUEUED | ter | Retry | Yes | Transaction queued |
| telINSUF_FEE_P | tel | Retry | Yes | Fee too low |

### Offer-Specific Considerations

1. **tecUNFUNDED_OFFER**: Common during realistic trading simulation
   - Mark as final failure but **not counted against success rate** (per FR-001)
   - Acceptable failure category (like tecPATH_DRY)

2. **tecKILLED**: Partial fill with tfImmediateOrCancel flag
   - Final result, offer partially executed
   - Track as partial success in metrics (new metric category)

3. **temBAD_OFFER**: Self-trade prevention
   - Workload must avoid creating offers that cross own orders
   - Builder logic: Check existing offers before creating new offer

4. **OfferCancel with non-existent sequence**: Returns temBAD_SEQUENCE (tem)
   - Final failure, indicates tracking error
   - Workload must track active offer sequences per account

### Offer Lifecycle States

```
Created ──> Active ──┬──> Crossed (partial/full)
                     ├──> Cancelled
                     ├──> Expired
                     └──> Unfunded (tecUNFUNDED_OFFER)
```

### Implementation Location

- Error categorization: `workload/src/workload/xrpl/error_codes.py`
- Offer builders: `workload/src/workload/txn_factory/offers.py` (new file)
- Offer tracking: `workload/src/workload/core/account_manager.py` (extend for active offers)

### References

- Constitution Principle III: Transaction Finality Awareness
- Constitution Principle VI: Error Code Categorization and Handling
- `docs/error_codes.md`: Base error code documentation
- XRPL Offer transaction documentation

---

## 5. Code Modularization Strategy

### Decision

**Extract workload_core.py and app.py into focused modules using FastAPI APIRouter pattern and domain-driven design principles.**

### Extraction Plan

#### Phase 1: Create Module Structure (non-breaking)
```bash
mkdir -p workload/src/workload/{api,core,storage,xrpl,dashboard/static,ws}
```

#### Phase 2: Extract workload_core.py (2245 lines → 5 files <500 lines each)

**workload_core.py decomposition**:
```
Lines  | Responsibility          | New File
-------|------------------------|----------------------------------
1-300  | Imports, data classes  | core/__init__.py, storage/models.py
301-600| Sequence allocation    | core/sequence.py (~300 lines)
601-1100| Transaction submission | core/submission.py (~500 lines)
1101-1600| Validation tracking   | core/validation.py (~500 lines)
1601-2245| Account management    | core/account_manager.py (~450 lines)
       | Workload orchestrator  | core/workload.py (~300 lines)
```

**Extraction strategy**:
1. Create new files with extracted code
2. Update imports in workload_core.py to re-export from new modules (backward compat)
3. Update app.py to import from new modules
4. Verify tests pass (if any)
5. Remove workload_core.py re-exports (breaking change, coordinate)

#### Phase 3: Extract app.py (1191 lines → 5 files <300 lines each)

**app.py decomposition**:
```
Lines  | Responsibility          | New File
-------|------------------------|----------------------------------
1-100  | FastAPI app, lifespan  | api/__init__.py (~100 lines)
101-300| Transaction endpoints  | api/transactions.py (~200 lines)
301-500| Account endpoints      | api/accounts.py (~200 lines)
501-700| State/metrics endpoints| api/state.py (~200 lines)
701-900| Dashboard endpoints    | api/dashboard.py (~200 lines)
901-1191| Models, utilities     | api/models.py (~290 lines)
```

**FastAPI APIRouter pattern**:
```python
# api/transactions.py
from fastapi import APIRouter

router = APIRouter(prefix="/transaction", tags=["transactions"])

@router.post("/payment")
async def submit_payment(...):
    ...

# api/__init__.py
from fastapi import FastAPI
from .transactions import router as txn_router
from .accounts import router as acct_router
from .state import router as state_router
from .dashboard import router as dash_router

app = FastAPI()
app.include_router(txn_router)
app.include_router(acct_router)
app.include_router(state_router)
app.include_router(dash_router)
```

#### Phase 4: Move Existing Files
```bash
mv workload/src/workload/ws.py workload/src/workload/ws/listener.py
mv workload/src/workload/ws_processor.py workload/src/workload/ws/processor.py
mv workload/src/workload/sqlite_store.py workload/src/workload/storage/sqlite_store.py
```

### File Size Targets

| Module | Target Lines | Complexity |
|--------|--------------|------------|
| api/* | <300 each | Low (routing only) |
| core/* | <500 each | Medium (business logic) |
| txn_factory/* | <300 each | Low (builders) |
| storage/* | <400 each | Medium (I/O) |
| xrpl/* | <400 each | Medium (protocol) |

### Backward Compatibility

**Strategy**: Two-phase migration
1. **Phase 1**: New modules coexist with old files (import re-exports)
2. **Phase 2**: Remove old files after full migration (coordinate in separate PR)

This allows gradual migration without breaking existing code.

### Testing Impact

- **Unit tests**: Must update imports to new module paths
- **Integration tests**: No changes (API unchanged)
- **Manual testing**: No changes (endpoints unchanged)

### Implementation Ordering

1. **P6 (Code Quality)**: Create module structure, extract files, verify
2. **P1-P5**: New features added to new module structure directly
3. **Post-P6**: Remove old files once verified (separate cleanup task)

### References

- Constitution Principle VIII: Code Quality and Maintainability
- FR-030: Large files (>500 lines) SHOULD be refactored
- FastAPI documentation: https://fastapi.tiangolo.com/tutorial/bigger-applications/

---

## 6. xrpl-py Minimal Usage Patterns

### Decision

**Use xrpl-py only for cryptographic operations (signing, hashing); implement custom transaction construction, RPC client, and error handling.**

### xrpl-py Usage Policy

**Use xrpl-py for**:
- Transaction signing: `xrpl.core.keypairs.sign()`
- Hash computation: `xrpl.core.binarycodec.encode_for_signing()`, `xrpl.core.addresscodec.encode_account_public_key()`
- Wallet management: `xrpl.wallet.Wallet` (for key generation only)

**Do NOT use xrpl-py for**:
- Transaction construction: Build dict manually, avoid `xrpl.models.*`
- RPC client: Custom `httpx` or `aiohttp` client with better error handling
- Error code categorization: Custom implementation based on error_codes.md
- WebSocket client: Custom implementation (already in `ws.py`)

### Rationale

xrpl-py pros:
- Well-tested cryptographic operations
- Handles key format conversions (base58, hex, etc.)

xrpl-py cons:
- Heavy dependency (pulls in many sub-dependencies)
- Transaction model classes add abstraction overhead
- Error handling too generic for workload needs
- RPC client lacks fine-grained control for fee escalation

**Custom implementation benefits**:
- Full control over transaction construction
- Custom error categorization aligned with Constitution Principle VI
- Better observability (log exactly what we send/receive)
- Easier to optimize for continuous submission workload

### Implementation Locations

**Custom XRPL module** (`workload/src/workload/xrpl/`):

1. **client.py**: Custom RPC client
   ```python
   class XRPLClient:
       async def submit_transaction(self, tx_blob: str) -> dict:
           # Custom httpx/aiohttp implementation
           # Better timeout handling, retry logic
           ...

       async def get_fee(self) -> FeeInfo:
           # Parse fee RPC response
           ...
   ```

2. **transactions.py**: Transaction construction
   ```python
   def build_payment(
       account: str,
       destination: str,
       amount: str | dict,
       sequence: int,
       last_ledger_sequence: int,
       fee: str = "10"
   ) -> dict:
       return {
           "TransactionType": "Payment",
           "Account": account,
           "Destination": destination,
           "Amount": amount,
           "Sequence": sequence,
           "LastLedgerSequence": last_ledger_sequence,
           "Fee": fee,
       }
   ```

3. **error_codes.py**: Error categorization
   ```python
   @dataclass
   class ErrorCodeInfo:
       code: str
       category: str  # tes, tec, tem, tef, ter, tel
       final: bool
       retryable: bool
       description: str

   ERROR_CODES: dict[str, ErrorCodeInfo] = {
       "tesSUCCESS": ErrorCodeInfo("tesSUCCESS", "tes", True, False, "Success"),
       "tecUNFUNDED_OFFER": ErrorCodeInfo(...),
       ...
   }
   ```

4. **types.py**: XRPL type definitions
   ```python
   from typing import TypedDict

   class Amount(TypedDict):
       currency: str
       issuer: str
       value: str

   class Transaction(TypedDict, total=False):
       TransactionType: str
       Account: str
       Sequence: int
       LastLedgerSequence: int
       Fee: str
       # ... other common fields
   ```

### Migration Strategy

1. **Phase 1**: Create custom xrpl module alongside xrpl-py usage
2. **Phase 2**: Gradually replace xrpl-py RPC calls with custom client
3. **Phase 3**: Replace transaction model usage with custom builders
4. **Phase 4**: Keep xrpl-py for crypto only (sign, hash, wallet)

### References

- User input: "Uses the xrpl-py sparingly and use our own implementation in our own module"
- xrpl-py documentation: https://xrpl-py.readthedocs.io/
- Existing usage: `workload/src/workload/workload_core.py` (imports from xrpl.models)

---

## 7. Swagger UI Example Generation

### Decision

**Use FastAPI Pydantic `Config.json_schema_extra` to provide conditional examples based on environment detection.**

### Implementation Pattern

```python
from pydantic import BaseModel, Field
from typing import Optional

class PaymentRequest(BaseModel):
    destination: str = Field(..., description="Destination account address")
    amount: str = Field(..., description="Amount in drops or issued currency")
    currency: Optional[str] = Field(None, description="Currency code (for issued currencies)")
    issuer: Optional[str] = Field(None, description="Issuer address (for issued currencies)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "destination": "rN7n7otQDd6FczFgLdlqtyMVrn3qHGbXcD",
                    "amount": "1000000",  # 1 XRP in drops
                },
                {
                    "destination": "rN7n7otQDd6FczFgLdlqtyMVrn3qHGbXcD",
                    "amount": "100",
                    "currency": "USD",
                    "issuer": "rPEPPER7kfTD9w2To4CQk6UCfuHM9c6GDY",
                },
            ]
        }
    }
```

### Environment-Based Examples

**Detection strategy**:
```python
import os

def get_testnet_funded_account() -> Optional[str]:
    """Return funded account address if testnet funding available."""
    if os.getenv("TESTNET_FUNDED_ACCOUNT"):
        return os.getenv("TESTNET_FUNDED_ACCOUNT")
    # Check config.toml for funding_account
    return config.get("funding_account", None)

def get_example_destination() -> str:
    """Return example destination (funded account or placeholder)."""
    funded = get_testnet_funded_account()
    return funded if funded else "rN7n7otQDd6FczFgLdlqtyMVrn3qHGbXcD"  # Placeholder
```

**Dynamic example injection**:
```python
def create_payment_examples() -> list[dict]:
    """Generate Payment examples with actual funded accounts if available."""
    dest = get_example_destination()
    return [
        {"destination": dest, "amount": "1000000"},  # 1 XRP
        {"destination": dest, "amount": "100", "currency": "USD", "issuer": "rPEPPER7kfTD9w2To4CQk6UCfuHM9c6GDY"},
    ]

PaymentRequest.model_config["json_schema_extra"]["examples"] = create_payment_examples()
```

### Swagger UI Testing

When testnet funding is available:
1. Swagger UI examples use actual funded account addresses
2. "Try it out" button submits working transactions
3. Responses show real transaction hashes and results

When testnet funding is NOT available:
1. Swagger UI examples use placeholder addresses
2. Submissions fail with tecNO_DST or similar (expected)
3. Still useful for API structure documentation

### Implementation Location

- Pydantic models: `workload/src/workload/api/models.py`
- Example generation: `workload/src/workload/api/models.py` (helper functions)
- Environment detection: `workload/src/workload/config.py` (extend)

### References

- User input: "All API endpoints should have working examples built in (if testnet funding available)"
- FastAPI examples documentation: https://fastapi.tiangolo.com/tutorial/schema-extra-example/
- Pydantic Config documentation: https://docs.pydantic.dev/latest/concepts/json_schema/

---

## 8. Pre-commit Hook Configuration

### Decision

**Use pre-commit framework with ruff hooks for format, lint, and import sorting; add custom hook for docstring checks.**

### Configuration File

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8  # Use latest stable version
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: check-docstrings
        name: Check docstrings and return types
        entry: python scripts/check_docstrings.py
        language: system
        types: [python]
        pass_filenames: true
```

### Custom Docstring Check Script

`scripts/check_docstrings.py`:
```python
#!/usr/bin/env python3
"""Check that all public functions have docstrings and return type annotations."""

import ast
import sys
from pathlib import Path

def check_file(filepath: Path) -> list[str]:
    """Check a Python file for missing docstrings and return types."""
    errors = []

    with filepath.open() as f:
        tree = ast.parse(f.read(), filename=str(filepath))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private functions (start with _)
            if node.name.startswith("_"):
                continue

            # Check docstring
            if not ast.get_docstring(node):
                errors.append(f"{filepath}:{node.lineno}: Missing docstring for {node.name}")

            # Check return type annotation
            if node.returns is None:
                errors.append(f"{filepath}:{node.lineno}: Missing return type for {node.name}")

    return errors

if __name__ == "__main__":
    errors = []
    for filepath in sys.argv[1:]:
        errors.extend(check_file(Path(filepath)))

    if errors:
        print("\n".join(errors))
        sys.exit(1)
```

### Installation and Usage

```bash
# Install pre-commit framework
pip install pre-commit

# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run on staged files only (automatic on commit)
git commit
```

### Hook Behavior

1. **ruff format**: Auto-formats code, fails if changes made
2. **ruff check**: Lints code, auto-fixes safe issues, fails if unfixed errors remain
3. **check-docstrings**: Fails if public methods lack docstrings or return types

### Integration with CI/CD

Add to `.github/workflows/lint.yml` (if using GitHub Actions):
```yaml
name: Lint
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.13'
      - run: pip install pre-commit
      - run: pre-commit run --all-files
```

### References

- Constitution Principle VIII: Code Quality and Maintainability
- FR-028, FR-029: Pre-commit hook requirements
- pre-commit documentation: https://pre-commit.com/
- ruff pre-commit integration: https://github.com/astral-sh/ruff-pre-commit

---

## Research Summary

All Phase 0 research tasks are complete. Key decisions:

1. ✅ **Dynamic Fee Adaptation**: Per-ledger fee querying with adaptive batch sizing
2. ✅ **Dashboard Updates**: FastAPI WebSocket with ledger-triggered push updates
3. ✅ **MPToken Error Codes**: Extended categorization table with disbursement-specific codes
4. ✅ **Offer Error Codes**: Order book-specific error handling with partial fill tracking
5. ✅ **Code Modularization**: FastAPI APIRouter pattern, domain-driven module structure
6. ✅ **xrpl-py Minimal Usage**: Custom implementation for everything except crypto operations
7. ✅ **Swagger Examples**: Pydantic Config with environment-based example generation
8. ✅ **Pre-commit Hooks**: pre-commit framework with ruff + custom docstring checks

**Status**: Ready for Phase 1 (Design & Contracts)
