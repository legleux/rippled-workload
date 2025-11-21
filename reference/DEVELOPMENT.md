# Development Principles

## Code Quality Standards

### Method Contracts
- **If your method cannot perform what its name states, throw an error**
  - Example: `get_user()` must raise if user doesn't exist, not return None silently
  - Use explicit error handling over implicit failure modes
  - Better to fail fast than continue with invalid state

### Documentation
- **Docstring in every public method**
  - Include: purpose, parameters, return value, exceptions raised
  - Use Google-style docstrings (configured in pyproject.toml)
  - Private methods (`_method`) should have docstrings if logic is non-trivial

### State Management
- **Fail loudly on state inconsistencies**
  - Assert invariants explicitly (number of currencies, accounts, etc.)
  - Log state transitions at INFO level for critical operations
  - Use Antithesis assertions for properties that must always/sometimes hold

### Type Safety
- **All public methods must have return type annotations** (enforced by ruff ANN201)
- **Use type hints for all parameters**
- **Avoid `Any` unless truly necessary**

### Logging
- **INFO**: User-visible state changes, major operations starting/completing
- **DEBUG**: Implementation details, loop iterations, cached values
- **WARNING**: Recoverable errors, degraded operation, timeouts with fallbacks
- **ERROR**: Unrecoverable errors, data inconsistencies

### Testing Principles
- **Workload must be resumable from any SQLite checkpoint**
  - All critical state (wallets, currencies, sequences) must persist
  - Load operations must validate consistency (e.g., gateways exist → currencies exist)
- **Transaction lifecycle tracking must be accurate**
  - Every state transition must be recorded in store
  - Terminal states are immutable once reached
  - Open states must eventually reach terminal state or timeout

### Separation of Concerns
- **Transaction generation (txn_factory) should be pure logic**
  - Context carries data, not state management
  - No direct access to workload state from builders
  - Keep account selection logic separate from transaction building
- **State management belongs in Workload class**
  - Sequence allocation, pending tracking, validation recording
  - Account lifecycle (funding, adoption, balance tracking)

### Error Recovery
- **Network errors (FAILED_NET) are not terminal**
  - Transaction may have been accepted despite timeout
  - Finality checker must poll for eventual validation or expiry
- **terPRE_SEQ means sequence conflict**
  - Mark as RETRYABLE, check for expiry
  - Do not retry automatically (would cause duplicate sequence consumption)
- **tem/tef codes are terminal rejections**
  - No retry possible, mark as REJECTED immediately

## Code Style
- Use `ruff` for linting and formatting (run before committing)
- Line length: 120 characters
- Prefer explicit over implicit
- Avoid premature optimization - clarity first, performance second

## Architecture Notes
- **Domain layer**: Pure data types (WalletModel, IssuedCurrencyModel)
- **Infrastructure layer**: External I/O (XRPL client, SQLite, network)
- **Application layer**: Coordination logic (txn_factory, account generation)
- **Interface layer**: Entry points (FastAPI, CLI)

## TODO: Tooling
- [ ] Apply GitHub's Speckit (or similar specification framework)
- [ ] Add pre-commit hooks for ruff checks
- [ ] Set up property-based testing for transaction generation
- [ ] Add mutation testing to verify error handling paths

## Performance Considerations
- **Batch operations where possible** (TrustSets, Payments, transaction submission)
- **Rate limit RPC calls** during heavy load (balance updates disabled when pending > 50)
- **Parallel submission** but sequential account usage (one tx per account at a time)
- **Lazy loading** - don't fetch account info until sequence needed

## Anti-Patterns to Avoid
- ❌ Silent failures (returning None/empty when operation fails)
- ❌ Mixing open/terminal states incorrectly (FAILED_NET is open, not terminal)
- ❌ Assuming RPC calls succeed (always check is_successful())
- ❌ Reusing accounts with pending transactions (causes terPRE_SEQ)
- ❌ Modifying terminal state (VALIDATED/REJECTED/EXPIRED are final)
- ❌ Using debug logs for critical user-visible information
