# Data Model: Priority Improvements for Fault-Tolerant XRPL Workload

**Feature**: Priority Improvements (001-priority-improvements)
**Date**: 2025-12-02
**Status**: Complete

This document defines all entities, state machines, and validation rules for the feature implementation.

---

## 1. Transaction State Machine

### States

```
CREATED ────> SUBMITTED ────> VALIDATED (final)
                 │
                 ├────> RETRYABLE ────> (retry cycle)
                 │
                 ├────> REJECTED (final)
                 │
                 ├────> EXPIRED (final)
                 │
                 └────> FAILED_NET (final)
```

### State Definitions

| State | Description | Final | Next States |
|-------|-------------|-------|-------------|
| CREATED | Transaction built and signed locally | No | SUBMITTED |
| SUBMITTED | Sent to rippled node | No | VALIDATED, RETRYABLE, REJECTED, EXPIRED, FAILED_NET |
| RETRYABLE | Temporary failure, can retry (ter/tel codes) | No | SUBMITTED, EXPIRED, FAILED_NET |
| VALIDATED | Confirmed in validated ledger (tes/tec) | Yes | None |
| REJECTED | Terminal rejection (tem/tef codes) | Yes | None |
| EXPIRED | Past LastLedgerSequence without validation | Yes | None |
| FAILED_NET | Network/timeout error | Yes | None |

### State Transitions

| From | To | Trigger | Error Codes |
|------|----|---------|-----------

--|
| CREATED | SUBMITTED | Transaction sent via RPC | - |
| SUBMITTED | VALIDATED | tesSUCCESS or tec* in validated ledger | tesSUCCESS, tecUNFUNDED_OFFER, tecPATH_DRY, tecNO_DST, etc. |
| SUBMITTED | RETRYABLE | ter* or tel* code received | terPRE_SEQ, terQUEUED, telINSUF_FEE_P, telCAN_NOT_QUEUE |
| SUBMITTED | REJECTED | tem* or specific tef* codes | temINVALID, temDISABLED, temBAD_OFFER |
| SUBMITTED | EXPIRED | Validated ledger > LastLedgerSequence | tefMAX_LEDGER or timeout |
| SUBMITTED | FAILED_NET | Network timeout or connection error | - |
| RETRYABLE | SUBMITTED | Retry after ledger close | - |
| RETRYABLE | EXPIRED | Retry timeout or LastLedgerSequence passed | - |
| RETRYABLE | FAILED_NET | Network failure during retry | - |

### Extended Error Code Categorization

**MPToken Transactions**:
| Error Code | Category | State | Description |
|------------|----------|-------|-------------|
| tecNO_DST | tec | VALIDATED | Disbursement destination does not exist |
| tecNO_LINE | tec | VALIDATED | Recipient lacks trust line for MPToken |
| tecNO_PERMISSION | tec | VALIDATED | Not authorized for MPToken operation |
| tecUNFUNDED | tec | VALIDATED | Insufficient MPToken balance |
| temDISABLED | tem | REJECTED | MPToken amendment not enabled |

**Offer Transactions**:
| Error Code | Category | State | Description |
|------------|----------|-------|-------------|
| tecUNFUNDED_OFFER | tec | VALIDATED | Account lacks balance to fulfill offer |
| tecKILLED | tec | VALIDATED | Offer crossed partially, remainder killed |
| tecEXPIRED | tec | VALIDATED | Offer already expired |
| temBAD_OFFER | tem | REJECTED | Offer crosses own order (self-trade) |

---

## 2. Account Entity

### Fields

```python
@dataclass
class Account:
    address: str                              # XRPL account address (rXXX...)
    sequence: int                             # Current sequence number
    next_seq: int | None                      # Next sequence to allocate (cached)
    lock: asyncio.Lock                        # Sequence allocation lock
    balance_xrp: int                          # XRP balance in drops
    balances: dict[str, dict[str, str]]       # Issued currency balances {currency: {issuer: amount}}
    trust_lines: set[tuple[str, str]]         # Trust lines {(currency, issuer)}
    pending_txns: dict[int, str]              # Pending transactions {sequence: tx_hash}
    mptoken_holdings: dict[str, int]          # MPToken balances {token_id: amount}
    active_offers: dict[int, OfferInfo]       # Active offers {offer_sequence: offer_info}
```

### Validation Rules

1. **Sequence Allocation**: Must use per-account lock (Constitution Principle IV)
   ```python
   async def alloc_seq(account: Account) -> int:
       async with account.lock:
           if account.next_seq is None:
               # Fetch from ledger once
               account.next_seq = await fetch_account_sequence(account.address)
           seq = account.next_seq
           account.next_seq += 1
           return seq
   ```

2. **Balance Checks**: Before creating offers, verify sufficient balance
   ```python
   def can_create_offer(account: Account, taker_pays: Amount) -> bool:
       if isinstance(taker_pays, str):  # XRP
           return account.balance_xrp >= int(taker_pays) + RESERVE
       else:  # Issued currency
           currency = taker_pays["currency"]
           issuer = taker_pays["issuer"]
           balance = account.balances.get(currency, {}).get(issuer, "0")
           return Decimal(balance) >= Decimal(taker_pays["value"])
   ```

3. **Trust Line Prerequisites**: Verify trust lines exist before disbursement
   ```python
   def has_trust_line(account: Account, currency: str, issuer: str) -> bool:
       return (currency, issuer) in account.trust_lines
   ```

### Relationships

- **Account → Transaction**: One-to-many (account creates many transactions)
- **Account → MPToken**: Many-to-many (account can hold multiple MPTokens, MPTokens have multiple holders)
- **Account → Offer**: One-to-many (account can have multiple active offers)

---

## 3. MPToken Entity

### Fields

```python
@dataclass
class MPToken:
    token_id: str                             # Unique MPToken identifier (64-char hex)
    issuer: str                               # Issuer account address
    maximum_amount: int                       # Maximum issuable amount
    current_supply: int                       # Currently issued amount
    holders: dict[str, int]                   # Holder balances {address: amount}
    transfer_restrictions: dict[str, Any]     # Transfer restriction flags
    minted_at_ledger: int                     # Ledger index when minted
    minted_at: float                          # Timestamp when minted
```

### Lifecycle States

```
NOT_MINTED ──> MINTED ──> DISBURSED ──> TRADED
                              │
                              └──> BURNED (optional, not in scope)
```

### Validation Rules

1. **Disbursement Prerequisites**:
   - MPToken must be MINTED
   - Recipient must have trust line for MPToken
   - Issuer must have sufficient balance
   - Total disbursed ≤ maximum_amount

2. **Offer Prerequisites** (for MPToken trading):
   - Holder must have MPToken balance > 0
   - Cannot offer more than current balance

### Relationships

- **MPToken → Account (issuer)**: Many-to-one (many MPTokens, one issuer per token)
- **MPToken → Account (holders)**: Many-to-many (via holders dict)
- **MPToken → Offer**: One-to-many (MPToken can have multiple offers)

---

## 4. Offer Entity

### Fields

```python
@dataclass
class OfferInfo:
    account: str                              # Account that created offer
    offer_sequence: int                       # Offer sequence number
    taker_gets: Amount                        # What taker receives (XRP or IOU)
    taker_pays: Amount                        # What taker pays (XRP or IOU)
    expiration: int | None                    # Optional expiration ledger index
    flags: int                                # Offer flags (tfImmediateOrCancel, etc.)
    created_at_ledger: int                    # Ledger index when created
    created_at: float                         # Timestamp when created
    status: OfferStatus                       # Current status

@dataclass
class Amount:
    """XRP amount (str of drops) or issued currency amount (dict)."""
    pass  # Type alias: str | dict[str, str]
```

### Lifecycle States

```
PENDING ──> ACTIVE ──┬──> CROSSED (partial/full)
                     ├──> CANCELLED
                     ├──> EXPIRED
                     └──> UNFUNDED
```

### State Definitions

| State | Description | Final |
|-------|-------------|-------|
| PENDING | Offer transaction submitted, not yet in validated ledger | No |
| ACTIVE | Offer in validated ledger, available for matching | No |
| CROSSED | Offer executed (partially or fully) | Yes |
| CANCELLED | Offer cancelled via OfferCancel transaction | Yes |
| EXPIRED | Offer expired (past expiration ledger) | Yes |
| UNFUNDED | Account no longer has balance to fulfill offer | Yes |

### Validation Rules

1. **Self-Trade Prevention**:
   ```python
   def crosses_own_offer(account: Account, new_offer: OfferInfo) -> bool:
       for existing_offer in account.active_offers.values():
           if offers_would_cross(existing_offer, new_offer):
               return True
       return False
   ```

2. **Balance Verification**:
   ```python
   def verify_offer_balance(account: Account, offer: OfferInfo) -> bool:
       return can_create_offer(account, offer.taker_pays)
   ```

3. **Cancellation Prerequisites**:
   - Offer must be ACTIVE
   - Must use correct offer_sequence
   - Only offer creator can cancel

### Relationships

- **Offer → Account**: Many-to-one (many offers per account)
- **Offer → Currency Pair**: Implicit via taker_gets/taker_pays

---

## 5. Node Metrics Entity

### Fields

```python
@dataclass
class NodeMetrics:
    url: str                                  # Node RPC URL
    server_state: str                         # Server state (full, syncing, etc.)
    ledger_index: int                         # Current validated ledger index
    complete_ledgers: str                     # Complete ledgers range (e.g., "1000-2000")
    queue_size: int                           # Current transaction queue size
    max_queue_size: int                       # Maximum queue size
    fee_base: int                             # Base fee in drops
    fee_ref: int                              # Reference fee level
    open_ledger_fee: int                      # Current open ledger fee in drops
    load_factor: float                        # Server load factor
    peers: int                                # Number of connected peers
    uptime: int                               # Server uptime in seconds
    last_updated: float                       # Timestamp of last update
    reachable: bool                           # Whether node is currently reachable
```

### Update Frequency

- **Triggered by**: Ledger close events (via WebSocket)
- **Fallback**: Periodic polling every 10 seconds if WebSocket fails
- **Retention**: Keep last 100 updates per node for historical queries

### Validation Rules

1. **Staleness Detection**:
   ```python
   def is_stale(metrics: NodeMetrics, max_age_seconds: float = 30) -> bool:
       return time.time() - metrics.last_updated > max_age_seconds
   ```

2. **Reachability Tracking**:
   ```python
   def mark_unreachable(metrics: NodeMetrics, error: Exception) -> None:
       metrics.reachable = False
       metrics.last_error = str(error)
       metrics.last_updated = time.time()
   ```

### Relationships

- **Node → Transaction**: Indirect (node processes transactions)
- **Node → Dashboard**: One-to-one (dashboard displays node metrics)

---

## 6. Dashboard Data Models

### NetworkOverview

```python
@dataclass
class NetworkOverview:
    total_nodes: int                          # Number of configured nodes
    reachable_nodes: int                      # Number of reachable nodes
    current_ledger: int                       # Highest validated ledger index
    ledger_close_time: float                  # Average ledger close time (seconds)
    total_transactions: int                   # Total workload transactions submitted
    validation_rate: float                    # Overall validation success rate (0.0-1.0)
    sequence_conflict_rate: float             # Sequence conflict rate (0.0-1.0)
    ledger_fill_rate: float                   # Ledger fill rate (txns/ledger)
    last_updated: float                       # Timestamp of last update
```

### QueueState

```python
@dataclass
class QueueState:
    node_url: str                             # Node RPC URL
    queue_size: int                           # Current queue size
    max_queue_size: int                       # Maximum queue size
    queue_utilization: float                  # queue_size / max_queue_size
    minimum_fee: int                          # Minimum fee to enter queue (drops)
    open_ledger_fee: int                      # Fee to enter open ledger immediately (drops)
    queued_transactions: list[str]            # List of queued transaction hashes
    last_updated: float                       # Timestamp of last update
```

### TransactionActivity

```python
@dataclass
class TransactionActivity:
    period: str                               # Time period ("last_minute", "last_hour", etc.)
    total_submitted: int                      # Transactions submitted
    validated: int                            # Successfully validated
    rejected: int                             # Terminally rejected (tem/tef)
    expired: int                              # Expired past LastLedgerSequence
    failed_net: int                           # Network failures
    retryable: int                            # Currently retryable (ter/tel)
    by_type: dict[str, int]                   # Transaction counts by type
    validation_latency_p50: float             # Median validation latency (seconds)
    validation_latency_p95: float             # 95th percentile validation latency
```

### ValidationMetrics

```python
@dataclass
class ValidationMetrics:
    total_validations: int                    # Total validated transactions
    validation_rate: float                    # Success rate (0.0-1.0)
    validated_by_polling: int                 # Validated via RPC polling
    validated_by_websocket: int               # Validated via WebSocket
    recent_validations: list[tuple[str, int]] # Recent validations [(tx_hash, ledger_index)]
    sequence_conflicts: int                   # terPRE_SEQ count
    queue_rejections: int                     # telCAN_NOT_QUEUE* count
    last_updated: float                       # Timestamp of last update
```

---

## Data Model Summary

### Entity Count

- **Core Entities**: 4 (Transaction, Account, MPToken, Offer)
- **Metrics Entities**: 5 (NodeMetrics, NetworkOverview, QueueState, TransactionActivity, ValidationMetrics)
- **Total**: 9 entities

### State Machines

- **Transaction**: 7 states (CREATED, SUBMITTED, RETRYABLE, VALIDATED, REJECTED, EXPIRED, FAILED_NET)
- **MPToken**: 3 lifecycle stages (MINTED, DISBURSED, TRADED)
- **Offer**: 5 states (PENDING, ACTIVE, CROSSED, CANCELLED, EXPIRED, UNFUNDED)

### Validation Rules

- **Sequence Allocation**: Per-account locks (Principle IV)
- **Balance Checks**: Before offers and disbursements
- **Trust Line Verification**: Before MPToken disbursement
- **Self-Trade Prevention**: Before offer creation
- **Staleness Detection**: For node metrics

**Status**: Data model design complete, ready for API contract generation
