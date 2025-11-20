# WebSocket Integration - Architecture Diagram

## Current Architecture (After Integration)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                          │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Lifespan TaskGroup                         │  │
│  │                                                               │  │
│  │  ┌────────────────┐     ┌──────────────┐     ┌────────────┐ │  │
│  │  │  ws_listener   │────▶│ Event Queue  │────▶│ ws_processor│ │  │
│  │  │                │     │              │     │            │ │  │
│  │  │ • Connects WS  │     │ maxsize:1000 │     │ • Consumes │ │  │
│  │  │ • Subscribes   │     │              │     │ • Updates  │ │  │
│  │  │ • Publishes    │     └──────────────┘     │   Workload │ │  │
│  │  └────────────────┘                          └─────┬──────┘ │  │
│  │          │                                          │        │  │
│  │          │                                          │        │  │
│  │          │         ┌────────────────────────────────┘        │  │
│  │          │         │                                         │  │
│  │          │         ▼                                         │  │
│  │          │  ┌──────────────────┐                            │  │
│  │          │  │  Workload        │                            │  │
│  │          │  │                  │                            │  │
│  │          │  │ • pending{}      │                            │  │
│  │          │  │ • store          │                            │  │
│  │          │  │ • wallets{}      │                            │  │
│  │          │  └────────┬─────────┘                            │  │
│  │          │           │                                      │  │
│  │          │           │ Fallback                             │  │
│  │          │           ▼                                      │  │
│  │          │  ┌─────────────────────┐                        │  │
│  │          │  │ finality_checker    │                        │  │
│  │          │  │                     │                        │  │
│  │          │  │ • Polls every 5s    │                        │  │
│  │          │  │ • RPC Tx() request  │                        │  │
│  │          │  │ • Catches edge cases│                        │  │
│  │          │  └──────────┬──────────┘                        │  │
│  │          │             │                                   │  │
│  └──────────┼─────────────┼───────────────────────────────────┘  │
│             │             │                                       │
└─────────────┼─────────────┼───────────────────────────────────────┘
              │             │
              │             │
              ▼             ▼
    ┌─────────────────────────────┐
    │       rippled Node          │
    │                             │
    │  ┌──────────────────────┐   │
    │  │  WebSocket (6006)    │   │
    │  │                      │   │
    │  │  Streams:            │   │
    │  │  • transactions      │   │
    │  │  • ledger           │   │
    │  └──────────────────────┘   │
    │                             │
    │  ┌──────────────────────┐   │
    │  │  JSON-RPC (5005)     │   │
    │  │                      │   │
    │  │  Methods:            │   │
    │  │  • submit            │   │
    │  │  • tx               │   │
    │  │  • account_info     │   │
    │  └──────────────────────┘   │
    └─────────────────────────────┘
```

---

## Transaction Lifecycle Flow

### 1. Transaction Submission (RPC - Unchanged)

```
User Request
    │
    ▼
FastAPI Endpoint
    │
    ▼
Workload.submit_random_txn()
    │
    ├─▶ build_sign_and_track()
    │       │
    │       └─▶ Creates PendingTx(state=CREATED)
    │
    ▼
Workload.submit_pending()
    │
    └─▶ RPC: SubmitOnly(tx_blob)
            │
            ▼
        rippled Node
            │
            └─▶ Returns engine_result
                    │
                    ▼
                PendingTx(state=SUBMITTED)
```

### 2. Transaction Validation (WebSocket - NEW)

```
rippled validates txn in ledger
    │
    ├─▶ Broadcasts to WS stream "transactions"
    │
    ▼
ws_listener receives message
    │
    ├─▶ Parses JSON
    ├─▶ Identifies type="transaction", validated=true
    ├─▶ Extracts tx_hash, ledger_index, meta
    │
    └─▶ queue.put(("tx_validated", data))
            │
            ▼
        Event Queue
            │
            ▼
    ws_processor.process_ws_events()
            │
            ├─▶ queue.get() with timeout
            ├─▶ Checks if tx_hash in workload.pending
            │
            └─▶ Calls workload.record_validated()
                    │
                    ├─▶ Updates PendingTx(state=VALIDATED)
                    ├─▶ Stores ValidationRecord(src="WS")
                    ├─▶ Increments validated_by_source["WS"]
                    └─▶ Adopts wallet if funding payment
```

### 3. Fallback Validation (RPC Polling - Existing)

```
periodic_finality_check() runs every 5s
    │
    ├─▶ For each tx in pending with state=SUBMITTED:
    │       │
    │       └─▶ RPC: Tx(transaction=tx_hash)
    │               │
    │               └─▶ If validated:
    │                       │
    │                       └─▶ record_validated(src="POLL")
    │
    └─▶ Catches transactions WS missed
        (network hiccup, subscription issue, etc.)
```

---

## State Machine

```
Transaction States:

CREATED ────────▶ SUBMITTED ────────▶ VALIDATED (terminal)
                     │                     ▲
                     │                     │
                     ├──▶ RETRYABLE        │
                     │        │            │
                     │        └────────────┘
                     │
                     ├──▶ REJECTED (terminal)
                     │
                     ├──▶ EXPIRED (terminal)
                     │
                     └──▶ FAILED_NET (terminal)


Validation Sources:

    ValidationSrc.WS ────┐
                         ├──▶ record_validated() ──▶ VALIDATED
    ValidationSrc.POLL ──┘

    (Only one ValidationRecord per (txn, ledger) even if both fire)
```

---

## Data Flow Diagram

```
                    ┌─────────────────────────────────────┐
                    │          InMemoryStore              │
                    │                                     │
Event Flow:         │  _records: {tx_hash: {...}}       │
                    │  validations: deque[...]           │
WS Stream ──┐       │  count_by_state: {...}            │
            │       │  validated_by_source: {           │
            ▼       │    "WS": 950,                     │
        ┌───────┐   │    "POLL": 50                     │
        │ Queue │   │  }                                │
        └───┬───┘   └─────────────────────────────────────┘
            │                      ▲
            │                      │
            ▼                      │
    ┌────────────────┐            │
    │  ws_processor  │────────────┘
    │                │
    │  • Gets event  │         ┌──────────────────┐
    │  • Validates   │────────▶│    Workload      │
    │  • Updates     │         │                  │
    └────────────────┘         │  pending: {      │
                               │    tx_hash: PendingTx(
            ┌──────────────────┤      state,      │
            │                  │      attempts,   │
    RPC Polling                │      validated_ledger,
    (Fallback)                 │      ...         │
            │                  │    )             │
            └──────────────────▶  }               │
                               └──────────────────┘
```

---

## Concurrent Tasks

```
FastAPI Lifespan:
    │
    ├─▶ [Task 1] ws_listener
    │       │
    │       ├─ Persistent WS connection
    │       ├─ Reconnects on failure
    │       └─ Publishes to queue
    │
    ├─▶ [Task 2] ws_processor
    │       │
    │       ├─ Consumes from queue
    │       ├─ Updates workload state
    │       └─ Handles errors gracefully
    │
    └─▶ [Task 3] finality_checker
            │
            ├─ Polls RPC every 5s
            ├─ Checks SUBMITTED transactions
            └─ Catches WS misses

All tasks share:
    • Same stop Event (graceful shutdown)
    • Same Workload instance (state coordination)
    • Independent error handling (failure isolation)
```

---

## Message Flow Example

```
Time: T+0s
User: POST /transaction/random
    │
    └─▶ Workload: submit_pending() via RPC
            │
            └─▶ rippled: Accepts (engine_result=tesSUCCESS)
                    │
                    └─▶ PendingTx(state=SUBMITTED)

Time: T+3s (next ledger closes)
rippled: Validates transaction in ledger 12345
    │
    ├─▶ WS stream: Broadcasts validation
    │       │
    │       └─▶ ws_listener: Receives
    │               │
    │               └─▶ Queue: Adds event
    │
    └─▶ (RPC poll hasn't fired yet - would at T+5s)

Time: T+3.1s
ws_processor: Gets event from queue
    │
    └─▶ Workload.record_validated(src=WS)
            │
            ├─▶ PendingTx(state=VALIDATED, validated_ledger=12345)
            ├─▶ ValidationRecord(txn, seq=12345, src="WS")
            └─▶ validated_by_source["WS"] += 1

Time: T+5s
finality_checker: Polls for validation
    │
    └─▶ RPC: Tx(transaction=tx_hash)
            │
            └─▶ Already VALIDATED (WS beat us to it!)
                    │
                    └─▶ No-op (deduplication in store.mark())

Result: User gets validation in ~3s instead of ~5s average
```

---

## Error Handling Matrix

```
Component: ws_listener
Error: Connection failed
Action:
    • Log error
    • Wait (exponential backoff)
    • Reconnect automatically
    • Continue with RPC polling

Component: ws_processor
Error: ValidationRecord fails
Action:
    • Log error with tx_hash
    • Continue processing next event
    • Transaction caught by RPC poll fallback

Component: finality_checker
Error: RPC timeout
Action:
    • Log error
    • Skip this check cycle
    • Retry on next interval (5s)

Component: Event Queue
Error: Queue full (1000 events)
Action:
    • ws_listener blocks on put()
    • Backpressure to rippled
    • Prevents memory overflow
```

---

## Metrics & Observability

```
Endpoint: /state/ws/stats

Returns:
{
  "queue_size": 3,              # Current events waiting
  "queue_maxsize": 1000,        # Maximum capacity
  "validations_by_source": {
    "WS": 847,                  # Fast path wins
    "POLL": 23                  # Fallback catches edge cases
  },
  "recent_validations_count": 870  # Total in deque
}

Health Indicators:
✓ queue_size < 100            (processor keeping up)
✓ WS > 80% of validations     (primary path working)
✓ POLL < 20% of validations   (fallback only for edge cases)
```

---

This architecture provides:
- **Performance:** Real-time validation via WS stream
- **Reliability:** RPC polling as automatic fallback
- **Observability:** Clear metrics on validation sources
- **Maintainability:** Isolated components with clean interfaces
- **Safety:** No single point of failure

**The best of both worlds!** 🚀
