# WebSocket Validation Architecture

How the workload tracks transaction validation in real time via a persistent WebSocket connection to rippled, with RPC polling as a fallback.

## Architecture

```
┌─────────────┐      ┌─────────────┐      ┌──────────────┐
│ WS Listener │ ───> │ Event Queue │ ───> │ WS Processor │
│  (ws.py)    │      │ (maxsize    │      │ (ws_proces-  │
│             │      │    1000)    │      │   sor.py)    │
└─────────────┘      └─────────────┘      └──────────────┘
       │                                         │
  subscribes to                          calls record_validated()
  rippled streams                        with src=WS
       │                                         │
       ▼                                         ▼
   rippled                                ┌──────────┐
       ▲                                  │ Workload │
       │                                  └──────────┘
  polls every 5s                                 ▲
       │                                         │
┌──────────────────┐          calls record_validated()
│ Finality Checker │          with src=POLL      │
│ (workload_core)  │─────────────────────────────┘
└──────────────────┘
```

Both paths call `workload.record_validated()`, which deduplicates by `(tx_hash, ledger_index)` so a transaction is only recorded once even if both paths see it.

## Components

### WS Listener (`ws.py`)

Maintains a persistent WebSocket connection to rippled and publishes typed events to an `asyncio.Queue`.

- Subscribes to `transactions`, `ledger`, and `server` streams
- Accepts an optional `accounts_provider` callback to narrow subscriptions to known accounts (falls back to broad streams during startup)
- Reconnects with exponential backoff (1s base, 10s cap)

Events published:

| Event type | Trigger |
|---|---|
| `tx_validated` | Transaction with `validated=true` |
| `tx_response` | Message with `engine_result` (reserved for future WS submission) |
| `ledger_closed` | New ledger close |
| `server_status` | Server status change |

### WS Processor (`ws_processor.py`)

Consumes events from the queue and dispatches them:

| Event | Handler | Effect |
|---|---|---|
| `tx_validated` | `_handle_tx_validated` | Records validation via `workload.record_validated(src=WS)` |
| `ledger_closed` | `_handle_ledger_closed` | Fetches ledger data, runs Antithesis assertions |
| `server_status` | `_handle_server_status` | Updates load factor multipliers |
| `tx_response` | `_handle_tx_response` | Placeholder for future WS submission |

Events are processed **sequentially** (parallel processing caused blocking issues).

### Finality Checker (`workload_core.py`)

RPC polling fallback that runs every 5 seconds. For each `SUBMITTED` transaction, issues an `Tx` RPC call and records validation with `src=POLL` if confirmed.

### Validation Sources

```python
class ValidationSrc(StrEnum):
    POLL = auto()   # From RPC polling
    WS = auto()     # From WebSocket stream
```

Under normal operation, the vast majority of validations come via WS. Polling catches edge cases (transactions validated during WS reconnects, etc.).

## Lifecycle

All three tasks run concurrently in an `asyncio.TaskGroup` spawned from `app.py:lifespan()`:

1. `ws_listener` — connects to rippled, publishes events to queue
2. `process_ws_events` — consumes queue, updates workload state
3. `periodic_finality_check` — polls RPC as fallback

## Deduplication

`InMemoryStore.mark()` checks `(tx_hash, ledger_index)` before appending a `ValidationRecord`. If the same transaction is seen by both WS and polling, only the first arrival is recorded.

## Monitoring

```bash
# Validation source breakdown and queue depth
curl http://localhost:8000/state/ws/stats

# Recent validations with source tags
curl http://localhost:8000/state/validations
```

The `/state/summary` endpoint includes `validated_by_source` counts (WS vs POLL).

## Troubleshooting

### Queue backing up
`/state/ws/stats` shows `queue_size` near 1000.

The processor can't keep up with the event rate. Profile `record_validated()` for bottlenecks or increase the queue size in `app.py`.

### No WS validations
`validated_by_source["WS"]` stays at 0.

Check in order:
1. WS listener connected — look for `"WS connected"` in logs
2. Subscription succeeded — look for `"WS subscription successful"`
3. Queue receiving events — check `queue_size` in `/state/ws/stats`
4. Processor running — look for `"WS event processor starting"`

### WS disconnects
The listener reconnects automatically with exponential backoff. During the reconnect window, polling covers validation tracking. No manual intervention needed.
