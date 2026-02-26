Here's some output during ctrl-c initiated shutdown.

```log
2026-02-26 14:00:23 INFO   workload.app:1599 Building batch @ ledger 29271: 32 txns (3136 accounts, 31342 slots, ledger_capacity=32, cap=100)
2026-02-26 14:00:24 INFO   workload.app:1643 📤 Submitting 14 transactions in parallel...
2026-02-26 14:00:26 INFO   workload.app:1599 Building batch @ ledger 29272: 32 txns (3136 accounts, 31339 slots, ledger_capacity=32, cap=100)
^CINFO:     Shutting down
INFO:     Waiting for connections to close. (CTRL+C to force quit)
2026-02-26 14:00:28 INFO   workload:930 Cascade check for rpnMbjFzHGnZuV5huBDvc4KKFDBFe4Yb3W: expired 0 txns with seq > 27733
INFO:     Waiting for application shutdown.
2026-02-26 14:00:28 INFO   workload.app:288 Shutting down...
2026-02-26 14:00:28 INFO   workload.ws:103 WS listener received stop signal
2026-02-26 14:00:28 INFO   workload.ws_processor:94 WS event processor stopped (processed 969 events)
2026-02-26 14:00:29 INFO   workload.app:1643 📤 Submitting 17 transactions in parallel...
2026-02-26 14:00:30 INFO   workload.app:1599 Building batch @ ledger 29273: 32 txns (3137 accounts, 31341 slots, ledger_capacity=32, cap=100)
2026-02-26 14:00:31 INFO   workload.app:1643 📤 Submitting 17 transactions in parallel...
2026-02-26 14:00:32 INFO   workload.app:1599 Building batch @ ledger 29274: 32 txns (3137 accounts, 31331 slots, ledger_capacity=32, cap=100)
2026-02-26 14:00:33 INFO   workload.app:293 Exiting TaskGroup (will cancel any remaining tasks)...
2026-02-26 14:00:33 INFO   workload.app:295 Shutdown complete
INFO:     Application shutdown complete.
INFO:     Finished server process [1756045]
```

Do we do any updating of the db or cancel current in-flight txns we have unaccounted for before shutting down? Just trying to make sure we're not slowing leaking accounts.
