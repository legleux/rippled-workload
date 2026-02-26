Network created with

`gen auto -o testnet -v 5 -n 40 -t "0:1:USD:1000000000"`

Log says:

```log
2026-02-26 15:19:41 INFO   workload:456 Loading from genesis: 1008 accounts (6 gateways, 1000 users)
2026-02-26 15:20:15 INFO   workload:484 Genesis loaded: 6 gateways, 1000 users, 24 currencies
2026-02-26 15:20:15 INFO   workload.ws_processor:57 WS event processor starting
2026-02-26 15:20:15 INFO   workload.ws:51 WS connected: ws://localhost:6006
2026-02-26 15:20:15 INFO   workload.ws:73 Subscribing to ledger and server + 1009 specific accounts
```

why'd it created 1000 accounts with those params. also why are 1000 accounts + 6 gw = 1008 what are the other 2 (the manually created AMMs?)
