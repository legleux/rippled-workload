# Future TODO's

## Workload app

### globals in app.py

Why are these globals in app.py and not referenincg workload attributes?
global workload_running, workload_stop_event, workload_task, workload_stats


### Add public methods for info from server_state method

in result.info value we get
      "uptime": 2627,
in result.info.validated_ledger we get
        "base_fee": 10,
        "close_time": 817279063,
        "hash": "89AC77018886354417FC4E8AA60E7098D877F21F403ACB1D94D2964EF590C305",
        "reserve_base": 1000000,
        "reserve_inc": 2000000,
        "seq": 863


## Dashboard
- [ ] Doesn't reflected state.db on hot reload.
- [ ] Move app.py dashboard html to separate file
- [ ] use variable for refresh time in app.py line 687 <div class="subtitle">Live monitoring • Auto-refresh every 1s • Ledger {fee_info.ledger_current_index} @ {hostname}</div>
- [ ] Add tab for Accounts on ledger
- [ ] In the "Top Failures" section, have the failure acutually be a link to the docs for the type of error it is. It's easy to determine since the page is the first three letters of the error code i.e.  terPRE_SEQ would be on the page "ter_codes" like: https://xrpl.org/docs/references/protocol/transactions/transaction-results/ter-codes

## Database
- [ ] Use ORM


## Open Questions

- Test how Batch txns deal with the 10-txn queue per account?
