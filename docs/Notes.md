
- Review the Workload's "INTERNAL_STATES" to determine how necessary they really are.
- generalize the "fan out" method so it can be used for the user creation so users create users and we init faster

# Endpoints
- /txn/flood/{txn_type}

- /txn/shape/{sine,ramp,log,impulse} Create an endpoint that is like "shape/contour" or something that modulates the submission rate to various functions. Can be stacked

- sort

# TODO

## txn submission

When a transaction gets `terPRE_SEQ`, it means its sequence number is too high (the account's current sequence is lower).
These transactions can't validate until earlier transactions complete, but they're blocking the account from submitting new transactions.


## workload_core

- Flesh out bulletproof "wait_for_ledger_close()" method. Separate dedicated thread? process even?
- Add sqlite3 for db debugging

## app.py

- use ajax for dashboard update instead of reloading the page

~92k txns 88M db
