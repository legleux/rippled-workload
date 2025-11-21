# On txn submission

When a transaction gets `terPRE_SEQ`, it means its sequence number is too high (the account's current sequence is lower).
These transactions can't validate until earlier transactions complete, but they're blocking the account from submitting new transactions.


# TODO

- Add sqlite3 for db debugging
