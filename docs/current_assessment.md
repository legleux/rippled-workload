Uh oh, we got a tecPATH_DRY again. raWpzNAhBu5FYrVqmuUhsPybw1TuWHCPgF sending USD to rUwHnaHDMJAiFzWm6rmwwTDb2vF9rGFjw5 who only has 15 trustlines because a tefPAST_SEQ was submitted presumably. Furthermore the logging after the TrustSet battery says Submission results: {'tecPATH_DRY': 1, 'terPRE_SEQ': 230, 'tesSUCCESS': 25} yet the dashboard doesn' show the terPRE_SEQ errors. I know we're treating them as retryable but I'd still like to know they occurred.
I don't like these "error" messages during trustset submissions during init: "INFO   workload:851 Local error (tel*): telCAN_NOT_QUEUE - TrustSet". How can we avoid that or at least show it's not really an error?

During contiuous workload, we still have large ledger gaps (5 or 6 empty ledgers between submissions) even with plenty of accounts issued and able to operate.
Dashboard improvements:
Speed up the dashboard refresh a little bit.Also the "queue utilization" section and "ledger utilization" doesn't seem not responsive enough, maybe that's just related to it resolving so much faster than the dashboard updates.
We have no set stop point to end trying to grow the ledger. We probably can't submit expected_ledger_size + 1 forever let's cap it to 200 for now.
Do not abbreviate the account_id in log in workload_core line 1191. I've added a "names" field to the gateways in config.toml.
We should associate these names with the gateways we create and use the name during logging instead of the address
When I hit "save" and we hot-reload, I don't want the init to run again if it aleady has. It should only run when we reset the network.
There's 2 ledger gap between TrustSet batches during init. Why is this and how can we mitigate that?

On line app.py line 220 we "# DISABLED: State loading causes sequence number conflicts (terPRE_SEQ)" but maybe since we're not doing that anymore it's ok to bring it back? It's nice to be able to continue the network during debugging rather than having to reset everything all the time. Perphaps we'll need to figure out how to deal with pending/in-flight/unvalidated txns? Trash and forget or save and attempt to continue? Trash and forget probably safer?

Make a TODO: in the docs/ to add a tab for that shows txns by state.

Stuff for Claude to remember:
Note in your testing command list in CLAUDE.md that "dcom down && dcom up -d" is the shell command to restart the docker compose network.
We also need to remove the db with "rm workload_state.db" but I want you to rename the db to "state.db" also so don't forget to note the correct command.
