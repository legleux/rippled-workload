## Dashboard

1. Node WebSocket Connection dropdown broken
2. TPS slider doesn't work
3. "Top Failures" list aging out? Also, clicking on "Top Failures" should bring you to "Full Failures" of similar form
4. Put "Ledger Stream" _above_ "Transaction Control"
5. Start Stop broken
6. Full HTML/CSS assessment/critique/refactor

find how many calls and assess if they are necessary
    1. sleep
    2. time methods we are using
    3. lambdas

Let's work on the fast API swagger UI and  make it easy to use
if I open the payment one, how quickly can I submit a paymwnt

We need a full itemization review of the API endpoints,
1. what's cruft
2. What's broken
3. What's duplicated/redundant


## CreateOffer

for "good" txns (definitely cross)
    1. Filter book_offers that _will_ cross (at least at time of lookup, assert sometimes)
    2. only pick from  txn_ctx that _will_ cross (guaranteed, assert always)

for bad offers
    1. Offer too low can't cross
    2. malformed

We'll have to peruse the xrpld source code and determine how to generate the ledger indexes of all the ledger objects
that _can_ be deterministically computed.

Dedupe any repeated code between generate_ledger's indices.py and what you just added so we can have that free-standing
module to compute indexes.


Rethink the submission cadence
  ledger close no good
  have N "producers" that are scheduled such a target txns/per second rate can be met?

generate_ledger has been refactored such that the command "gen" is just the sane default so to create a network that's
all we need. However for normal testing, we'll just assume the testnet dir is already generated. and all you'll need to
do is "docker compose -d down/up" and uv run workload (or compose or whatever the eff we call it now)
