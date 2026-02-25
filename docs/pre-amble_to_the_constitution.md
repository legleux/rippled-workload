# Workload - A Working Title for the App

Welcome! We're neck deep in a ambitious project of creating a "Workload" application responsible for generating traffic for the XRPL ledger for a company called [Antithesis](https://antithesis.com/docs/introduction/how_antithesis_works/) to do fuzz-testing while we run our Workload. There's a slew off accompanying docs in the `docs` dir to check out.
Start with `DEVELOPMENT.md` obviously but at least read these right now also:
- `reliable-tx-submission.svg` - This is our roadmap, our Bible. This is what we based our workflow on and we must improve upon it.
- `Finality_of_Results.md` - name says it all.
- `FeeEscalation.md` - insights on how txns are processed
- `error_codes.md` - insights on how to handle errors we receive

There's also some other files, todos, random notes and `rippled` source files but we can take a look at them later.

## Priority **NOW** - ASAP - Like YESTERDAY!

**We _must_ get these items done as quickly as possible, as soon as possible.**

1. Get the continous submission phase to 90% successes and <10% outright failures. Do not count unfunded offers, tecPATH_DRY, and `ter` codes as outright failures as they are legitimate traffic a ledgere might experience and also should be replicated. We have some significant issues with sequence tracking now but as long as we can get decent throughput, I'm working on that in another project. The key result I want now is contiguous ledgers being filled and if it's going to require more accounts at the outset, so be it.
2. Complete a MPToken txns workflow. We only have minting at the moment we need to disburse and create offers on them also.
3. We need to get offers crossing. We need OfferCreate txns and OfferCancel txns.
4. Each txn-specific endpoint must have the API correct parameters avaible in the Swagger UI to allow arbitrary txns submissions by a user.
5. The dashboard should have tabs with some basic info about the nodes in the network (output from server_info, current queue, txns, validations etc) maybe this should be a separate page alltogether.
6. The ruff formatting must be finalized and all the code linted, formatted and public methods documented. Subsequent submissions must be gated with pre-commit configs. The code should also be made more modular as it's very few files right now and they're getting pretty large and inconsistent.

## The Road to 1.0
- We must have versionable packaging via uv-build or hatch.
- Must be able to be pointed up the the actual Testnet and Devnet vs only being used for custom local networks.
- Actual per-account Balance trackfing for all assets not just XRP.
- Some semblance of testing
- Tiny Python app deployed with each `rippled` node so the workload can remotely
  - 1. Start/stop
  - 2. Upgrade/downgrade


Create a template from the `latest_workload_run_info.md` file to be used to assess the efficacy of changes made in each iterations of our work.


## Later... but soon
We'll need to use the primitives and tools we've created here to use Antithesis' [Test Composer](https://antithesis.com/docs/test_templates/) to create interesting scenarios to fuzz. Some ideas off the top of my head.
- Muliple validators updating at the same time
- validators updating to incompatible versions
- Multiple UNLs defined with varying degrees of overlapping nodes
- Memecoin drop! A huge influx of traffic around a particular NFT, MPT which requires tons of AccountSet TrustSet txns all directed at a few account's assets.
- I have another project that will be capable of pre-generating our ledger so that the initial setup phases can be omitted since our primitives will already exist from the genesis ledger. We'll retain the setup phases though because sometimes we'll want purely "organic" evolution.

We should probably have an agent dedicated to checking our code is up to our standards and is unbiased in critisizing Python anti-patterns and suggesting DRY-ness.

## Some Less Organized Leavings

These should be incorparted into future features/specs

- Review the Workload's `INTERNAL_STATES` to determine how necessary they really are.
- Generalize the "fan out" method so it can be used for the user creation so users create users and we init faster

### Additional Endpoints to Create
- `/txn/flood/{txn_type}`

- `/txn/shape/{sine,ramp,log,impulse}` Create an endpoint that is like "shape/contour" or something that modulates the submission rate to various functions. Can be stacked

### Txn Submission

When a transaction gets `terPRE_SEQ`, it means its sequence number is too high (the account's current sequence is lower).
These transactions can't validate until earlier transactions complete, but they're blocking the account from submitting new transactions.

### File-specific Improvements

#### `workload_core.py`

- Flesh out bulletproof `wait_for_ledger_close()` method. Separate dedicated thread or processes even?
- Add sqlite3 for db debugging

#### `app.py`

- Remove the dashboard from `app.py` and convert it to use ajax for dashboard update instead of reloading the whole page.
