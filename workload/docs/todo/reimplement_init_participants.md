# Reimplement `init_participants` — Organic Genesis Init

## Status: P0 TODO (removed from codebase 2026-03-26)

## What this was

`init_participants()` was an 8-phase init sequence that built a full testnet state from scratch — no pre-generated `ledger.json`. It funded gateways, set flags, funded users, established trust lines, distributed tokens via a fan-out tree, and created AMM pools. All by submitting real transactions and waiting for validation.

## Why it was removed

1. **Broken** — It predated the `generate_ledger` package which now handles all this via pre-genesis `ledger.json`. The code was never updated to work with the current account/sequence management.
2. **Absurdly complex** — CC 94, cognitive complexity 145. Single worst function in the codebase.
3. **Painfully slow** — Sequentially submitted transactions through consensus, waiting for validation barriers between phases. Funding 1000 accounts took minutes because it bottlenecked on one funding wallet's sequence.

## What needs to be reimplemented

An efficient way to build testnet state from an empty ledger without a pre-generated `ledger.json`. This is needed for:
- Public devnet/testnet targeting (faucet-funded, no genesis control)
- Environments where `generate_ledger` isn't available
- Dynamic account creation during long-running tests

## Key design constraints

- **The funding wallet is the bottleneck** — one account, sequential sequences. Can't parallelize submissions from the same account without sequence management.
- **Ledger throughput is finite** — ~200-300 txns per ledger close (~3.5s). Stuffing more into the queue doesn't help if they expire.
- **Account creation requires funding first** — new accounts don't exist until a Payment creates them. Trust lines require the account to exist. Token distribution requires trust lines. AMM creation requires tokens. The dependency chain is real.

## The old approach (8 phases, sequential)

1. Fund gateways (sequential from funding wallet)
2. Set gateway flags (AccountSet)
3. Fund users in batches of 10 (sequential from funding wallet)
4. TrustSets (N users × M currencies)
5. Token seeding to sqrt(N) seed users from gateways
6. Fan-out from seed users to remaining users
7. Gateway AMM pool creation (XRP/IOU)
8. User AMM pool creation (IOU/IOU)

Each phase waited for ALL transactions to validate before starting the next. Phase 3 alone (funding 1000 users) took dozens of ledger closes because it submitted in batches of 10 from a single funding wallet.

## Better approach (ideas, not finalized)

**Goal:** Fill ledgers as full as possible, minimize sequence tracking overhead, avoid waiting between phases where dependencies don't require it.

- **Parallel funding wallets** — Pre-fund N intermediate wallets in the genesis account, then fan out from all N simultaneously. Each intermediate wallet handles its own sequence independently.
- **Pipeline phases** — Don't wait for ALL of phase 3 before starting phase 4. As soon as a user is funded and confirmed, start their trust lines immediately. Overlap phases where the dependency is per-account, not per-phase.
- **Batch submission without per-account tracking** — During init, we don't need the full `build_sign_and_track` / `submit_pending` pipeline. We know the sequences ahead of time (starting from 0 for new accounts). Build all txns upfront, blast them at rippled, and poll for finality in bulk.
- **Accept partial success** — If 950/1000 trust lines validate, that's fine. Don't block on the last 50.

## Helper methods also removed

- `_init_batch()` — build/sign/submit a list of (Transaction, Wallet) pairs
- `_submit_and_wait_batched()` — submit + poll for validation
- `_submit_batched_by_account()` — group submissions by account to manage sequences
- `_wait_all_validated()` — poll until all PendingTx reach terminal state

These were tightly coupled to `init_participants` and not used elsewhere.

## Where the call site was

`bootstrap.py:lifespan()` — the `else` branch when neither SQLite nor genesis state is available:

```python
# This now logs an error instead of calling init_participants
else:
    log.error("No genesis accounts found. Use 'gen' to create a testnet with pre-provisioned accounts.")
```
