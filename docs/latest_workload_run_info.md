# Init info

Results of [✓] or [✗]

## Gateway and user setup

### Account creation via Payment
- [✓] [2] ledgers of account creation
    - [✓] [4] Gateways with [4] AccountSets
    - [✓] [16] User accounts created successfully.

### TrustSets and token distribution
- [✗] [4] empty ledgers gap then [5] ledgers of User account TrustSets for gateways 33 + 40 + 49 + 56 + 8 + 53 + 5 = 244
    - Oof, we missed even more!!!! Nooooooo!
    Didn't even get sumission results this time!
2025-11-24 11:00:54 INFO   workload:1324   TrustSet: Submitting batch of 160 txns (160/256 total)
- [✓] [7] empty ledgers gap then [3] ledgesrs of Payments of token disbursement 86 + 104 + 66 = 256
2025-11-24 11:01:42 INFO   workload:1433   Submission results: {'tesSUCCESS': 12, 'terPRE_SEQ': 27, 'tecPATH_DRY': 1}
2025-11-24 11:01:42 INFO   workload:1438 Token distribution complete at 107: 13/256 validated

### Completion

`/state/summary` reports [] validated txns should be [20] Payments for gw and users, + [4] AccountSets for , [256] TrustSets [256] Payments
[✓] 512 + 20 + 4 = 536

# Continuous Workload Info
[✓] At least one of every txn type enabled succeed.
    - Maybe not offers crossed? didn't check
[✗] At least 90% VALIDATED rate
    - VALIDATED rate 65.6%
[✗] At most 15% REJECTED rate
    - REJECTED rate 11.5% - but I don't even trust that!

# Notes
"workload.ws:86 Subscribing to ledger and server + 2 specific accounts" only 2?
We've gotten much worse init this time
Dashboard states:
    290 EXPIRED
    telCAN_NOT_QUEUE 1
    terPRE_SEQ 22

Stopped it around ledger 250 and when we settled EXPIRED and terPRE_SEQ + None! matched at 1837 txns :(.

Ledger txn mix still looks good but the tefPAST_SEQ is just rampant!
stopped barely at 200 with final results:
155 REJECTED

tefPAST_SEQ 155
terPRE_SEQ 35
telCAN_NOT_QUEUE 1
