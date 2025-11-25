# Initialization Info

REsults:
[✓] - Pass
[-] - Meh
[✗] - Fail
[?] - No metric

### Phase 1 - Gateway funding

[✓] - Fine

### Phase 2 - Gateway configuration

[✓] - Fine

### Phase 3 - User funding

[-] - Took 53s for the 96 users, let's not mess with it unless that's where the errors came form

### Phase 4 - User TrustSets configuration

- [-] Still took Over 3 minutes. Like to see it under 1 but let's stick with this setup for now since it actually works.
    The TaskGroup did great fill the ledger up immediately but we still had gaps between ledgers that killed our time!

### Phase 5 - Token distribution

[✗] -  Reported before validation completed "Phase 5: Token seeding: 28/144 validated, 14.11s, 5 ledgers"
    All 144 showed up in the ledger

### Init Completion

[-] - We did have  6 terPRE_SEQ during init
[-] - Added 32 more users and still too over 4 minutes.  just a bit long.
[✗] - Also reported "TOTAL: 1668/1784 validated, 283.05s, 93 ledgers" but the summary showed everyone got validated actually:
oof, we _did_ get some EXPIREDs, I don't like that but we might need to deal with it for now.
{
  "total_tracked": 1784,
  "by_state": {
    "VALIDATED": 1668,
    "EXPIRED": 116
  },
  "gateways": 4,
  "users": 96
}
# Continuous Workload Info

[?] At least one of every txn type enabled succeed.
    - Maybe not offers crossed? We should start watching for this somehow
[✗] At least 90% VALIDATED rate
    - VALIDATED rate 81.9%
[✓] At most 15% REJECTED rate
    - REJECTED rate 13.0%
[?] 50% offers crossed
    - Not tracking yet

# Notes
Let it run for a loooong time
First txn was almost 200 (192) but still had gap of 5 ledgers until next txn-filled ledger so more accounts didn't help that much.
It also just immediately started with "tefPAST_SEQ" responses still!!!
We really have so many "Reset next_seq for ..." earnings is there a way to mitigate that?
Still tracking _tons_ of tefPAST_SEQ and terPRE_SEQ errors! We got to find a way to manage these better!
Still getting gaps in between full ledgers - how can we smooth out the traffic?
Having more accounts did _NOT_ smooth the throughput out appreciably.
Perhaps insteead always trying to max a ledger out we can divided our pool of potential accounts for the next submission into a
few batches and line them up to submit over serveral ledgers?

I'm skeptical about the handling of tec errors too because really only 12 unfunded offers and 1 tecPATH_DRY in almost 100k txns submitted?


Dashboard states:
TOTAL TXNS 92858
VALIDATED 76041
REJECTED 12055
EXPIRED 4762

Top Failures:
tefPAST_SEQ 8728
terPRE_SEQ 2355
tecUNFUNDED_OFFER 12
tecPATH_DRY 1
