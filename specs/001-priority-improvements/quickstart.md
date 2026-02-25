# Quickstart: Priority Improvements Manual Testing Guide

**Feature**: Priority Improvements (001-priority-improvements)
**Date**: 2025-12-02

This guide provides step-by-step manual testing procedures for all 6 priority user stories.

---

## Prerequisites

1. **Docker Network Running**:
   ```bash
   cd /home/emel/dev/Ripple/rippled-workload/prepare-workload/testnet
   docker compose up -d
   ```

2. **Workload Container Started**:
   ```bash
   docker ps | grep workload
   # Verify workload container is running
   ```

3. **Get Workload IP**:
   ```bash
   workload_ip=$(docker inspect workload | jq -r '.[0].NetworkSettings.Networks[].IPAddress')
   echo "Workload API: http://${workload_ip}:8000"
   ```

4. **Verify Connectivity**:
   ```bash
   curl -s "http://${workload_ip}:8000/health" | jq
   # Should return {"status": "healthy"}
   ```

---

## P1: Continuous Submission Reliability

**Objective**: Verify 90% validation rate, <10% sequence conflicts, zero ledger gaps

### Test 1: Validation Success Rate

```bash
# Start continuous submission (if not already running)
curl -X POST "http://${workload_ip}:8000/workload/start"

# Wait for 100 ledgers (~6-7 minutes at 4s/ledger)
sleep 420

# Check validation metrics
curl -s "http://${workload_ip}:8000/state/summary" | jq '{
  total: .total_transactions,
  validated: .by_state.VALIDATED,
  rejected: .by_state.REJECTED,
  expired: .by_state.EXPIRED,
  validation_rate: (.by_state.VALIDATED / .total_transactions)
}'

# Expected: validation_rate >= 0.90 (90%)
```

### Test 2: Sequence Conflict Rate

```bash
# Check sequence conflict metrics
curl -s "http://${workload_ip}:8000/dashboard/validations" | jq '{
  total_submitted: .total_submitted,
  sequence_conflicts: .sequence_conflicts,
  conflict_rate: (.sequence_conflicts / .total_submitted)
}'

# Expected: conflict_rate < 0.10 (10%)
```

### Test 3: Ledger Fill Rate

```bash
# Query ledger fill metrics (requires custom endpoint or rippled query)
for i in {1..10}; do
  ledger_idx=$(($(curl -s "http://${workload_ip}:8000/dashboard/overview" | jq '.current_ledger') - i))
  txn_count=$(docker exec rippled rippled --silent ledger ${ledger_idx} | jq '.result.ledger.transactions | length')
  echo "Ledger $ledger_idx: $txn_count transactions"
done

# Expected: All ledgers have at least 1 workload transaction (no gaps)
```

### Test 4: Increased Account Count

```bash
# Modify config.toml to increase user account count
# users.count = 100 (from 50)

# Restart workload
docker restart workload

# Re-run Test 1 and verify throughput increases
```

**Success Criteria**:
- ✅ Validation rate ≥ 90%
- ✅ Sequence conflict rate < 10%
- ✅ Zero ledger gaps in 100 consecutive ledgers
- ✅ Sustained 10+ transactions/ledger

---

## P2: Complete MPToken Transaction Workflow

**Objective**: Verify full MPToken lifecycle (mint, disburse, offer)

### Test 1: MPToken Minting

```bash
# Mint MPToken
curl -X POST "http://${workload_ip}:8000/transaction/mptoken/mint" \
  -H "Content-Type: application/json" \
  -d '{
    "max_amount": "1000000"
  }' | jq

# Note the tx_hash and token_id from response
TOKEN_ID="<token_id_from_response>"
```

### Test 2: MPToken Disbursement (NEW)

```bash
# Get recipient accounts
RECIPIENTS=$(curl -s "http://${workload_ip}:8000/accounts" | jq -r '.[0:5] | .[].address')

# Ensure recipients have trust lines (may need TrustSet first)
for recipient in $RECIPIENTS; do
  curl -X POST "http://${workload_ip}:8000/transaction/trustset" \
    -H "Content-Type: application/json" \
    -d "{
      \"mptoken_id\": \"$TOKEN_ID\",
      \"destination\": \"$recipient\"
    }"
done

# Wait for trust lines to validate
sleep 10

# Disburse to 5 recipients
for recipient in $RECIPIENTS; do
  curl -X POST "http://${workload_ip}:8000/transaction/mptoken/disburse" \
    -H "Content-Type: application/json" \
    -d "{
      \"token_id\": \"$TOKEN_ID\",
      \"destination\": \"$recipient\",
      \"amount\": \"1000\"
    }" | jq
done

# Verify disbursements validated
curl -s "http://${workload_ip}:8000/state/validations" | jq '.[-5:]'
```

### Test 3: MPToken Offers (NEW)

```bash
# Create offer to trade MPToken for XRP
curl -X POST "http://${workload_ip}:8000/transaction/offer/create" \
  -H "Content-Type: application/json" \
  -d "{
    \"taker_pays\": \"1000000\",
    \"taker_gets\": {
      \"mptoken_id\": \"$TOKEN_ID\",
      \"value\": \"100\"
    }
  }" | jq

# Check offer book
docker exec rippled rippled --silent book_offers $TOKEN_ID XRP | jq
```

**Success Criteria**:
- ✅ MPToken mints successfully
- ✅ All 5 disbursements validate (100% success)
- ✅ Offers created successfully
- ✅ MPToken operations represent ≥15% of transaction mix

---

## P3: Offer Crossing and Order Book Activity

**Objective**: Verify OfferCreate and OfferCancel transactions

### Test 1: Create Offers

```bash
# Create 10 offers across different currency pairs
for i in {1..10}; do
  curl -X POST "http://${workload_ip}:8000/transaction/offer/create" \
    -H "Content-Type: application/json" \
    -d '{
      "taker_pays": {"currency": "USD", "issuer": "rPEPPER7kfTD9w2To4CQk6UCfuHM9c6GDY", "value": "100"},
      "taker_gets": "1000000"
    }' | jq '.tx_hash'
done

# Verify offers in order books
docker exec rippled rippled --silent book_offers USD rPEPPER7kfTD9w2To4CQk6UCfuHM9c6GDY XRP | jq '.result.offers | length'
# Expected: 10 offers
```

### Test 2: Cancel Offers

```bash
# Get offer sequences from account
ACCOUNT=$(curl -s "http://${workload_ip}:8000/accounts" | jq -r '.[0].address')
OFFER_SEQS=$(docker exec rippled rippled --silent account_offers $ACCOUNT | jq -r '.result.offers[0:3] | .[].seq')

# Cancel 3 offers
for seq in $OFFER_SEQS; do
  curl -X POST "http://${workload_ip}:8000/transaction/offer/cancel" \
    -H "Content-Type: application/json" \
    -d "{\"offer_sequence\": $seq}" | jq
done

# Verify offers removed
docker exec rippled rippled --silent account_offers $ACCOUNT | jq '.result.offers | length'
# Expected: 7 offers remaining
```

### Test 3: Offer Crossing

```bash
# Create crossing offer (opposite side of book)
curl -X POST "http://${workload_ip}:8000/transaction/offer/create" \
  -H "Content-Type: application/json" \
  -d '{
    "taker_pays": "1000000",
    "taker_gets": {"currency": "USD", "issuer": "rPEPPER7kfTD9w2To4CQk6UCfuHM9c6GDY", "value": "100"}
  }' | jq

# Check if offer crossed (tx result should show partial/full execution)
sleep 5
curl -s "http://${workload_ip}:8000/state/validations" | jq '.[-1]'
```

**Success Criteria**:
- ✅ OfferCreate success rate ≥95%
- ✅ OfferCancel success rate ≥95%
- ✅ At least 10% of offers result in crossing
- ✅ Offers represent ≥20% of transaction mix

---

## P4: Arbitrary Transaction Submission via API

**Objective**: Verify Swagger UI endpoints with proper parameters

### Test 1: Access Swagger UI

```bash
# Open Swagger UI in browser
xdg-open "http://${workload_ip}:8000/docs"

# Or curl the OpenAPI spec
curl -s "http://${workload_ip}:8000/openapi.json" | jq '.paths | keys'
```

### Test 2: Test Payment Endpoint

1. Open Swagger UI: `http://${workload_ip}:8000/docs`
2. Navigate to **POST /transaction/payment**
3. Click "Try it out"
4. Verify fields are present:
   - `destination` (required, string)
   - `amount` (required, string or object)
   - `currency` (optional, string)
   - `issuer` (optional, string)
5. Fill in example values (should be pre-populated if testnet funding available)
6. Click "Execute"
7. Verify response contains `tx_hash` and `result`

### Test 3: Test All Transaction Types

Repeat Test 2 for each transaction endpoint:
- ✅ `/transaction/payment`
- ✅ `/transaction/trustset`
- ✅ `/transaction/accountset`
- ✅ `/transaction/nftoken/mint`
- ✅ `/transaction/mptoken/mint`
- ✅ `/transaction/mptoken/disburse` (NEW)
- ✅ `/transaction/offer/create` (NEW)
- ✅ `/transaction/offer/cancel` (NEW)

### Test 4: Parameter Validation

```bash
# Submit invalid payment (negative amount)
curl -X POST "http://${workload_ip}:8000/transaction/payment" \
  -H "Content-Type: application/json" \
  -d '{"destination": "rN7n7otQDd6FczFgLdlqtyMVrn3qHGbXcD", "amount": "-1000"}' | jq

# Expected: 400 Bad Request with validation error
```

**Success Criteria**:
- ✅ All 8 transaction types have Swagger UI endpoints
- ✅ All required/optional parameters documented
- ✅ 100% of valid submissions succeed
- ✅ 100% of invalid submissions return clear errors

---

## P5: Network Node Observability Dashboard

**Objective**: Verify dashboard displays node metrics with real-time updates

### Test 1: Access Dashboard

```bash
# Open dashboard in browser
xdg-open "http://${workload_ip}:8000/dashboard"
```

### Test 2: Network Overview Tab

Verify dashboard displays:
- ✅ Total nodes count
- ✅ Reachable nodes count
- ✅ Current ledger index
- ✅ Validation rate
- ✅ Sequence conflict rate

### Test 3: Queue Tab

Verify dashboard displays for each node:
- ✅ Queue size
- ✅ Max queue size
- ✅ Minimum fee
- ✅ Open ledger fee

### Test 4: Transactions Tab

Verify dashboard displays:
- ✅ Recent transaction counts (submitted, validated, rejected, expired)
- ✅ Transaction breakdown by state
- ✅ Validation source (polling vs WebSocket)

### Test 5: Validations Tab

Verify dashboard displays:
- ✅ Validator participation
- ✅ Consensus agreement metrics
- ✅ Recent validation messages

### Test 6: Real-Time Updates

```bash
# Open browser console and check WebSocket connection
# Should see: WebSocket connected to ws://${workload_ip}:8000/dashboard/updates

# Submit transaction and verify dashboard updates within 5 seconds
curl -X POST "http://${workload_ip}:8000/transaction/payment" \
  -H "Content-Type: application/json" \
  -d '{"destination": "rN7n7otQDd6FczFgLdlqtyMVrn3qHGbXcD", "amount": "1000000"}'

# Watch dashboard - transaction count should increment immediately
```

### Test 7: Node Unreachability

```bash
# Stop one rippled node
docker stop rippled-1

# Verify dashboard shows node as unreachable
# Other nodes should still display normally
```

**Success Criteria**:
- ✅ Dashboard displays all required metrics
- ✅ Updates within 5 seconds of ledger close
- ✅ Functional when up to 50% nodes unreachable

---

## P6: Code Quality Enforcement

**Objective**: Verify ruff formatting, linting, docstrings, and pre-commit hooks

### Test 1: Ruff Format

```bash
cd /home/emel/dev/Ripple/rippled-workload/workload

# Run ruff format
uv run ruff format

# Re-run should show no changes
uv run ruff format
# Expected: "No files reformatted"
```

### Test 2: Ruff Check

```bash
# Run ruff check
uv run ruff check

# Expected: No errors
# If errors, fix them:
uv run ruff check --fix
```

### Test 3: Docstring and Return Type Audit

```bash
# Run custom docstring checker
python scripts/check_docstrings.py workload/src/workload/**/*.py

# Expected: No errors
# Each public method should have:
# - Google-style docstring
# - Return type annotation
```

### Test 4: Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test on all files
pre-commit run --all-files

# Expected: All hooks pass
```

### Test 5: Commit with Improper Formatting

```bash
# Create poorly formatted file
echo "def bad_function():return None" > test_bad.py
git add test_bad.py
git commit -m "test: bad formatting"

# Expected: Commit rejected with formatting errors

# Fix and re-commit
uv run ruff format test_bad.py
git add test_bad.py
git commit -m "test: bad formatting"

# Expected: Commit succeeds
git reset --soft HEAD~1  # Undo test commit
rm test_bad.py
```

### Test 6: File Size Audit

```bash
# Check file sizes
find workload/src/workload -name "*.py" -exec wc -l {} + | sort -n | tail -20

# Expected: No files >500 lines (after refactoring)
```

**Success Criteria**:
- ✅ Zero ruff format/check errors
- ✅ 100% of public methods have docstrings and return types
- ✅ Pre-commit hooks prevent bad commits
- ✅ No files >500 lines

---

## Summary

This quickstart provides manual testing procedures for all 6 priority improvements. Each test includes:
- Specific commands to run
- Expected outcomes
- Success criteria from spec.md

**Testing Order**: P6 (Code Quality) should be completed first to establish quality baseline, then P1-P5 can be tested in any order.
