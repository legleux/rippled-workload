#!/usr/bin/env bash
# Exercise every transaction type once via /txn/create/{Type}.
# Prints pass/fail per type and a summary at the end.
#
# Usage:
#   ./exercise_all_types.sh                 # default: http://workload:8000
#   ./exercise_all_types.sh localhost:8000   # override host

set -euo pipefail

HOST="${1:-workload:8000}"
BASE="http://${HOST}/txn/create"

# Every type in _BUILDERS (same order as TxType enum)
TYPES=(
  # Core
  Payment
  TrustSet
  AccountSet
  OfferCreate
  TicketCreate
  # NFT
  NFTokenMint
  # MPT
  MPTokenIssuanceCreate
  # AMM
  AMMCreate
  # Batch
  Batch
  # Delegation
  DelegateSet
  # Credentials
  CredentialCreate
  # Permissioned Domains
  PermissionedDomainSet
  # Vaults
  VaultCreate
  VaultDeposit
  # --- State-dependent types below ---
  # These may return errors if no state exists yet (e.g. no NFTs to burn).
  # That's expected — we still verify the endpoint responds.
  NFTokenBurn
  NFTokenCreateOffer
  NFTokenCancelOffer
  NFTokenAcceptOffer
  OfferCancel
  MPTokenIssuanceSet
  MPTokenAuthorize
  MPTokenIssuanceDestroy
  AMMDeposit
  AMMWithdraw
  CredentialAccept
  CredentialDelete
  PermissionedDomainDelete
  VaultSet
  VaultDelete
  VaultWithdraw
  VaultClawback
)

pass=0
fail=0
errors=()

for type in "${TYPES[@]}"; do
  # -s silent, -w httpcode, -o /dev/null discards body
  http_code=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/${type}" 2>/dev/null || echo "000")

  if [[ "$http_code" == "200" ]]; then
    printf "  \033[32mPASS\033[0m  %s\n" "$type"
    ((pass++))
  else
    printf "  \033[31mFAIL\033[0m  %s  (HTTP %s)\n" "$type" "$http_code"
    ((fail++))
    errors+=("$type ($http_code)")
  fi
done

echo ""
echo "────────────────────────────────"
printf "  Total: %d  |  Pass: \033[32m%d\033[0m  |  Fail: \033[31m%d\033[0m\n" $((pass + fail)) "$pass" "$fail"

if [[ ${#errors[@]} -gt 0 ]]; then
  echo ""
  echo "  Failed types:"
  for e in "${errors[@]}"; do
    echo "    - $e"
  done
  echo ""
  exit 1
fi

echo ""
