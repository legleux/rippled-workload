MPToken Transaction Examples

  The endpoints are available at /transaction/create/{transaction_type}. Here are curl commands you can run:

  1. Create an MPToken Issuance

  curl -s http://localhost:8000/transaction/create/MPTokenIssuanceCreate

  This creates a new MPToken with random metadata from an existing account.

  2. Authorize/Opt-in to an MPToken

  curl -s http://localhost:8000/transaction/create/MPTokenAuthorize

  This creates an MPToken holder object (opt-in) or authorizes a holder (if called by issuer).

  Note: This requires an existing MPToken issuance ID to be tracked in the workload context. You'll need to create an issuance first.

  3. Modify MPToken Properties (Set)

  curl -s http://localhost:8000/transaction/create/MPTokenIssuanceSet

  This can lock/unlock tokens or modify properties (requires DynamicMPT).

  4. Destroy an MPToken Issuance

  curl -s http://localhost:8000/transaction/create/MPTokenIssuanceDestroy

  Destroys an MPToken issuance (only works if no outstanding balance).

  5. Submit Random Transaction (any type)

  curl -s http://localhost:8000/transaction/random

  Important Notes
Can
