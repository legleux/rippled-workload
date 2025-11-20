# MPToken (Multi-Purpose Token) Guide

Comprehensive guide to MPToken functionality in the XRP Ledger, based on unit test analysis.

## Overview

MPTokens are native tokens on the XRP Ledger that provide flexible tokenization with features like:
- Authorization controls
- Transfer fees
- Locking/unlocking
- Clawback functionality
- Permissioned domains integration
- Dynamic property updates

## Transaction Types

### 1. MPTokenIssuanceCreate

Creates a new MPToken issuance.

**Required Fields:**
- `Account`: Issuer's address
- `MPTokenIssuanceID`: Auto-generated from sequence number and issuer

**Optional Fields:**
- `MaximumAmount`: Max tokens that can be minted (default: 9,223,372,036,854,775,807)
- `AssetScale`: Decimal precision 0-19 (e.g., 2 = supports values like 100.25)
- `TransferFee`: Transfer fee in basis points, 0-50,000 (0%-50%)
- `MPTokenMetadata`: Metadata string (cannot be empty, max 1024 bytes)
- `DomainID`: Permissioned domain ID (requires `tfMPTRequireAuth` flag)
- `MutableFlags`: Flags indicating which properties can be changed later (requires DynamicMPT)

**Flags:**
- `tfMPTCanLock` (0x0001): Allows issuer to lock/unlock tokens
- `tfMPTRequireAuth` (0x0002): Requires issuer authorization for holders
- `tfMPTCanEscrow` (0x0004): Allows tokens in escrow
- `tfMPTCanTrade` (0x0008): Allows trading
- `tfMPTCanTransfer` (0x0010): Allows peer-to-peer transfers
- `tfMPTCanClawback` (0x0020): Allows issuer clawback

**Mutable Flags (DynamicMPT amendment):**
- `tmfMPTCanMutateMetadata`: Allow metadata changes
- `tmfMPTCanMutateCanLock`: Allow toggling CanLock flag
- `tmfMPTCanMutateRequireAuth`: Allow toggling RequireAuth flag
- `tmfMPTCanMutateCanEscrow`: Allow toggling CanEscrow flag
- `tmfMPTCanMutateCanTrade`: Allow toggling CanTrade flag
- `tmfMPTCanMutateCanTransfer`: Allow toggling CanTransfer flag
- `tmfMPTCanMutateCanClawback`: Allow toggling CanClawback flag
- `tmfMPTCanMutateTransferFee`: Allow transfer fee changes

**Example:**
```python
{
    "TransactionType": "MPTokenIssuanceCreate",
    "Account": "rIssuer...",
    "MaximumAmount": "9223372036854775807",
    "AssetScale": 2,
    "TransferFee": 10000,  # 10% in basis points
    "MPTokenMetadata": "My Token",
    "Flags": tfMPTCanLock | tfMPTRequireAuth | tfMPTCanTransfer | tfMPTCanClawback
}
```

**Common Errors:**
- `temDISABLED`: MPTokensV1 amendment not enabled
- `temMALFORMED`: Transfer fee without `tfMPTCanTransfer`, empty metadata, MaximumAmount is 0
- `temBAD_TRANSFER_FEE`: Transfer fee exceeds 50,000

---

### 2. MPTokenIssuanceDestroy

Destroys an MPToken issuance (must have no outstanding balance).

**Required Fields:**
- `Account`: Issuer's address
- `MPTokenIssuanceID`: The issuance to destroy

**Example:**
```python
{
    "TransactionType": "MPTokenIssuanceDestroy",
    "Account": "rIssuer...",
    "MPTokenIssuanceID": "..."
}
```

**Common Errors:**
- `tecNO_PERMISSION`: Non-issuer trying to destroy
- `tecHAS_OBLIGATIONS`: Outstanding balance exists (holders still have tokens)

---

### 3. MPTokenAuthorize

Creates/deletes a holder's MPToken object, or issuer authorizes/unauthorizes a holder.

**Required Fields:**
- `Account`: Transaction submitter
- `MPTokenIssuanceID`: The issuance

**Optional Fields:**
- `Holder`: Holder account (used by issuer for authorization)

**Flags:**
- `tfMPTUnauthorize` (0x0001): Delete MPToken or revoke authorization

**Usage Patterns:**

**Holder creates MPToken (opt-in):**
```python
{
    "TransactionType": "MPTokenAuthorize",
    "Account": "rHolder...",
    "MPTokenIssuanceID": "..."
}
```

**Holder deletes MPToken:**
```python
{
    "TransactionType": "MPTokenAuthorize",
    "Account": "rHolder...",
    "MPTokenIssuanceID": "...",
    "Flags": tfMPTUnauthorize
}
```

**Issuer authorizes holder (when tfMPTRequireAuth is set):**
```python
{
    "TransactionType": "MPTokenAuthorize",
    "Account": "rIssuer...",
    "MPTokenIssuanceID": "...",
    "Holder": "rHolder..."
}
```

**Common Errors:**
- `tecDUPLICATE`: MPToken already exists
- `tecHAS_OBLIGATIONS`: Trying to delete MPToken with non-zero balance
- `tecINSUFFICIENT_RESERVE`: Insufficient XRP reserve (first 2 MPTokens are free per account)

---

### 4. MPTokenIssuanceSet

Lock/unlock tokens, set/clear flags, update metadata, or change transfer fee.

**Required Fields:**
- `Account`: Issuer's address
- `MPTokenIssuanceID`: The issuance to modify

**Optional Fields:**
- `Holder`: Specific holder to lock/unlock
- `DomainID`: Update permissioned domain
- `MPTokenMetadata`: Update metadata (requires DynamicMPT)
- `TransferFee`: Update transfer fee (requires DynamicMPT)
- `MutableFlags`: Dynamic flag changes (requires DynamicMPT)

**Flags (for lock/unlock):**
- `tfMPTLock` (0x0001): Lock the issuance or specific holder
- `tfMPTUnlock` (0x0002): Unlock the issuance or specific holder

**Mutable Flags for Dynamic Changes:**
- Set flags: `tmfMPTSetCanLock`, `tmfMPTSetRequireAuth`, `tmfMPTSetCanEscrow`, etc.
- Clear flags: `tmfMPTClearCanLock`, `tmfMPTClearRequireAuth`, `tmfMPTClearCanEscrow`, etc.

**Examples:**

**Lock globally:**
```python
{
    "TransactionType": "MPTokenIssuanceSet",
    "Account": "rIssuer...",
    "MPTokenIssuanceID": "...",
    "Flags": tfMPTLock
}
```

**Lock specific holder:**
```python
{
    "TransactionType": "MPTokenIssuanceSet",
    "Account": "rIssuer...",
    "MPTokenIssuanceID": "...",
    "Holder": "rHolder...",
    "Flags": tfMPTLock
}
```

**Update metadata (DynamicMPT):**
```python
{
    "TransactionType": "MPTokenIssuanceSet",
    "Account": "rIssuer...",
    "MPTokenIssuanceID": "...",
    "MPTokenMetadata": "Updated metadata"
}
```

**Common Errors:**
- `temINVALID_FLAG`: Invalid flag combination (e.g., both lock and unlock)
- `temMALFORMED`: Nothing being changed, or invalid field combinations
- `tecNO_PERMISSION`: Locking disabled or trying to mutate non-mutable flag

---

### 5. Payment (with MPToken)

Transfer MPTokens between accounts.

**Required Fields:**
- `Account`: Sender
- `Destination`: Receiver
- `Amount`: MPToken amount (using MPTIssue structure)

**Optional Fields:**
- `SendMax`: Maximum amount to send (required for transfers with fees)
- `DeliverMin`: Minimum amount to deliver (for partial payments)

**Flags:**
- `tfPartialPayment`: Allow partial delivery

**Payment Types:**
1. **Issuer → Holder**: No transfer fee
2. **Holder → Issuer**: No transfer fee
3. **Holder → Holder**: Transfer fee applies (if set)

**Transfer Fee Calculation:**
- Fee in basis points (10,000 = 1%)
- Fee goes to issuer
- Sender pays: `amount + (amount * transferFee / 100,000)`
- Example: 10% fee, sending 100 tokens requires SendMax of 110

**Examples:**

**Simple payment (issuer to holder):**
```python
{
    "TransactionType": "Payment",
    "Account": "rIssuer...",
    "Destination": "rHolder...",
    "Amount": {
        "mpt_issuance_id": "...",
        "value": "100"
    }
}
```

**Payment with transfer fee (holder to holder):**
```python
{
    "TransactionType": "Payment",
    "Account": "rHolder1...",
    "Destination": "rHolder2...",
    "Amount": {
        "mpt_issuance_id": "...",
        "value": "100"
    },
    "SendMax": {
        "mpt_issuance_id": "...",
        "value": "110"  # Includes 10% fee
    }
}
```

**Common Errors:**
- `tecNO_AUTH`: Sender or receiver not authorized (when `tfMPTRequireAuth` is set)
- `tecLOCKED`: Tokens are locked
- `tecPATH_PARTIAL`: Insufficient funds or SendMax doesn't cover transfer fee
- `tecINSUFFICIENT_FUNDS`: Insufficient balance

---

### 6. Clawback (MPToken)

Issuer reclaims tokens from a holder.

**Required Fields:**
- `Account`: Issuer
- `Amount`: MPToken amount to claw back
- `Holder`: The holder to claw back from (in Amount structure)

**Requirements:**
- Issuance must have `tfMPTCanClawback` flag
- Holder must have sufficient balance

**Special Properties:**
- Can claw back locked tokens
- Can claw back from unauthorized holders
- Cannot claw back from issuer's own account

**Example:**
```python
{
    "TransactionType": "Clawback",
    "Account": "rIssuer...",
    "Amount": {
        "mpt_issuance_id": "...",
        "value": "50"
    },
    "Holder": "rHolder..."
}
```

**Common Errors:**
- `temMALFORMED`: Clawing back from issuer
- `tecNO_PERMISSION`: Clawback not enabled
- `tecINSUFFICIENT_FUNDS`: Holder has insufficient balance

---

## Key Concepts

### Asset Scale
- Defines decimal precision (0-19)
- Represents decimal places the token supports
- Example: `assetScale = 2` → values like 100.25

### Transfer Fees
- Expressed in basis points (1 bp = 0.01%)
- Range: 0-50,000 (0%-50%)
- Only applies to holder-to-holder transfers
- Fee amount goes to issuer
- Requires `tfMPTCanTransfer` flag

### Authorization Models

**1. Public (no authorization required):**
```python
# Create without tfMPTRequireAuth
# Holders can opt-in and receive immediately
```

**2. Require Authorization:**
```python
# Create with tfMPTRequireAuth
# Holder opts-in
# Issuer must explicitly authorize before holder can receive
```

**3. Permissioned Domains (with credentials):**
```python
# Create with tfMPTRequireAuth + DomainID
# Holders with matching credentials are auto-authorized
```

### Locking/Unlocking

**Global Lock (affects all holders):**
- Holder-to-holder transfers blocked
- Issuer can still send/receive

**Individual Lock (affects specific holder):**
- Locked holder cannot send to other holders
- Other holders cannot send to locked holder
- Issuer can still interact with locked holder

**Lock Behavior:**
- Locked tokens can still be clawed back by issuer
- Requires `tfMPTCanLock` flag

### Clawback
- Issuer can reclaim tokens from any holder
- Works even if tokens are locked
- Works even if holder is unauthorized
- Cannot claw back from issuer
- Requires `tfMPTCanClawback` flag

### Maximum Amounts
- Default maximum: 9,223,372,036,854,775,807 (2^63 - 1)
- Custom maximum via `MaximumAmount` field
- Outstanding amount = total tokens held by all holders (excluding issuer)
- Issuer cannot mint more than MaximumAmount total

---

## Dynamic MPT (DynamicMPT Amendment)

Allows post-creation modification of properties.

### Workflow

**1. Create with mutable flags:**
```python
{
    "TransactionType": "MPTokenIssuanceCreate",
    "Account": "rIssuer...",
    "TransferFee": 100,
    "Flags": tfMPTCanTransfer,
    "MutableFlags": tmfMPTCanMutateTransferFee | tmfMPTCanMutateMetadata
}
```

**2. Later, update properties:**
```python
# Update transfer fee
{
    "TransactionType": "MPTokenIssuanceSet",
    "Account": "rIssuer...",
    "MPTokenIssuanceID": "...",
    "TransferFee": 200
}

# Update metadata
{
    "TransactionType": "MPTokenIssuanceSet",
    "Account": "rIssuer...",
    "MPTokenIssuanceID": "...",
    "MPTokenMetadata": "Updated metadata"
}

# Toggle flags
{
    "TransactionType": "MPTokenIssuanceSet",
    "Account": "rIssuer...",
    "MPTokenIssuanceID": "...",
    "MutableFlags": tmfMPTClearCanTransfer
}
```

---

## Constants and Limits

```python
MAX_MPTOKEN_AMOUNT = 9_223_372_036_854_775_807  # 2^63 - 1
MAX_TRANSFER_FEE = 50_000  # 50% in basis points
MAX_METADATA_LENGTH = 1024  # bytes
```

---

## Reserve Requirements
- First 2 MPTokens per account: Free
- Additional MPTokens: Require reserve increment
- MPToken reserve separate from account reserve

---

## Complete Transaction Flow Example

```python
# 1. Create issuance
create_txn = {
    "TransactionType": "MPTokenIssuanceCreate",
    "Account": "rIssuer...",
    "MaximumAmount": "1000000",
    "AssetScale": 2,
    "TransferFee": 10000,  # 10%
    "MPTokenMetadata": "My Token",
    "Flags": tfMPTCanLock | tfMPTRequireAuth | tfMPTCanTransfer | tfMPTCanClawback
}

# 2. Holders opt-in
authorize_bob = {
    "TransactionType": "MPTokenAuthorize",
    "Account": "rBob...",
    "MPTokenIssuanceID": "..."
}

# 3. Issuer authorizes holders
authorize_holder = {
    "TransactionType": "MPTokenAuthorize",
    "Account": "rIssuer...",
    "MPTokenIssuanceID": "...",
    "Holder": "rBob..."
}

# 4. Issue tokens
issue_payment = {
    "TransactionType": "Payment",
    "Account": "rIssuer...",
    "Destination": "rBob...",
    "Amount": {"mpt_issuance_id": "...", "value": "1000"}
}

# 5. Transfer between holders (with fee)
holder_payment = {
    "TransactionType": "Payment",
    "Account": "rBob...",
    "Destination": "rCarol...",
    "Amount": {"mpt_issuance_id": "...", "value": "100"},
    "SendMax": {"mpt_issuance_id": "...", "value": "110"}  # 10% fee
}

# 6. Lock if needed
lock_txn = {
    "TransactionType": "MPTokenIssuanceSet",
    "Account": "rIssuer...",
    "MPTokenIssuanceID": "...",
    "Holder": "rBob...",
    "Flags": tfMPTLock
}

# 7. Claw back
clawback_txn = {
    "TransactionType": "Clawback",
    "Account": "rIssuer...",
    "Amount": {"mpt_issuance_id": "...", "value": "500"},
    "Holder": "rBob..."
}

# 8. Unauthorize
unauthorize_txn = {
    "TransactionType": "MPTokenAuthorize",
    "Account": "rIssuer...",
    "MPTokenIssuanceID": "...",
    "Holder": "rBob...",
    "Flags": tfMPTUnauthorize
}

# 9. Return tokens
return_payment = {
    "TransactionType": "Payment",
    "Account": "rBob...",
    "Destination": "rIssuer...",
    "Amount": {"mpt_issuance_id": "...", "value": "400"}
}

# 10. Delete MPToken
delete_mptoken = {
    "TransactionType": "MPTokenAuthorize",
    "Account": "rBob...",
    "MPTokenIssuanceID": "...",
    "Flags": tfMPTUnauthorize
}

# 11. Destroy issuance
destroy_txn = {
    "TransactionType": "MPTokenIssuanceDestroy",
    "Account": "rIssuer...",
    "MPTokenIssuanceID": "..."
}
```

---

## Common Error Patterns & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `tecNO_AUTH` | Payment without authorization | Issuer must authorize holder when `tfMPTRequireAuth` is set |
| `tecPATH_PARTIAL` | Insufficient SendMax | Include transfer fee in SendMax calculation |
| `tecHAS_OBLIGATIONS` | Destroy with outstanding balance | All tokens must be returned to issuer first |
| `temMALFORMED` | Transfer fee without CanTransfer | Set `tfMPTCanTransfer` flag when creating |
| `tecLOCKED` | Payment while locked | Unlock tokens or use issuer account |
| `tecINSUFFICIENT_RESERVE` | Creating MPToken without reserve | Ensure account has sufficient XRP reserve |

---

## Workload Integration Notes

When implementing MPToken support in the workload generator:

1. **Track MPToken state per holder:**
   - MPToken exists (authorized or not)
   - Current balance
   - Lock status

2. **Calculate SendMax for holder-to-holder payments:**
   ```python
   if transfer_fee > 0 and not (src == issuer or dst == issuer):
       send_max = amount + (amount * transfer_fee // 100_000)
   ```

3. **Handle authorization workflow:**
   - Holder creates MPToken first
   - Wait for validation
   - Issuer authorizes (if required)
   - Wait for validation
   - Then payments can proceed

4. **Respect locks:**
   - Track global lock state
   - Track per-holder lock state
   - Skip holder-to-holder payments when locked

5. **Sequence destruction properly:**
   - Claw back or receive all tokens
   - Holders delete their MPTokens
   - Destroy issuance when outstanding amount is 0
