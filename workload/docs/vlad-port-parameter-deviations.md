# Parameter Deviations from `upstream/vvysokikh1/new_transaction_types`

Differences between the upstream branch's `params.py` values and what we
actually use, with reasons.

## Architecture difference that drives most deviations

**Upstream** creates xrpl-py model objects directly and submits via
`submit_and_wait()`, which auto-fills fee/sequence/LLS.

**Our branch** returns raw dicts from builders, converts via
`Model.from_xrpl(dict)`, then manually signs and sets fee/sequence/LLS in
`build_sign_and_track()`.

The `from_xrpl()` path runs **model validation** (field length limits, type
checks) that `submit_and_wait()` also runs — but since upstream was still
in development, several of their `params.py` ranges silently exceed xrpl-py
validation limits.  Those would crash at model creation time on their branch
too; they just hadn't hit the edge cases yet.

---

## Credential URI — CAPPED (upstream bug)

| | Upstream (`params.py`) | Ours (`config.toml`) |
|---|---|---|
| **Range** | 10–256 bytes (20–512 hex chars) | 10–128 bytes (20–256 hex chars) |
| **Config key** | n/a (hardcoded) | `credential_create.uri_max_bytes = 128` |

**Why**: xrpl-py `CredentialCreate` validates `len(uri) <= 256` on the
*hex string*, not the byte count.  256 bytes → 512 hex chars → model
raises `XRPLModelException('Length cannot exceed 256 characters.')`.
Upstream has the same bug: `params.credential_uri()` generates up to
512 hex chars, which would crash ~50% of the time at
`CredentialCreate(uri=...)`.

We cap at 128 bytes (256 hex chars) to stay within xrpl-py's limit.

## Credential Type — SAME

| | Upstream | Ours |
|---|---|---|
| **Range** | 1–64 bytes (2–128 hex chars) | 1–64 bytes (2–128 hex chars) |
| **Config key** | n/a | `credential_create.credential_type_max_bytes = 64` |

No deviation.  xrpl-py limit is 128 hex chars (64 bytes).

## Credential Expiration — SAME (Ripple epoch adjusted)

| | Upstream | Ours |
|---|---|---|
| **Range** | `time.time() + randint(3600, 2592000)` | `time.time() - 946684800 + randrange(3600, 2592001)` |
| **Config key** | n/a | `credential_create.expiration_min/max_offset` |

Upstream passes Unix epoch timestamps but XRPL uses Ripple epoch
(Unix - 946684800).  We subtract the offset.  The `Expiration` field
would be rejected or interpreted incorrectly without this.

## DelegateSet Permissions — SAME (different format)

| | Upstream | Ours |
|---|---|---|
| **Count** | `randint(1, 3)` | `randrange(1, max_permissions + 1)` where `max_permissions = 3` |
| **Source** | `GranularPermission` enum | Same enum, serialised as `{"Permission": {"PermissionValue": p.value}}` |
| **Config key** | n/a | `delegate_set.max_permissions = 3` |

Same behaviour.  Upstream creates `Permission(permission_value=p)` model
objects; we emit the equivalent dict since our builders return dicts.

## PermissionedDomainSet Credentials — SAME

| | Upstream | Ours |
|---|---|---|
| **Count** | `randint(1, 10)` | `randrange(1, max_credentials + 1)` where `max_credentials = 10` |
| **Config key** | n/a | `permissioned_domain_set.max_credentials = 10` |

Same range.  Upstream uses `XRPLCredential(...)` model objects; we emit
`{"Credential": {"Issuer": ..., "CredentialType": ...}}` dicts (the
JSON serialisation format that `from_xrpl()` expects — note: the wrapper
key is `Credential`, NOT `AcceptedCredential`; we found this by
roundtripping through `to_xrpl()`).

## VaultCreate AssetsMaximum — SAME

| | Upstream | Ours |
|---|---|---|
| **Range** | `randint(100_000_000, 10_000_000_000)` | `randrange(assets_maximum_min, assets_maximum_max + 1)` |
| **Config key** | n/a | `vault_create.assets_maximum_min/max` |

Same range.

## VaultCreate/Set Data — SAME

| | Upstream | Ours |
|---|---|---|
| **Range** | 1–256 bytes (2–512 hex chars) | 1–256 bytes (2–512 hex chars) |
| **Config key** | n/a | `vault_create.data_max_bytes = 256` |

xrpl-py limit is 512 hex chars (256 bytes).  Both branches use the full
range.

## Vault Deposit/Withdraw Amounts — SAME

| | Upstream | Ours |
|---|---|---|
| **IOU** | `randint(1, 10_000)` | `randrange(iou_amount_min, iou_amount_max + 1)` with min=1, max=10000 |
| **MPT** | `randint(1, 10_000)` | `randrange(mpt_amount_min, mpt_amount_max + 1)` with min=1, max=10000 |
| **XRP** | `randint(1_000_000, 100_000_000)` | `randrange(xrp_drops_min, xrp_drops_max + 1)` with min=1M, max=100M |
| **Config key** | n/a | `vault_deposit.*` |

Same ranges, but configurable via `config.toml`.

## VaultClawback semantics — MATCHED (upstream's choice)

| | Upstream | Ours |
|---|---|---|
| **Account** | vault owner | vault owner |
| **Holder** | random other account | random other account |
| **Amount** | omitted | omitted |

Upstream has the vault **owner** as the clawback initiator, not the asset
issuer.  We matched this exactly.  (The correct XRPL semantics may differ
once the Vault amendment spec is finalised — this is what their branch
tested.)

## Fee handling — ADDED (upstream didn't need it)

VaultCreate and PermissionedDomainSet create ledger objects and require
owner_reserve (2 XRP) as the fee.  Upstream uses `submit_and_wait()` which
auto-fills fees.  Our `build_sign_and_track()` sets fees manually, so we
added these types to the owner_reserve fee path alongside AMMCreate and
Batch.

---

## Summary

| Parameter | Upstream | Ours | Deviation? |
|---|---|---|---|
| credential_type | 1–64 bytes | 1–64 bytes | None |
| credential URI | 10–256 bytes | 10–128 bytes | **Capped** (xrpl-py bug) |
| credential expiration | Unix epoch | Ripple epoch | **Fixed** |
| delegate permissions | 1–3 | 1–3 | None |
| domain credentials | 1–10 | 1–10 | None |
| vault assets_maximum | 100M–10B | 100M–10B | None |
| vault data | 1–256 bytes | 1–256 bytes | None |
| vault deposit IOU | 1–10k | 1–10k | None |
| vault deposit MPT | 1–10k | 1–10k | None |
| vault deposit XRP | 1M–100M drops | 1M–100M drops | None |
| vault clawback | owner, no amount | owner, no amount | None |
| fee (VaultCreate) | auto (submit_and_wait) | 2M drops (owner_reserve) | **Added** |
| fee (PermDomainSet) | auto (submit_and_wait) | 2M drops (owner_reserve) | **Added** |
