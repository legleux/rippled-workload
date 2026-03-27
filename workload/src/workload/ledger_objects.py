"""Deterministic XRPL ledger object ID computation.

Computes ledger object indices and IDs locally from transaction fields,
eliminating the need to RPC-fetch metadata from rippled.

Self-contained module — designed for eventual contribution to xrpl-py.
Only dependency: xrpl.core.addresscodec for r-address decoding.

Algorithms sourced from rippled:
  - include/xrpl/protocol/Indexes.h (keylet definitions)
  - src/libxrpl/protocol/Indexes.cpp (keylet implementations)
  - include/xrpl/protocol/nftPageMask.h (NFT page mask)

Sections:
  - Core: SHA512Half, namespace prefixes, account decoding
  - Sequence-keyed objects: Offer, NFTokenOffer, Check, Escrow, Ticket,
    Vault, PermissionedDomain
  - NFTokenID: deterministic token ID (packed struct, not hashed)
  - Credential: variable-length credential type
"""

from __future__ import annotations

import hashlib
import struct

from xrpl.core.addresscodec import decode_classic_address


# ---------------------------------------------------------------------------
# Core: SHA512Half + namespace prefixes
# ---------------------------------------------------------------------------


def sha512_half(data: bytes) -> bytes:
    """First 32 bytes of SHA-512 — the XRPL 'SHA512Half' operation."""
    return hashlib.sha512(data).digest()[:32]


def _ns_prefix(code: int) -> bytes:
    """XRPL index key prefix: 0x00 + namespace byte."""
    return b"\x00" + bytes([code])


def _compute_index(namespace: int, payload: bytes) -> str:
    """SHA512Half(namespace_prefix + payload), returned as 64-char uppercase hex."""
    return sha512_half(_ns_prefix(namespace) + payload).hex().upper()


# Namespace bytes — from rippled LedgerNameSpace enum
_OFFER = 0x6F  # 'o'
_NFTOKEN_OFFER = 0x71  # 'q'
_CHECK = 0x43  # 'C'
_ESCROW = 0x75  # 'u'
_TICKET = 0x54  # 'T'
_VAULT = 0x56  # 'V'
_PERMISSIONED_DOMAIN = 0x6D  # 'm'
_CREDENTIAL = 0x44  # 'D'


# ---------------------------------------------------------------------------
# Account decoding
# ---------------------------------------------------------------------------


def _account_id(classic_address: str) -> bytes:
    """Classic r-address → 20-byte AccountID."""
    acct = decode_classic_address(classic_address)
    if len(acct) != 20:
        raise ValueError(f"AccountID must be 20 bytes, got {len(acct)}")
    return acct


# ---------------------------------------------------------------------------
# Sequence-keyed objects: SHA512Half(namespace + AccountID + uint32 sequence)
#
# All follow the same pattern from rippled:
#   indexHash(LedgerNameSpace::TYPE, account_id, sequence)
# ---------------------------------------------------------------------------


def _account_seq_index(namespace: int, account: str, sequence: int) -> str:
    """Generic index for objects keyed by (account, sequence)."""
    payload = _account_id(account) + struct.pack(">I", sequence)
    return _compute_index(namespace, payload)


def offer_index(account: str, sequence: int) -> str:
    """Compute DEX Offer ledger index.

    From rippled keylet::offer(AccountID, uint32).
    """
    return _account_seq_index(_OFFER, account, sequence)


def nftoken_offer_index(account: str, sequence: int) -> str:
    """Compute NFTokenOffer ledger index.

    From rippled keylet::nftoffer(AccountID, uint32).
    """
    return _account_seq_index(_NFTOKEN_OFFER, account, sequence)


def check_index(account: str, sequence: int) -> str:
    """Compute Check ledger index.

    From rippled keylet::check(AccountID, uint32).
    """
    return _account_seq_index(_CHECK, account, sequence)


def escrow_index(account: str, sequence: int) -> str:
    """Compute Escrow ledger index.

    From rippled keylet::escrow(AccountID, uint32).
    """
    return _account_seq_index(_ESCROW, account, sequence)


def ticket_index(account: str, sequence: int) -> str:
    """Compute Ticket ledger index.

    From rippled keylet::ticket(AccountID, uint32).
    """
    return _account_seq_index(_TICKET, account, sequence)


def vault_index(account: str, sequence: int) -> str:
    """Compute Vault ledger index.

    From rippled keylet::vault(AccountID, uint32).
    """
    return _account_seq_index(_VAULT, account, sequence)


def permissioned_domain_index(account: str, sequence: int) -> str:
    """Compute PermissionedDomain ledger index.

    From rippled keylet::permissionedDomain(AccountID, uint32).
    """
    return _account_seq_index(_PERMISSIONED_DOMAIN, account, sequence)


# ---------------------------------------------------------------------------
# MPToken: MPTID = sequence(big-endian u32) + account_id(20 bytes)
#
# The MPTID is a 192-bit (24-byte, 48-hex-char) packed value used as the
# MPTokenIssuanceID field in Set/Authorize/Destroy transactions.
# The ledger object index is SHA512Half(0x00 + '~' + MPTID).
#
# From rippled: makeMptID() in Indexes.cpp, keylet::mptIssuance() in Indexes.cpp
# ---------------------------------------------------------------------------

_MPTOKEN_ISSUANCE = 0x7E  # '~'


def mptid(account: str, sequence: int) -> str:
    """Compute MPToken Issuance ID (192-bit packed value, 48 hex chars).

    This is the value used in the MPTokenIssuanceID transaction field,
    NOT the ledger object index. Structure: sequence(big-endian u32) + AccountID(20 bytes).
    """
    return (struct.pack(">I", sequence) + _account_id(account)).hex().upper()


def mptoken_issuance_index(account: str, sequence: int) -> str:
    """Compute MPTokenIssuance ledger object index (256-bit hash, 64 hex chars).

    From rippled keylet::mptIssuance(MPTID) = indexHash(MPTOKEN_ISSUANCE, MPTID).
    """
    mpt = struct.pack(">I", sequence) + _account_id(account)
    return _compute_index(_MPTOKEN_ISSUANCE, mpt)


# ---------------------------------------------------------------------------
# Credential: SHA512Half(namespace + Subject(20) + Issuer(20) + CredType(var))
# ---------------------------------------------------------------------------


def credential_index(subject: str, issuer: str, credential_type_hex: str) -> str:
    """Compute Credential ledger index.

    From rippled keylet::credential(subject, issuer, credType).

    Args:
        subject: Subject's classic r-address.
        issuer: Issuer's classic r-address.
        credential_type_hex: Credential type as hex string (variable length).
    """
    payload = _account_id(subject) + _account_id(issuer) + bytes.fromhex(credential_type_hex)
    return _compute_index(_CREDENTIAL, payload)


# ---------------------------------------------------------------------------
# NFTokenID: deterministic 256-bit token ID (packed struct, NOT hashed)
#
# Layout (32 bytes, big-endian):
#   [flags: uint16][transfer_fee: uint16][issuer: 20 bytes]
#   [ciphered_taxon: uint32][token_seq: uint32]
#
# The taxon is ciphered with an LCG to prevent minters from controlling
# which NFT page their tokens land on.
#
# From rippled: src/libxrpl/protocol/NFTokenID.cpp
# ---------------------------------------------------------------------------


def _ciphered_taxon(token_seq: int, taxon: int) -> int:
    """Cipher a taxon using the LCG scramble from rippled.

    Constants from rippled nft::cipheredTaxon():
      scramble = (384160001 * token_seq + 2459) mod 2^32
      result = taxon XOR scramble
    """
    scramble = (384160001 * token_seq + 2459) & 0xFFFF_FFFF
    return (taxon ^ scramble) & 0xFFFF_FFFF


def nftoken_id(
    account: str,
    sequence: int,
    taxon: int,
    flags: int = 0,
    transfer_fee: int = 0,
) -> str:
    """Compute the deterministic 256-bit NFTokenID.

    This is NOT a hash — it's a packed 32-byte struct that uniquely identifies
    an NFToken. The ID encodes the minting parameters directly.

    Args:
        account: Minter's classic r-address.
        sequence: Minting transaction's sequence number (MintedNFTokens counter).
        taxon: NFTokenTaxon value from the mint transaction.
        flags: NFToken flags (e.g., tfBurnable=0x0001, tfTransferable=0x0008).
        transfer_fee: Transfer fee in basis points (0-50000).

    Returns:
        64-char uppercase hex string.
    """
    issuer_bytes = _account_id(account)
    ciphered = _ciphered_taxon(sequence, taxon)

    buf = struct.pack(
        ">HH20sII",
        flags & 0xFFFF,
        transfer_fee & 0xFFFF,
        issuer_bytes,
        ciphered,
        sequence & 0xFFFF_FFFF,
    )
    return buf.hex().upper()
