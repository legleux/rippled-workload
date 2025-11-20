"""Test the encode_nftoken_id util."""

from __future__ import annotations

from unittest import TestCase

from xrpl import XRPLException
from xrpl.utils import parse_nftoken_id


def scramble_taxon(taxon: int, sequence: int, taxon_length: int) -> bytes:
    # These values are defined in rippled/include/xrpl/protocol/nft.h in cipheredTaxon
    modulus = 384160001
    increment = 2459
    scramble = modulus * sequence + increment
    taxon ^= scramble
    taxon &= 0xFFFFFFFF  # I think using "to_bytes()",  is clearer but we cannot overflow taxon_length
    return taxon.to_bytes(
        taxon_length, byteorder="big"
    )  # TODO: Can remove byteorder="big" at Python 3.11


def encode_nft_id(
    flags: int,
    issuer: str,
    taxon: int,
    transfer_fee: int,
    sequence: int,
) -> str:
    """Create a unique 256-bit NFTokenID.

        Example:
            flags: 11,
            transfer_fee: 1337,
            issuer: "rJoxBSzpXhPtAuqFmqxQtGKjA13jUJWthE",
            taxon: 1337,
            sequence: 12,
        Should return:
            nft_id = "000B0539C35B55AA096BA6D87A6E6C965A6534150DC56E5E12C5D09E0000000C".

    Returns:
        A 64-character hex string (256 bits).

    """
    # All lengths in bytes
    nft_id_length = 32
    flags_length = 2
    transfer_fee_length = 2
    issuer_length = 20
    taxon_length = 4
    sequence_length = 4

    issuer_bytes = base58.b58decode_check(issuer, alphabet=base58.XRP_ALPHABET)
    issuer_bytes = (
        issuer_bytes[1:] if len(issuer_bytes) > issuer_length else issuer_bytes
    )
    if len(issuer_bytes) != issuer_length:
        msg = f"issuer must be {issuer_length} bytes"
        raise ValueError(msg)
    taxon_bytes = scramble_taxon(taxon, sequence, taxon_length)
    nftoken_id_bytes = (
        flags.to_bytes(flags_length, byteorder="big")
        + transfer_fee.to_bytes(transfer_fee_length, byteorder="big")
        + issuer_bytes
        + taxon_bytes
        + sequence.to_bytes(sequence_length, byteorder="big")
    )

    if len(nftoken_id_bytes) != nft_id_length:
        msg = f"NFT ID is not {nft_id_length} bytes!"
        raise ValueError(msg)
    nftoken_id = nftoken_id_bytes.hex().upper()
    log.info(f"Encoded {nftoken_id=}")
    return nftoken_id
