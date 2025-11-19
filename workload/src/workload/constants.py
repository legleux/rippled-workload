from typing import Final
from enum import StrEnum

genesis_account: Final = {
    "address": "rHb9CJAWyB4rj91VRWn96DkukG4bwdtyTh",
    "seed": "snoPBrXtMeMyMHUVTgbuqAfg1SUTb",
    "public_key": "...",
    "private_key": "...",
}

GENESIS = genesis_account

class TxType(StrEnum):
    ACCOUNT_SET                = "AccountSet"
    AMM_CREATE                 = "AMMCreate"
    BATCH                      = "Batch"
    MPTOKEN_ISSUANCE_CREATE    = "MPTokenIssuanceCreate"
    MPTOKEN_ISSUANCE_SET       = "MPTokenIssuanceSet"
    MPTOKEN_AUTHORIZE          = "MPTokenAuthorize"
    MPTOKEN_ISSUANCE_DESTROY   = "MPTokenIssuanceDestroy"
    NFTOKEN_MINT               = "NFTokenMint"
    PAYMENT                    = "Payment"
    TRUSTSET                   = "TrustSet"

class TxState(StrEnum):
    CREATED    = "CREATED"
    SUBMITTED  = "SUBMITTED"
    RETRYABLE  = "RETRYABLE"
    VALIDATED  = "VALIDATED"
    REJECTED   = "REJECTED"
    EXPIRED    = "EXPIRED"
    FAILED_NET = "FAILED_NET"



# TODO: organize these
DEFAULT_CREATE_AMOUNT = int(100 * 1e6)
MAX_CREATE_AMOUNT = int(100e6 * 1e6) # alot?
HORIZON = 10  # Transactions expire if not validated within 10 ledgers (~30-40 seconds)
RPC_TIMEOUT = 2.0
SUBMIT_TIMEOUT = 20
LOCK_TIMEOUT = 2.0

__all__ = [
    "DEFAULT_CREATE_AMOUNT",
    "HORIZON",
    "LOCK_TIMEOUT",
    "MAX_CREATE_AMOUNT",
    "RPC_TIMEOUT",
    "SUBMIT_TIMEOUT",

    ######
    "TxType",
    "TxState",
]
