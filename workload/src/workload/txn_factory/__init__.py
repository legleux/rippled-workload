"""Transaction factory — build, compose, and taint XRPL transactions.

Re-exports the public API so existing imports continue to work.
"""

from workload.txn_factory.context import TxnContext, choice_omit, deep_update, sample_omit, token_metadata
from workload.txn_factory.taint import taint_txn
from workload.txn_factory.registry import (
    _BUILDERS,
    build_txn_dict,
    compose_submission_set,
    create_accountset,
    create_amm_create,
    create_batch,
    create_mptoken_issuance_create,
    create_nftoken_mint,
    create_payment,
    create_trustset,
    create_xrp_payment,
    generate_txn,
    global_eligible_types,
    is_account_eligible,
    pick_eligible_txn_type,
    txn_model_cls,
    update_transaction,
)

__all__ = [
    "_BUILDERS",
    "TxnContext",
    "build_txn_dict",
    "compose_submission_set",
    "choice_omit",
    "create_accountset",
    "create_amm_create",
    "create_batch",
    "create_mptoken_issuance_create",
    "create_nftoken_mint",
    "create_payment",
    "create_trustset",
    "create_xrp_payment",
    "deep_update",
    "generate_txn",
    "global_eligible_types",
    "is_account_eligible",
    "pick_eligible_txn_type",
    "sample_omit",
    "taint_txn",
    "token_metadata",
    "txn_model_cls",
    "update_transaction",
]
