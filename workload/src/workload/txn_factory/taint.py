"""Semantic tainting for intentionally invalid transactions.

Applies random semantic mutations to valid transaction dicts. The tainted dicts
remain structurally valid (pass xrpl-py model validation and binary codec encoding)
but are semantically wrong — rippled rejects them with tem/tef/tec codes.

Tainting strategies are collected from builder modules (TAINTERS dicts) and
supplemented with a generic fallback.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from random import choice

from workload.txn_factory.builders import batch, credential, dex, domain, mptoken, nft, payment, vault

log = logging.getLogger("workload.txn")

# Collect TAINTERS from all builder modules
_TAINTERS: dict[str, list[Callable[[dict], dict]]] = {}

for _mod in [payment, dex, nft, mptoken, vault, credential, domain, batch]:
    _TAINTERS.update(getattr(_mod, "TAINTERS", {}))


def _generic_taint(tx: dict) -> dict:
    """Fallback tainting: zero Amount or self-Destination."""
    if "Amount" in tx:
        if isinstance(tx["Amount"], str):
            tx["Amount"] = "0"
        elif isinstance(tx["Amount"], dict):
            tx["Amount"]["value"] = "0"
    elif "Destination" in tx and "Account" in tx:
        tx["Destination"] = tx["Account"]
    return tx


def taint_txn(tx_dict: dict, txn_type: str) -> dict:
    """Apply a random semantic tainting strategy to a valid transaction dict.

    The dict is mutated in place and returned. If no type-specific tainter exists,
    a generic fallback is used.
    """
    tainters = _TAINTERS.get(txn_type)
    if tainters:
        taint_fn = choice(tainters)
    else:
        taint_fn = _generic_taint
    result = taint_fn(tx_dict)
    log.debug("Tainted %s with %s", txn_type, taint_fn.__doc__ or taint_fn.__name__)
    return result
