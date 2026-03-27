"""Validation hooks — called when transactions are confirmed in a validated ledger.

Each hook inspects the validated PendingTx and updates tracking state on the
Workload instance (NFTs, offers, tickets, checks, escrows, balances, etc.).

Extracted from workload_core.py for maintainability. These are standalone
functions that take the Workload instance as first argument.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from xrpl.models.requests import AccountInfo, Tx

import workload.constants as C
from workload.ledger_objects import check_index, escrow_index, mptid, nftoken_id, nftoken_offer_index
from workload.validation import ValidationRecord

if TYPE_CHECKING:
    from workload.workload_core import PendingTx, Workload

log = logging.getLogger("workload")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


async def dispatch_validation_hooks(
    wl: Workload,
    p_live: PendingTx | None,
    rec: ValidationRecord,
    meta_result: str | None,
) -> None:
    """Run all post-validation hooks. Called from Workload.record_validated()."""
    on_account_adopted(wl, p_live, rec)
    on_payment_validated(wl, p_live, meta_result)
    on_mptoken_created(wl, p_live)
    await on_batch_validated(wl, p_live)
    on_amm_created(wl, p_live, meta_result)
    on_dex_activity(wl, p_live, meta_result)
    on_credential_created(wl, p_live)
    on_credential_accepted(wl, p_live)
    on_credential_deleted(wl, p_live)
    await on_vault_created(wl, p_live, rec)
    on_vault_deleted(wl, p_live)
    await on_domain_created(wl, p_live, rec)
    on_domain_deleted(wl, p_live)
    on_nftoken_minted(wl, p_live)
    on_nftoken_burned(wl, p_live)
    on_offer_created(wl, p_live)
    on_offer_cancelled(wl, p_live)
    on_nft_offer_created(wl, p_live)
    on_nft_offer_accepted(wl, p_live)
    on_ticket_created(wl, p_live)
    on_ticket_consumed(wl, p_live)
    on_check_created(wl, p_live)
    on_check_cashed(wl, p_live)
    on_check_cancelled(wl, p_live)
    on_escrow_created(wl, p_live)
    on_escrow_finished(wl, p_live)
    on_escrow_cancelled(wl, p_live)


# ---------------------------------------------------------------------------
# Account adoption
# ---------------------------------------------------------------------------


def _is_viable_for_pool(wl: Workload, balances: dict) -> bool:
    """Return True if an account has enough balance to send at least one transaction.

    Checks each asset class against its configured minimum send amount plus
    a small fee/reserve buffer. Accounts that can't afford anything should
    not be added to the submission pool.

    Args:
        balances: Dict of {asset_key: amount} where asset_key is:
            - "XRP"             → amount in drops (float/int)
            - (currency, issuer) → IOU amount (float)
            - ("MPT", mpt_id)   → MPToken amount (float)

    Returns:
        True if the account can afford at least one configured transaction type.
    """
    BASE_RESERVE_DROPS = 2_000_000  # 2 XRP minimum account reserve
    FEE_BUFFER_DROPS = 10_000  # headroom for fees on a few txns

    # XRP check: must cover reserve + fee buffer + at least one payment
    xrp_balance = balances.get("XRP", 0.0)
    xrp_payment = int(wl.config.get("transactions", {}).get("payment", {}).get("amount", 0))
    if xrp_balance >= BASE_RESERVE_DROPS + FEE_BUFFER_DROPS + xrp_payment:
        return True

    # IOU check: any non-zero IOU balance is sufficient to attempt a payment
    for key, amount in balances.items():
        if isinstance(key, tuple) and len(key) == 2 and key[0] != "MPT" and amount > 0:
            return True

    # MPToken check: any non-zero MPT balance
    for key, amount in balances.items():
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "MPT" and amount > 0:
            return True

    return False


def on_account_adopted(wl: Workload, p_live: PendingTx | None, rec: ValidationRecord) -> None:
    """Adopt a newly funded account into the submission pool if it's viable."""
    if p_live is None:
        return
    w = getattr(p_live, "wallet", None)
    if w is None:
        return
    funded_drops = 0
    tx_amount = (p_live.tx_json or {}).get("Amount")
    if isinstance(tx_amount, str):
        funded_drops = int(tx_amount)
    if _is_viable_for_pool(wl, {"XRP": float(funded_drops)}):
        wl.wallets[w.address] = w
        wl._record_for(w.address)
        wl.users.append(w)
        wl.save_wallet_to_store(w, is_user=True, funded_ledger_index=rec.seq)
        wl.update_txn_context()
        # Fire-and-forget sequence warmup so it's cached before next build iteration
        asyncio.create_task(wl.warm_sequences([w.address]))
        log.debug("Adopted new account %s (funded: %d drops)", w.address, funded_drops)
    else:
        log.debug(
            "Skipped adopting %s — funded amount %d drops insufficient for any configured payment "
            "(need > %d drops for XRP payment + reserve + fees)",
            w.address,
            funded_drops,
            int(wl.config.get("transactions", {}).get("payment", {}).get("amount", 0)) + 2_010_000,
        )


# ---------------------------------------------------------------------------
# Payment
# ---------------------------------------------------------------------------


def on_payment_validated(wl: Workload, p_live: PendingTx | None, meta_result: str | None) -> None:
    """Update in-memory balances for a successful Payment."""
    if not (p_live and meta_result == "tesSUCCESS" and p_live.transaction_type == C.TxType.PAYMENT):
        return
    try:
        tx_json = p_live.tx_json
        if not tx_json:
            return
        sender = tx_json.get("Account")
        destination = tx_json.get("Destination")
        amount = tx_json.get("Amount")
        if not (sender and destination and amount):
            return
        if isinstance(amount, str):
            amount_val = float(amount)
            wl._update_balance(sender, "XRP", -amount_val)
            wl._update_balance(destination, "XRP", amount_val)
        elif isinstance(amount, dict):
            currency = amount.get("currency")
            issuer = amount.get("issuer")
            value = float(amount.get("value", 0))
            if currency and issuer:
                if sender != issuer:
                    wl._update_balance(sender, currency, -value, issuer)
                if destination != issuer:
                    wl._update_balance(destination, currency, value, issuer)
                log.debug("Balance update: %s -> %s: %s %s", sender, destination, value, currency)
    except Exception as e:
        log.debug("Failed to update in-memory balances for %s: %s", p_live.tx_hash, e)


# ---------------------------------------------------------------------------
# MPToken
# ---------------------------------------------------------------------------


def on_mptoken_created(wl: Workload, p_live: PendingTx | None) -> None:
    """Track new MPToken issuance ID — computed deterministically, no RPC needed."""
    if not (p_live and p_live.transaction_type == C.TxType.MPTOKEN_ISSUANCE_CREATE):
        return
    if not (p_live.account and p_live.sequence is not None):
        log.warning("MPTokenIssuanceCreate missing account/sequence: %s", p_live.tx_hash)
        return
    mpt_id = mptid(p_live.account, p_live.sequence)
    if mpt_id not in wl._mptoken_issuance_ids:
        wl._mptoken_issuance_ids[mpt_id] = p_live.account
        wl.update_txn_context()
        log.info("Tracked MPToken issuance: %s (issuer=%s seq=%d)", mpt_id, p_live.account, p_live.sequence)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


async def on_batch_validated(wl: Workload, p_live: PendingTx | None) -> None:
    """Sync account sequence from ledger after a Batch transaction validates."""
    if not (p_live and p_live.transaction_type == C.TxType.BATCH and p_live.account):
        return
    try:
        ai = await wl._rpc(AccountInfo(account=p_live.account, ledger_index="validated"))
        rec_acct = wl._record_for(p_live.account)
        async with rec_acct.lock:
            old_seq = rec_acct.next_seq
            rec_acct.next_seq = ai.result["account_data"]["Sequence"]
            log.debug("Batch validated: synced %s sequence %s -> %s", p_live.account, old_seq, rec_acct.next_seq)
    except Exception as e:
        log.warning("Failed to sync sequence after Batch validation: %s", e)


# ---------------------------------------------------------------------------
# AMM / DEX
# ---------------------------------------------------------------------------


def on_amm_created(wl: Workload, p_live: PendingTx | None, meta_result: str | None) -> None:
    """Register a new AMM pool after a successful AMMCreate."""
    if not (p_live and meta_result == "tesSUCCESS" and p_live.transaction_type == C.TxType.AMM_CREATE):
        return
    try:
        tx_json = p_live.tx_json
        if not tx_json:
            return
        amount1 = tx_json.get("Amount")
        amount2 = tx_json.get("Amount2")
        if not (amount1 and amount2):
            return
        asset1 = (
            {"currency": "XRP"}
            if isinstance(amount1, str)
            else {"currency": amount1["currency"], "issuer": amount1["issuer"]}
        )
        asset2 = (
            {"currency": "XRP"}
            if isinstance(amount2, str)
            else {"currency": amount2["currency"], "issuer": amount2["issuer"]}
        )
        wl._register_amm_pool(asset1, asset2, p_live.account)
    except Exception as e:
        log.warning("Failed to register AMM pool from %s: %s", p_live.tx_hash, e)


def on_dex_activity(wl: Workload, p_live: PendingTx | None, meta_result: str | None) -> None:
    """Update DEX metrics counters for successful AMM deposit/withdraw and offer creation."""
    if not (p_live and meta_result == "tesSUCCESS"):
        return
    if p_live.transaction_type == C.TxType.AMM_DEPOSIT:
        wl.dex_metrics.total_deposits += 1
        tx_json = p_live.tx_json or {}
        dep_asset1 = tx_json.get("Asset")
        dep_asset2 = tx_json.get("Asset2")
        if dep_asset1 and dep_asset2 and p_live.account:
            wl.amm.add_lp_holder(dep_asset1, dep_asset2, p_live.account)
    elif p_live.transaction_type == C.TxType.AMM_WITHDRAW:
        wl.dex_metrics.total_withdrawals += 1
    elif p_live.transaction_type == C.TxType.OFFER_CREATE:
        wl.dex_metrics.total_offers += 1


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def on_credential_created(wl: Workload, p_live: PendingTx | None) -> None:
    if not (p_live and p_live.transaction_type == C.TxType.CREDENTIAL_CREATE):
        return
    tx_json = p_live.tx_json or {}
    cred = {
        "issuer": tx_json.get("Account"),
        "subject": tx_json.get("Subject"),
        "credential_type": tx_json.get("CredentialType"),
        "accepted": False,
    }
    if cred["issuer"] and cred["subject"] and cred["credential_type"]:
        wl._credentials.append(cred)
        wl.update_txn_context()
        log.debug("Tracked credential: issuer=%s subject=%s", cred["issuer"], cred["subject"])


def on_credential_accepted(wl: Workload, p_live: PendingTx | None) -> None:
    if not (p_live and p_live.transaction_type == C.TxType.CREDENTIAL_ACCEPT):
        return
    tx_json = p_live.tx_json or {}
    issuer = tx_json.get("Issuer")
    subject = tx_json.get("Account")  # The acceptor is the subject
    cred_type = tx_json.get("CredentialType")
    for c in wl._credentials:
        if c["issuer"] == issuer and c["subject"] == subject and c["credential_type"] == cred_type:
            c["accepted"] = True
            wl.update_txn_context()
            break


def on_credential_deleted(wl: Workload, p_live: PendingTx | None) -> None:
    if not (p_live and p_live.transaction_type == C.TxType.CREDENTIAL_DELETE):
        return
    tx_json = p_live.tx_json or {}
    issuer = tx_json.get("Issuer")
    subject = tx_json.get("Subject")
    cred_type = tx_json.get("CredentialType")
    wl._credentials = [
        c
        for c in wl._credentials
        if not (c["issuer"] == issuer and c["subject"] == subject and c["credential_type"] == cred_type)
    ]
    wl.update_txn_context()


# ---------------------------------------------------------------------------
# Vaults
# ---------------------------------------------------------------------------


async def on_vault_created(wl: Workload, p_live: PendingTx | None, rec: ValidationRecord) -> None:
    if not (p_live and p_live.transaction_type == C.TxType.VAULT_CREATE):
        return
    try:
        tx_result = await wl._rpc(Tx(transaction=rec.txn))
        meta = tx_result.result.get("meta", {})
        for node in meta.get("AffectedNodes", []):
            created = node.get("CreatedNode", {})
            if created.get("LedgerEntryType") == "Vault":
                vault_id = created.get("LedgerIndex")
                if vault_id:
                    tx_json = p_live.tx_json or {}
                    vault = {
                        "vault_id": vault_id,
                        "owner": p_live.account,
                        "asset": tx_json.get("Asset"),
                    }
                    wl._vaults.append(vault)
                    wl.update_txn_context()
                    log.debug("Tracked vault: %s owner=%s", vault_id, p_live.account)
                    return
    except Exception as e:
        log.warning("Failed to extract vault ID from %s: %s", rec.txn, e)


def on_vault_deleted(wl: Workload, p_live: PendingTx | None) -> None:
    if not (p_live and p_live.transaction_type == C.TxType.VAULT_DELETE):
        return
    tx_json = p_live.tx_json or {}
    vault_id = tx_json.get("VaultID")
    if vault_id:
        wl._vaults = [v for v in wl._vaults if v["vault_id"] != vault_id]
        wl.update_txn_context()


# ---------------------------------------------------------------------------
# Permissioned Domains
# ---------------------------------------------------------------------------


async def on_domain_created(wl: Workload, p_live: PendingTx | None, rec: ValidationRecord) -> None:
    if not (p_live and p_live.transaction_type == C.TxType.PERMISSIONED_DOMAIN_SET):
        return
    # If DomainID is present, this is an update — not a new domain
    if (p_live.tx_json or {}).get("DomainID"):
        return
    try:
        tx_result = await wl._rpc(Tx(transaction=rec.txn))
        meta = tx_result.result.get("meta", {})
        for node in meta.get("AffectedNodes", []):
            created = node.get("CreatedNode", {})
            if created.get("LedgerEntryType") == "PermissionedDomain":
                domain_id = created.get("LedgerIndex")
                if domain_id:
                    domain = {"domain_id": domain_id, "owner": p_live.account}
                    wl._domains.append(domain)
                    wl.update_txn_context()
                    log.debug("Tracked domain: %s owner=%s", domain_id, p_live.account)
                    return
    except Exception as e:
        log.warning("Failed to extract domain ID from %s: %s", rec.txn, e)


def on_domain_deleted(wl: Workload, p_live: PendingTx | None) -> None:
    if not (p_live and p_live.transaction_type == C.TxType.PERMISSIONED_DOMAIN_DELETE):
        return
    tx_json = p_live.tx_json or {}
    domain_id = tx_json.get("DomainID")
    if domain_id:
        wl._domains = [d for d in wl._domains if d["domain_id"] != domain_id]
        wl.update_txn_context()


# ---------------------------------------------------------------------------
# NFTokens
# ---------------------------------------------------------------------------


def on_nftoken_minted(wl: Workload, p_live: PendingTx | None) -> None:
    """Track a newly minted NFToken — ID is deterministic, no RPC needed."""
    if not (p_live and p_live.transaction_type == C.TxType.NFTOKEN_MINT):
        return
    tx_json = p_live.tx_json or {}
    sequence = tx_json.get("Sequence")
    if sequence is None:
        return
    nft_id_hex = nftoken_id(
        account=p_live.account,
        sequence=sequence,
        taxon=tx_json.get("NFTokenTaxon", 0),
        flags=tx_json.get("Flags", 0),
        transfer_fee=tx_json.get("TransferFee", 0),
    )
    wl._nfts[nft_id_hex] = p_live.account
    wl.update_txn_context()
    log.debug("Tracked NFToken: %s owner=%s", nft_id_hex, p_live.account)


def on_nftoken_burned(wl: Workload, p_live: PendingTx | None) -> None:
    if not (p_live and p_live.transaction_type == C.TxType.NFTOKEN_BURN):
        return
    nft_id_hex = (p_live.tx_json or {}).get("NFTokenID", "").upper()
    if nft_id_hex and nft_id_hex in wl._nfts:
        del wl._nfts[nft_id_hex]
        wl.update_txn_context()
        log.debug("Removed burned NFToken: %s", nft_id_hex)


# ---------------------------------------------------------------------------
# DEX Offers
# ---------------------------------------------------------------------------


def on_offer_created(wl: Workload, p_live: PendingTx | None) -> None:
    """Track a new DEX offer — index computed from account + sequence."""
    if not (p_live and p_live.transaction_type == C.TxType.OFFER_CREATE):
        return
    tx_json = p_live.tx_json or {}
    sequence = tx_json.get("Sequence")
    if sequence is None:
        return
    offer_key = f"{p_live.account}:{sequence}"
    wl._offers[offer_key] = {
        "type": "IOUOffer",
        "owner": p_live.account,
        "sequence": sequence,
    }
    wl.update_txn_context()
    log.debug("Tracked offer: %s", offer_key)


def on_offer_cancelled(wl: Workload, p_live: PendingTx | None) -> None:
    if not (p_live and p_live.transaction_type == C.TxType.OFFER_CANCEL):
        return
    tx_json = p_live.tx_json or {}
    offer_seq = tx_json.get("OfferSequence")
    if offer_seq is not None:
        offer_key = f"{p_live.account}:{offer_seq}"
        if offer_key in wl._offers:
            del wl._offers[offer_key]
            wl.update_txn_context()


# ---------------------------------------------------------------------------
# NFToken Offers
# ---------------------------------------------------------------------------


def on_nft_offer_created(wl: Workload, p_live: PendingTx | None) -> None:
    """Track a new NFToken offer — ledger index computed locally."""
    if not (p_live and p_live.transaction_type == C.TxType.NFTOKEN_CREATE_OFFER):
        return
    tx_json = p_live.tx_json or {}
    sequence = tx_json.get("Sequence")
    if sequence is None:
        return
    offer_idx = nftoken_offer_index(p_live.account, sequence)
    is_sell = bool(tx_json.get("Flags", 0) & 1)  # tfSellNFToken = 0x0001
    wl._offers[offer_idx] = {
        "type": "NFTokenOffer",
        "owner": p_live.account,
        "is_sell_offer": is_sell,
        "nft_id": tx_json.get("NFTokenID", ""),
        "sequence": sequence,
    }
    wl.update_txn_context()
    log.debug("Tracked NFToken offer: %s sell=%s", offer_idx, is_sell)


def on_nft_offer_accepted(wl: Workload, p_live: PendingTx | None) -> None:
    """Remove accepted NFToken offer and update NFT ownership if sell offer."""
    if not (p_live and p_live.transaction_type == C.TxType.NFTOKEN_ACCEPT_OFFER):
        return
    tx_json = p_live.tx_json or {}
    changed = False
    for offer_field in ("NFTokenSellOffer", "NFTokenBuyOffer"):
        offer_id = tx_json.get(offer_field)
        if offer_id and offer_id in wl._offers:
            offer_data = wl._offers.pop(offer_id)
            changed = True
            # Update NFT ownership for sell offers — buyer is the acceptor
            if offer_data.get("is_sell_offer") and offer_data.get("nft_id"):
                nft_id_upper = offer_data["nft_id"].upper()
                if nft_id_upper in wl._nfts:
                    wl._nfts[nft_id_upper] = p_live.account
    if changed:
        wl.update_txn_context()


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------


def on_ticket_created(wl: Workload, p_live: PendingTx | None) -> None:
    """Track tickets created by a validated TicketCreate transaction.

    TicketCreate with Sequence=S and TicketCount=N creates ticket sequences
    S+1 through S+N.
    """
    if not (p_live and p_live.transaction_type == C.TxType.TICKET_CREATE):
        return
    tx_json = p_live.tx_json or {}
    sequence = tx_json.get("Sequence")
    ticket_count = tx_json.get("TicketCount")
    if sequence is None or ticket_count is None:
        return
    account = p_live.account
    new_tickets = {sequence + 1 + i for i in range(ticket_count)}
    if account not in wl._tickets:
        wl._tickets[account] = new_tickets
    else:
        wl._tickets[account] |= new_tickets
    wl.update_txn_context()
    log.debug("Tracked %d tickets for %s: seqs %s", ticket_count, account, sorted(new_tickets))


def on_ticket_consumed(wl: Workload, p_live: PendingTx | None) -> None:
    """Remove a consumed ticket when a txn using TicketSequence validates."""
    if not p_live:
        return
    tx_json = p_live.tx_json or {}
    ticket_seq = tx_json.get("TicketSequence")
    if ticket_seq is None:
        return
    account = p_live.account
    acct_tickets = wl._tickets.get(account)
    if acct_tickets and ticket_seq in acct_tickets:
        acct_tickets.discard(ticket_seq)
        if not acct_tickets:
            del wl._tickets[account]
        wl.update_txn_context()
        log.debug("Consumed ticket %d for %s", ticket_seq, account)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def on_check_created(wl: Workload, p_live: PendingTx | None) -> None:
    """Track a newly created Check — ID computed deterministically."""
    if not (p_live and p_live.transaction_type == C.TxType.CHECK_CREATE):
        return
    tx_json = p_live.tx_json or {}
    sequence = tx_json.get("Sequence")
    if sequence is None:
        return
    cid = check_index(p_live.account, sequence)
    wl._checks[cid] = {
        "sender": p_live.account,
        "destination": tx_json.get("Destination"),
        "send_max": tx_json.get("SendMax"),
    }
    wl.update_txn_context()
    log.debug("Tracked Check: %s sender=%s dst=%s", cid, p_live.account, tx_json.get("Destination"))


def on_check_cashed(wl: Workload, p_live: PendingTx | None) -> None:
    """Remove a cashed Check from tracking."""
    if not (p_live and p_live.transaction_type == C.TxType.CHECK_CASH):
        return
    check_id = (p_live.tx_json or {}).get("CheckID")
    if check_id and check_id in wl._checks:
        del wl._checks[check_id]
        wl.update_txn_context()
        log.debug("Removed cashed Check: %s", check_id)


def on_check_cancelled(wl: Workload, p_live: PendingTx | None) -> None:
    """Remove a cancelled Check from tracking."""
    if not (p_live and p_live.transaction_type == C.TxType.CHECK_CANCEL):
        return
    check_id = (p_live.tx_json or {}).get("CheckID")
    if check_id and check_id in wl._checks:
        del wl._checks[check_id]
        wl.update_txn_context()
        log.debug("Removed cancelled Check: %s", check_id)


# ---------------------------------------------------------------------------
# Escrows
# ---------------------------------------------------------------------------


def on_escrow_created(wl: Workload, p_live: PendingTx | None) -> None:
    """Track a newly created Escrow — ID computed deterministically."""
    if not (p_live and p_live.transaction_type == C.TxType.ESCROW_CREATE):
        return
    tx_json = p_live.tx_json or {}
    sequence = tx_json.get("Sequence")
    if sequence is None:
        return
    eid = escrow_index(p_live.account, sequence)
    wl._escrows[eid] = {
        "owner": p_live.account,
        "sequence": sequence,
        "destination": tx_json.get("Destination"),
        "finish_after": tx_json.get("FinishAfter", 0),
        "cancel_after": tx_json.get("CancelAfter", 0),
    }
    wl.update_txn_context()
    log.debug("Tracked Escrow: %s owner=%s dst=%s", eid, p_live.account, tx_json.get("Destination"))


def on_escrow_finished(wl: Workload, p_live: PendingTx | None) -> None:
    """Remove a finished Escrow from tracking."""
    if not (p_live and p_live.transaction_type == C.TxType.ESCROW_FINISH):
        return
    tx_json = p_live.tx_json or {}
    owner = tx_json.get("Owner")
    offer_seq = tx_json.get("OfferSequence")
    if owner and offer_seq is not None:
        eid = escrow_index(owner, offer_seq)
        if eid in wl._escrows:
            del wl._escrows[eid]
            wl.update_txn_context()
            log.debug("Removed finished Escrow: %s", eid)


def on_escrow_cancelled(wl: Workload, p_live: PendingTx | None) -> None:
    """Remove a cancelled Escrow from tracking."""
    if not (p_live and p_live.transaction_type == C.TxType.ESCROW_CANCEL):
        return
    tx_json = p_live.tx_json or {}
    owner = tx_json.get("Owner")
    offer_seq = tx_json.get("OfferSequence")
    if owner and offer_seq is not None:
        eid = escrow_index(owner, offer_seq)
        if eid in wl._escrows:
            del wl._escrows[eid]
            wl.update_txn_context()
            log.debug("Removed cancelled Escrow: %s", eid)
