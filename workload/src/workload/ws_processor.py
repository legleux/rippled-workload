# workload/ws_processor.py
"""
WebSocket event processor that consumes events from the WS listener queue
and updates transaction states in the Workload.

This is the bridge between the passive WS listener and the active Workload state machine.
"""
import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from workload.workload_core import Workload

from workload.workload_core import ValidationRecord, ValidationSrc
import workload.constants as C

try:
    from antithesis.assertions import sometimes, always
    ANTITHESIS_AVAILABLE = True
except ImportError:
    ANTITHESIS_AVAILABLE = False
    # No-op fallbacks for when not in Antithesis environment
    def sometimes(condition, message, details=None):
        pass
    def always(condition, message, details=None):
        pass

log = logging.getLogger("workload.ws_processor")


async def process_ws_events(
    workload: "Workload",
    event_queue: asyncio.Queue,
    stop: asyncio.Event,
) -> None:
    """
    Consume events from WebSocket listener and update workload state.

    This task runs for the lifetime of the application, processing:
    - Transaction validations from the stream (SEQUENTIAL - parallel caused blocking)
    - Ledger close notifications
    - Immediate submission responses (if/when we switch to WS submission)

    Parameters
    ----------
    workload:
        The Workload instance to update
    event_queue:
        Queue receiving events from ws_listener
    stop:
        Event to signal graceful shutdown
    """
    log.info("WS event processor starting")
    processed_count = 0

    try:
        while not stop.is_set():
            try:
                # Wait for event with timeout so we can check stop signal
                event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                event_type, data = event

                if event_type == "tx_validated":
                    # Process sequentially to avoid flooding event loop with concurrent tasks
                    await _handle_tx_validated(workload, data)
                    processed_count += 1

                elif event_type == "ledger_closed":
                    await _handle_ledger_closed(workload, data)

                elif event_type == "tx_response":
                    # For future: when we switch to WS submission
                    await _handle_tx_response(workload, data)

                elif event_type == "raw":
                    # Unknown messages - just log
                    pass

                # Periodic stats logging
                if processed_count > 0 and processed_count % 100 == 0:
                    log.info("WS processor: %d validations processed", processed_count)

            except asyncio.TimeoutError:
                # No event in 1s, check stop signal and continue
                continue
            except Exception as e:
                log.error("Error processing WS event: %s", e, exc_info=True)
                # Continue processing despite errors

    except asyncio.CancelledError:
        log.info("WS event processor cancelled")
        raise
    finally:
        log.info("WS event processor stopped (processed %d events)", processed_count)


async def _handle_tx_validated(workload: "Workload", msg: dict) -> None:
    """
    Handle a validated transaction from the WebSocket stream.

    We subscribe to specific accounts (our wallets), so we should only receive
    transactions affecting our accounts. However, we still need to check if
    it's in pending since we receive notifications for:
    - Transactions we submitted (in pending)
    - Transactions TO our accounts from external sources (not in pending)

    Message structure:
    {
        "type": "transaction",
        "validated": true,
        "transaction": {
            "hash": "ABC123...",
            "Account": "rXXX...",
            ...
        },
        "meta": {
            "TransactionResult": "tesSUCCESS",
            ...
        },
        "ledger_index": 12345
    }
    """
    tx_data = msg.get("transaction", {})
    tx_hash = tx_data.get("hash")

    if not tx_hash:
        log.debug("WS validation missing tx hash, ignoring")
        return

    # Check if this is a transaction we're tracking
    pending = workload.pending.get(tx_hash)
    if not pending:
        # Transaction affecting our account but we didn't submit it
        # (e.g., payment TO us from external source, or system transaction)
        log.debug("WS validation for non-pending tx affecting our accounts: %s", tx_hash[:8])
        return

    # Extract validation data
    ledger_index = msg.get("ledger_index")
    meta = msg.get("meta", {})
    meta_result = meta.get("TransactionResult")

    if not ledger_index:
        log.warning("WS validation missing ledger_index for tx %s", tx_hash)
        return

    log.debug("WS validation: tx=%s ledger=%s result=%s account=%s",
              tx_hash[:8], ledger_index, meta_result, pending.account[:8])

    # Record validation through the official path
    # This will:
    # 1. Update pending transaction state to VALIDATED
    # 2. Store validation in the deque
    # 3. Update per-source counters
    # 4. Adopt new wallets if this was a funding payment
    validation_record = ValidationRecord(
        txn=tx_hash,
        seq=ledger_index,
        src=ValidationSrc.WS,
    )

    try:
        await workload.record_validated(validation_record, meta_result=meta_result)
    except Exception as e:
        log.error("Failed to record WS validation for %s: %s", tx_hash, e, exc_info=True)


async def _handle_ledger_closed(workload: "Workload", msg: dict) -> None:
    """
    Handle a ledger close notification.

    Fetches ledger transaction count and runs Antithesis assertions to validate:
    - Ledgers contain transactions at least sometimes (workload is functioning)
    - Network is processing submitted transactions

    Also submits heartbeat transaction for this ledger.

    Message structure:
    {
        "type": "ledgerClosed",
        "ledger_index": 12345,
        "ledger_hash": "ABC123...",
        "ledger_time": 742623456,
        ...
    }
    """
    ledger_index = msg.get("ledger_index")
    ledger_hash = msg.get("ledger_hash")

    log.debug("Ledger %s closed (hash: %s)", ledger_index, ledger_hash)

    # Submit heartbeat for this ledger - our canary!
    try:
        await workload.submit_heartbeat(ledger_index)
    except Exception as e:
        log.error("Failed to submit heartbeat for ledger %s: %s", ledger_index, e)

    # Fetch ledger to get transaction count for Antithesis assertions
    try:
        from xrpl.models.requests import Ledger

        ledger_req = Ledger(
            ledger_index=ledger_index,
            transactions=True,  # Include tx hashes (not full txns)
            expand=False,       # Don't expand to full transaction objects
        )

        ledger_resp = await workload._rpc(ledger_req)

        if ledger_resp.is_successful():
            ledger_data = ledger_resp.result.get("ledger", {})
            transactions = ledger_data.get("transactions", [])
            txn_count = len(transactions)

            log.debug("Ledger %s closed with %d transactions", ledger_index, txn_count)

            # Antithesis assertion: we should see transactions in ledgers at least sometimes
            # This validates the workload is functioning and transactions are being processed
            sometimes(
                txn_count > 0,
                "ledger_contains_transactions",
                {
                    "ledger_index": ledger_index,
                    "txn_count": txn_count,
                    "ledger_hash": ledger_hash,
                }
            )

            # Additional assertion: if workload is running, ledgers should have transactions
            # This is more strict - useful for detecting workload stalls
            if hasattr(workload, '_workload_started') and workload._workload_started:
                always(
                    txn_count > 0,
                    "active_workload_produces_transactions",
                    {
                        "ledger_index": ledger_index,
                        "txn_count": txn_count,
                    }
                )
        else:
            log.warning("Failed to fetch ledger %s: %s", ledger_index, ledger_resp.result)

    except Exception as e:
        log.error("Error fetching ledger %s for assertions: %s", ledger_index, e)


async def _handle_tx_response(workload: "Workload", msg: dict) -> None:
    """
    Handle immediate submission response from WebSocket.

    This is for FUTURE USE when we switch to WebSocket-based submission.
    Currently submissions go via RPC, so this won't fire.

    Message structure (after WS submit):
    {
        "engine_result": "tesSUCCESS",
        "engine_result_code": 0,
        "engine_result_message": "The transaction was applied...",
        "tx_json": {
            "hash": "ABC123...",
            ...
        },
        ...
    }
    """
    engine_result = msg.get("engine_result")
    tx_json = msg.get("tx_json", {})
    tx_hash = tx_json.get("hash")

    if not tx_hash:
        log.debug("WS response missing tx hash")
        return

    log.info("WS submission response: tx=%s result=%s", tx_hash[:8], engine_result)

    # TODO: When we switch to WS submission, this would call record_submitted()
    # For now, just log that we saw it
