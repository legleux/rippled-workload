# workload/ws.py
"""
WebSocket listener that:
1. Maintains persistent connection to rippled WS endpoint
2. Subscribes to transaction and ledger streams
3. Publishes events to a queue for workload processing
4. Handles reconnection with exponential backoff
"""
import asyncio
import json
import logging
import websockets
from typing import Literal

log = logging.getLogger("workload.ws")

RECV_TIMEOUT = 90.0
RECONNECT_BASE = 1.0
RECONNECT_MAX = 10.0

# Event types we publish to the queue
EventType = Literal["tx_validated", "tx_response", "ledger_closed", "raw"]


async def ws_listener(
    stop: asyncio.Event,
    ws_url: str,
    event_queue: asyncio.Queue,
    accounts_provider: callable = None,
) -> None:
    """
    Connect to rippled WebSocket, subscribe to streams, and publish events to queue.

    Parameters
    ----------
    stop:
        Event to signal graceful shutdown
    ws_url:
        WebSocket URL (e.g., "ws://rippled:6006")
    event_queue:
        Queue to publish parsed events for workload consumption
    accounts_provider:
        Optional callable that returns list of account addresses to subscribe to.
        If None, subscribes to all transactions (inefficient but simple).
    """
    backoff = RECONNECT_BASE

    while not stop.is_set():
        try:
            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=1
            ) as ws:
                log.info("WS connected: %s", ws_url)

                # Get account addresses to subscribe to
                accounts = None
                if accounts_provider:
                    try:
                        accounts = accounts_provider()
                        if accounts:
                            log.debug(f"Got {len(accounts)} accounts from provider")
                        else:
                            log.debug("No accounts available yet (likely during startup)")
                    except Exception as e:
                        log.warning(f"Failed to get accounts from provider: {e}, falling back to transaction stream")

                # Subscribe to ledger stream + either specific accounts or all transactions
                if accounts:
                    # Efficient: subscribe only to our accounts
                    subscribe_msg = {
                        "id": 1,
                        "command": "subscribe",
                        "streams": ["ledger"],
                        "accounts": accounts
                    }
                    log.info(f"✓ Subscribing to ledger + {len(accounts)} specific accounts (efficient)")
                else:
                    # Fallback: subscribe to all transactions (used during init before accounts exist)
                    subscribe_msg = {
                        "id": 1,
                        "command": "subscribe",
                        "streams": ["transactions", "ledger"]
                    }
                    log.info("Subscribing to ALL transactions (fallback - will switch to accounts after init/reconnect)")

                await ws.send(json.dumps(subscribe_msg))

                # Wait for subscription acknowledgment
                try:
                    ack = await asyncio.wait_for(ws.recv(), timeout=10)
                    ack_obj = json.loads(ack)
                    if ack_obj.get("status") != "success":
                        raise RuntimeError(f"subscribe failed: {ack}")
                    log.info("WS subscription successful")
                except asyncio.TimeoutError:
                    log.warning("WS subscription ack timeout, continuing anyway")

                # Reset backoff on successful connection
                backoff = RECONNECT_BASE

                # Main message loop
                while not stop.is_set():
                    recv_task = asyncio.create_task(ws.recv())
                    halt_task = asyncio.create_task(stop.wait())

                    done, pending = await asyncio.wait(
                        {recv_task, halt_task},
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # Cancel pending task
                    for t in pending:
                        t.cancel()

                    # Check if we're shutting down
                    if halt_task in done:
                        log.info("WS listener received stop signal")
                        return

                    # Process the message
                    try:
                        msg = recv_task.result()
                        await _process_message(msg, event_queue)
                    except Exception as e:
                        log.error("Error processing WS message: %s", e, exc_info=True)

        except asyncio.CancelledError:
            log.info("WS listener cancelled")
            raise
        except Exception as e:
            log.error("WS connection error: %s", e)

        # Don't reconnect if we're stopping
        if stop.is_set():
            break

        # Exponential backoff before reconnecting
        log.info("WS reconnecting in %.1fs", backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, RECONNECT_MAX)

    log.info("WS listener stopped")


async def _process_message(raw_msg: str, queue: asyncio.Queue) -> None:
    """
    Parse WebSocket message and publish appropriate event to queue.

    Message types we handle:
    - type="transaction" + validated=true → tx_validated event
    - type="ledgerClosed" → ledger_closed event
    - engine_result present → tx_response event (immediate submission feedback)
    """
    try:
        obj = json.loads(raw_msg)
    except json.JSONDecodeError:
        log.debug("WS raw (non-JSON): %s", raw_msg[:200])
        await queue.put(("raw", raw_msg))
        return

    msg_type = obj.get("type")

    # Transaction validation notification from stream
    if msg_type == "transaction" and obj.get("validated"):
        log.debug("WS tx_validated: hash=%s ledger=%s",
                 obj.get("transaction", {}).get("hash"),
                 obj.get("ledger_index"))
        await queue.put(("tx_validated", obj))
        return

    # Ledger close notification
    if msg_type == "ledgerClosed":
        ledger_idx = obj.get("ledger_index")
        log.debug("WS ledger_closed: %s", ledger_idx)
        await queue.put(("ledger_closed", obj))
        return

    # Immediate submission response (has engine_result)
    engine_result = obj.get("engine_result")
    if engine_result:
        log.debug("WS tx_response: result=%s", engine_result)
        await queue.put(("tx_response", obj))
        return

    # Subscription acknowledgments, status messages, etc.
    status = obj.get("status")
    if status:
        log.debug("WS status: %s", status)
        return

    # Unknown message type - log and continue
    log.debug("WS unknown message type: %s", obj.get("type") or "no_type")
