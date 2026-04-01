"""
WebSocket listener that:
1. Maintains persistent connection to rippled WS endpoint
2. Subscribes to account streams for validated + proposed txns
3. Publishes events to a queue for workload processing
4. Handles reconnection with exponential backoff
5. Defers account subscription until accounts are loaded (accounts_ready event)
"""

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Literal

import websockets

log = logging.getLogger("workload.ws")
ws_accounts_log = logging.getLogger("workload.ws.accounts")

RECV_TIMEOUT = 90.0
RECONNECT_BASE = 1.0
RECONNECT_MAX = 10.0

EventType = Literal["tx_validated", "tx_proposed", "tx_response", "ledger_closed", "server_status", "raw"]


async def _subscribe_accounts(ws, accounts: list[str]) -> bool:
    """Send a subscribe command for accounts + accounts_proposed. Returns True on success."""
    subscribe_msg = {
        "id": 2,
        "command": "subscribe",
        "accounts": accounts,
        "accounts_proposed": accounts,
    }
    ws_accounts_log.info("Subscribing to %d accounts + accounts_proposed", len(accounts))
    await ws.send(json.dumps(subscribe_msg))
    try:
        ack = await asyncio.wait_for(ws.recv(), timeout=10)
        ack_obj = json.loads(ack)
        if ack_obj.get("status") != "success":
            ws_accounts_log.error("Account subscription failed: %s", ack)
            return False
        ws_accounts_log.info("Account subscription successful (%d accounts)", len(accounts))
        return True
    except asyncio.TimeoutError:
        ws_accounts_log.warning("Account subscription ack timeout, continuing anyway")
        return True


async def ws_listener(
    stop: asyncio.Event,
    ws_url: str,
    event_queue: asyncio.Queue,
    accounts_provider: Callable | None = None,
    accounts_ready: asyncio.Event | None = None,
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
    accounts_ready:
        Optional event that signals accounts have been loaded (genesis/db).
        WS subscribes to ledger+server immediately but defers account subscription
        until this event is set.
    """
    backoff = RECONNECT_BASE

    while not stop.is_set():
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20, close_timeout=1) as ws:
                log.info("WS connected: %s", ws_url)

                # Phase 1: Subscribe to ledger + server streams immediately
                base_msg = {"id": 1, "command": "subscribe", "streams": ["ledger", "server"]}
                await ws.send(json.dumps(base_msg))
                try:
                    ack = await asyncio.wait_for(ws.recv(), timeout=10)
                    ack_obj = json.loads(ack)
                    if ack_obj.get("status") != "success":
                        raise RuntimeError(f"base subscribe failed: {ack}")
                    log.info("WS base subscription successful (ledger + server)")
                except asyncio.TimeoutError:
                    log.warning("WS base subscription ack timeout, continuing anyway")

                # Phase 2: Wait for accounts to be ready, then subscribe
                if accounts_ready and not accounts_ready.is_set():
                    ws_accounts_log.info("Waiting for accounts_ready before subscribing to accounts...")
                    # Wait for accounts_ready OR stop, whichever comes first
                    ready_task = asyncio.create_task(accounts_ready.wait())
                    halt_task = asyncio.create_task(stop.wait())
                    done, pending = await asyncio.wait({ready_task, halt_task}, return_when=asyncio.FIRST_COMPLETED)
                    for t in pending:
                        t.cancel()
                    if halt_task in done or stop.is_set():
                        log.info("WS listener received stop signal while waiting for accounts")
                        return
                    ws_accounts_log.info("accounts_ready fired, proceeding with account subscription")

                # Now subscribe to accounts
                if accounts_provider:
                    try:
                        accounts = accounts_provider()
                        if accounts:
                            await _subscribe_accounts(ws, accounts)
                        else:
                            ws_accounts_log.warning("accounts_provider returned empty list")
                    except Exception as e:
                        ws_accounts_log.error("Failed to get accounts from provider: %s", e)

                backoff = RECONNECT_BASE

                while not stop.is_set():
                    recv_task = asyncio.create_task(ws.recv())
                    halt_task = asyncio.create_task(stop.wait())

                    done, pending = await asyncio.wait({recv_task, halt_task}, return_when=asyncio.FIRST_COMPLETED)

                    for t in pending:
                        t.cancel()

                    if halt_task in done or stop.is_set():
                        log.info("WS listener received stop signal")
                        return

                    try:
                        msg = recv_task.result()
                        await _process_message(msg, event_queue)
                    except websockets.exceptions.ConnectionClosedError:
                        if stop.is_set():
                            return  # Expected during shutdown
                        log.warning("WS connection closed unexpectedly, will reconnect")
                        break  # Break inner loop to trigger reconnect
                    except Exception as e:
                        if stop.is_set():
                            return
                        log.error("Error processing WS message: %s", e)

        except asyncio.CancelledError:
            log.info("WS listener cancelled")
            raise
        except websockets.exceptions.ConnectionClosedError:
            if stop.is_set():
                log.info("WS connection closed during shutdown")
                return
            log.warning("WS connection closed, will reconnect")
        except Exception as e:
            if stop.is_set():
                return
            log.error("WS connection error: %s", e)

        if stop.is_set():
            break

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

    if msg_type == "transaction":
        if obj.get("validated"):
            tx_hash = obj.get("hash") or obj.get("transaction", {}).get("hash")
            account = obj.get("transaction", {}).get("Account", "?")
            ws_accounts_log.debug("validated: hash=%s account=%s ledger=%s", tx_hash, account, obj.get("ledger_index"))
            await queue.put(("tx_validated", obj))
        elif obj.get("engine_result"):
            tx_hash = obj.get("hash") or obj.get("transaction", {}).get("hash")
            ws_accounts_log.debug("proposed: hash=%s result=%s", tx_hash, obj.get("engine_result"))
            await queue.put(("tx_proposed", obj))
        return

    if msg_type == "ledgerClosed":
        ledger_idx = obj.get("ledger_index")
        log.debug("WS ledger_closed: %s", ledger_idx)
        await queue.put(("ledger_closed", obj))
        return

    if msg_type == "serverStatus":
        log.debug("WS server_status: load_factor=%s", obj.get("load_factor"))
        await queue.put(("server_status", obj))
        return

    engine_result = obj.get("engine_result")
    if engine_result:
        log.debug("WS tx_response: result=%s", engine_result)
        await queue.put(("tx_response", obj))
        return

    status = obj.get("status")
    if status:
        log.debug("WS status: %s", status)
        return

    log.debug("WS unknown message type: %s", obj.get("type") or "no_type")
