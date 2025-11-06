# workload/ws.py
import asyncio, json, logging, websockets
log = logging.getLogger("workload.ws")

RECV_TIMEOUT = 90.0
RECONNECT_BASE = 1.0
RECONNECT_MAX = 10.0

async def ws_listener(stop: asyncio.Event, ws_url: str) -> None:
    backoff = RECONNECT_BASE
    while not stop.is_set():
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20, close_timeout=1) as ws:
                log.info("WS connected: %s", ws_url)
                await ws.send(json.dumps({"id": 1, "command": "subscribe", "streams": ["transactions", "ledger"]}))
                try:
                    ack = await asyncio.wait_for(ws.recv(), timeout=10)
                    if json.loads(ack).get("status") != "success":
                        raise RuntimeError(f"subscribe failed: {ack}")
                except asyncio.TimeoutError:
                    pass

                backoff = RECONNECT_BASE
                while not stop.is_set():
                    recv = asyncio.create_task(ws.recv())
                    halt = asyncio.create_task(stop.wait())
                    done, pending = await asyncio.wait({recv, halt}, return_when=asyncio.FIRST_COMPLETED)
                    for t in pending: t.cancel()
                    if halt in done:
                        return
                    msg = recv.result()
                    try:
                        obj = json.loads(msg)
                        kind = obj.get("type") or obj.get("status") or obj.get("engine_result") or "message"
                        log.debug("WS %s: %s", kind, msg)
                    except Exception:
                        log.debug("WS raw: %s", msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("WS error: %s", e)

        if stop.is_set(): break
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, RECONNECT_MAX)

    log.info("WS listener stopped")
