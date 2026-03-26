import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from xrpl.asyncio.clients import AsyncJsonRpcClient, AsyncWebsocketClient
from xrpl.models import StreamParameter, Subscribe

from workload.ws import ws_listener
from workload.ws_processor import process_ws_events

from workload.assertions import setup_complete


from workload.config import cfg
from workload.logging_config import setup_logging
from workload.workload_core import Workload, periodic_dex_metrics, periodic_finality_check
from workload import workload_runner

setup_logging()
log = logging.getLogger("workload.app")


def _log_task_exception(task: asyncio.Task) -> None:
    """Done callback: log unhandled exceptions from fire-and-forget tasks."""
    if not task.cancelled() and (exc := task.exception()):
        log.error("Task %r crashed: %s: %s", task.get_name(), type(exc).__name__, exc, exc_info=exc)


if Path("/.dockerenv").is_file():
    rippled = cfg["rippled"]["docker"]
else:
    rippled = cfg["rippled"]["local"]

rpc_port = cfg["rippled"]["rpc_port"]
ws_port = cfg["rippled"]["ws_port"]
rippled_ip = os.getenv("RIPPLED_IP", rippled)

RPC = os.getenv("RPC_URL", f"http://{rippled_ip}:{rpc_port}")
WS = os.getenv("WS_URL", f"ws://{rippled_ip}:{ws_port}")

to = cfg["timeout"]
TIMEOUT = 3.0
OVERALL = to["overall"]
OVERALL_STARTUP_TIMEOUT = to["startup"]
LEDGERS_TO_WAIT = to["initial_ledgers"]


async def _probe_rippled(url: str, max_retries: int = 30, retry_delay: float = 2.0) -> None:
    """Probe rippled RPC endpoint with retries until it responds.


    Args:
        url: RPC endpoint URL
        max_retries: Maximum number of retry attempts (default: 30 = 1 minute with 2s delay)
        retry_delay: Seconds to wait between retries
    """
    payload = {"method": "server_info", "params": [{}]}

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as http:
                r = await http.post(url, json=payload)
                r.raise_for_status()
                log.info(f"RPC endpoint responding (attempt {attempt}/{max_retries})")
                return
        except Exception as e:
            if attempt < max_retries:
                log.info(
                    f"RPC not ready yet (attempt {attempt}/{max_retries}): {e.__class__.__name__} - retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
            else:
                log.error(
                    "\n"
                    "========================================================\n"
                    f"  Cannot reach rippled at {url}\n"
                    f"  Failed after {max_retries} attempts ({e.__class__.__name__})\n"
                    "\n"
                    "  Is the network running? Start it with:\n"
                    "    cd /path/to/testnet && docker compose up -d\n"
                    "\n"
                    "  Or point at a different node:\n"
                    "    RPC_URL=http://<host>:5005 WS_URL=ws://<host>:6006\n"
                    "========================================================"
                )
                raise SystemExit(1)


async def wait_for_ledgers(url: str, count: int, timeout: float = 120.0, retry_delay: float = 5.0) -> None:
    """Connect to the rippled WebSocket and wait for *count* ledger closes.

    Retries the WS connection on failure (the hub may not be ready yet).
    The overall *timeout* caps total wall-clock time across all attempts.
    """
    log.info(f"Connecting to WebSocket {url} to wait for {count} ledgers...")
    deadline = asyncio.get_event_loop().time() + timeout
    attempt = 0
    while True:
        attempt += 1
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            async with asyncio.timeout(remaining):
                async with AsyncWebsocketClient(url) as client:
                    await client.send(Subscribe(streams=[StreamParameter.LEDGER]))
                    ledger_count = 0
                    async for msg in client:
                        if msg.get("type") == "ledgerClosed":
                            ledger_count += 1
                            log.debug("Ledger %s closed. (%s/%s)", msg.get("ledger_index"), ledger_count, count)
                            if ledger_count >= count:
                                log.info("Observed %s ledgers closed. Convinced network is progressing.", ledger_count)
                                return
        except Exception as e:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= retry_delay:
                log.error(
                    "\n"
                    "========================================================\n"
                    f"  Cannot connect to rippled WebSocket at {url}\n"
                    f"  {e.__class__.__name__}: {e}\n"
                    "\n"
                    "  The RPC endpoint responded but the WebSocket is not\n"
                    "  reachable or no ledgers are closing yet.\n"
                    "========================================================"
                )
                raise SystemExit(1)
            log.info(f"WebSocket not ready (attempt {attempt}): {e.__class__.__name__} - retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_interval = 2
    stop = asyncio.Event()

    log.info("=" * 60)
    log.info("XRPL Workload starting up...")
    log.info("  RPC: %s", RPC)
    log.info("  WS:  %s", WS)
    log.info("=" * 60)

    async with asyncio.timeout(OVERALL_STARTUP_TIMEOUT):
        log.info("[1/4] Probing RPC endpoint...")
        await _probe_rippled(RPC)
        log.info("[2/4] Waiting for ledger progress (%d closes)...", LEDGERS_TO_WAIT)
        await wait_for_ledgers(WS, LEDGERS_TO_WAIT)

    log.info(
        "[3/4] Network confirmed. Loading state...\n"
        "  Dashboard:  http://localhost:8000/state/dashboard\n"
        "  API docs:   http://localhost:8000/docs\n"
        "  Explorer:   https://custom.xrpl.org/localhost:6006"
    )

    from workload.sqlite_store import SQLiteStore

    client = AsyncJsonRpcClient(RPC)
    use_sqlite = os.getenv("WORKLOAD_PERSIST", "0") == "1"
    if use_sqlite:
        sqlite_store = SQLiteStore(db_path="state.db")
        log.info("SQLite persistence enabled (WORKLOAD_PERSIST=1)")
    else:
        sqlite_store = None
        log.info("SQLite persistence disabled (set WORKLOAD_PERSIST=1 to enable)")
    app.state.workload = Workload(cfg, client, store=sqlite_store)
    app.state.stop = stop

    app.state.ws_queue = asyncio.Queue(maxsize=1000)  # TODO: Constant
    log.debug("Created WS event queue (maxsize=1000)")

    async with asyncio.TaskGroup() as tg:
        tg.create_task(
            ws_listener(
                app.state.stop, WS, app.state.ws_queue, accounts_provider=app.state.workload.get_all_account_addresses
            ),
            name="ws_listener",
        )

        tg.create_task(
            periodic_finality_check(app.state.workload, app.state.stop, check_interval), name="finality_checker"
        )

        tg.create_task(
            process_ws_events(app.state.workload, app.state.ws_queue, app.state.stop),
            name="ws_processor",
        )

        tg.create_task(
            periodic_dex_metrics(app.state.workload, app.state.stop),
            name="dex_metrics_poller",
        )

        log.info("Background tasks started: ws_listener, finality_checker, ws_processor, dex_metrics_poller")

        log.debug("Checking for persisted state...")
        state_loaded = app.state.workload.load_state_from_store()

        if state_loaded:
            log.info("Loaded existing state from database, skipping network provisioning")
            log.info(
                "  Wallets: %s (Gateways: %s, Users: %s)",
                len(app.state.workload.wallets),
                len(app.state.workload.gateways),
                len(app.state.workload.users),
            )
        else:
            # Try loading from pre-generated genesis accounts
            genesis_cfg = cfg.get("genesis", {})
            accounts_json = genesis_cfg.get("accounts_json", "")

            if accounts_json:
                # Resolve relative to CWD (typically workload/ project root)
                accounts_path = Path(accounts_json)
                if not accounts_path.is_absolute() and not accounts_path.exists():
                    # Try relative to bootstrap.py's directory as fallback
                    accounts_path = Path(__file__).parent / accounts_json
                log.info("Genesis accounts path: %s (exists=%s)", accounts_path, accounts_path.exists())

                genesis_loaded = await app.state.workload.load_from_genesis(str(accounts_path))
            else:
                genesis_loaded = False

            if genesis_loaded:
                log.info(
                    "Loaded from genesis: %s gateways, %s users, %s AMM pools",
                    len(app.state.workload.gateways),
                    len(app.state.workload.users),
                    len(app.state.workload.amm.pools),
                )
            else:
                log.error(
                    "No genesis accounts found and no persisted state.\n"
                    "Use 'gen' to create a testnet with pre-provisioned accounts:\n"
                    "  uv run gen --amendment-profile develop -o testnet"
                )
                raise SystemExit(1)

        init_ledger = await app.state.workload._current_ledger_index()
        app.state.workload.first_ledger_index = init_ledger
        setup_complete(
            {
                "gateways": len(app.state.workload.gateways),
                "users": len(app.state.workload.users),
                "total_wallets": len(app.state.workload.wallets),
                "currencies": len(app.state.workload.ctx.currencies),
                "available_txn_types": app.state.workload.ctx.config.get("transactions", {}).get("available", []),
                "state_loaded_from_db": state_loaded,
                "mptoken_ids": len(app.state.workload._mptoken_issuance_ids),
                "init_completed_ledger": init_ledger,
            }
        )
        log.info("[4/4] Initialization complete at ledger %d. Starting workload.", init_ledger)
        await asyncio.sleep(5)

        # Pre-warm account sequences so the first build iteration doesn't pay RPC latency
        await app.state.workload.warm_sequences(list(app.state.workload.wallets.keys()))

        await workload_runner.start(app.state.workload)
        try:
            yield
        finally:
            log.info("Shutting down...")

            # Stop workload first to prevent new submissions during flush
            await workload_runner.force_stop()

            stop.set()

            # Flush in-memory state to SQLite before exit (skip if no persistent store)
            if app.state.workload.persistent_store is not None:
                log.info("Flushing state to persistent store... (Ctrl-C again to skip)")
                try:
                    await app.state.workload.flush_to_persistent_store()
                except (asyncio.CancelledError, KeyboardInterrupt):
                    log.warning("Flush interrupted, skipping")

            await asyncio.sleep(0.5)
            log.info("Exiting TaskGroup (will cancel any remaining tasks)...")

    log.info("Shutdown complete")
