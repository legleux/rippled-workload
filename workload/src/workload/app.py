import os
import asyncio
import logging
import contextlib
from fastapi import FastAPI, APIRouter
from contextlib import asynccontextmanager
import httpx
from pydantic import BaseModel, PositiveInt
from xrpl.asyncio.clients import AsyncJsonRpcClient, AsyncWebsocketClient
from xrpl.models.transactions import Payment
from xrpl.models import Subscribe, StreamParameter

# Import updated WS components
from workload.ws import ws_listener
from workload.ws_processor import process_ws_events

from xrpl.asyncio.clients import AsyncWebsocketClient
from contextlib import asynccontextmanager
import asyncio, contextlib
import asyncio
import contextlib
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyUrl
from xrpl.asyncio.clients import AsyncWebsocketClient
from xrpl.models import Subscribe, StreamParameter
from xrpl.models import Subscribe, StreamParameter
from xrpl.asyncio.clients import AsyncWebsocketClient
from workload.logging_config import setup_logging

from workload.workload_core import Workload, periodic_finality_check
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from workload.workload_core import ValidationRecord
from fastapi.templating import Jinja2Templates
import json
from fastapi.middleware.cors import CORSMiddleware
from workload.config import cfg
import xrpl
from pathlib import Path

setup_logging()
log = logging.getLogger("workload.app")

if Path("/.dockerenv").is_file():
    rippled = cfg["rippled"]["docker"]
else:
    rippled = cfg["rippled"]["local"]

rpc_port = cfg["rippled"]["rpc_port"]
ws_port = cfg["rippled"]["ws_port"]
rippled_ip = os.getenv("RIPPLED_IP", rippled)

RPC = os.getenv("RPC_URL", f"http://{rippled_ip}:{rpc_port}")
WS = os.getenv("WS_URL", f"ws://{rippled_ip}:{ws_port}")

# TODO: Move to constants
to = cfg["timeout"]
TIMEOUT = 3.0
OVERALL = to["overall"]
OVERALL_STARTUP_TIMEOUT = to["startup"]
LEDGERS_TO_WAIT = 0
WS = "ws://rippled:6006"

# LEDGERS_TO_WAIT = to["initial_ledgers"]

async def _probe_rippled(url: str) -> None:
    # NOTE: Rely on workload from communication?
    payload = {"method": "server_info", "params": [{}]}
    async with httpx.AsyncClient(timeout=TIMEOUT) as http:
        r = await http.post(url, json=payload)
        r.raise_for_status()

async def wait_for_ledgers(url: str, count: int) -> None:
    """
    Connects to the rippled WebSocket and waits for 'count' ledgers to close.
    """
    log.info(f"Connecting to WebSocket {url} to wait for {count} ledgers...")
    try:
        async with AsyncWebsocketClient(url) as client:
            await client.send(Subscribe(streams=[StreamParameter.LEDGER]))
            ledger_count = 0
            async for msg in client:
                if msg.get("type") == "ledgerClosed":
                    ledger_count += 1
                    log.info(f"Ledger {msg.get('ledger_index')} closed. ({ledger_count}/{count})")
                    if ledger_count >= count:
                        log.info("Sufficient ledgers seen. Network is ready.")
                        break
    except Exception as e:
        log.error(f"Failed to wait for ledgers via WebSocket: {e}")
        raise  # Fail startup if we can't confirm network status

async def _dump_tasks(tag: str):
    log.warning("=== TASK DUMP: %s ===", tag)
    for t in asyncio.all_tasks():
        if t is asyncio.current_task():
            continue
        log.warning("task %r done=%s cancelled=%s", t.get_name(), t.done(), t.cancelled())
        for frame in t.get_stack(limit=5):
            log.warning("  at %s:%s in %s", frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)

@asynccontextmanager
async def lifespan(app: FastAPI):
    check_interval = 5
    stop = asyncio.Event()

    # Startup probes to make sure network is ready
    async with asyncio.timeout(OVERALL_STARTUP_TIMEOUT):
        log.info("Probing RPC endpoint...")
        await _probe_rippled(RPC)
        log.info("RPC OK. Waiting for network to be ready (seeing ledger progress)")
        await wait_for_ledgers(WS, LEDGERS_TO_WAIT)

    log.info("Network is ready. Initializing workload...")

    # Initialize workload with SQLite persistence
    from workload.sqlite_store import SQLiteStore

    client = AsyncJsonRpcClient(RPC)
    sqlite_store = SQLiteStore(db_path="workload_state.db")
    app.state.workload = Workload(cfg, client, store=sqlite_store)
    app.state.stop = stop

    # Try to load existing state from database
    state_loaded = app.state.workload.load_state_from_store()

    if state_loaded:
        log.info("✓ Loaded existing state from database, skipping network provisioning")
        log.info(
            f"  Wallets: {len(app.state.workload.wallets)} "
            f"(Gateways: {len(app.state.workload.gateways)}, Users: {len(app.state.workload.users)})"
        )
    else:
        # Initialize participants (provision network)
        gw, u = cfg["gateways"], cfg["users"]
        log.info("No persisted state found. Initializing participants (gateways=%s, users=%s)...", gw, u)
        init_result = await app.state.workload.init_participants(gateway_cfg=gw, user_cfg=u)
        app.state.workload.update_txn_context()  # refresh ctx with our brand new users and gateways
        log.info("Accounts initialized: %s gateways, %s users.", len(init_result["gateways"]), len(init_result["users"]))

    # ============================================================
    # NEW: Create WebSocket event queue for communication between
    # the WS listener and the workload processor
    # ============================================================
    app.state.ws_queue = asyncio.Queue(maxsize=1000)  # Buffer up to 1000 events
    log.info("Created WS event queue (maxsize=1000)")

    # --- concurrent services via TaskGroup ---
    app.state.ws_stop_event = asyncio.Event()
    async with asyncio.TaskGroup() as tg:
        app.state.tg = tg

        # Existing finality checker (now acts as fallback for transactions WS doesn't catch)
        tg.create_task(
            periodic_finality_check(app.state.workload, app.state.stop, check_interval),
            name="finality_checker"
        )

        # ============================================================
        # NEW: WebSocket listener now publishes to queue
        # ============================================================
        tg.create_task(
            ws_listener(app.state.stop, WS, app.state.ws_queue),
            name="ws_listener"
        )

        # ============================================================
        # NEW: WebSocket event processor consumes from queue
        # ============================================================
        tg.create_task(
            process_ws_events(app.state.workload, app.state.ws_queue, app.state.stop),
            name="ws_processor"
        )

        log.info("Startup OK. Ready to accept requests.")
        log.info("Active tasks: finality_checker, ws_listener, ws_processor")

        try:
            yield
            stop.set()
        finally:
            # For debugging
            await _dump_tasks("begin shutdown")
            app.state.ws_stop_event.set()
            # exiting the TaskGroup cancels any still-running tasks after the stop signal

    await _dump_tasks("end shutdown")

app = FastAPI(
    title="XRPL Workload",
    debug=True,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Accounts", "description": "Create and query accounts"},
        {"name": "Payments", "description": "Send and track payments"},
        {"name": "Transactions", "description": "Transactions"},
        {"name": "State", "description": "Send and track general state"},
    ],
    swagger_ui_parameters={
        "tagsSorter": "alpha",       # checkout "order"
        "operationsSorter": "alpha", # checkout "method"
    },
)

r_accounts = APIRouter(prefix="/accounts", tags=["Accounts"])
r_pay = APIRouter(prefix="/payments", tags=["Payments"])
r_transaction = APIRouter(tags=["Transactions"])
r_state = APIRouter(prefix="/state", tags=["State"])


class TxnReq(BaseModel):
    type: str


class CreateAccountReq(BaseModel):
    seed: str | None = None
    address: str | None = None
    drops: int | None = None
    algorithm: str | None = None
    wait: bool | None = False


class CreateAccountResp(BaseModel):
    address: str
    seed: str | None = None
    funded: bool
    tx_hash: str | None = None


class PaymentReq(BaseModel):
    sender_address: str = cfg["funding_account"]["address"]
    receiver_address: str = xrpl.wallet.Wallet.create().address
    drops: PositiveInt = 10


@app.get("/health")
def health():
    return {"status": "ok"} # Not the most thorough of healthchecks...

@r_accounts.get("/create")
async def api_create_account():
    return await app.state.workload.create_account()

@r_accounts.post("/create", response_model=CreateAccountResp)
async def accounts_create(req: CreateAccountReq):
    data = req.model_dump(exclude_unset=True)
    return await app.state.workload.create_account(data, wait=req.wait)


@r_accounts.get("/create/random", response_model=CreateAccountResp)
async def accounts_create_random():
    return await app.state.workload.create_account({}, wait=False)


@r_transaction.get("/random")
async def transaction_random():
    w = app.state.workload
    res = await w.submit_random_txn()
    return res


@r_transaction.get("/create/{transaction}")
async def create(transaction: str):
    w: Workload = app.state.workload
    log.info(f"Creating a {transaction}")
    r = await w.create_transaction(transaction)
    return r

@app.post("/debug/fund")
async def debug_fund(dest: str):
    """Manually fund an address from the workload's configured `funding_account` and return the unvalidated result."""
    w: Workload = app.state.workload
    log.info("funding_wallet %s", w.funding_wallet.address,)
    fund_tx = Payment(
        account=w.funding_wallet.address,
        destination=dest,
        amount=str(1_000_000_000),
    )
    log.info("submitting payment...")
    log.info(json.dumps(fund_tx.to_dict(), indent=2))
    p = await w.build_sign_and_track(fund_tx, w.funding_wallet)
    log.info("bsat: %s", p)
    res = await w.submit_pending(p)
    log.info("response frmo submit_pending() %s", res)
    print("Submit result:", res)
    return res


@r_state.get("/summary")
async def state_summary():
    return app.state.workload.snapshot_stats()


@r_state.get("/pending")
async def state_pending():
    return {"pending": app.state.workload.snapshot_pending()}


@r_state.get("/failed")
async def state_failed():
    return {"failed": app.state.workload.snapshot_failed()}


@r_state.get("/tx/{tx_hash}")
async def state_tx(tx_hash: str):
    data = app.state.workload.snapshot_tx(tx_hash)
    if not data:
        raise HTTPException(404, "tx not tracked")
    return data


@r_state.get("/accounts")
async def state_accounts():
    wl = app.state.workload
    return {
        "count": len(wl.accounts),
        "addresses": list(wl.accounts.keys()),
    }

@r_state.get("/validations")
async def state_validations(limit: int = 100):
    """
    Return recent validation records from the in-memory store.
    Parameters
    ----------
    limit:
        Optional maximum number of records to return (default 100).
    """
    vals = list(app.state.workload.store.validations)[-limit:]
    return [{"txn": v.txn, "ledger": v.seq, "source": v.src} for v in reversed(vals)]

@r_state.get("/wallets")
def api_state_wallets():
    ws = app.state.workload.wallets
    return {"count": len(ws), "addresses": list(ws.keys())}

@r_state.get("/finality")
async def check_finality():
    wl = app.state.workload
    await wl.check_finality


# ============================================================
# NEW: Diagnostic endpoints for WebSocket integration
# ============================================================
@r_state.get("/ws/stats")
async def ws_stats():
    """Return stats about WebSocket event processing."""
    queue_size = app.state.ws_queue.qsize()
    store_stats = app.state.workload.store.snapshot_stats()

    return {
        "queue_size": queue_size,
        "queue_maxsize": app.state.ws_queue.maxsize,
        "validations_by_source": store_stats.get("validated_by_source", {}),
        "recent_validations_count": store_stats.get("recent_validations", 0),
    }


app.include_router(r_accounts)
app.include_router(r_pay)
app.include_router(r_transaction, prefix="/transaction")
app.include_router(r_transaction, prefix="/txn", include_in_schema=False)  # alias /txn/ because I'm sick of typing...
app.include_router(r_state)
