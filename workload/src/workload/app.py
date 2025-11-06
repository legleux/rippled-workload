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
from xrpl.models import (
    Subscribe,
    StreamParameter
)
from workload.ws import ws_listener
from xrpl.models import Subscribe, StreamParameter
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
    # --- startup probes ---
    async with asyncio.timeout(OVERALL_STARTUP_TIMEOUT):
        log.info("Probing RPC endpoint...")
        await _probe_rippled(RPC)
        log.info("RPC OK. Waiting for network to be ready (seeing ledger progress)")
        await wait_for_ledgers(WS, LEDGERS_TO_WAIT)

    # --- init workload ---
    log.info("Network is ready. Initializing workload...")
    client = AsyncJsonRpcClient(RPC)
    app.state.workload = Workload(cfg, client)
    app.state.stop = stop
    gw, u = cfg["gateways"], cfg["users"]
    log.info("Initializing participants (gateways=%s, users=%s)...", gw, u)
    init_result = await app.state.workload.init_participants(gateway_cfg=gw, user_cfg=u)
    log.info("Accounts initialized: %s gateways, %s users.", len(init_result["gateways"]), len(init_result["users"]))

    # --- concurrent services via TaskGroup ---
    app.state.ws_stop_event = asyncio.Event()
    async with asyncio.TaskGroup() as tg:
        app.state.tg = tg
        tg.create_task(periodic_finality_check(app.state.workload, app.state.stop, check_interval), name="finality")
        tg.create_task(ws_listener(app.state.stop, WS), name="ws")
        log.info("Startup OK. Ready to accept requests.")
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

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

r_accounts = APIRouter(prefix="/accounts", tags=["Accounts"])
r_pay = APIRouter(prefix="/payments", tags=["Payments"])
# r_transaction = APIRouter(prefix="/transaction", tags=["Transactions"])
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
    # algorithm: str


class PaymentReq(BaseModel):
    sender_address: str = cfg["funding_account"]["address"]
    receiver_address: str = xrpl.wallet.Wallet.create().address
    drops: PositiveInt = 10


@app.get("/health")
def health():
    return {"status": "ok"} # Not the most thorough of healthchecks...


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

@r_state.get("/finality")
async def check_finality():
    wl = app.state.workload
    await wl.check_finality
## Websocket listener maintainence
# TODO: Add to router
# @app.get("/ws/listeners")
# async def ws_list():
#     _ensure_ws_state(app)
#     return [
#         {
#             "url": url,
#             "done": t.task.done(),
#             "cancelled": t.task.cancelled(),
#             "exception": (repr(t.task.exception())
#                           if t.task.done() and not t.task.cancelled()
#                           else None),
#         }
#         for url, t in app.state.ws_listeners.items()
#     ]

# @app.post("/ws/listeners")
# async def ws_add(req: WSAddReq):
#     await _start_ws(app, str(req.url))
#     return {"added": str(req.url)}

# @app.delete("/ws/listeners")
# async def ws_del(req: WSAddReq):
#     ok = await _stop_ws(app, str(req.url))
#     if not ok:
#         raise HTTPException(404, "listener not found")
#     return {"removed": str(req.url)}


app.include_router(r_accounts)
app.include_router(r_pay)
app.include_router(r_transaction, prefix="/transaction")
app.include_router(r_transaction, prefix="/txn", include_in_schema=False)  # alias /txn/ because I'm sick of typing...
app.include_router(r_state)

# @app.get("/validator/{n}", response=HTMLResponse)
# async def validator_state(n):
#     vstate = app.state.workload.validator_state(n)
#     vtmpl = Template(filename="templates/validator_layout.html.mako"))
#     v = vtmpl.render(vstate)
#     return v.render_unicode()

# from mako.template import Template
# @app.get("/", response=HTMLResponse)
# async def home():
#     template = mylookup.get_template('templates/')
#     return template.render_unicode('validator.html')

# templates = Jinja2Templates(directory="templates")



# from pathlib import Path
# BASE_DIR = Path(__file__).resolve().parent
# templates = Jinja2Templates(directory=BASE_DIR / "templates")
# # templates = Jinja2Templates(directory="workload/src/workload/templates")
# @app.get("/val{n}", response_class=HTMLResponse)
# async def val_n(n: int, request: Request):
#     val_url = await app.state.workload.validator_state(n)

#     return templates.TemplateResponse(
#         request=request, name="val.html.jinja", context={"val": n}
#     )
#     return templates.TemplateResponse("val.html.jinja", data)

# @app.get("/server_info/{n}")
# async def server_info(n):
#     log.info(f"got request for {n}")
#     async with httpx.AsyncClient(timeout=10) as c:
#         url = await app.state.workload.validator_state(n)
#         log.info(f"Hitting val{n} at {url}")
#         r = await c.post(url, json={"method": "server_info"})
#     log.info("got request!")
#     r.raise_for_status()
#     return r.json()

# from pathlib import Path
# @app.get("/val", response_class=HTMLResponse)
# async def val():
#     vf = Path(__file__).parent / 'templates/validator.html'
#     return vf.read_text()
#     return templates.TemplateResponse("some-file.html",
#           {
#               "request": request,
#                // pass your variables to HTML template here
#               "my_variable": my_variable
#           }
# )
