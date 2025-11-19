import os
import asyncio
import logging
import contextlib
from time import perf_counter
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

try:
    from antithesis.assertions import setup_complete
    ANTITHESIS_AVAILABLE = True
except ImportError:
    ANTITHESIS_AVAILABLE = False
    # No-op fallback for when not in Antithesis environment
    def setup_complete(details=None):
        pass

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

from workload.workload_core import Workload, periodic_finality_check, periodic_state_monitor
from workload.txn_factory.builder import generate_txn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from workload.workload_core import ValidationRecord
import workload.constants as C
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
LEDGERS_TO_WAIT = to["initial_ledgers"]
WS = "ws://rippled:6006"

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
                    log.info("Ledger %s closed. (%s/%s)", msg.get('ledger_index'), ledger_count, count)
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
    check_interval = 2
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

    # ============================================================
    # Create WebSocket event queue then start the background tasks
    # ============================================================
    app.state.ws_queue = asyncio.Queue(maxsize=1000) # TODO: Constant
    log.debug("Created WS event queue (maxsize=1000)")

    app.state.ws_stop_event = asyncio.Event()
    async with asyncio.TaskGroup() as tg:
        app.state.tg = tg

        # Finality checker via RPC polling
        tg.create_task(
            periodic_finality_check(app.state.workload, app.state.stop, check_interval),
            name="finality_checker"
        )
        # using polling only for now
        # ============================================================
        # WebSocket listener
        # ============================================================
        # tg.create_task(
        #     ws_listener(
        #         app.state.stop,
        #         WS,
        #         app.state.ws_queue,
        #         accounts_provider=app.state.workload.get_all_account_addresses
        #     ),
        #     name="ws_listener"
        # )
        #
        # tg.create_task(
        #     process_ws_events(app.state.workload, app.state.ws_queue, app.state.stop),
        #     name="ws_processor"
        # )

        log.info("Background tasks started: finality_checker (polling only), state_monitor (every 10s)")

        # No WS listener, no need to wait
        # await asyncio.sleep(2)

        # DISABLED: State loading causes sequence number conflicts (terPRE_SEQ)
        # When we load wallets but have pending txns, we re-query AccountInfo which doesn't
        # account for allocated sequences from pending txns -> reuse sequence numbers
        state_loaded = False  # app.state.workload.load_state_from_store()

        if state_loaded:
            log.debug("✓ Loaded existing state from database, skipping network provisioning")
            log.debug(
                "  Wallets: %s (Gateways: %s, Users: %s)",
                len(app.state.workload.wallets),
                len(app.state.workload.gateways),
                len(app.state.workload.users)
            )
        else:
            gw, u = cfg["gateways"], cfg["users"]
            log.info("No persisted state found. Initializing participants (gateways=%s, users=%s)...", gw, u)
            init_result = await app.state.workload.init_participants(gateway_cfg=gw, user_cfg=u)
            app.state.workload.update_txn_context()  # refresh ctx with our brand new users and gateways
            log.info("Accounts initialized: %s gateways, %s users.", len(init_result["gateways"]), len(init_result["users"]))

        # Signal setup is complete and fuzzing can begin
        setup_complete({
            "gateways": len(app.state.workload.gateways),
            "users": len(app.state.workload.users),
            "total_wallets": len(app.state.workload.wallets),
            "currencies": len(app.state.workload.ctx.currencies),
            "available_txn_types": app.state.workload.ctx.config.get("transactions", {}).get("available", []),
            "state_loaded_from_db": state_loaded,
            "mptoken_ids": len(app.state.workload._mptoken_issuance_ids),
        })
        log.warning("✓ STARTUP COMPLETE - Workload ready to accept requests.")

        try:
            yield
        finally:
            # Graceful shutdown: set stop signals and give tasks time to exit
            log.info("Shutdown initiated...")
            stop.set()
            app.state.ws_stop_event.set()

            # Give tasks 5 seconds to shut down gracefully before TaskGroup cancels them
            await asyncio.sleep(5)
            log.info("Exiting TaskGroup (will cancel any remaining tasks)...")
            # exiting the TaskGroup cancels any still-running tasks after the stop signal

    log.info("Shutdown complete")

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
r_pay = APIRouter(prefix="/payment", tags=["Payments"])
r_transaction = APIRouter(tags=["Transactions"])
r_state = APIRouter(prefix="/state", tags=["State"])
r_workload = APIRouter(prefix="/workload", tags=["Workload"])


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


class SendPaymentReq(BaseModel):
    source: str
    destination: str
    amount: str | dict  # XRP drops as string, or IOU as {"currency": "USD", "issuer": "r...", "value": "100"}


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


@r_accounts.get("/{account_id}")
async def get_account_info(account_id: str):
    """Get account_info for a specific account."""
    from xrpl.models.requests import AccountInfo

    w: Workload = app.state.workload
    try:
        result = await w.client.request(AccountInfo(account=account_id, ledger_index="validated"))
        return result.result
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Account not found or error: {str(e)}")


@r_accounts.get("/{account_id}/balances")
async def get_account_balances(account_id: str):
    """Get all balances for a specific account from database."""
    from workload.sqlite_store import SQLiteStore

    w: Workload = app.state.workload
    if isinstance(w.store, SQLiteStore):
        balances = w.store.get_balances(account_id)
        return {"account": account_id, "balances": balances}
    else:
        raise HTTPException(status_code=503, detail="Balance tracking not available (not using SQLiteStore)")


@r_accounts.get("/{account_id}/lines")
async def get_account_lines(account_id: str):
    """Get trust lines for a specific account from the ledger."""
    from xrpl.models.requests import AccountLines

    w: Workload = app.state.workload
    try:
        result = await w.client.request(AccountLines(account=account_id, ledger_index="validated"))
        return result.result
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Account lines not found or error: {str(e)}")


@r_pay.post("")
async def send_payment(req: SendPaymentReq):
    """Send a payment from source to destination. Works for both XRP and issued currencies."""
    w: Workload = app.state.workload

    # Look up source wallet from memory
    source_wallet = w.wallets.get(req.source)
    if not source_wallet:
        raise HTTPException(status_code=404, detail=f"Source wallet not found: {req.source}")

    # Build payment (amount works for both XRP string and IOU dict)
    payment = Payment(
        account=req.source,
        destination=req.destination,
        amount=req.amount,
    )

    # Build, sign, and submit
    pending = await w.build_sign_and_track(payment, source_wallet)
    result = await w.submit_pending(pending)

    return {
        "tx_hash": pending.tx_hash,
        "engine_result": result.get("engine_result") if result else None,
        "state": pending.state.name,
        "source": req.source,
        "destination": req.destination,
        "amount": req.amount,
    }


@r_transaction.get("/random")
async def transaction_random():
    w = app.state.workload
    res = await w.submit_random_txn()
    return res


@r_transaction.get("/create/{transaction}")
async def create(transaction: str):
    w: Workload = app.state.workload
    log.debug("Creating a %s", transaction)
    r = await w.create_transaction(transaction)
    return r

@app.post("/debug/fund")
async def debug_fund(dest: str):
    """Manually fund an address from the workload's configured `funding_account` and return the unvalidated result."""
    w: Workload = app.state.workload
    log.debug("funding_wallet %s", w.funding_wallet.address,)
    fund_tx = Payment(
        account=w.funding_wallet.address,
        destination=dest,
        amount=str(1_000_000_000),
    )
    log.debug("submitting payment...")
    log.debug(json.dumps(fund_tx.to_dict(), indent=2))
    p = await w.build_sign_and_track(fund_tx, w.funding_wallet)
    log.debug("bsat: %s", p)
    res = await w.submit_pending(p)
    log.debug("response frmo submit_pending() %s", res)
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


@r_state.get("/expired")
async def state_expired():
    """Get transactions that expired without validating."""
    wl = app.state.workload
    expired = [r for r in wl.snapshot_pending(open_only=False) if r["state"] == "EXPIRED"]
    return {"expired": expired}


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


@r_state.get("/users")
def api_state_users():
    """Get all user wallets with addresses and seeds."""
    wl = app.state.workload
    users = [
        {
            "address": user.address,
            "seed": user.seed,
        }
        for user in wl.users
    ]
    return {"count": len(users), "users": users}


@r_state.get("/gateways")
def api_state_gateways():
    """Get all gateway wallets with addresses, seeds, and issued currencies."""
    wl = app.state.workload
    gateways = []
    for gateway in wl.gateways:
        # Find all currencies issued by this gateway
        issued_currencies = [
            curr.currency
            for curr in wl._currencies
            if curr.issuer == gateway.address
        ]
        gateways.append({
            "address": gateway.address,
            "seed": gateway.seed,
            "currencies": issued_currencies,
        })
    return {"count": len(gateways), "gateways": gateways}


@r_state.get("/currencies")
def get_currencies():
    """Get all configured/issued currencies."""
    wl = app.state.workload
    currencies = [
        {
            "currency": curr.currency,
            "issuer": curr.issuer,
        }
        for curr in wl._currencies
    ]
    return {"count": len(currencies), "currencies": currencies}


@r_state.get("/mptokens")
def get_mptokens():
    """Get all tracked MPToken issuance IDs."""
    wl = app.state.workload
    mptoken_ids = getattr(wl, '_mptoken_issuance_ids', [])
    return {
        "count": len(mptoken_ids),
        "mptoken_issuance_ids": mptoken_ids,
        "note": "MPToken IDs are tracked automatically when MPTokenIssuanceCreate transactions validate"
    }


@r_state.get("/finality")
async def check_finality():
    """Manually trigger finality check for all pending submitted transactions."""
    wl = app.state.workload
    results = []

    for p in wl.find_by_state(C.TxState.SUBMITTED):
        try:
            state, ledger_index = await wl.check_finality(p)
            results.append({
                "tx_hash": p.tx_hash,
                "state": state.name,
                "ledger_index": ledger_index,
            })
        except Exception as e:
            results.append({
                "tx_hash": p.tx_hash,
                "error": str(e),
            })

    return {
        "checked": len(results),
        "results": results,
    }


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


workload_running = False
workload_stop_event = None
workload_task = None
workload_stats = {"submitted": 0, "validated": 0, "failed": 0, "started_at": None}


async def continuous_workload():
    """Continuously submit random transactions, ramping up with expected_ledger_size."""
    from random import random
    global workload_stats
    wl = app.state.workload

    log.debug("🚀 Continuous workload started")
    workload_stats["started_at"] = perf_counter()

    try:
        while not workload_stop_event.is_set():
            # Get current expected ledger size
            ledger_size = await wl._expected_ledger_size()

            # Occasionally create a new account
            if random() < 0.10:
                try:
                    default_balance = wl.config["users"]["default_balance"]
                    large_balance = str(int(default_balance) * 10)

                    log.debug(f"Creating new account with {large_balance} drops")
                    result = await wl.create_account(initial_xrp_drops=large_balance)
                    workload_stats["submitted"] += 1

                    log.debug(f"✓ New account created: {result['address'][:12]}... (will be adopted on validation)")
                except Exception as e:
                    log.error(f"Failed to create new account: {e}")
                    workload_stats["failed"] += 1

            batch_size = max(10, ledger_size // 2)
            log.info(f"📊 Expected ledger size: {ledger_size} - submitting {batch_size} txns (throttled to avoid queue overflow)")

            try:
                # Build and sign all transactions first
                pending_txns = []
                build_errors = 0
                for i in range(batch_size):
                    if workload_stop_event.is_set():
                        break
                    try:
                        # Generate transaction and prepare it
                        txn = await generate_txn(wl.ctx)
                        pending = await wl.build_sign_and_track(txn, wl.wallets[txn.account])
                        pending_txns.append(pending)
                    except Exception as e:
                        log.error(f"Failed to build txn {i+1}/{batch_size}: {e}", exc_info=True)
                        build_errors += 1
                        workload_stats["failed"] += 1

                if build_errors > 0:
                    log.warning(f"⚠️  {build_errors} transactions failed to build, only submitting {len(pending_txns)}/{batch_size}")

                if not pending_txns:
                    log.error("No transactions to submit!")
                    continue

                log.info(f"📤 Submitting {len(pending_txns)} transactions with throttling...")

                # Submit with slight throttling to avoid overwhelming the queue
                # Submit in smaller sub-batches with delays
                submit_errors = 0
                submit_tasks = []
                SUB_BATCH_SIZE = 5  # Submit 5 at a time

                for i in range(0, len(pending_txns), SUB_BATCH_SIZE):
                    sub_batch = pending_txns[i:i+SUB_BATCH_SIZE]
                    async with asyncio.TaskGroup() as tg:
                        batch_tasks = [tg.create_task(wl.submit_pending(p)) for p in sub_batch]
                        submit_tasks.extend(batch_tasks)
                    # Small delay between sub-batches to let queue drain
                    if i + SUB_BATCH_SIZE < len(pending_txns):
                        await asyncio.sleep(0.1)

                # Count results
                for task in submit_tasks:
                    try:
                        result = task.result()
                        workload_stats["submitted"] += 1

                        if result.get("state") == "VALIDATED":
                            workload_stats["validated"] += 1
                        elif result.get("state") in ["REJECTED", "FAILED_NET", "EXPIRED"]:
                            workload_stats["failed"] += 1
                    except Exception as e:
                        log.error(f"Error in parallel submission: {e}")
                        submit_errors += 1
                        workload_stats["failed"] += 1

                if submit_errors > 0:
                    log.warning(f"⚠️  {submit_errors} transactions failed to submit")

            except* Exception as eg:
                # Handle ExceptionGroup from TaskGroup failures
                for exc in eg.exceptions:
                    log.error(f"Task failed in batch submission: {type(exc).__name__}: {exc}", exc_info=exc)
                workload_stats["failed"] += batch_size

            # Wait for next ledger to close before submitting next batch
            # This prevents flooding the queue and keeps expiration rate low
            current_ledger = await wl._current_ledger_index()
            next_ledger = current_ledger + 1
            while await wl._current_ledger_index() < next_ledger and not workload_stop_event.is_set():
                await asyncio.sleep(0.5)

    except asyncio.CancelledError:
        log.debug("Continuous workload cancelled")
        raise
    finally:
        log.debug(f"🛑 Continuous workload stopped - Stats: {workload_stats}")


@r_workload.post("/start")
async def start_workload():
    """Start continuous random transaction workload."""
    global workload_running, workload_stop_event, workload_task, workload_stats

    if workload_running:
        raise HTTPException(status_code=400, detail="Workload already running")

    # Reset stats
    workload_stats = {"submitted": 0, "validated": 0, "failed": 0, "started_at": perf_counter()}

    log.info("Starting workload")
    # Create stop event and start task
    workload_stop_event = asyncio.Event()
    workload_task = asyncio.create_task(continuous_workload())
    workload_running = True

    return {
        "status": "started",
        "message": "Continuous workload started - submitting random transactions at expected_ledger_size + 1 per ledger"
    }


@r_workload.post("/stop")
async def stop_workload():
    """Stop continuous workload."""
    global workload_running, workload_stop_event, workload_task

    if not workload_running:
        raise HTTPException(status_code=400, detail="Workload not running")

    # Signal stop and wait for task to complete
    log.info("Stopping workload")
    workload_stop_event.set()
    await workload_task
    workload_running = False
    log.info("Stopped workload")

    return {
        "status": "stopped",
        "stats": workload_stats
    }


@r_workload.get("/status")
async def workload_status():
    """Get current workload status and statistics."""
    return {
        "running": workload_running,
        "stats": workload_stats,
        "uptime_seconds": perf_counter() - workload_stats["started_at"] if workload_stats["started_at"] else 0
    }


app.include_router(r_accounts)
app.include_router(r_pay)
app.include_router(r_pay, prefix="/pay", include_in_schema=False)  # alias /pay/ for convenience
app.include_router(r_transaction, prefix="/transaction")
app.include_router(r_transaction, prefix="/txn", include_in_schema=False)  # alias /txn/ because I'm sick of typing...
app.include_router(r_state)
app.include_router(r_workload)
