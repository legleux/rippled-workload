import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager
from time import perf_counter

import httpx
from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, PositiveInt
from xrpl.asyncio.clients import AsyncJsonRpcClient, AsyncWebsocketClient
from xrpl.models import StreamParameter, Subscribe
from xrpl.models.transactions import Payment

from workload.ws import ws_listener
from workload.ws_processor import process_ws_events

try:
    from antithesis.lifecycle import setup_complete

    ANTITHESIS_AVAILABLE = True
except ImportError:
    ANTITHESIS_AVAILABLE = False

    def setup_complete(details=None):
        pass


import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import xrpl
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import AnyUrl, BaseModel
from xrpl.asyncio.clients import AsyncWebsocketClient
from xrpl.models import StreamParameter, Subscribe

import workload.constants as C
from workload.config import cfg
from workload.logging_config import setup_logging
from workload.txn_factory.builder import generate_txn
from workload.workload_core import ValidationRecord, Workload, periodic_dex_metrics, periodic_finality_check

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
                log.error(f"RPC failed after {max_retries} attempts")
                raise


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
                    log.info("Ledger %s closed. (%s/%s)", msg.get("ledger_index"), ledger_count, count)
                    if ledger_count >= count:
                        log.info("Observed %s ledgers closed. Convinced network is progessing.", ledger_count)
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

    async with asyncio.timeout(OVERALL_STARTUP_TIMEOUT):
        log.info("Probing RPC endpoint...")
        await _probe_rippled(RPC)
        log.info("RPC OK. Waiting for network to be ready (seeing ledger progress)")
        await wait_for_ledgers(WS, LEDGERS_TO_WAIT)

    log.info("Network is ready. Initializing workload...")

    from workload.sqlite_store import SQLiteStore

    client = AsyncJsonRpcClient(RPC)
    sqlite_store = SQLiteStore(db_path="state.db")
    app.state.workload = Workload(cfg, client, store=sqlite_store)
    app.state.stop = stop

    app.state.ws_queue = asyncio.Queue(maxsize=1000)  # TODO: Constant
    log.debug("Created WS event queue (maxsize=1000)")

    app.state.ws_stop_event = asyncio.Event()
    async with asyncio.TaskGroup() as tg:
        app.state.tg = tg

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
                from pathlib import Path

                # Resolve relative to CWD (typically workload/ project root)
                accounts_path = Path(accounts_json)
                if not accounts_path.is_absolute() and not accounts_path.exists():
                    # Try relative to app.py's directory as fallback
                    accounts_path = Path(__file__).parent / accounts_json
                if not accounts_path.exists():
                    # Try absolute from repo root
                    accounts_path = Path(__file__).parent.parent.parent.parent / "prepare-workload" / "testnet" / "accounts.json"
                log.info("Genesis accounts path: %s (exists=%s)", accounts_path, accounts_path.exists())

                genesis_loaded = await app.state.workload.load_from_genesis(str(accounts_path))
            else:
                genesis_loaded = False

            if genesis_loaded:
                log.info(
                    "Loaded from genesis: %s gateways, %s users, %s AMM pools",
                    len(app.state.workload.gateways),
                    len(app.state.workload.users),
                    len(app.state.workload._amm_pool_registry),
                )
            else:
                gw, u = cfg["gateways"], cfg["users"]
                log.info("No persisted/genesis state. Initializing participants (gateways=%s, users=%s)...", gw, u)
                init_result = await app.state.workload.init_participants(gateway_cfg=gw, user_cfg=u)
                app.state.workload.update_txn_context()
                log.info(
                    "Accounts initialized: %s gateways, %s users.",
                    len(init_result["gateways"]),
                    len(init_result["users"]),
                )

        init_ledger = await app.state.workload._current_ledger_index()
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
        workload_ready_msg = "Workload initialization complete"
        log.info(f"Network initialization complete at ledger {init_ledger}. Ready to accept requests!")
        setup_complete(details={"message": workload_ready_msg})
        await asyncio.sleep(5)
        await start_workload()
        try:
            yield
        finally:
            log.info("Shutting down...")
            stop.set()
            app.state.ws_stop_event.set()

            await asyncio.sleep(5)
            log.info("Exiting TaskGroup (will cancel any remaining tasks)...")

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
        "tagsSorter": "alpha",  # See what "order" does...
        "operationsSorter": "alpha",  # See what "method" does...
    },
)

r_accounts = APIRouter(prefix="/accounts", tags=["Accounts"])
r_pay = APIRouter(prefix="/payment", tags=["Payments"])
r_transaction = APIRouter(tags=["Transactions"])
r_state = APIRouter(prefix="/state", tags=["State"])
r_workload = APIRouter(prefix="/workload", tags=["Workload"])
r_dex = APIRouter(prefix="/dex", tags=["DEX"])


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
    return {"status": "ok"}  # Not the most thorough of healthchecks...


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

    source_wallet = w.wallets.get(req.source)
    if not source_wallet:
        raise HTTPException(status_code=404, detail=f"Source wallet not found: {req.source}")

    payment = Payment(
        account=req.source,
        destination=req.destination,
        amount=req.amount,
    )

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


@r_transaction.post("/payment")
async def create_payment():
    """Create and submit a Payment transaction."""
    return await create("Payment")


@r_transaction.post("/trustset")
async def create_trustset():
    """Create and submit a TrustSet transaction."""
    return await create("TrustSet")


@r_transaction.post("/accountset")
async def create_accountset():
    """Create and submit an AccountSet transaction."""
    return await create("AccountSet")


@r_transaction.post("/ammcreate")
async def create_ammcreate():
    """Create and submit an AMMCreate transaction."""
    return await create("AMMCreate")


@r_transaction.post("/nftokenmint")
async def create_nftokenmint():
    """Create and submit an NFTokenMint transaction."""
    return await create("NFTokenMint")


@r_transaction.post("/mptokenissuancecreate")
async def create_mptokenissuancecreate():
    """Create and submit an MPTokenIssuanceCreate transaction."""
    return await create("MPTokenIssuanceCreate")


@r_transaction.post("/mptokenissuanceset")
async def create_mptokenissuanceset():
    """Create and submit an MPTokenIssuanceSet transaction."""
    return await create("MPTokenIssuanceSet")


@r_transaction.post("/mptokenauthorize")
async def create_mptokenauthorize():
    """Create and submit an MPTokenAuthorize transaction."""
    return await create("MPTokenAuthorize")


@r_transaction.post("/mptokenissuancedestroy")
async def create_mptokenissuancedestroy():
    """Create and submit an MPTokenIssuanceDestroy transaction."""
    return await create("MPTokenIssuanceDestroy")


@r_transaction.post("/batch")
async def create_batch():
    """Create and submit a Batch transaction."""
    return await create("Batch")


@app.post("/debug/fund")
async def debug_fund(dest: str):
    """Manually fund an address from the workload's configured `funding_account` and return the unvalidated result."""
    w: Workload = app.state.workload
    log.debug(
        "funding_wallet %s",
        w.funding_wallet.address,
    )
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


@r_state.get("/dashboard", response_class=HTMLResponse)
async def state_dashboard():
    """HTML dashboard with live stats, explorer embed, and WS terminal."""
    hostname = RPC.split("//")[1].split(":")[0] if "//" in RPC else RPC.split(":")[0]

    # Build node list from compose config for the WS terminal dropdown
    from generate_ledger.config import ComposeConfig
    cc = ComposeConfig()
    nodes = []
    for i in range(cc.num_validators):
        name = f"{cc.validator_name}{i}"
        ws = cc.ws_port + i + cc.num_hubs
        nodes.append({"name": name, "ws": ws})
    for i in range(cc.num_hubs):
        name = cc.hub_name if cc.num_hubs == 1 else f"{cc.hub_name}{i}"
        ws = cc.ws_port + i
        nodes.append({"name": name, "ws": ws})
    import json as _json
    nodes_json = _json.dumps(nodes)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Workload Dashboard</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0d1117;
                color: #c9d1d9;
                margin: 0;
                padding: 20px;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            h1 {{ color: #58a6ff; margin-bottom: 10px; }}
            h2 {{ color: #c9d1d9; margin: 0 0 12px 0; font-size: 16px; }}
            .subtitle {{ color: #8b949e; margin-bottom: 20px; }}
            .subtitle a {{ color: #58a6ff; text-decoration: none; }}
            .subtitle a:hover {{ text-decoration: underline; }}

            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 16px;
                margin-bottom: 20px;
            }}
            .stat-card {{
                background: #161b22; border: 1px solid #30363d;
                border-radius: 6px; padding: 16px;
            }}
            .stat-label {{
                color: #8b949e; font-size: 11px;
                text-transform: uppercase; margin-bottom: 6px;
            }}
            .stat-value {{
                font-size: 28px; font-weight: bold; margin-bottom: 2px;
            }}
            .stat-value.success {{ color: #3fb950; }}
            .stat-value.error {{ color: #f85149; }}
            .stat-value.warning {{ color: #d29922; }}
            .stat-value.info {{ color: #58a6ff; }}
            .stat-percentage {{ color: #8b949e; font-size: 13px; }}
            .progress-bar {{
                background: #21262d; border-radius: 6px;
                height: 6px; overflow: hidden; margin-top: 6px;
            }}
            .progress-fill {{ height: 100%; transition: width 0.3s ease; }}
            .progress-fill.success {{ background: #3fb950; }}
            .progress-fill.error {{ background: #f85149; }}
            .progress-fill.info {{ background: #58a6ff; }}

            .panel {{
                background: #161b22; border: 1px solid #30363d;
                border-radius: 6px; padding: 20px; margin-bottom: 20px;
            }}
            .failures-table {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #21262d; }}
            th {{ color: #8b949e; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
            tr:last-child td {{ border-bottom: none; }}
            .badge {{
                display: inline-block; padding: 2px 8px; border-radius: 4px;
                font-size: 12px; font-weight: 600;
            }}
            .badge.success {{ background: #3fb9501a; color: #3fb950; }}
            .badge.warning {{ background: #d299221a; color: #d29922; }}
            .badge.error {{ background: #f851491a; color: #f85149; }}
            .badge.info {{ background: #58a6ff1a; color: #58a6ff; }}

            .controls {{
                margin-bottom: 20px; display: flex;
                gap: 10px; align-items: center; flex-wrap: wrap;
            }}
            .btn {{
                padding: 8px 16px; border: none; border-radius: 6px;
                font-size: 13px; font-weight: 600; cursor: pointer;
                transition: opacity 0.2s;
            }}
            .btn:hover {{ opacity: 0.8; }}
            .btn-start {{ background: #3fb950; color: white; }}
            .btn-stop {{ background: #f85149; color: white; }}
            .fill-control {{
                display: flex; align-items: center; gap: 8px;
                background: #161b22; border: 1px solid #30363d;
                border-radius: 6px; padding: 6px 14px;
            }}
            .fill-control label {{
                color: #8b949e; font-size: 12px; font-weight: 600;
                text-transform: uppercase; white-space: nowrap;
            }}
            .fill-control input[type=range] {{ width: 140px; accent-color: #58a6ff; }}
            .fill-control .fill-value {{
                color: #58a6ff; font-weight: 700; font-size: 15px;
                min-width: 36px; text-align: right;
            }}
            .link-btn {{
                padding: 8px 16px; border: 1px solid #30363d; border-radius: 6px;
                font-size: 13px; font-weight: 600; cursor: pointer;
                background: #161b22; color: #58a6ff; text-decoration: none;
                transition: border-color 0.2s;
            }}
            .link-btn:hover {{ border-color: #58a6ff; }}

            .explorer-viewport {{
                position: relative; width: 100%; height: 500px;
                overflow: hidden; border-radius: 4px;
            }}
            .explorer-viewport iframe {{
                position: absolute; top: -60px; left: 0;
                width: 100%; height: calc(100% + 60px); border: none;
            }}

            /* WS Terminal */
            .ws-terminal-bar {{
                display: flex; gap: 8px; align-items: center;
                margin-bottom: 10px; flex-wrap: wrap;
            }}
            .ws-terminal-bar select, .ws-terminal-bar button {{
                background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
                border-radius: 4px; padding: 6px 10px; font-size: 13px;
            }}
            .ws-terminal-bar select {{ min-width: 120px; }}
            .ws-terminal-bar button {{ cursor: pointer; font-weight: 600; }}
            .ws-terminal-bar button:hover {{ border-color: #58a6ff; }}
            .ws-terminal-bar button.active {{ background: #3fb950; color: #0d1117; border-color: #3fb950; }}
            .stream-filters {{
                display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
            }}
            .stream-filters label {{
                display: flex; align-items: center; gap: 4px;
                font-size: 12px; color: #8b949e; cursor: pointer;
                background: #0d1117; border: 1px solid #30363d;
                border-radius: 4px; padding: 4px 8px;
            }}
            .stream-filters label.checked {{
                border-color: #58a6ff; color: #58a6ff;
            }}
            .stream-filters input {{ display: none; }}
            #ws-output {{
                background: #010409; border: 1px solid #21262d; border-radius: 4px;
                height: 400px; overflow-y: auto; padding: 10px;
                font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
                font-size: 12px; line-height: 1.5;
                scroll-behavior: smooth;
            }}
            .ws-line {{ margin: 0; white-space: pre-wrap; word-break: break-all; }}
            .ws-line.ledger {{ color: #58a6ff; }}
            .ws-line.transaction {{ color: #3fb950; }}
            .ws-line.validation {{ color: #d29922; }}
            .ws-line.server {{ color: #8b949e; }}
            .ws-line.consensus {{ color: #bc8cff; }}
            .ws-line.peer {{ color: #f0883e; }}
            .ws-line.error {{ color: #f85149; }}
            .ws-line.info {{ color: #8b949e; font-style: italic; }}
            .ws-line.txn-Payment {{ color: #3fb950; }}
            .ws-line.txn-OfferCreate {{ color: #58a6ff; }}
            .ws-line.txn-OfferCancel {{ color: #6cb6ff; }}
            .ws-line.txn-TrustSet {{ color: #d29922; }}
            .ws-line.txn-AccountSet {{ color: #8b949e; }}
            .ws-line.txn-NFTokenMint {{ color: #bc8cff; }}
            .ws-line.txn-NFTokenBurn {{ color: #986ee2; }}
            .ws-line.txn-NFTokenCreateOffer {{ color: #a371f7; }}
            .ws-line.txn-NFTokenCancelOffer {{ color: #8957e5; }}
            .ws-line.txn-NFTokenAcceptOffer {{ color: #c297ff; }}
            .ws-line.txn-TicketCreate {{ color: #7ee787; }}
            .ws-line.txn-MPTokenIssuanceCreate {{ color: #f0883e; }}
            .ws-line.txn-MPTokenIssuanceSet {{ color: #d4762c; }}
            .ws-line.txn-MPTokenAuthorize {{ color: #e09b4f; }}
            .ws-line.txn-MPTokenIssuanceDestroy {{ color: #c45e1a; }}
            .ws-line.txn-AMMCreate {{ color: #f778ba; }}
            .ws-line.txn-Batch {{ color: #79c0ff; }}
            .stream-filters label.txn-Payment {{ border-color: #3fb95066; }}
            .stream-filters label.txn-Payment.checked {{ border-color: #3fb950; color: #3fb950; }}
            .stream-filters label.txn-OfferCreate {{ border-color: #58a6ff66; }}
            .stream-filters label.txn-OfferCreate.checked {{ border-color: #58a6ff; color: #58a6ff; }}
            .stream-filters label.txn-OfferCancel {{ border-color: #6cb6ff66; }}
            .stream-filters label.txn-OfferCancel.checked {{ border-color: #6cb6ff; color: #6cb6ff; }}
            .stream-filters label.txn-TrustSet {{ border-color: #d2992266; }}
            .stream-filters label.txn-TrustSet.checked {{ border-color: #d29922; color: #d29922; }}
            .stream-filters label.txn-AccountSet {{ border-color: #8b949e66; }}
            .stream-filters label.txn-AccountSet.checked {{ border-color: #8b949e; color: #8b949e; }}
            .stream-filters label.txn-NFTokenMint {{ border-color: #bc8cff66; }}
            .stream-filters label.txn-NFTokenMint.checked {{ border-color: #bc8cff; color: #bc8cff; }}
            .stream-filters label.txn-NFTokenBurn {{ border-color: #986ee266; }}
            .stream-filters label.txn-NFTokenBurn.checked {{ border-color: #986ee2; color: #986ee2; }}
            .stream-filters label.txn-NFTokenCreateOffer {{ border-color: #a371f766; }}
            .stream-filters label.txn-NFTokenCreateOffer.checked {{ border-color: #a371f7; color: #a371f7; }}
            .stream-filters label.txn-NFTokenCancelOffer {{ border-color: #8957e566; }}
            .stream-filters label.txn-NFTokenCancelOffer.checked {{ border-color: #8957e5; color: #8957e5; }}
            .stream-filters label.txn-NFTokenAcceptOffer {{ border-color: #c297ff66; }}
            .stream-filters label.txn-NFTokenAcceptOffer.checked {{ border-color: #c297ff; color: #c297ff; }}
            .stream-filters label.txn-TicketCreate {{ border-color: #7ee78766; }}
            .stream-filters label.txn-TicketCreate.checked {{ border-color: #7ee787; color: #7ee787; }}
            .stream-filters label.txn-MPTokenIssuanceCreate {{ border-color: #f0883e66; }}
            .stream-filters label.txn-MPTokenIssuanceCreate.checked {{ border-color: #f0883e; color: #f0883e; }}
            .stream-filters label.txn-MPTokenIssuanceSet {{ border-color: #d4762c66; }}
            .stream-filters label.txn-MPTokenIssuanceSet.checked {{ border-color: #d4762c; color: #d4762c; }}
            .stream-filters label.txn-MPTokenAuthorize {{ border-color: #e09b4f66; }}
            .stream-filters label.txn-MPTokenAuthorize.checked {{ border-color: #e09b4f; color: #e09b4f; }}
            .stream-filters label.txn-MPTokenIssuanceDestroy {{ border-color: #c45e1a66; }}
            .stream-filters label.txn-MPTokenIssuanceDestroy.checked {{ border-color: #c45e1a; color: #c45e1a; }}
            .stream-filters label.txn-AMMCreate {{ border-color: #f778ba66; }}
            .stream-filters label.txn-AMMCreate.checked {{ border-color: #f778ba; color: #f778ba; }}
            .stream-filters label.txn-Batch {{ border-color: #79c0ff66; }}
            .stream-filters label.txn-Batch.checked {{ border-color: #79c0ff; color: #79c0ff; }}
            .ws-status {{ font-size: 12px; margin-left: auto; }}
            .ws-status.connected {{ color: #3fb950; }}
            .ws-status.disconnected {{ color: #f85149; }}
            .msg-count {{ color: #8b949e; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Workload Dashboard</h1>
            <div class="subtitle" id="subtitle">Loading...</div>

            <div class="controls">
                <button class="btn btn-start" onclick="fetch('/workload/start', {{method:'POST'}})">Start</button>
                <button class="btn btn-stop" onclick="fetch('/workload/stop', {{method:'POST'}})">Stop</button>
                <div class="fill-control">
                    <label>Fill</label>
                    <input type="range" id="fill-slider" min="0" max="100" value="50"
                           oninput="document.getElementById('fill-val').textContent=this.value+'%'"
                           onchange="fetch('/workload/fill-fraction',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{fill_fraction:this.value/100}})}})">
                    <span class="fill-value" id="fill-val">50%</span>
                </div>
                <a class="link-btn" href="https://custom.xrpl.org/localhost:6006" target="_blank">XRPL Explorer</a>
                <a class="link-btn" href="/docs" target="_blank">API Docs</a>
            </div>

            <!-- Stats cards (updated via JS) -->
            <div class="stats-grid" id="fee-stats"></div>
            <div class="stats-grid" id="txn-stats"></div>

            <!-- Explorer embed -->
            <div class="panel">
                <h2>Ledger Stream</h2>
                <div class="explorer-viewport">
                    <iframe src="https://custom.xrpl.org/localhost:6006" id="explorer-frame"></iframe>
                </div>
            </div>

            <!-- WS Terminal -->
            <div class="panel">
                <h2>Node WebSocket</h2>
                <div class="ws-terminal-bar">
                    <select id="ws-node"></select>
                    <button id="ws-connect-btn" onclick="toggleWs()">Connect</button>
                    <div class="stream-filters" id="stream-filters"></div>
                    <span class="msg-count" id="ws-msg-count"></span>
                    <span class="ws-status disconnected" id="ws-status">disconnected</span>
                </div>
                <div class="ws-terminal-bar" style="margin-top:0">
                    <span style="color:#8b949e;font-size:11px;text-transform:uppercase;font-weight:600">Txn types</span>
                    <div class="stream-filters" id="txn-type-filters"></div>
                </div>
                <div id="ws-output"></div>
            </div>

            <!-- Submission results + failures (updated via JS) -->
            <div id="tables-container"></div>
        </div>

        <script>
        // --- Nodes ---
        const NODES = {nodes_json};
        const STREAMS = [
            {{name:'ledger', label:'Ledger', on:true}},
            {{name:'transactions', label:'Transactions', on:true}},
            {{name:'validations', label:'Validations', on:false}},
            {{name:'consensus', label:'Consensus', on:false}},
            {{name:'peer_status', label:'Peers', on:false}},
            {{name:'server', label:'Server', on:false}},
        ];
        let ws = null;
        let msgCount = 0;
        let activeStreams = new Set(STREAMS.filter(s=>s.on).map(s=>s.name));
        const MAX_LINES = 500;

        // Populate node dropdown
        const nodeSelect = document.getElementById('ws-node');
        NODES.forEach(n => {{
            const opt = document.createElement('option');
            opt.value = n.ws;
            opt.textContent = n.name + ' (:' + n.ws + ')';
            nodeSelect.appendChild(opt);
        }});

        // Populate stream filter checkboxes
        const filtersEl = document.getElementById('stream-filters');
        STREAMS.forEach(s => {{
            const lbl = document.createElement('label');
            lbl.className = s.on ? 'checked' : '';
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = s.on;
            cb.dataset.stream = s.name;
            cb.onchange = function() {{
                if (this.checked) activeStreams.add(s.name);
                else activeStreams.delete(s.name);
                lbl.className = this.checked ? 'checked' : '';
                if (ws && ws.readyState === WebSocket.OPEN) resubscribe();
            }};
            lbl.appendChild(cb);
            lbl.appendChild(document.createTextNode(s.label));
            filtersEl.appendChild(lbl);
        }});

        // Transaction type filters
        const TXN_TYPES = [
            'Payment','OfferCreate','OfferCancel','TrustSet','AccountSet',
            'NFTokenMint','NFTokenBurn','NFTokenCreateOffer','NFTokenCancelOffer','NFTokenAcceptOffer',
            'TicketCreate',
            'MPTokenIssuanceCreate','MPTokenIssuanceSet','MPTokenAuthorize','MPTokenIssuanceDestroy',
            'AMMCreate','Batch',
        ];
        let activeTxnTypes = new Set(TXN_TYPES); // all on by default
        const txnFiltersEl = document.getElementById('txn-type-filters');
        TXN_TYPES.forEach(tt => {{
            const lbl = document.createElement('label');
            lbl.className = 'checked txn-' + tt;
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = true;
            cb.onchange = function() {{
                if (this.checked) activeTxnTypes.add(tt);
                else activeTxnTypes.delete(tt);
                lbl.className = (this.checked ? 'checked ' : '') + 'txn-' + tt;
            }};
            lbl.appendChild(cb);
            lbl.appendChild(document.createTextNode(tt));
            txnFiltersEl.appendChild(lbl);
        }});

        function wsLog(text, cls) {{
            const out = document.getElementById('ws-output');
            const line = document.createElement('div');
            line.className = 'ws-line ' + (cls || '');
            const ts = new Date().toLocaleTimeString('en-US', {{hour12:false}});
            line.textContent = ts + '  ' + text;
            out.appendChild(line);
            while (out.children.length > MAX_LINES) out.removeChild(out.firstChild);
            out.scrollTop = out.scrollHeight;
        }}

        // Returns [text, cssClass, txnType|null]
        function formatMsg(data) {{
            const t = data.type || '';
            if (t === 'ledgerClosed') {{
                return [
                    `LEDGER #${{data.ledger_index}}  txns=${{data.txn_count}}  close=${{data.ledger_time}}`,
                    'ledger', null
                ];
            }}
            if (t === 'transaction') {{
                const tx = data.transaction || {{}};
                const tt = tx.TransactionType || '?';
                const v = data.validated ? 'validated' : 'proposed';
                const r = data.engine_result || data.meta?.TransactionResult || '';
                return [
                    `${{tt}}  ${{tx.Account?.slice(0,12)}}..  ${{r}}  [${{v}}]  ${{tx.hash?.slice(0,16)}}..`,
                    'txn-' + tt, tt
                ];
            }}
            if (t === 'validationReceived') {{
                return [
                    `VALIDATION  ledger=${{data.ledger_index}}  key=${{data.validation_public_key?.slice(0,16)}}..`,
                    'validation', null
                ];
            }}
            if (t === 'serverStatus') {{
                return [
                    `SERVER  load=${{data.load_factor}}  state=${{data.server_status}}`,
                    'server', null
                ];
            }}
            if (t === 'consensusPhase') {{
                return [`CONSENSUS  phase=${{data.consensus}}`, 'consensus', null];
            }}
            if (t === 'peerStatusChange') {{
                return [`PEER  ${{data.action}}  ${{data.address || ''}}`, 'peer', null];
            }}
            return [JSON.stringify(data).slice(0, 200), '', null];
        }}

        function resubscribe() {{
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            // Unsubscribe all, then resubscribe active
            ws.send(JSON.stringify({{command:'unsubscribe', streams:STREAMS.map(s=>s.name)}}));
            const streams = Array.from(activeStreams);
            if (streams.length > 0) {{
                ws.send(JSON.stringify({{command:'subscribe', streams}}));
                wsLog('Subscribed: ' + streams.join(', '), 'info');
            }}
        }}

        function toggleWs() {{
            const btn = document.getElementById('ws-connect-btn');
            const statusEl = document.getElementById('ws-status');
            if (ws && ws.readyState <= WebSocket.OPEN) {{
                ws.close();
                return;
            }}
            const port = nodeSelect.value;
            const url = 'ws://{hostname}:' + port;
            wsLog('Connecting to ' + url + '...', 'info');
            ws = new WebSocket(url);
            ws.onopen = () => {{
                statusEl.textContent = 'connected';
                statusEl.className = 'ws-status connected';
                btn.textContent = 'Disconnect';
                btn.className = 'active';
                const streams = Array.from(activeStreams);
                ws.send(JSON.stringify({{command:'subscribe', streams}}));
                wsLog('Connected. Subscribed: ' + streams.join(', '), 'info');
            }};
            ws.onmessage = (ev) => {{
                const data = JSON.parse(ev.data);
                if (data.type === 'response') return; // subscribe ack
                const [text, cls, txnType] = formatMsg(data);
                // Filter: if it's a transaction, check txn type filter
                if (txnType && !activeTxnTypes.has(txnType)) return;
                wsLog(text, cls);
                msgCount++;
                document.getElementById('ws-msg-count').textContent = msgCount + ' msgs';
            }};
            ws.onclose = () => {{
                statusEl.textContent = 'disconnected';
                statusEl.className = 'ws-status disconnected';
                btn.textContent = 'Connect';
                btn.className = '';
                wsLog('Disconnected', 'info');
            }};
            ws.onerror = () => wsLog('WebSocket error', 'error');
        }}

        // --- Stats polling (no page reload, keeps WS alive) ---
        function fmt(n) {{ return n == null ? '—' : Number(n).toLocaleString(); }}
        function pct(a, b) {{ return b > 0 ? (a/b*100).toFixed(1) : '0.0'; }}

        function statCard(label, value, cls, extra, barPct) {{
            let html = '<div class="stat-card"><div class="stat-label">' + label + '</div>';
            html += '<div class="stat-value ' + (cls||'') + '">' + value + '</div>';
            if (extra) html += '<div class="stat-percentage">' + extra + '</div>';
            if (barPct != null) {{
                html += '<div class="progress-bar"><div class="progress-fill ' + (cls||'') + '" style="width:' + barPct + '%"></div></div>';
            }}
            return html + '</div>';
        }}

        async function refreshStats() {{
            try {{
                const [statsRes, feeRes, ffRes, failedRes] = await Promise.all([
                    fetch('/state/summary').then(r=>r.json()),
                    fetch('/state/fees').then(r=>r.json()),
                    fetch('/workload/fill-fraction').then(r=>r.json()),
                    fetch('/state/failed').then(r=>r.json()),
                ]);
                const s = statsRes;
                const f = feeRes;
                const bs = s.by_state || {{}};
                const total = s.total_tracked || 0;
                const validated = bs.VALIDATED || 0;
                const rejected = bs.REJECTED || 0;
                const submitted = bs.SUBMITTED || 0;
                const created = bs.CREATED || 0;
                const retryable = bs.RETRYABLE || 0;
                const expired = bs.EXPIRED || 0;

                // Subtitle
                document.getElementById('subtitle').innerHTML =
                    'Live monitoring &bull; Ledger ' + f.ledger_current_index +
                    ' @ {hostname} &bull; <a href="https://custom.xrpl.org/localhost:6006" target="_blank">Explorer</a>';

                // Fill slider (don't overwrite while user is dragging)
                const slider = document.getElementById('fill-slider');
                if (document.activeElement !== slider) {{
                    slider.value = Math.round(ffRes.fill_fraction * 100);
                    document.getElementById('fill-val').textContent = slider.value + '%';
                }}

                // Fee stats
                const feeWarn = f.minimum_fee > f.base_fee ? 'warning' : 'success';
                const qPct = f.max_queue_size > 0 ? (f.current_queue_size/f.max_queue_size*100) : 0;
                const lPct = f.expected_ledger_size > 0 ? (f.current_ledger_size/f.expected_ledger_size*100) : 0;
                document.getElementById('fee-stats').innerHTML =
                    statCard('Fee (min/open/base)', f.minimum_fee+'/'+f.open_ledger_fee+'/'+f.base_fee, feeWarn, 'drops') +
                    statCard('Queue Utilization', f.current_queue_size+'/'+f.max_queue_size, 'info', qPct.toFixed(1)+'%', qPct) +
                    statCard('Ledger Utilization', f.current_ledger_size+'/'+f.expected_ledger_size, 'info', lPct.toFixed(1)+'%', lPct);

                // Txn stats
                document.getElementById('txn-stats').innerHTML =
                    statCard('Total Transactions', fmt(total), 'info') +
                    statCard('Validated', fmt(validated), 'success', pct(validated,total)+'%', pct(validated,total)) +
                    statCard('Rejected', fmt(rejected), 'error', pct(rejected,total)+'%', pct(rejected,total)) +
                    statCard('In-Flight', fmt(submitted+created), 'warning', 'Submitted: '+submitted+' | Created: '+created) +
                    statCard('Retryable', fmt(retryable), 'warning', 'terPRE_SEQ waiting') +
                    statCard('Expired', fmt(expired), '');

                // Tables
                const sr = s.submission_results || {{}};
                const sorted_sr = Object.entries(sr).sort((a,b) => b[1]-a[1]);
                let tablesHtml = '';
                if (sorted_sr.length) {{
                    tablesHtml += '<div class="failures-table"><h2>Submission Results</h2><table><thead><tr><th>Engine Result</th><th>Count</th></tr></thead><tbody>';
                    sorted_sr.forEach(([r,c]) => {{
                        let cls = 'info';
                        if (r === 'tesSUCCESS') cls = 'success';
                        else if (r.startsWith('ter')) cls = 'warning';
                        else if (/^(tel|tec|tem|tef)/.test(r)) cls = 'error';
                        tablesHtml += '<tr><td><span class="badge '+cls+'">'+r+'</span></td><td>'+fmt(c)+'</td></tr>';
                    }});
                    tablesHtml += '</tbody></table></div>';
                }}
                // Top failures
                const INTERNAL = new Set(['CASCADE_EXPIRED','unknown','']);
                const failMap = {{}};
                (failedRes.failed||[]).forEach(f => {{
                    const r = f.engine_result_first || 'unknown';
                    if (!INTERNAL.has(r) && !r.startsWith('tes')) failMap[r] = (failMap[r]||0) + 1;
                }});
                const topFail = Object.entries(failMap).sort((a,b)=>b[1]-a[1]).slice(0,10);
                if (topFail.length) {{
                    tablesHtml += '<div class="failures-table"><h2>Top Failures</h2><table><thead><tr><th>Error Code</th><th>Count</th></tr></thead><tbody>';
                    topFail.forEach(([r,c]) => {{
                        tablesHtml += '<tr><td><span class="badge error">'+r+'</span></td><td>'+fmt(c)+'</td></tr>';
                    }});
                    tablesHtml += '</tbody></table></div>';
                }}
                document.getElementById('tables-container').innerHTML = tablesHtml;

            }} catch(e) {{
                console.error('Stats refresh error:', e);
            }}
        }}

        // Initial load + periodic refresh
        refreshStats();
        setInterval(refreshStats, 3000);

        // Try to hide explorer chrome
        document.getElementById('explorer-frame').addEventListener('load', function() {{
            try {{
                const doc = this.contentDocument || this.contentWindow.document;
                const s = doc.createElement('style');
                s.textContent = 'body>*:not(.ledger-list){{display:none!important}}.ledger-list{{margin:0!important;padding:10px!important}}nav,header,footer,.navbar,.header,.sidebar{{display:none!important}}';
                doc.head.appendChild(s);
            }} catch(e) {{}}
        }});
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


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


@r_state.get("/fees")
async def state_fees():
    """Get current fee escalation state from rippled."""
    wl = app.state.workload
    fee_info = await wl.get_fee_info()
    return {
        "expected_ledger_size": fee_info.expected_ledger_size,
        "current_ledger_size": fee_info.current_ledger_size,
        "current_queue_size": fee_info.current_queue_size,
        "max_queue_size": fee_info.max_queue_size,
        "base_fee": fee_info.base_fee,
        "minimum_fee": fee_info.minimum_fee,
        "median_fee": fee_info.median_fee,
        "open_ledger_fee": fee_info.open_ledger_fee,
        "ledger_current_index": fee_info.ledger_current_index,
        "queue_utilization": f"{fee_info.current_queue_size}/{fee_info.max_queue_size}",
        "ledger_utilization": f"{fee_info.current_ledger_size}/{fee_info.expected_ledger_size}",
    }


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


@r_state.get("/heartbeat")
async def state_heartbeat():
    """
    Return heartbeat status - our canary for ledger health.

    We should see exactly ONE heartbeat transaction per ledger.
    Missing heartbeats indicate network issues, WS disconnection, or other problems.

    Returns:
        - last_heartbeat_ledger: Most recent ledger where heartbeat was attempted
        - total_heartbeats: Total number of heartbeats submitted
        - missed_heartbeats: Ledger indices where heartbeat failed
        - missed_count: Total number of missed heartbeats
        - recent_heartbeats: Last 20 heartbeat attempts with status
    """
    return app.state.workload.snapshot_heartbeat()


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
        issued_currencies = [curr.currency for curr in wl._currencies if curr.issuer == gateway.address]
        gateways.append(
            {
                "address": gateway.address,
                "seed": gateway.seed,
                "currencies": issued_currencies,
            }
        )
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
    mptoken_ids = getattr(wl, "_mptoken_issuance_ids", [])
    return {
        "count": len(mptoken_ids),
        "mptoken_issuance_ids": mptoken_ids,
        "note": "MPToken IDs are tracked automatically when MPTokenIssuanceCreate transactions validate",
    }


@r_state.get("/finality")
async def check_finality():
    """Manually trigger finality check for all pending submitted transactions."""
    wl = app.state.workload
    results = []

    for p in wl.find_by_state(C.TxState.SUBMITTED):
        try:
            state = await wl.check_finality(p)
            results.append(
                {
                    "tx_hash": p.tx_hash,
                    "state": state.name,
                    "ledger_index": p.validated_ledger,  # Get from PendingTx object
                }
            )
        except Exception as e:
            results.append(
                {
                    "tx_hash": p.tx_hash,
                    "error": str(e),
                }
            )

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
    """Continuously submit XRP payments, respecting 1 pending txn per account.

    Key constraint: Only ONE transaction per account can be in-flight at a time.
    This prevents sequence number conflicts entirely - no resyncs needed.

    We can still submit many transactions in PARALLEL as long as each is from
    a different account.

    Uses XRP-only payments for simplicity and predictable base fees.
    """
    from random import random, sample

    from xrpl.models.transactions import Payment

    global workload_stats
    wl = app.state.workload

    log.debug("🚀 Continuous workload started (XRP payments only)")
    workload_stats["started_at"] = perf_counter()

    try:
        while not workload_stop_event.is_set():
            ledger_size = await wl._expected_ledger_size()

            if random() < 0.50:
                funding_pending = wl.get_pending_txn_counts_by_account().get(wl.funding_wallet.address, 0)
                if funding_pending == 0:
                    try:
                        default_balance = wl.config["users"]["default_balance"]
                        large_balance = str(int(default_balance) * 10)
                        result = await wl.create_account(initial_xrp_drops=large_balance)
                        workload_stats["submitted"] += 1
                        log.debug(f"✓ New account created: {result['address']}...")
                    except Exception as e:
                        log.error(f"Failed to create new account: {e}")
                        workload_stats["failed"] += 1

            pending_counts = wl.get_pending_txn_counts_by_account()

            account_slots = {
                addr: wl.max_pending_per_account - pending_counts.get(addr, 0) for addr in wl.wallets.keys()
            }
            available_accounts = [addr for addr, slots in account_slots.items() if slots > 0]
            total_available_slots = sum(slots for slots in account_slots.values() if slots > 0)

            batch_size = min(total_available_slots, wl.target_txns_per_ledger)

            if batch_size == 0:
                log.debug("No available slots (all accounts at max pending), waiting...")
                await asyncio.sleep(0.5)  # TODO: Remove time
                continue
            current_ledger = await wl._current_ledger_index()
            log.info(
                f"📊 Building batch @ ledger {current_ledger}: {batch_size} txns ({len(available_accounts)} accounts, {total_available_slots} slots, target_size={ledger_size})"
            )

            try:
                from workload.txn_factory.builder import generate_txn

                pending_txns = []
                txns_built = 0
                max_retries = batch_size * 2  # Avoid infinite loop
                retries = 0

                while txns_built < batch_size and retries < max_retries and not workload_stop_event.is_set():
                    retries += 1
                    try:
                        wl.ctx.wallets = list(wl.wallets.values())

                        txn = await generate_txn(wl.ctx)

                        src_addr = txn.account

                        current_pending = wl.get_pending_txn_counts_by_account().get(src_addr, 0)
                        if current_pending >= wl.max_pending_per_account:
                            continue  # Try another transaction

                        pending = await wl.build_sign_and_track(txn, wl.wallets[src_addr])
                        pending_txns.append(pending)
                        txns_built += 1

                    except Exception as e:
                        log.error(f"Failed to build transaction: {e}")
                        workload_stats["failed"] += 1

                if not pending_txns:
                    log.warning("No transactions built this batch")
                    await asyncio.sleep(0.5)  # TODO: Remove time - we tick on LEDGERS not time!
                    continue

                log.info(f"📤 Submitting {len(pending_txns)} transactions in parallel...")

                async with asyncio.TaskGroup() as tg:
                    submit_tasks = [tg.create_task(wl.submit_pending(p)) for p in pending_txns]

                for task in submit_tasks:
                    try:
                        result = task.result()
                        workload_stats["submitted"] += 1
                        er = result.get("engine_result") if result else None
                        if er and er.startswith(("ter", "tem", "tef", "tel")):
                            workload_stats["failed"] += 1
                    except Exception as e:
                        log.error(f"Submit error: {e}")
                        workload_stats["failed"] += 1

            except* Exception as eg:
                for exc in eg.exceptions:
                    log.error(f"Batch error: {type(exc).__name__}: {exc}")
                workload_stats["failed"] += len(pending_txns) if pending_txns else 0

            next_ledger = current_ledger + 1
            while await wl._current_ledger_index() < next_ledger and not workload_stop_event.is_set():
                await asyncio.sleep(0.5)  # TODO: Remove time

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

    workload_stats = {"submitted": 0, "validated": 0, "failed": 0, "started_at": perf_counter()}

    log.info("Starting workload")
    workload_stop_event = asyncio.Event()
    workload_task = asyncio.create_task(continuous_workload())
    workload_running = True

    return {
        "status": "started",
        "message": "Continuous workload started - submitting random transactions at expected_ledger_size + 1 per ledger (max 200)",
    }


@r_workload.post("/stop")
async def stop_workload():
    """Stop continuous workload."""
    global workload_running, workload_stop_event, workload_task

    if not workload_running:
        raise HTTPException(status_code=400, detail="Workload not running")

    log.info("Stopping workload")
    workload_stop_event.set()
    await workload_task
    stop_ledger = await app.state.workload._current_ledger_index()
    log.info("Stopped workload at ledger %s", stop_ledger)
    workload_running = False

    return {"status": "stopped", "stats": workload_stats}


@r_workload.get("/status")
async def workload_status():
    """Get current workload status and statistics."""
    return {
        "running": workload_running,
        "stats": workload_stats,
        "uptime_seconds": perf_counter() - workload_stats["started_at"] if workload_stats["started_at"] else 0,
    }


@r_workload.get("/fill-fraction")
async def get_fill_fraction():
    """Get current ledger fill fraction for continuous workload.

    Returns the fraction (0.0 to 1.0) of expected_ledger_size used for batch sizing.
    Lower values = smoother distribution across ledgers, higher = more aggressive filling.
    """
    return {
        "fill_fraction": app.state.workload.ledger_fill_fraction,
        "description": "Fraction of ledger_size to fill per batch (0.0 to 1.0)",
        "recommendation": "0.3-0.4 = conservative/smooth, 0.5 = balanced, 0.7-0.8 = aggressive",
    }


class FillFractionReq(BaseModel):
    fill_fraction: float


@r_workload.post("/fill-fraction")
async def set_fill_fraction(req: FillFractionReq):
    """Set ledger fill fraction for continuous workload.

    Controls batch size as fraction of expected_ledger_size.
    - Lower (0.3-0.4): More conservative, smoother distribution, less throughput
    - Medium (0.5): Balanced approach
    - Higher (0.7-0.8): More aggressive, higher throughput, risk of gaps

    Takes effect immediately on next batch.
    """
    if not 0.0 < req.fill_fraction <= 1.0:
        raise HTTPException(status_code=400, detail="fill_fraction must be between 0.0 and 1.0")

    old_value = app.state.workload.ledger_fill_fraction
    app.state.workload.ledger_fill_fraction = req.fill_fraction

    log.info(f"ledger_fill_fraction changed: {old_value} -> {app.state.workload.ledger_fill_fraction}")

    return {
        "old_value": old_value,
        "new_value": app.state.workload.ledger_fill_fraction,
        "status": "updated",
        "note": "Change takes effect on next workload batch",
    }


class TargetTxnsReq(BaseModel):
    target_txns: int


@r_workload.get("/target-txns")
async def get_target_txns():
    """Get current target transactions per ledger for continuous workload.

    Returns the hard cap on transactions submitted per ledger.
    """
    return {
        "target_txns_per_ledger": app.state.workload.target_txns_per_ledger,
        "description": "Hard cap on transactions per ledger",
    }


@r_workload.post("/target-txns")
async def set_target_txns(req: TargetTxnsReq):
    """Set target transactions per ledger for continuous workload.

    Controls how many transactions to submit per ledger.
    - Lower (10-20): Very conservative, smooth, low throughput
    - Medium (30-50): Balanced approach
    - Higher (80-100): Aggressive, high throughput

    Takes effect immediately on next batch.
    """
    if req.target_txns < 1 or req.target_txns > 500:
        raise HTTPException(status_code=400, detail="target_txns must be between 1 and 500")

    old_value = app.state.workload.target_txns_per_ledger
    app.state.workload.target_txns_per_ledger = req.target_txns

    log.info(f"target_txns_per_ledger changed: {old_value} -> {app.state.workload.target_txns_per_ledger}")

    return {
        "old_value": old_value,
        "new_value": app.state.workload.target_txns_per_ledger,
        "status": "updated",
        "note": "Change takes effect on next workload batch",
    }


r_network = APIRouter(prefix="/network", tags=["Network"])


@r_network.post("/reset")
async def network_reset():
    """Reset the network: stop workload, regenerate testnet, restart containers, restart workload.

    This calls `gen auto` to regenerate the testnet, then `docker compose down/up`.
    Requires gen CLI and docker compose to be available on the host.
    """
    import shutil
    import subprocess

    # 1. Stop workload if running
    global workload_running, workload_stop_event, workload_task
    if workload_running and workload_stop_event:
        workload_stop_event.set()
        if workload_task:
            try:
                await workload_task
            except Exception:
                pass
        workload_running = False

    testnet_dir = os.getenv("TESTNET_DIR", str(Path(__file__).resolve().parents[3] / "prepare-workload" / "testnet"))
    gen_bin = os.getenv("GEN_BIN", "gen")

    steps = []

    # 2. Docker compose down
    try:
        r = subprocess.run(
            ["docker", "compose", "down"],
            cwd=testnet_dir, capture_output=True, text=True, timeout=30,
        )
        steps.append({"step": "docker compose down", "returncode": r.returncode, "stderr": r.stderr.strip()})
    except Exception as e:
        steps.append({"step": "docker compose down", "error": str(e)})

    # 3. Clean old artifacts
    for name in ["volumes", "ledger.json", "accounts.json", "docker-compose.yml"]:
        target = Path(testnet_dir) / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.is_file():
            target.unlink()
    # Clean state.db (check both workload cwd and parent)
    for db_path in [Path("state.db"), Path(__file__).resolve().parents[3] / "state.db"]:
        if db_path.is_file():
            db_path.unlink()
    steps.append({"step": "clean artifacts", "status": "ok"})

    # 4. Regenerate with gen auto
    try:
        r = subprocess.run(
            [gen_bin, "auto", "-o", testnet_dir, "-v", "5", "-n", "40",
             "-t", "0:1:USD:1000000000", "--amendment-majority-time", "15 minutes"],
            capture_output=True, text=True, timeout=60,
        )
        steps.append({"step": "gen auto", "returncode": r.returncode, "stdout": r.stdout.strip()[-200:]})
        if r.returncode != 0:
            steps.append({"step": "gen auto stderr", "stderr": r.stderr.strip()[-500:]})
    except Exception as e:
        steps.append({"step": "gen auto", "error": str(e)})

    # 5. Docker compose up
    try:
        r = subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=testnet_dir, capture_output=True, text=True, timeout=60,
        )
        steps.append({"step": "docker compose up", "returncode": r.returncode, "stderr": r.stderr.strip()[-200:]})
    except Exception as e:
        steps.append({"step": "docker compose up", "error": str(e)})

    return {
        "status": "reset complete — restart the workload process to reconnect",
        "steps": steps,
    }


@r_transaction.post("/ammdeposit")
async def create_ammdeposit():
    """Create and submit an AMMDeposit transaction."""
    return await create("AMMDeposit")


@r_transaction.post("/ammwithdraw")
async def create_ammwithdraw():
    """Create and submit an AMMWithdraw transaction."""
    return await create("AMMWithdraw")


@r_dex.get("/metrics")
async def dex_metrics():
    """Get DEX metrics including AMM pool states, trading activity counts."""
    return app.state.workload.snapshot_dex_metrics()


@r_dex.get("/pools")
async def dex_pools():
    """List all tracked AMM pools."""
    w: Workload = app.state.workload
    return {
        "total_pools": len(w._amm_pool_registry),
        "pools": w._amm_pool_registry,
    }


@r_dex.get("/pools/{index}")
async def dex_pool_detail(index: int):
    """Get detailed amm_info for a specific pool by index."""
    from xrpl.models.currencies import XRP as XRPCurrency
    from xrpl.models.requests import AMMInfo

    w: Workload = app.state.workload
    if index >= len(w._amm_pool_registry):
        raise HTTPException(status_code=404, detail=f"Pool index {index} not found")

    pool = w._amm_pool_registry[index]
    asset1, asset2 = pool["asset1"], pool["asset2"]

    if asset1.get("currency") == "XRP":
        a1 = XRPCurrency()
    else:
        from xrpl.models import IssuedCurrency

        a1 = IssuedCurrency(currency=asset1["currency"], issuer=asset1["issuer"])

    if asset2.get("currency") == "XRP":
        a2 = XRPCurrency()
    else:
        from xrpl.models import IssuedCurrency

        a2 = IssuedCurrency(currency=asset2["currency"], issuer=asset2["issuer"])

    resp = await w._rpc(AMMInfo(asset=a1, asset2=a2))
    return resp.result


@r_dex.post("/poll")
async def dex_poll_now():
    """Manually trigger a DEX metrics poll."""
    return await app.state.workload.poll_dex_metrics()


app.include_router(r_accounts)
app.include_router(r_pay)
app.include_router(r_pay, prefix="/pay", include_in_schema=False)  # alias /pay/ for convenience
app.include_router(r_transaction, prefix="/transaction")
app.include_router(r_transaction, prefix="/txn", include_in_schema=False)  # alias /txn/ because I'm sick of typing...
app.include_router(r_state)
app.include_router(r_workload)
app.include_router(r_dex)
app.include_router(r_network)
