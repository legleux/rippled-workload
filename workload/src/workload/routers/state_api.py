import logging

from fastapi import APIRouter, HTTPException, Request

import workload.constants as C
from workload.ws_processor import get_ws_counters

log = logging.getLogger("workload.app")

router = APIRouter(prefix="/state", tags=["State"])


@router.get("/summary")
async def state_summary(request: Request):
    return request.app.state.workload.snapshot_stats()


@router.get("/pending")
async def state_pending(request: Request):
    return {"pending": request.app.state.workload.snapshot_pending()}


@router.get("/failed")
async def state_failed(request: Request):
    return {"failed": request.app.state.workload.snapshot_failed()}


@router.get("/failed/{error_code}")
async def state_failed_by_code(error_code: str, request: Request):
    """Get failed transactions filtered by engine result code."""
    all_failed = request.app.state.workload.snapshot_failed()
    filtered = [
        f
        for f in all_failed
        if f.get("engine_result_final") == error_code or f.get("engine_result_first") == error_code
    ]
    return {"error_code": error_code, "count": len(filtered), "transactions": filtered}


@router.get("/expired")
async def state_expired(request: Request):
    """Get transactions that expired without validating."""
    wl = request.app.state.workload
    expired = [r for r in wl.snapshot_pending(open_only=False) if r["state"] == "EXPIRED"]
    return {"expired": expired}


@router.get("/type/{txn_type}")
async def state_type_json(txn_type: str, request: Request):
    """Get transactions filtered by transaction type."""
    wl = request.app.state.workload
    filtered = [r for r in wl.snapshot_pending(open_only=False) if r.get("transaction_type") == txn_type]
    return {"transaction_type": txn_type, "count": len(filtered), "transactions": filtered}


@router.get("/tx/{tx_hash}")
async def state_tx(tx_hash: str, request: Request):
    data = request.app.state.workload.snapshot_tx(tx_hash)
    if not data:
        raise HTTPException(404, "tx not tracked")
    return data


@router.get("/fees")
async def state_fees(request: Request):
    """Get current fee escalation state from rippled."""
    wl = request.app.state.workload
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
        "last_closed_txn_count": wl.last_closed_ledger_txn_count,
    }


@router.get("/accounts")
async def state_accounts(request: Request):
    wl = request.app.state.workload
    return {
        "count": len(wl.accounts),
        "addresses": list(wl.accounts.keys()),
    }


@router.get("/validations")
async def state_validations(request: Request, limit: int = 100):
    """
    Return recent validation records from the in-memory store.
    Parameters
    ----------
    limit:
        Optional maximum number of records to return (default 100).
    """
    vals = list(request.app.state.workload.store.validations)[-limit:]
    return [{"txn": v.txn, "ledger": v.seq, "source": v.src} for v in reversed(vals)]


@router.get("/wallets")
def api_state_wallets(request: Request):
    ws = request.app.state.workload.wallets
    return {"count": len(ws), "addresses": list(ws.keys())}


@router.get("/users")
def api_state_users(request: Request):
    """Get all user wallets with addresses and seeds."""
    wl = request.app.state.workload
    users = [
        {
            "address": user.address,
            "seed": user.seed,
        }
        for user in wl.users
    ]
    return {"count": len(users), "users": users}


@router.get("/gateways")
def api_state_gateways(request: Request):
    """Get all gateway wallets with addresses, seeds, and issued currencies."""
    wl = request.app.state.workload
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


@router.get("/currencies")
def get_currencies(request: Request):
    """Get all configured/issued currencies."""
    wl = request.app.state.workload
    currencies = [
        {
            "currency": curr.currency,
            "issuer": curr.issuer,
        }
        for curr in wl._currencies
    ]
    return {"count": len(currencies), "currencies": currencies}


@router.get("/mptokens")
def get_mptokens(request: Request):
    """Get all tracked MPToken issuance IDs."""
    wl = request.app.state.workload
    mptoken_ids = getattr(wl, "_mptoken_issuance_ids", [])
    return {
        "count": len(mptoken_ids),
        "mptoken_issuance_ids": mptoken_ids,
        "note": "MPToken IDs are tracked automatically when MPTokenIssuanceCreate transactions validate",
    }


@router.get("/finality")
async def check_finality(request: Request):
    """Manually trigger finality check for all pending submitted transactions."""
    wl = request.app.state.workload
    results = []

    for p in wl.find_by_state(C.TxState.SUBMITTED):
        try:
            state = await wl.check_finality(p)
            results.append(
                {
                    "tx_hash": p.tx_hash,
                    "state": state.name,
                    "ledger_index": p.validated_ledger,
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


@router.get("/ws/stats")
async def ws_stats(request: Request):
    """Return stats about WebSocket event processing."""
    queue_size = request.app.state.ws_queue.qsize()
    store_stats = request.app.state.workload.store.snapshot_stats()

    return {
        "queue_size": queue_size,
        "queue_maxsize": request.app.state.ws_queue.maxsize,
        "validations_by_source": store_stats.get("validated_by_source", {}),
        "recent_validations_count": store_stats.get("recent_validations", 0),
        "ws_event_counters": get_ws_counters(),
    }


@router.get("/diagnostics")
async def diagnostics(request: Request):
    """Return diagnostic data about pending txn and account health."""
    wl = request.app.state.workload
    return wl.diagnostics_snapshot()
