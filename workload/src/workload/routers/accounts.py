from fastapi import APIRouter, HTTPException, Request

from workload.schemas import CreateAccountReq, CreateAccountResp
from workload.workload_core import Workload

router = APIRouter(prefix="/accounts", tags=["Accounts"])


@router.get("")
async def list_accounts(request: Request):
    """List all accounts available to the workload.

    Returns the total count, breakdown by role (funding, gateway, user),
    and the addresses in each pool. These are the accounts used by
    generate_txn() for random transaction generation.
    """
    wl = request.app.state.workload
    pending_counts = wl.get_pending_txn_counts_by_account()
    available = sum(1 for addr in wl.wallets if pending_counts.get(addr, 0) < wl.max_pending_per_account)

    return {
        "total_wallets": len(wl.wallets),
        "available_for_txn": available,
        "max_pending_per_account": wl.max_pending_per_account,
        "funding": wl.funding_wallet.address,
        "gateways": {
            "count": len(wl.gateways),
            "addresses": [g.address for g in wl.gateways],
            "names": wl.gateway_names,
        },
        "users": {
            "count": len(wl.users),
            "addresses": [u.address for u in wl.users],
        },
    }


@router.get("/create")
async def api_create_account(request: Request):
    return await request.app.state.workload.create_account()


@router.post("/create", response_model=CreateAccountResp)
async def accounts_create(req: CreateAccountReq, request: Request):
    data = req.model_dump(exclude_unset=True)
    return await request.app.state.workload.create_account(data, wait=req.wait)


@router.get("/create/random", response_model=CreateAccountResp)
async def accounts_create_random(request: Request):
    return await request.app.state.workload.create_account({}, wait=False)


@router.get("/{account_id}")
async def get_account_info(account_id: str, request: Request):
    """Get account_info for a specific account."""
    from xrpl.models.requests import AccountInfo

    w: Workload = request.app.state.workload
    try:
        result = await w.client.request(AccountInfo(account=account_id, ledger_index="validated"))
        return result.result
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Account not found or error: {str(e)}")


@router.get("/{account_id}/balances")
async def get_account_balances(account_id: str, request: Request):
    """Get all balances for a specific account from database."""
    from workload.sqlite_store import SQLiteStore

    w: Workload = request.app.state.workload
    if isinstance(w.persistent_store, SQLiteStore):
        balances = w.persistent_store.get_balances(account_id)
        return {"account": account_id, "balances": balances}
    else:
        raise HTTPException(status_code=503, detail="Balance tracking not available (no persistent store)")


@router.get("/{account_id}/lines")
async def get_account_lines(account_id: str, request: Request):
    """Get trust lines for a specific account from the ledger."""
    from xrpl.models.requests import AccountLines

    w: Workload = request.app.state.workload
    try:
        result = await w.client.request(AccountLines(account=account_id, ledger_index="validated"))
        return result.result
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Account lines not found or error: {str(e)}")
