from fastapi import APIRouter, HTTPException, Request

from workload.workload_core import Workload

router = APIRouter(prefix="/dex", tags=["DEX"])


@router.get("/metrics")
async def dex_metrics(request: Request):
    """Get DEX metrics including AMM pool states, trading activity counts."""
    return request.app.state.workload.snapshot_dex_metrics()


@router.get("/pools")
async def dex_pools(request: Request):
    """List all tracked AMM pools."""
    w: Workload = request.app.state.workload
    return {
        "total_pools": len(w.amm.pools),
        "pools": w.amm.pools,
    }


@router.get("/pools/{index}")
async def dex_pool_detail(index: int, request: Request):
    """Get detailed amm_info for a specific pool by index."""
    from xrpl.models.currencies import XRP as XRPCurrency
    from xrpl.models.requests import AMMInfo

    w: Workload = request.app.state.workload
    if index >= len(w.amm.pools):
        raise HTTPException(status_code=404, detail=f"Pool index {index} not found")

    pool = w.amm.pools[index]
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


@router.post("/poll")
async def dex_poll_now(request: Request):
    """Manually trigger a DEX metrics poll."""
    return await request.app.state.workload.poll_dex_metrics()
