from fastapi import APIRouter, HTTPException, Request

from workload.schemas import SendPaymentReq
from workload.workload_core import Workload
from xrpl.models.transactions import Payment

router = APIRouter(prefix="/payment", tags=["Payments"])


@router.post("")
async def send_payment(req: SendPaymentReq, request: Request):
    """Send a payment from source to destination. Works for both XRP and issued currencies."""
    w: Workload = request.app.state.workload

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
