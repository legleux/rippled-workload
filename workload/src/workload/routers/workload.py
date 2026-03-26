import logging

from fastapi import APIRouter, HTTPException, Request

from workload import workload_runner
from workload.schemas import IntentReq, MaxPendingReq, TargetTPSReq, ToggleTypeReq

log = logging.getLogger("workload.app")

router = APIRouter(prefix="/workload", tags=["Workload"])


@router.post("/start")
async def start_workload(request: Request):
    """Start continuous random transaction workload."""
    try:
        result = await workload_runner.start(request.app.state.workload)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.post("/stop")
async def stop_workload(request: Request):
    """Stop continuous workload."""
    try:
        result = await workload_runner.stop(request.app.state.workload)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.get("/status")
async def workload_status(request: Request):
    """Get current workload status and statistics."""
    return workload_runner.status(request.app.state.workload)


@router.get("/rate-controls")
async def get_rate_controls(request: Request):
    """Get all rate control settings."""
    wl = request.app.state.workload
    return {
        "target_tps": wl.target_tps,
        "max_pending_per_account": wl.max_pending_per_account,
        "submission_set_size": wl.submission_set_size,
    }


@router.get("/target-tps")
async def get_target_tps(request: Request):
    """Get current target TPS. 0 = unlimited."""
    return {"target_tps": request.app.state.workload.target_tps}


@router.post("/target-tps")
async def set_target_tps(req: TargetTPSReq, request: Request):
    """Set target transactions per second. 0 = unlimited (firehose)."""
    if req.target_tps < 0 or req.target_tps > 10000:
        raise HTTPException(status_code=400, detail="target_tps must be between 0 and 10000")

    old_value = request.app.state.workload.target_tps
    request.app.state.workload.target_tps = req.target_tps
    log.info(f"target_tps changed: {old_value} -> {req.target_tps}")
    return {"old_value": old_value, "new_value": req.target_tps, "status": "updated"}


@router.get("/max-pending")
async def get_max_pending(request: Request):
    """Get max pending transactions per account."""
    return {"max_pending_per_account": request.app.state.workload.max_pending_per_account}


@router.post("/max-pending")
async def set_max_pending(req: MaxPendingReq, request: Request):
    """Set max pending transactions per account. Range: 1-10."""
    if req.max_pending < 1 or req.max_pending > 10:
        raise HTTPException(status_code=400, detail="max_pending must be between 1 and 10")

    old_value = request.app.state.workload.max_pending_per_account
    request.app.state.workload.max_pending_per_account = req.max_pending
    log.info(f"max_pending_per_account changed: {old_value} -> {req.max_pending}")
    return {"old_value": old_value, "new_value": req.max_pending, "status": "updated"}


@router.get("/intent")
async def get_intent_ratio(request) -> dict:
    """Get current valid/invalid intent ratio for transaction generation."""
    intent_cfg = request.app.state.workload.config.get("transactions", {}).get("intent", {})
    return {
        "valid": intent_cfg.get("valid", 0.90),
        "invalid": intent_cfg.get("invalid", 0.10),
        "per_type": intent_cfg.get("per_type", {}),
    }


@router.post("/intent")
async def set_intent_ratio(req: IntentReq, request) -> dict:
    """Set valid/invalid intent ratio. Takes effect immediately.

    Provide either valid or invalid (the other is computed as 1 - value).
    Values must be between 0.0 and 1.0.
    """
    intent_cfg = request.app.state.workload.config.setdefault("transactions", {}).setdefault("intent", {})
    old_valid = intent_cfg.get("valid", 0.90)
    old_invalid = intent_cfg.get("invalid", 0.10)

    if req.invalid is not None:
        if not 0.0 <= req.invalid <= 1.0:
            raise HTTPException(status_code=400, detail="invalid must be between 0.0 and 1.0")
        intent_cfg["invalid"] = req.invalid
        intent_cfg["valid"] = 1.0 - req.invalid
    elif req.valid is not None:
        if not 0.0 <= req.valid <= 1.0:
            raise HTTPException(status_code=400, detail="valid must be between 0.0 and 1.0")
        intent_cfg["valid"] = req.valid
        intent_cfg["invalid"] = 1.0 - req.valid
    else:
        raise HTTPException(status_code=400, detail="Provide either 'valid' or 'invalid'")

    log.info(
        "Intent ratio changed: valid=%.2f->%.2f invalid=%.2f->%.2f",
        old_valid,
        intent_cfg["valid"],
        old_invalid,
        intent_cfg["invalid"],
    )
    return {
        "old": {"valid": old_valid, "invalid": old_invalid},
        "new": {"valid": intent_cfg["valid"], "invalid": intent_cfg["invalid"]},
        "status": "updated",
    }


@router.get("/disabled-types")
async def get_disabled_types(request: Request):
    """Get currently disabled transaction types for random generation."""
    from workload.txn_factory import _BUILDERS

    all_types = list(_BUILDERS.keys())
    disabled = sorted(request.app.state.workload.disabled_txn_types)
    enabled = [t for t in all_types if t not in request.app.state.workload.disabled_txn_types]
    return {
        "all_types": all_types,
        "enabled_types": enabled,
        "disabled_types": disabled,
        "config_disabled": sorted(request.app.state.workload._config_disabled_types),
    }


@router.post("/toggle-type")
async def toggle_txn_type(req: ToggleTypeReq, request: Request):
    """Toggle a single transaction type on or off for random generation.

    Takes effect immediately on the next transaction generation.
    """
    from workload.txn_factory import _BUILDERS

    if req.txn_type not in _BUILDERS:
        raise HTTPException(
            status_code=400, detail=f"Unknown transaction type: {req.txn_type}. Valid: {list(_BUILDERS.keys())}"
        )

    if req.txn_type in request.app.state.workload._config_disabled_types:
        raise HTTPException(
            status_code=400,
            detail=f"{req.txn_type} is disabled in config.toml (amendment not available). Cannot toggle at runtime.",
        )

    if req.enabled:
        request.app.state.workload.disabled_txn_types.discard(req.txn_type)
    else:
        request.app.state.workload.disabled_txn_types.add(req.txn_type)

    return {
        "txn_type": req.txn_type,
        "enabled": req.enabled,
        "disabled_types": sorted(request.app.state.workload.disabled_txn_types),
    }
