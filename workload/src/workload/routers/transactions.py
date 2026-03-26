import logging

from fastapi import APIRouter, Request

from workload.workload_core import Workload

log = logging.getLogger("workload.app")

router = APIRouter(tags=["Transactions"])


@router.get("/random")
async def transaction_random(request: Request):
    w = request.app.state.workload
    res = await w.submit_random_txn()
    return res


@router.get("/create/{transaction}")
async def create(transaction: str, request: Request):
    w: Workload = request.app.state.workload
    log.debug("Creating a %s", transaction)
    r = await w.create_transaction(transaction)
    return r


@router.post("/payment")
async def create_payment(request: Request):
    """Create and submit a Payment transaction."""
    return await create("Payment", request)


@router.post("/trustset")
async def create_trustset(request: Request):
    """Create and submit a TrustSet transaction."""
    return await create("TrustSet", request)


@router.post("/accountset")
async def create_accountset(request: Request):
    """Create and submit an AccountSet transaction."""
    return await create("AccountSet", request)


@router.post("/ammcreate")
async def create_ammcreate(request: Request):
    """Create and submit an AMMCreate transaction."""
    return await create("AMMCreate", request)


@router.post("/nftokenmint")
async def create_nftokenmint(request: Request):
    """Create and submit an NFTokenMint transaction."""
    return await create("NFTokenMint", request)


@router.post("/mptokenissuancecreate")
async def create_mptokenissuancecreate(request: Request):
    """Create and submit an MPTokenIssuanceCreate transaction."""
    return await create("MPTokenIssuanceCreate", request)


@router.post("/mptokenissuanceset")
async def create_mptokenissuanceset(request: Request):
    """Create and submit an MPTokenIssuanceSet transaction."""
    return await create("MPTokenIssuanceSet", request)


@router.post("/mptokenauthorize")
async def create_mptokenauthorize(request: Request):
    """Create and submit an MPTokenAuthorize transaction."""
    return await create("MPTokenAuthorize", request)


@router.post("/mptokenissuancedestroy")
async def create_mptokenissuancedestroy(request: Request):
    """Create and submit an MPTokenIssuanceDestroy transaction."""
    return await create("MPTokenIssuanceDestroy", request)


@router.post("/batch")
async def create_batch(request: Request):
    """Create and submit a Batch transaction."""
    return await create("Batch", request)


@router.post("/ammdeposit")
async def create_ammdeposit(request: Request):
    """Create and submit an AMMDeposit transaction."""
    return await create("AMMDeposit", request)


@router.post("/ammwithdraw")
async def create_ammwithdraw(request: Request):
    """Create and submit an AMMWithdraw transaction."""
    return await create("AMMWithdraw", request)


@router.post("/delegateset")
async def create_delegateset(request: Request):
    """Create and submit a DelegateSet transaction."""
    return await create("DelegateSet", request)


@router.post("/credentialcreate")
async def create_credentialcreate(request: Request):
    """Create and submit a CredentialCreate transaction."""
    return await create("CredentialCreate", request)


@router.post("/credentialaccept")
async def create_credentialaccept(request: Request):
    """Create and submit a CredentialAccept transaction."""
    return await create("CredentialAccept", request)


@router.post("/credentialdelete")
async def create_credentialdelete(request: Request):
    """Create and submit a CredentialDelete transaction."""
    return await create("CredentialDelete", request)


@router.post("/permissioneddomainset")
async def create_permissioneddomainset(request: Request):
    """Create and submit a PermissionedDomainSet transaction."""
    return await create("PermissionedDomainSet", request)


@router.post("/permissioneddomaindelete")
async def create_permissioneddomaindelete(request: Request):
    """Create and submit a PermissionedDomainDelete transaction."""
    return await create("PermissionedDomainDelete", request)


@router.post("/vaultcreate")
async def create_vaultcreate(request: Request):
    """Create and submit a VaultCreate transaction."""
    return await create("VaultCreate", request)


@router.post("/vaultset")
async def create_vaultset(request: Request):
    """Create and submit a VaultSet transaction."""
    return await create("VaultSet", request)


@router.post("/vaultdelete")
async def create_vaultdelete(request: Request):
    """Create and submit a VaultDelete transaction."""
    return await create("VaultDelete", request)


@router.post("/vaultdeposit")
async def create_vaultdeposit(request: Request):
    """Create and submit a VaultDeposit transaction."""
    return await create("VaultDeposit", request)


@router.post("/vaultwithdraw")
async def create_vaultwithdraw(request: Request):
    """Create and submit a VaultWithdraw transaction."""
    return await create("VaultWithdraw", request)


@router.post("/vaultclawback")
async def create_vaultclawback(request: Request):
    """Create and submit a VaultClawback transaction."""
    return await create("VaultClawback", request)


