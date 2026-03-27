import json
import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from xrpl.models.transactions import Payment

from workload import __version__
from workload.bootstrap import lifespan
from workload.routers import accounts, dex, logs, network, payments, state_api, state_pages, transactions, workload
from workload.workload_core import Workload

log = logging.getLogger("workload.app")

app = FastAPI(
    title="XRPL Workload",
    version=__version__,
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


@app.get("/")
def root():
    return RedirectResponse(url="/state/dashboard")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def get_version():
    return {"version": __version__}


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
    log.debug("response from submit_pending() %s", res)
    return res


# Router registration — aliases preserved exactly
app.include_router(accounts.router)
app.include_router(payments.router)
app.include_router(payments.router, prefix="/pay", include_in_schema=False)  # alias /pay/ for convenience
app.include_router(transactions.router, prefix="/transaction")
app.include_router(
    transactions.router, prefix="/txn", include_in_schema=False
)  # alias /txn/ because I'm sick of typing...
app.include_router(state_api.router)
app.include_router(state_pages.router)
app.include_router(workload.router)
app.include_router(dex.router)
app.include_router(network.router)
app.include_router(logs.router)
