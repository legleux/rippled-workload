import pytest
import httpx

API_URL = "http://localhost:8000"


@pytest.fixture(scope="session")
def api():
    with httpx.Client(base_url=API_URL, timeout=10) as client:
        yield client


@pytest.fixture(scope="session")
def live_account(api):
    resp = api.get("/state/accounts")
    data = resp.json()
    addresses = data.get("addresses", data) if isinstance(data, dict) else data
    return addresses[1] if len(addresses) > 1 else addresses[0]


@pytest.fixture(scope="session")
def live_failure_code(api):
    resp = api.get("/state/failure-codes")
    codes = resp.json().get("failure_codes", [])
    return codes[0][0] if codes else None


@pytest.fixture(scope="session")
def live_txn_type():
    return "Payment"


@pytest.fixture(scope="session")
def live_tx_hash(api):
    resp = api.get("/state/validations", params={"limit": 1})
    validations = resp.json()
    if validations and len(validations) > 0:
        return validations[0].get("tx_hash")
    return None


@pytest.fixture(scope="session")
def live_pool_count(api):
    resp = api.get("/dex/pools")
    return len(resp.json())
