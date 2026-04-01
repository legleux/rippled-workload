import pytest


def test_summary(api):
    resp = api.get("/state/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "by_state" in data
    assert "ledger_index" in data


def test_pending(api):
    resp = api.get("/state/pending")
    assert resp.status_code == 200


def test_failed(api):
    resp = api.get("/state/failed")
    assert resp.status_code == 200


def test_failed_by_code(api, live_failure_code):
    if live_failure_code is None:
        pytest.skip("No failure codes available")
    resp = api.get(f"/state/failed/{live_failure_code}")
    assert resp.status_code == 200


def test_failure_codes(api):
    resp = api.get("/state/failure-codes")
    assert resp.status_code == 200
    assert "failure_codes" in resp.json()


def test_expired(api):
    resp = api.get("/state/expired")
    assert resp.status_code == 200


def test_type(api, live_txn_type):
    resp = api.get(f"/state/type/{live_txn_type}")
    assert resp.status_code == 200


def test_tx_by_hash(api, live_tx_hash):
    if live_tx_hash is None:
        pytest.skip("No tx hash available")
    resp = api.get(f"/state/tx/{live_tx_hash}")
    assert resp.status_code == 200


def test_fees(api):
    resp = api.get("/state/fees")
    assert resp.status_code == 200


def test_accounts(api):
    resp = api.get("/state/accounts")
    assert resp.status_code == 200
    assert len(resp.json()) > 0


def test_validations(api):
    resp = api.get("/state/validations")
    assert resp.status_code == 200


def test_validations_with_limit(api):
    resp = api.get("/state/validations", params={"limit": 5})
    assert resp.status_code == 200
    assert len(resp.json()) <= 5


def test_wallets(api):
    resp = api.get("/state/wallets")
    assert resp.status_code == 200


def test_users(api):
    resp = api.get("/state/users")
    assert resp.status_code == 200


def test_gateways(api):
    resp = api.get("/state/gateways")
    assert resp.status_code == 200


def test_currencies(api):
    resp = api.get("/state/currencies")
    assert resp.status_code == 200


def test_mptokens(api):
    resp = api.get("/state/mptokens")
    assert resp.status_code == 200


def test_finality(api):
    import httpx

    # This endpoint triggers RPC lookups for all pending txns — needs a long timeout
    with httpx.Client(base_url="http://localhost:8000", timeout=60) as long_client:
        resp = long_client.get("/state/finality")
    assert resp.status_code == 200


def test_ws_stats(api):
    resp = api.get("/state/ws/stats")
    assert resp.status_code == 200
    assert "ws_event_counters" in resp.json()


def test_diagnostics(api):
    resp = api.get("/state/diagnostics")
    assert resp.status_code == 200
