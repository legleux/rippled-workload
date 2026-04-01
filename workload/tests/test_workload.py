import pytest


def test_status(api):
    resp = api.get("/workload/status")
    assert resp.status_code == 200


def test_rate_controls(api):
    resp = api.get("/workload/rate-controls")
    assert resp.status_code == 200


def test_target_tps_get(api):
    resp = api.get("/workload/target-tps")
    assert resp.status_code == 200


def test_max_pending(api):
    resp = api.get("/workload/max-pending")
    assert resp.status_code == 200
    assert resp.json().get("max_pending_per_account") == 1


def test_intent_get(api):
    resp = api.get("/workload/intent")
    assert resp.status_code == 200
    data = resp.json()
    assert "valid" in data
    assert "invalid" in data


def test_disabled_types(api):
    resp = api.get("/workload/disabled-types")
    assert resp.status_code == 200


@pytest.mark.mutating
def test_start(api):
    resp = api.post("/workload/start")
    assert resp.status_code == 200


@pytest.mark.mutating
def test_stop(api):
    resp = api.post("/workload/stop")
    assert resp.status_code == 200


@pytest.mark.mutating
def test_target_tps_post(api):
    current = api.get("/workload/target-tps").json()
    resp = api.post("/workload/target-tps", json={"target_tps": current.get("target_tps", 0)})
    assert resp.status_code == 200


@pytest.mark.mutating
def test_intent_post(api):
    resp = api.post("/workload/intent", json={"invalid": 0.0})
    assert resp.status_code == 200


@pytest.mark.mutating
def test_toggle_type(api):
    resp = api.post("/workload/toggle-type", json={"txn_type": "Payment", "enabled": True})
    assert resp.status_code == 200
