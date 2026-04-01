import pytest


def test_metrics(api):
    resp = api.get("/dex/metrics")
    assert resp.status_code == 200


def test_pools(api):
    resp = api.get("/dex/pools")
    assert resp.status_code == 200


def test_amm_pools_page(api):
    resp = api.get("/dex/amm-pools")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_pool_by_index(api, live_pool_count):
    if live_pool_count == 0:
        pytest.skip("No AMM pools available")
    resp = api.get("/dex/pools/0")
    assert resp.status_code == 200


@pytest.mark.mutating
def test_poll(api):
    resp = api.post("/dex/poll")
    assert resp.status_code == 200
