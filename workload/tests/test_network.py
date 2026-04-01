import pytest


@pytest.mark.reset
def test_network_reset(api):
    resp = api.post("/network/reset")
    assert resp.status_code == 200
