def test_root_redirects_to_dashboard(api):
    resp = api.get("/", follow_redirects=False)
    assert resp.status_code in (200, 307), f"Expected 200 or redirect, got {resp.status_code}"


def test_health(api):
    resp = api.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_version(api):
    resp = api.get("/version")
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data
