def test_logs(api):
    resp = api.get("/logs")
    assert resp.status_code == 200


def test_logs_with_params(api):
    resp = api.get("/logs", params={"n": 10, "level": "WARNING"})
    assert resp.status_code == 200


def test_logs_page(api):
    resp = api.get("/logs/page")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
