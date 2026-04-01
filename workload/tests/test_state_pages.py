import pytest


def _assert_html(resp):
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_dashboard(api):
    _assert_html(api.get("/state/dashboard"))


def test_failures_page(api):
    _assert_html(api.get("/state/failures"))


def test_failures_by_code_page(api, live_failure_code):
    if live_failure_code is None:
        pytest.skip("No failure codes available")
    _assert_html(api.get(f"/state/failures/{live_failure_code}"))


def test_types_page(api, live_txn_type):
    _assert_html(api.get(f"/state/types/{live_txn_type}"))


def test_mpt_issuances_page(api):
    _assert_html(api.get("/state/mpt-issuances"))
