import pytest


def test_list_accounts(api):
    resp = api.get("/accounts")
    assert resp.status_code == 200


def test_account_info(api, live_account):
    resp = api.get(f"/accounts/{live_account}")
    assert resp.status_code == 200


def test_account_balances(api, live_account):
    resp = api.get(f"/accounts/{live_account}/balances")
    assert resp.status_code == 200


def test_account_lines(api, live_account):
    resp = api.get(f"/accounts/{live_account}/lines")
    assert resp.status_code == 200


@pytest.mark.mutating
def test_create_account_get(api):
    resp = api.get("/accounts/create")
    assert resp.status_code == 200


@pytest.mark.mutating
def test_create_account_post(api):
    resp = api.post("/accounts/create", json={})
    assert resp.status_code == 200
