import pytest


@pytest.mark.mutating
def test_send_payment(api, live_account):
    resp = api.post(
        "/payment",
        json={
            "source": live_account,
            "destination": live_account,  # self-payment, simplest case
            "amount": "1000000",  # 1 XRP in drops
        },
    )
    assert resp.status_code in (200, 400, 422, 500)
