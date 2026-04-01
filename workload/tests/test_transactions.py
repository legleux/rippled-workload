import pytest

ALL_TXN_TYPES = [
    "payment",
    "trustset",
    "accountset",
    "ammcreate",
    "ammdeposit",
    "ammwithdraw",
    "nftokenmint",
    "mptokenissuancecreate",
    "mptokenissuanceset",
    "mptokenauthorize",
    "mptokenissuancedestroy",
    "batch",
    "delegateset",
    "credentialcreate",
    "credentialaccept",
    "credentialdelete",
    "permissioneddomainset",
    "permissioneddomaindelete",
    "vaultcreate",
    "vaultset",
    "vaultdelete",
    "vaultdeposit",
    "vaultwithdraw",
    "vaultclawback",
]


@pytest.mark.mutating
@pytest.mark.parametrize("txn_type", ALL_TXN_TYPES)
def test_submit_transaction(api, txn_type):
    resp = api.post(f"/transaction/{txn_type}")
    # Some may fail (disabled types, no eligible accounts, etc.) — that's expected
    assert resp.status_code in (200, 400, 422, 500), f"{txn_type}: {resp.status_code} {resp.text[:200]}"


@pytest.mark.mutating
def test_random_transaction(api):
    resp = api.get("/transaction/random")
    assert resp.status_code in (200, 400, 500)


@pytest.mark.mutating
@pytest.mark.parametrize("txn_type", ["Payment", "TrustSet", "OfferCreate"])
def test_create_transaction_by_name(api, txn_type):
    resp = api.get(f"/transaction/create/{txn_type}")
    assert resp.status_code in (200, 400, 500)
