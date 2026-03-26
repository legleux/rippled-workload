from pydantic import BaseModel


class TxnReq(BaseModel):
    type: str


class CreateAccountReq(BaseModel):
    seed: str | None = None
    address: str | None = None
    drops: int | None = None
    algorithm: str | None = None
    wait: bool | None = False


class CreateAccountResp(BaseModel):
    address: str
    seed: str | None = None
    funded: bool
    tx_hash: str | None = None


class SendPaymentReq(BaseModel):
    source: str
    destination: str
    amount: str | dict  # XRP drops as string, or IOU as {"currency": "USD", "issuer": "r...", "value": "100"}


class TargetTPSReq(BaseModel):
    target_tps: float


class MaxPendingReq(BaseModel):
    max_pending: int


class IntentReq(BaseModel):
    valid: float | None = None
    invalid: float | None = None


class ToggleTypeReq(BaseModel):
    txn_type: str
    enabled: bool
