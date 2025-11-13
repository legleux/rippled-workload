import xrpl
import httpx
import xrpl.constants
from xrpl.models import Batch, BatchFlag, Payment


rippled_url = "172.17.0.6"
rippled_port = "5005"
url = f"http://{rippled_url}:{rippled_port}"
client = xrpl.clients.JsonRpcClient(url=url)

seed = "snZC2894DPEvhGtuZiQAHQaVJFtrV"
dst = "rAByFkaef3Mw6VU6UapDuMJSihorzJpKd"

wallet = xrpl.wallet.Wallet.from_seed(seed=seed, algorithm=xrpl.CryptoAlgorithm.SECP256K1)

sequence = xrpl.account.get_next_valid_seq_number(address=wallet.address, client=client)
amount = "1000000"
raw_transactions = [
    Payment(
        account=wallet.address,
        amount=amount,
        destination=dst,
        flags=xrpl.models.TransactionFlag.TF_INNER_BATCH_TXN,
        sequence=sequence + 1,
        fee="0",
        signing_pub_key=""
    ),
    Payment(
        account=wallet.address,
        amount=amount,
        destination=dst,
        flags=xrpl.models.TransactionFlag.TF_INNER_BATCH_TXN,
        sequence=sequence + 2,
        fee="0",
        signing_pub_key=""
    )
]
txn = Batch(
    account=wallet.address,
    flags=BatchFlag.TF_ALL_OR_NOTHING,
    raw_transactions=[*raw_transactions],
    sequence=sequence,
)

response = xrpl.transaction.submit_and_wait(transaction=txn, client=client, wallet=wallet)
pass
