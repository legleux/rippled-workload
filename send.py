import asyncio
import json
import time
from asyncio import TaskGroup
from itertools import takewhile
from pathlib import Path
from random import choice, randint, sample

import httpx
import xrpl
from xrpl.asyncio.account import get_next_valid_seq_number
from xrpl.asyncio.clients import AsyncJsonRpcClient
from xrpl.asyncio.ledger import get_latest_validated_ledger_sequence
from xrpl.asyncio.transaction import sign_and_submit, submit, submit_and_wait
from xrpl.constants import CryptoAlgorithm
from xrpl.core.binarycodec import encode, encode_for_signing
from xrpl.core.keypairs import sign
from xrpl.models import IssuedCurrency
from xrpl.models.transactions import (
    NFTokenBurn,
    NFTokenCreateOffer,
    NFTokenCreateOfferFlag,
    Payment,
)
from xrpl.wallet import Wallet

accounts = json.loads(Path("workload/accounts.json").read_text())


def gen_wallets(n):
    account_data = sample(accounts, n)
    wallets = [
        Wallet.from_seed(seed=seed, algorithm=CryptoAlgorithm.SECP256K1)
        for address, seed in account_data
    ]
    return accounts


async def generate_wallet():
    return Wallet.from_seed(
        seed=xrpl.core.keypairs.generate_seed(algorithm=CryptoAlgorithm.SECP256K1)
    )


async def generate_wallets(n):
    wallets = []
    async with TaskGroup() as tg:
        for _ in range(n):
            wallets.append(tg.create_task(generate_wallet()))
    return [w.result() for w in wallets]


# def create_payment_txn():
#     payment =  Payment(
#                 account=wallet.address,
#                 amount="1000000",
#                 destination=destination_address,
#                 sequence=seq + i,
#                 fee="500",
#                 last_ledger_sequence=latest_ledger + 10,
#                 signing_pub_key=wallet.public_key,
#             )
#     return


async def make_wallets(n):
    wallets = await generate_wallets(n)
    return wallets
    # async with TaskGroup() as tg:
    #     for blob in signed_blobs:
    #         tg.create_task(self.submit_via_http(blob, responses))
    # await asyncio.sleep(2)
    # print("Hello, Async World!")


async def select_destination(account):
    # ensure the destination is not the source
    accounts_list = accounts.copy()
    a = choice(accounts)
    accounts_list.remove(a)
    return choice(accounts_list)


# async def select_destinations
# for idx, aa in enumerate(accounts_list):
#     if a == aa[0]:
#         accounts.
# while next()

# print(sum(1 for _ in takewhile(lambda x: x[0] != a, account_list)))


async def flood_ledger(n):
    # wallets = await make_wallets(n)
    # for w in wallets:
    #     print(w.address)
    async with TaskGroup() as tg:
        wallets_1 = tg.create_task(make_wallets(n))
        wallets_2 = tg.create_task(make_wallets(n))
    r = []
    wallets = [*wallets_1.result(), *wallets_2.result()]

    # wallets = [w.result() for ww in [wallets_1, wallets_2] for w in ww]
    # return wallets


if __name__ == "__main__":
    start = time.time()
    asyncio.run(flood_ledger(100))
    end = time.time()
    elapsed = end - start
    elapsed = int(elapsed) if elapsed > 1 else f"{elapsed:.3f}"
    print(f"took {elapsed} seconds.")
    # print(f"took {int(end - start)}")
