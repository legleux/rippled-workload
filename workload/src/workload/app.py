import json
import os
import pathlib
import time
from antithesis import lifecycle
from typing import Any
from workload.check_rippled_sync_state import is_rippled_synced # TODO:git use rippled_sync.py
from workload.create import create_accounts
from xrpl.models import IssuedCurrency
import asyncio
import httpx
from asyncio import TaskGroup
from xrpl.asyncio.account import get_next_valid_seq_number
from xrpl.asyncio.clients import AsyncJsonRpcClient
from xrpl.asyncio.ledger import get_latest_validated_ledger_sequence
from xrpl.core.binarycodec import encode_for_signing, encode
from xrpl.core.keypairs import sign
from xrpl.models.transactions import Payment
from xrpl.constants import CryptoAlgorithm
from xrpl.wallet import Wallet
from workload.models import UserAccount
from xrpl.account import does_account_exist
from xrpl.asyncio.clients import AsyncJsonRpcClient
from xrpl.asyncio.transaction.reliable_submission import XRPLReliableSubmissionException
import sys
import xrpl
from xrpl.asyncio.transaction import submit_and_wait, sign_and_submit
from xrpl.models.transactions import NFTokenCreateOffer, NFTokenCreateOfferFlag

import json
# from xrpl.transaction import sign_and_submit, submit_and_wait
from workload.balances import get_account_tokens
from workload.config import conf_file, config_file
from workload.randoms import sample, choice
from workload import utils, logger
from pathlib import Path
from fastapi import FastAPI, Depends
import uvicorn
from workload.nft import mint_nft

class Workload:
    def __init__(self, conf: dict[str, Any]):
        print("Starting workload")

        self.config = conf
        self.accounts = {}
        # TODO: Lookup account by nfts owned, tickets, etc
        self.gateways = []
        self.amms = []
        self.nfts = []
        self.currencies = []
        self.currency_codes = conf["currencies"]["codes"]
        self.start_time = time.time()
        rippled_host = os.environ.get("RIPPLED_NAME", conf["rippled"]["local"])
        rippled_rpc_port = os.environ.get("RIPPLED_RPC_PORT", conf["rippled"]["json_rpc_port"])
        self.rippled = f"http://{rippled_host}:{rippled_rpc_port}"
        logger.info("Connecting to rippled at: %s", self.rippled)
        self.load_initial_accounts()
        self.client = AsyncJsonRpcClient(self.rippled)
        # self.wait_for_network(self.rippled)


        # utils.check_validator_proposing() or sys.exit("All validators not in 'proposing' state!")

        account_type = xrpl.models.requests.LedgerEntryType.ACCOUNT
        ledger_data_request = xrpl.models.requests.LedgerData(type=account_type)

        logger.info("%s after %ss", "Workload initialization complete", int(time.time() - self.start_time))
        logger.info("Workload going to sleep...")
        # local_path = pathlib.Path(__file__).parents[3] / "tc/workload.json"
        workload_ready_msg =  "Workload initialization complete"
        lifecycle.setup_complete(details={"message": workload_ready_msg})
        print('{"antithesis_setup": { "status": "complete", "details": "" }}')
        logger.info("Called lifecycle setup_complete()")

    def load_initial_accounts(self):
        try:
            accounts = json.loads(Path("/accounts.json").read_text())
            print(json.dumps(self.account_data, indent=2))
        except FileNotFoundError:
            logger.error("accounts.json not found.")
            if True:
                local_path = "accounts.json"
            # local_path = input("Enter local file path:")
            accounts_json = Path(local_path)
            logger.info(f"Using {accounts_json.resolve()}")
            accounts = json.loads(accounts_json.read_text())
            logger.info(f"{len(accounts)} accounts found!")
            for idx, i in enumerate(accounts):
                logger.info(f"{idx}: {i}")
        self.account_data = accounts

        default_algo = CryptoAlgorithm[conf_file["workload"]["accounts"]["default_crypto_algorithm"]]

        def generate_wallet_from_seed(seed: str, algorithm: CryptoAlgorithm = default_algo) -> Wallet:
            wallet = Wallet.from_seed(seed=seed, algorithm=algorithm)
            return wallet
        for _, seed in self.account_data:
            wallet = generate_wallet_from_seed(seed)
            self.accounts[wallet.address] = UserAccount(wallet=wallet)
        logger.info(f"Loaded {len(self.accounts)} initial accounts")

    # def configure_gateways(self, number: int, balance: str) -> None:
    #     """Configure the gateways for the network.

    #     Creates the accounts, enables default_ripple and creates some initial currencies.
    #     """
    #     logger.info("Configuring %s gateways", number)
    #     # BUG: Configuring 2 gateways causes error in xrpl-py
    #     gateway_config_start = time.time()
    #     gateway_wallets, responses = create_accounts(number=number, client=self.client, amount=balance)
    #     self.gateways = [Gateway(wallet) for idx, wallet in enumerate(gateway_wallets)]
    #     for gateway in self.gateways:
    #         logger.info("Setting up gateway %s", gateway.address)
    #         # Enable rippling on gateway's trustlines so tokens can be transferred
    #         accountset_txn = AccountSet(
    #             account=gateway.address,
    #             set_flag=AccountSetAsfFlag.ASF_DEFAULT_RIPPLE,
    #         )
    #         utils.wait_for_ledger_close(self.client)
    #         response = submit_and_wait(accountset_txn, self.client, gateway.wallet)

        #     for ic in utils.issue_currencies(gateway.address, self.currency_codes):
        #         gateway.issued_currencies[ic.currency] = ic
        #         self.currencies.append(ic)
        # logger.info("%s gateways configured in %ss", len(self.gateways), int(time.time() - gateway_config_start))

    # def configure_accounts(self, number: int, balance: str) -> None:
    #     # TODO: Too many variables, too complex
    #     trustset_limit = self.config["transactions"]["trustset"]["limit"]
    #     logger.info("Configuring %s accounts", number)
    #     account_create_start = time.time()
    #     wallets, responses = create_accounts(number=number, client=self.client, amount=balance)
    #     self.accounts = [UserAccount(wallet=wallet) for wallet in wallets]
    #     for account in self.accounts:
    #         does_account_exist(account.address, self.client)
    #     logger.debug("%s accounts created in %s ms", number, time.time() - account_create_start)
    #     utils.wait_for_ledger_close(self.client)
    #     accounts = self.accounts
    #     trustset_txns = []
    #     c = 1
    #     utils.wait_for_ledger_close(self.client)
    #     for gateway in self.gateways:
    #         for ic in gateway.issued_currencies.values():
    #             for account in accounts:
    #                 trustset_txns.append((account, TrustSet(account=account.address, limit_amount=ic.to_amount(value=trustset_limit))))
    #                 c += 1
    #     trustset_responses = []
    #     for account, txn in trustset_txns:
    #         trustset_responses.append(sign_and_submit(txn, self.client, account.wallet))
    #     # Simulate the accounts have bought tokens from the gateways
    #     payments = []
    #     for gateway in self.gateways:
    #         for account in accounts:
    #             for ic in gateway.issued_currencies.values():
    #                 # TODO: Get rid of magic numbers
    #                 usd_deposit = 10e3
    #                 rate = self.config["currencies"]["rate"][ic.currency]
    #                 token_disbursement = str(round(usd_deposit * 1e6 / int(rate), 10))
    #                 payment_amount = ic.to_amount(value=token_disbursement)
    #                 logger.info("Account %s depositing %s fiat USD for %s",
    #                             account.address, usd_deposit, utils.format_currency(payment_amount)
    #                 )
    #                 payments.append((gateway, Payment(account=gateway.address,
    #                                          amount=payment_amount,
    #                                          destination=account.address)))
    #     payment_responses = []
    #     for gw, txn in payments:
    #         payment_responses.append(sign_and_submit(txn, self.client, gw.wallet))
    #     for idx, account in enumerate(self.accounts):
    #         self.accounts[idx].balances = get_account_tokens(self.accounts[idx], self.client)["held"] # TODO: Fix

    @classmethod
    def issue_currencies(cls, issuer: str, currency_code: list[str]) -> list[IssuedCurrency]:
        """Use a fixed set of currency codes to create IssuedCurrencies for a specific gateway.

        Args:
            issuer (str): Account_id of the gateway for all currencies
            currency_code (list[str], optional): _description_. Defaults to config.currency_codes.

        Returns:
            list[IssuedCurrency]: List of IssuedCurrencies a gateway provides

        """
        issued_currencies = [IssuedCurrency.from_dict(dict(issuer=issuer, currency=cc)) for cc in currency_code]
        logger.debug("Issued %s currencies", len(issued_currencies))
        if Workload.verbose:
            for c in issued_currencies:
                logger.debug(c)
        return issued_currencies

    def wait_for_network(self, rippled) -> None:
        timeout = self.config["rippled"]["timeout"]  # Wait at most 10 minutes
        wait_start = time.time()
        logger.debug("Waiting %ss for rippled at %s to be running.", timeout, rippled)
        while not (is_rippled_synced(rippled)):
            irs = is_rippled_synced(rippled)
            logger.info(f"is_rippled_synced returning: {irs}")

            if (rippled_ready_time := int(time.time() - self.start_time)) > timeout:
                logger.info("rippled ready after %ss", rippled_ready_time)
            logger.info("Waited %ss so far", int(time.time() - wait_start))
            wait_time = 10
            time.sleep(wait_time)
        logger.info("rippled ready...")

    async def submit_payments(self, n: int, wallet: Wallet, destination_address: str):
        seq = await get_next_valid_seq_number(wallet.address, client=self.client)
        latest_ledger = await get_latest_validated_ledger_sequence(client=self.client)

        signed_blobs = []
        for i in range(n):
            tx_json = Payment(
                account=wallet.address,
                amount="1000000",
                destination=destination_address,
                sequence=seq + i,
                fee="500",
                last_ledger_sequence=latest_ledger + 10,
                signing_pub_key=wallet.public_key
            ).to_xrpl()

            signing_blob = encode_for_signing(tx_json)
            signature = sign(signing_blob, wallet.private_key)
            tx_json["TxnSignature"] = signature

            signed_blob = encode(tx_json)
            signed_blobs.append(signed_blob)

        responses = []
        async with TaskGroup() as tg:
            for blob in signed_blobs:
                tg.create_task(self.submit_via_http(blob, responses))

        return responses

    async def submit_via_http(self, blob: str, responses: list):
        payload = {
            "method": "submit",
            "params": [{"tx_blob": blob}]
        }
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(self.rippled, json=payload)
                responses.append(resp.json())
            except Exception as e:
                responses.append({"error": str(e)})

    async def pay(self):
        src, dst = sample(self.account_data, 2)
        src_secret = src[1]
        dst_address = dst[0]
        src_wallet = Wallet.from_secret(src_secret, algorithm=CryptoAlgorithm.SECP256K1)
        responses = await self.submit_payments(100, src_wallet, dst_address)
        for i in responses:
            print(i)

        return {"cool": "beans"}

    async def mint_random_nft(self):
        account_id = choice(list(self.accounts))
        account = self.accounts[account_id]
        sequence = await get_next_valid_seq_number(account.address, self.client)
        result = await mint_nft(account, sequence, self.client)
        logger.info(json.dumps(result, indent=2))
        logger.info(json.dumps(result["meta"]["nftoken_id"], indent=2))
        logger.info(json.dumps(result["tx_json"]["Account"], indent=2))
        nft_owner = result["tx_json"]["Account"]
        nftoken_id = result["meta"]["nftoken_id"]
        from workload.models import NFT
        nft = NFT(owner=account.address, nftoken_id=nftoken_id)
        self.nfts.append(nft)
        account.nfts.add(nftoken_id)
        # for a in self.accounts:
        #     if a.address == nft_owner:
        #         nft = NFT(owner=a, nftoken_id=nftoken_id)
        #         self.nfts.append(nft)
        #         logger.info(f"Added NFT {nftoken_id} with ownder {a}")
        #         break

        logger.info(f"Account {account.address}'s NFTs:")
        for idx, nft in enumerate(account.nfts):
            logger.info(f"{idx}: {nft}")

    def get_accounts(self):
        return self.accounts

    def get_nfts(self):
        return self.nfts

    async def nftoken_create_offer(self, account, nft_id, wallet):
        create_amount = "1000000"
        logger.info("Creating offer for %s's nft [%s]", account, nft_id)
        nftoken_offer_create_txn = NFTokenCreateOffer(
            account=account,
            nftoken_id=nft_id,
            amount=create_amount,
            flags=NFTokenCreateOfferFlag.TF_SELL_NFTOKEN,
        )
        logger.debug(json.dumps(nftoken_offer_create_txn.to_xrpl(), indent=2))
        nftoken_offer_create_txn_response = await submit_and_wait(transaction=nftoken_offer_create_txn, client=self.client, wallet=wallet)
        return nftoken_offer_create_txn_response.result

    async def create_random_nft_offer(self):
        nft = choice(self.nfts)
        res = await self.nftoken_create_offer(nft.owner.address, nft.nftoken_id, nft.owner.wallet)
        logger.info(res)
        return None

    async def nft_burn_random(self):
        from xrpl.models.transactions import NFTokenBurn
        nft = choice(self.nfts)
        nft_owner = nft.owner
        nftburn_txn = NFTokenBurn(account=nft_owner.address, nftoken_id=nft.nftoken_id)
        nftburn_txn_response = await submit_and_wait(transaction=nftburn_txn, client=self.client, wallet=nft_owner.wallet)
        return nftburn_txn_response.result

    async def payment_random(self):
        amount = str(1_000_000)
        src, dst = sample(list(self.accounts), 2)
        sequence = await get_next_valid_seq_number(src.address, self.client)
        payment_txn = Payment(account=src.address, amount=amount, destination=dst.address, sequence=sequence)
        response = await sign_and_submit(payment_txn, self.client, src.wallet)
        logger.debug("Payment from %s to %s for %s submitted.", src.address, dst.address, amount)
        return response, src.address, dst.address, amount

    async def create_ticket(self):
        ticket_count = 5
        logger.info(choice(list(self.accounts)))
        account_id  = choice(list(self.accounts))
        logger.info(f"{account_id=}")
        account = self.accounts[account_id]
        logger.info(f"Chose account: {account}")
        ticket_create_txn = xrpl.models.TicketCreate(
            account=account.address,
            ticket_count=ticket_count,
        )
        response = await submit_and_wait(ticket_create_txn, self.client, account.wallet)
        result = response.result
        logger.info(json.dumps(result, indent=2))

        ticket_seq = result["tx_json"]["Sequence"] + 1
        tix = [ticket_seq for ticket_seq in range(ticket_seq, ticket_seq + ticket_count)]
        account.tickets.update(tix)
        logger.info(f"Account {account.address} tickets: {account.tickets=}")
        # logger.info(f"Created tickets: {tickets}")
        # self.accounts
        return None

    async def use_random_ticket(self):
        account_ids = list(self.accounts)
        len_account_ids = len(account_ids)
        logger.info(f"Length of account_ids: {len_account_ids}")
        for aid in account_ids:
            logger.info(f"{self.accounts[aid].tickets}")
            len_tickets = len(self.accounts[aid].tickets)
            if len_tickets > 0:
                account_id = aid
                logger.info(f"Found {aid} to have tickets {self.accounts[aid].tickets}")
                break
            else:
                logger.info(f"removing {aid} from list")
                account_ids.remove(aid)
                logger.info(f"list now {len(account_ids)} long")

        # Our ticket holder is the source account
        src = self.accounts[account_id]
        logger.info(f"Ticket holder: {src.address}")
        # Use a random ticker of theirs
        ticket_sequence = choice(src.tickets)
        logger.info(f"Using ticket: {ticket_sequence}")
        # Pick a destination account that's not the source
        account_ids = list(self.accounts)
        account_ids.remove(account_id)
        dst = choice(account_ids)

        amount = str(1_000_000)
        payment_txn = xrpl.models.Payment(
            account=src.address,
            destination=dst,
            amount=amount,
            sequence=0,
            ticket_sequence=ticket_sequence,
        )
        return await submit_and_wait(payment_txn, self.client, src.wallet)

def create_app(workload: Workload) -> FastAPI:
    app = FastAPI()

    def get_workload():
        return workload

    @app.get("/accounts")
    def get_accounts(w: Workload = Depends(get_workload)):
        accounts = w.get_accounts()
        for a in accounts:
            logger.info(a)
        return {}

    @app.get("/nft/list")
    def get_nfts(w: Workload = Depends(get_workload)):
        nfts = w.get_nfts()
        for n in nfts:
            logger.info(n)
        return {}

    @app.get("/nft/mint/random")
    async def mint_random_nft(w: Workload = Depends(get_workload)):
        return await w.mint_random_nft()

    @app.get("/nft/create_offer/random")
    async def create_random_nft_offer(w: Workload = Depends(get_workload)):
        return await w.create_random_nft_offer()

    @app.get("/nft/burn/random")
    async def burn_nft(w: Workload = Depends(get_workload)):
        return await w.nft_burn_random()

    @app.get("/pay")
    async def payment_random(w: Workload = Depends(get_workload)):
        return await w.pay()

    @app.get("/payment/random")
    async def make_payment(w: Workload = Depends(get_workload)):
        return await w.payment_random()

    @app.get("/tickets/create/random")
    async def create_ticket(w: Workload = Depends(get_workload)):
        return await w.create_ticket()

    @app.get("/tickets/use/random")
    async def use_random_ticket(w: Workload = Depends(get_workload)):
        return await w.use_random_ticket()

    ## This requires issued currencies
    # @app.get("/offers/create/random")
    # async def cancel_create_random_offer(w: Workload = Depends(get_workload)):
    #     return await w.cancel_create_random_offer()
    # @app.get("/offers/cancel/random")
    # async def cancel_random_offer(w: Workload = Depends(get_workload)):
    #     return await w.cancel_random_offer()

    return app


    # @app.get("/ticket/use")
    # def

def main():
    print("IN main()")
    logger.info("Loaded config from %s", config_file)
    conf = conf_file["workload"]
    logger.info("Config %s", json.dumps(conf, indent=2))
    workload = Workload(conf)
    app = create_app(workload)
    uvicorn.run(app, host="0.0.0.0", port=8000)
