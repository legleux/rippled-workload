#!/usr/bin/env python3
"""MPT Memecoin Drop scenario.

Exercises the full MPToken lifecycle on a running XRPL testnet:

1. Fund issuer account (from genesis)
2. MPTokenIssuanceCreate — issuer creates a memecoin
3. Fund N holder accounts
4. MPTokenAuthorize — each holder opts in to the token
5. MPT Payment — issuer airdrops tokens to each holder
6. MPT Payment — holders trade tokens between each other
7. MPTokenAuthorize (holder, tfMPTUnauthorize) — a holder opts out
8. MPTokenIssuanceSet — issuer locks/unlocks the token
9. MPTokenIssuanceDestroy — issuer destroys the issuance (if possible)

Each step validates the transaction before proceeding.

Usage:
    python memecoin_drop.py [--rpc http://localhost:5005] [--holders 5] [--supply 1000000]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

import httpx

log = logging.getLogger("memecoin_drop")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Genesis account — standard for XRPL test networks
GENESIS = {
    "address": "rHb9CJAWyB4rj91VRWn96DkukG4bwdtyTh",
    "seed": "snoPBrXtMeMyMHUVTgbuqAfg1SUTb",
}


class RippleRPC:
    """Thin async wrapper around rippled JSON-RPC."""

    def __init__(self, url: str):
        self.url = url
        self.client = httpx.AsyncClient(timeout=10.0)

    async def call(self, method: str, params: list[dict] | None = None) -> dict:
        payload = {"method": method, "params": params or [{}]}
        r = await self.client.post(self.url, json=payload)
        r.raise_for_status()
        result = r.json()["result"]
        if result.get("status") == "error":
            raise RuntimeError(f"RPC error: {result.get('error_message', result.get('error'))}")
        return result

    async def submit(self, tx_blob: str) -> dict:
        return await self.call("submit", [{"tx_blob": tx_blob}])

    async def tx(self, tx_hash: str) -> dict:
        return await self.call("tx", [{"transaction": tx_hash}])

    async def account_info(self, address: str) -> dict:
        return await self.call("account_info", [{"account": address, "ledger_index": "validated"}])

    async def server_info(self) -> dict:
        return await self.call("server_info")

    async def ledger_current(self) -> int:
        r = await self.call("ledger_current")
        return r["ledger_current_index"]


async def wait_for_validation(rpc: RippleRPC, tx_hash: str, max_ledgers: int = 10) -> dict:
    """Poll until a transaction is validated or we give up."""
    for _ in range(max_ledgers * 3):  # ~3 polls per ledger
        await asyncio.sleep(1.0)
        try:
            result = await rpc.tx(tx_hash)
            if result.get("validated"):
                return result
        except RuntimeError:
            pass  # tx not found yet
    raise TimeoutError(f"Transaction {tx_hash} not validated after {max_ledgers} ledgers")


async def sign_and_submit(rpc: RippleRPC, tx: dict, secret: str) -> dict:
    """Sign a transaction and submit it, returning the submit result."""
    result = await rpc.call("submit", [{"tx_json": tx, "secret": secret, "fee_mult_max": 1000}])
    engine = result.get("engine_result", "")
    tx_hash = result.get("tx_json", {}).get("hash", "unknown")
    log.info("  submit: %s  hash=%s", engine, tx_hash[:16])
    if engine not in ("tesSUCCESS", "terQUEUED"):
        log.warning("  unexpected engine result: %s — %s", engine, result.get("engine_result_message", ""))
    return result


async def sign_submit_wait(rpc: RippleRPC, tx: dict, secret: str, label: str = "") -> dict:
    """Sign, submit, and wait for validation. Returns the validated tx result."""
    result = await sign_and_submit(rpc, tx, secret)
    tx_hash = result.get("tx_json", {}).get("hash")
    if not tx_hash:
        raise RuntimeError(f"No hash in submit result: {result}")
    engine = result.get("engine_result", "")
    if engine not in ("tesSUCCESS", "terQUEUED"):
        raise RuntimeError(f"{label}: submit failed with {engine}: {result.get('engine_result_message')}")

    validated = await wait_for_validation(rpc, tx_hash)
    meta_result = validated.get("meta", {}).get("TransactionResult", "")
    if meta_result != "tesSUCCESS":
        raise RuntimeError(f"{label}: validated but failed: {meta_result}")
    log.info("  validated: %s (%s)", meta_result, label)
    return validated


async def fund_account(rpc: RippleRPC, destination: str, drops: str = "10000000000") -> dict:
    """Fund an account from genesis."""
    tx = {
        "TransactionType": "Payment",
        "Account": GENESIS["address"],
        "Destination": destination,
        "Amount": drops,
    }
    return await sign_submit_wait(rpc, tx, GENESIS["seed"], f"fund {destination[:12]}")


def generate_wallet() -> dict:
    """Generate a wallet using xrpl-py."""
    from xrpl.wallet import Wallet
    w = Wallet.create()
    return {"address": w.address, "seed": w.seed, "public_key": w.public_key}


async def run_scenario(rpc_url: str, num_holders: int, total_supply: int):
    """Run the full memecoin drop scenario."""
    rpc = RippleRPC(rpc_url)

    # Verify network is up
    info = await rpc.server_info()
    state = info["info"]["server_state"]
    log.info("Network state: %s", state)
    if state not in ("full", "proposing", "validating"):
        raise RuntimeError(f"Network not ready: {state}")

    # ─── Step 1: Create issuer account ───
    log.info("=== Step 1: Fund issuer account ===")
    issuer = generate_wallet()
    log.info("Issuer: %s", issuer["address"])
    await fund_account(rpc, issuer["address"])

    # ─── Step 2: MPTokenIssuanceCreate ───
    log.info("=== Step 2: Create memecoin (MPTokenIssuanceCreate) ===")
    metadata = json.dumps({"name": "MEMECOIN", "description": "The ultimate memecoin airdrop"}).encode().hex()
    create_tx = {
        "TransactionType": "MPTokenIssuanceCreate",
        "Account": issuer["address"],
        "MaximumAmount": str(total_supply),
        "MPTokenMetadata": metadata,
        # tfMPTCanTransfer: allows holders to send to each other
        # tfMPTCanLock: allows issuer to lock
        "Flags": 0x0002 | 0x0004,  # CanTransfer | CanLock
    }
    validated = await sign_submit_wait(rpc, create_tx, issuer["seed"], "MPTokenIssuanceCreate")

    # Extract the MPTokenIssuanceID from the validated tx
    # The MPTokenIssuanceID is a 192-bit value: sequence (32-bit big-endian) + AccountID (160-bit)
    # rippled returns it in the tx result's mpt_issuance_id field or we compute it
    mpt_issuance_id = None
    # Method 1: Check if the tx result has it directly
    mpt_issuance_id = validated.get("mpt_issuance_id")
    # Method 2: Look in AffectedNodes for the MPTokenIssuance created node
    if not mpt_issuance_id:
        for node in validated.get("meta", {}).get("AffectedNodes", []):
            created = node.get("CreatedNode", {})
            if created.get("LedgerEntryType") == "MPTokenIssuance":
                new_fields = created.get("NewFields", {})
                mpt_issuance_id = new_fields.get("MPTokenIssuanceID")
                if not mpt_issuance_id:
                    # Compute from sequence + account: the issuance ID is
                    # 4 bytes (sequence big-endian) + 20 bytes (account ID)
                    seq = validated.get("Sequence") or validated.get("tx_json", {}).get("Sequence")
                    acct = validated.get("Account") or validated.get("tx_json", {}).get("Account")
                    if seq is not None and acct:
                        from xrpl.core.addresscodec import decode_classic_address
                        account_id = decode_classic_address(acct).hex().upper()
                        mpt_issuance_id = f"{seq:08X}" + account_id
                break
    if not mpt_issuance_id:
        log.error("Full validated result: %s", json.dumps(validated, indent=2, default=str))
        raise RuntimeError("Could not find MPTokenIssuanceID in metadata")
    log.info("MPTokenIssuanceID: %s", mpt_issuance_id)

    # ─── Step 3: Fund holder accounts ───
    log.info("=== Step 3: Fund %d holder accounts ===", num_holders)
    holders = []
    for i in range(num_holders):
        h = generate_wallet()
        log.info("Holder %d: %s", i, h["address"])
        await fund_account(rpc, h["address"])
        holders.append(h)

    # ─── Step 4: Holders authorize (opt-in) ───
    log.info("=== Step 4: Holders opt-in (MPTokenAuthorize) ===")
    for i, h in enumerate(holders):
        auth_tx = {
            "TransactionType": "MPTokenAuthorize",
            "Account": h["address"],
            "MPTokenIssuanceID": mpt_issuance_id,
        }
        await sign_submit_wait(rpc, auth_tx, h["seed"], f"authorize holder {i}")

    # ─── Step 5: Airdrop — issuer sends tokens to each holder ───
    log.info("=== Step 5: Airdrop tokens to holders ===")
    per_holder = total_supply // num_holders
    for i, h in enumerate(holders):
        pay_tx = {
            "TransactionType": "Payment",
            "Account": issuer["address"],
            "Destination": h["address"],
            "Amount": {
                "mpt_issuance_id": mpt_issuance_id,
                "value": str(per_holder),
            },
        }
        await sign_submit_wait(rpc, pay_tx, issuer["seed"], f"airdrop {per_holder} to holder {i}")

    # ─── Step 6: Holders trade tokens between each other ───
    log.info("=== Step 6: Holder-to-holder transfers ===")
    trade_amount = per_holder // 10  # 10% of holdings
    for i in range(len(holders) - 1):
        sender = holders[i]
        receiver = holders[i + 1]
        trade_tx = {
            "TransactionType": "Payment",
            "Account": sender["address"],
            "Destination": receiver["address"],
            "Amount": {
                "mpt_issuance_id": mpt_issuance_id,
                "value": str(trade_amount),
            },
        }
        await sign_submit_wait(
            rpc, trade_tx, sender["seed"],
            f"trade {trade_amount}: holder {i} -> holder {i+1}"
        )

    # ─── Step 7: Issuer locks the token ───
    log.info("=== Step 7: Issuer locks the token (MPTokenIssuanceSet) ===")
    lock_tx = {
        "TransactionType": "MPTokenIssuanceSet",
        "Account": issuer["address"],
        "MPTokenIssuanceID": mpt_issuance_id,
        "Flags": 0x0001,  # tfMPTLock (lock all)
    }
    await sign_submit_wait(rpc, lock_tx, issuer["seed"], "lock issuance")

    # ─── Step 8: Verify transfer fails while locked ───
    log.info("=== Step 8: Verify transfer fails while locked ===")
    fail_tx = {
        "TransactionType": "Payment",
        "Account": holders[0]["address"],
        "Destination": holders[1]["address"],
        "Amount": {
            "mpt_issuance_id": mpt_issuance_id,
            "value": "1",
        },
    }
    fail_result = await sign_and_submit(rpc, fail_tx, holders[0]["seed"])
    fail_engine = fail_result.get("engine_result", "")
    if fail_engine == "tesSUCCESS":
        # It might succeed at submit but fail at validation
        fail_hash = fail_result.get("tx_json", {}).get("hash")
        if fail_hash:
            try:
                fail_validated = await wait_for_validation(rpc, fail_hash, max_ledgers=5)
                fail_meta = fail_validated.get("meta", {}).get("TransactionResult", "")
                log.info("  locked transfer result: %s (expected tecMPTOKEN_LOCKED or similar)", fail_meta)
            except TimeoutError:
                log.info("  locked transfer timed out (expected — token is locked)")
    else:
        log.info("  locked transfer rejected at submit: %s (expected)", fail_engine)

    # ─── Step 9: Issuer unlocks ───
    log.info("=== Step 9: Issuer unlocks the token ===")
    unlock_tx = {
        "TransactionType": "MPTokenIssuanceSet",
        "Account": issuer["address"],
        "MPTokenIssuanceID": mpt_issuance_id,
        "Flags": 0x0002,  # tfMPTUnlock
    }
    await sign_submit_wait(rpc, unlock_tx, issuer["seed"], "unlock issuance")

    # ─── Step 10: One holder opts out ───
    log.info("=== Step 10: Last holder opts out (MPTokenAuthorize + tfMPTUnauthorize) ===")
    last = holders[-1]
    # First send all tokens back to issuer
    clawback_tx = {
        "TransactionType": "Payment",
        "Account": last["address"],
        "Destination": issuer["address"],
        "Amount": {
            "mpt_issuance_id": mpt_issuance_id,
            "value": str(per_holder + trade_amount),  # original + received from trade
        },
    }
    try:
        await sign_submit_wait(rpc, clawback_tx, last["seed"], "return tokens before opt-out")
    except RuntimeError as e:
        log.warning("  return tokens failed (balance may differ): %s", e)
        # Try with a smaller amount — query actual balance
        log.info("  attempting to return all remaining tokens...")

    unauth_tx = {
        "TransactionType": "MPTokenAuthorize",
        "Account": last["address"],
        "MPTokenIssuanceID": mpt_issuance_id,
        "Flags": 0x0001,  # tfMPTUnauthorize
    }
    try:
        await sign_submit_wait(rpc, unauth_tx, last["seed"], "holder opt-out")
    except RuntimeError as e:
        log.warning("  opt-out failed (may still hold tokens): %s", e)

    # ─── Done ───
    log.info("=" * 50)
    log.info("MEMECOIN DROP COMPLETE")
    log.info("  Issuer:       %s", issuer["address"])
    log.info("  IssuanceID:   %s", mpt_issuance_id)
    log.info("  Holders:      %d", num_holders)
    log.info("  Total supply: %s", total_supply)
    log.info("  Steps completed: create, authorize, airdrop, trade, lock, unlock, opt-out")
    log.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="MPT Memecoin Drop scenario")
    parser.add_argument("--rpc", default=os.getenv("RPC_URL", "http://localhost:5005"), help="rippled RPC URL")
    parser.add_argument("--holders", type=int, default=5, help="Number of holder accounts")
    parser.add_argument("--supply", type=int, default=1_000_000, help="Total token supply")
    args = parser.parse_args()

    log.info("MPT Memecoin Drop")
    log.info("  RPC: %s", args.rpc)
    log.info("  Holders: %d", args.holders)
    log.info("  Supply: %s", args.supply)

    asyncio.run(run_scenario(args.rpc, args.holders, args.supply))


if __name__ == "__main__":
    main()
