import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import StrEnum, auto
import httpx
from typing import Protocol
import xrpl
from collections import Counter, deque
import time
from xrpl.core.binarycodec import encode, encode_for_signing
from xrpl.core.keypairs import sign
from xrpl.asyncio.clients import AsyncJsonRpcClient
from xrpl.models import IssuedCurrency, Transaction, SubmitOnly
from xrpl.asyncio.ledger import get_latest_validated_ledger_sequence
from xrpl.wallet import Wallet
from xrpl.core.keypairs import generate_seed
from xrpl import CryptoAlgorithm
from xrpl.models.transactions import (
    AccountSet,
    AccountSetAsfFlag,
    Payment,
)
from typing import Any
from xrpl.models.transactions import AccountSet, AccountSetAsfFlag
from typing import Any
from xrpl.models.requests import (
    AccountInfo,
    Ledger,
    Tx,
    ServerState,
)
import sys
import logging
from workload.txn_factory.builder import TxnContext, TxnDefaults, generate_txn
import multiprocessing
import workload.constants as C

num_cpus = multiprocessing.cpu_count()

log = logging.getLogger("workload.core")


# # TODO: Temporary constant store
# C = SimpleNamespace(
#     default_create_amount=int(100 * 1e6),
#     max_create_amount=int(100e6 * 1e6), # alot?
#     horizon=20,  # If it's not validated/failed after 20 ledgers it's gone...
#     rpc_timeout=2.0,
#     submit_timeout=20,
#     lock_timeout=2.0,
# )

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

log = logging.getLogger("workload")

FINAL_STATES = {"VALIDATED", "REJECTED", "EXPIRED"}


@dataclass(slots=True)
class PendingTx:
    tx_hash: str
    signed_blob_hex: str
    account: str
    sequence: int | None
    last_ledger_seq: int
    transaction_type: C.TxType | None
    created_ledger: int
    state: C.TxState = C.TxState.CREATED
    attempts: int = 0
    engine_result_first: str | None = None
    validated_ledger: int | None = None
    meta_txn_result: str | None= None
    created_at: float = field(default_factory=time.time)
    finalized_at: float | None = None


class Store(Protocol):
    async def upsert(self, p: PendingTx) -> None: ...
    async def get(self, tx_hash: str) -> PendingTx | None: ...
    async def mark(self, tx_hash: str, **fields) -> None: ...
    async def rekey(self, old_hash: str, new_hash: str) -> None: ...
    async def find_by_state(self, *states: C.TxState) -> list[PendingTx]: ...
    async def all(self) -> list[PendingTx]: ...


class InMemoryStore:
    """Current snapshot of transaction states and validation history for metrics."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._records: dict[str, dict] = {}
        self.validations: deque[ValidationRecord] = deque(maxlen=5000)
        self.count_by_state: dict[str, int] = {}
        self.validated_by_source: dict[str, int] = {}

    def _recount(self) -> None:
        # per-state tallies
        self.count_by_state = Counter(rec.get("state", "UNKNOWN") for rec in self._records.values())
        # how many VALIDATEDs came from which path
        self.validated_by_source = Counter(v.src for v in self.validations)

    async def update_record(self, tx: dict) -> None:
        """Insert or update a flat transaction record and recompute metrics."""
        txh = tx.get("tx_hash")
        if not txh:
            raise ValueError("update_record() requires 'tx_hash'")
        async with self._lock:
            self._records[txh] = tx
            self._recount()

    async def get(self, tx_hash: str) -> dict | None:
        async with self._lock:
            return self._records.get(tx_hash)

    async def mark(self, tx_hash: str, *, source: str | None = None, **fields) -> None:
        """
        Update or insert a transaction record.

        tx_hash:
            Unique transaction identifier.
        source:
            Origin of the update ("ws", "poll", etc.).
        **fields:
            Additional fields to merge (e.g., state, validated_ledger, meta_txn_result).

        Behavior:
        - Creates/updates the flat record under tx_hash.
        - On first transition to a final state (VALIDATED/REJECTED/EXPIRED), stamps 'finalized_at'.
        - When state == VALIDATED, appends a ValidationRecord(txn, ledger, src).
        - Recomputes per-state and per-source counters.
        """
        async with self._lock:
            rec = self._records.get(tx_hash, {})
            rec.update(fields)
            if source is not None:
                rec["source"] = source

            state = rec.get("state")
            if isinstance(state, C.TxState):        # normalize enum to string
                state = state.name
                rec["state"] = state

            if state in FINAL_STATES:
                rec.setdefault("finalized_at", time.time())
                # Only VALIDATED gets validation history
                if state == "VALIDATED":
                    self.validations.append(
                        ValidationRecord(
                            txn=tx_hash,
                            seq=rec.get("validated_ledger") or 0,
                            src=source or rec.get("source", "unknown"),
                        )
                    )

            self._records[tx_hash] = rec
            self._recount()

    async def rekey(self, old_hash: str, new_hash: str) -> None:
        """
        Replace a record's key when a tx's canonical hash changes.
        - Moves the flat record from old_hash -> new_hash.
        - Updates inner 'tx_hash' field if present.
        - No-op if old_hash doesn't exist.
        """
        async with self._lock:
            rec = self._records.pop(old_hash, None)
            if rec is None:
                return
            rec["tx_hash"] = new_hash
            self._records[new_hash] = rec
            # (no recount needed unless you want to)

    # Snapshot-oriented queries (flat dicts). Live PendingTx queries belong on Workload.
    async def find_by_state(self, *states: C.TxState | str) -> list[dict]:
        """Return flat records whose 'state' matches any of the given states."""
        wanted = {s.name if isinstance(s, C.TxState) else s for s in states}
        async with self._lock:
            return [rec for rec in self._records.values() if rec.get("state") in wanted]

    async def all_records(self) -> list[dict]:
        async with self._lock:
            return list(self._records.values())

    def snapshot_stats(self) -> dict:
        return {
            "by_state": dict(self.count_by_state),
            "validated_by_source": dict(self.validated_by_source),
            "total_tracked": len(self._records),
            "recent_validations": len(self.validations),
        }

@dataclass
class AccountRecord:
    lock: asyncio.Lock
    next_seq: int | None = None

class ValidationSrc(StrEnum):
    POLL = auto()
    WS = auto()

@dataclass
class ValidationRecord:
    txn: str
    seq: int
    src: str

def _sha512half(b: bytes) -> bytes:
    return hashlib.sha512(b).digest()[:32]


def _txid_from_signed_blob_hex(signed_blob_hex: str) -> str:
    # XRPL txid = SHA512Half(0x54584E00 || signed_bytes)
    return _sha512half(bytes.fromhex("54584E00") + bytes.fromhex(signed_blob_hex)).hex().upper()


def issue_currencies(issuer: str, currency_code: list[str]) -> list[IssuedCurrency]:
    issued_currencies = [IssuedCurrency.from_dict(dict(issuer=issuer, currency=cc)) for cc in currency_code]
    return issued_currencies


class Workload:
    def __init__(self, config: dict, client: AsyncJsonRpcClient, *, store: Store | None = None):
        self.config = config
        self.client = client

        # TODO: Load from pre-generated accounts.json
        self.funding_wallet = Wallet.from_seed("snoPBrXtMeMyMHUVTgbuqAfg1SUTb", algorithm=xrpl.CryptoAlgorithm.SECP256K1)

        self.accounts: dict[str, AccountRecord] = {}
        self.wallets:  dict[str, Wallet] = {}
        self.gateways: list[Wallet] = []
        self.users:    list[Wallet] = []

        # Live txns that are going on. Not finalized yet. Go to self.store after we figure it out.
        self.pending: dict[str, PendingTx] = {}

        # tracks the recorded state of all transactions (past and present)—
        self.store: Store = store or InMemoryStore()

        self._fee_cache: int | None = None
        self._fee_lock = asyncio.Lock()

        # Configure the initial currencies available.
        self._currencies = issue_currencies(self.funding_wallet.address, self.config["currencies"]["codes"][:4])

        # Finally, set up the txn_context for generic txn use.
        self.ctx = self.configure_txn_context(wallets=self.wallets,
                                              funding_wallet=self.funding_wallet,
                                              defaults=None,
                                              )

    # Set up the txn_context if we want random transactons
    def configure_txn_context(
        self,
        *,
        funding_wallet: "Wallet",
        wallets: dict[str, "Wallet"],
        currencies: list["IssuedCurrency"] | None = None,
        defaults: TxnDefaults | None = None,
    ) -> TxnContext:
        currs = currencies if currencies is not None else self._currencies
        if not currs:
            raise ValueError("No currencies configured")
        return TxnContext.build(
            funding_wallet=funding_wallet,
            wallets=wallets,
            currencies=currs,
            defaults=defaults,
            base_fee_drops=self._open_ledger_fee,
            next_sequence=self.alloc_seq,
        )

    # Will it be sufficient to do this every time an account is created? or intermittently and mark some accounts as
    # not usable yet?
    def update_txn_context(self):
        """Re-builds the transaction context with the current list of wallets."""
        log.info(f"Updating txn_context with {len(self.wallets)} wallets.")
        self.ctx = self.configure_txn_context(wallets=list(self.wallets.values()), funding_wallet=self.funding_wallet)
    # Use Fee for dynamic value when submitting txns with possibility of fee-escalation.
    # async def _open_ledger_fee(self) -> int:
    #     async with self._fee_lock:
    #         r = await self.client.request(Fee())
    #         self._fee_cache = int(r.result["drops"]["open_ledger_fee"])
    #         return self._fee_cache

    async def _latest_validated_ledger(self) -> int:
        return await get_latest_validated_ledger_sequence(client=self.client)

    def _record_for(self, addr: str) -> AccountRecord:
        rec = self.accounts.get(addr)
        if rec is None:
            rec = AccountRecord(lock=asyncio.Lock(), next_seq=None)
            self.accounts[addr] = rec
        return rec

    async def _rpc(self, req, *, t=C.RPC_TIMEOUT):
        return await asyncio.wait_for(self.client.request(req), timeout=t)

    async def alloc_seq(self, addr: str) -> int:
        rec = self._record_for(addr)
        if rec.next_seq is None:
            ai = await self._rpc(AccountInfo(account=addr, ledger_index="current", strict=True))
            rec.next_seq = ai.result["account_data"]["Sequence"]

        async with rec.lock:
            assert rec.next_seq is not None
            s = rec.next_seq
            rec.next_seq += 1
            return s

    async def _open_ledger_fee(self) -> int:
        ss = await self._rpc(ServerState(), t=2.0)
        base = float(ss.result["state"]["validated_ledger"]["base_fee"])
        return int(base * 10)  # TODO: Tie this down, need to be able to handle fee elevation.

    async def _last_ledger_sequence_offset(self, off: int) -> int:
        ss = await self._rpc(ServerState(), t=2.0)
        return ss.result["state"]["validated_ledger"]["seq"] + off

    async def _current_ledger_index(self) -> int:
        r = await self.client.request(Ledger(ledger_index="current", transactions=False, expand=False))
        return int(r.result["ledger_index"])

    async def record_created(self, p: PendingTx) -> None:
        # store pending txn keyed by local hash
        self.pending[p.tx_hash] = p
        p.state = C.TxState.CREATED
        await self.store.update_record({
            "tx_hash": p.tx_hash,
            "state": p.state.name,
            "created_ledger": p.created_ledger,
        })

    async def record_submitted(self, p: PendingTx, engine_result: str | None, srv_txid: str | None):
        old = p.tx_hash
        new_hash = srv_txid or old
        if srv_txid and srv_txid != old:
            self.pending[new_hash] = self.pending.pop(old, p)
            p.tx_hash = new_hash
            await self.store.rekey(old, new_hash)
        p.state = C.TxState.SUBMITTED
        p.engine_result_first = p.engine_result_first or engine_result
        self.pending[new_hash] = p
        await self.store.mark(new_hash, state=C.TxState.SUBMITTED, engine_result_first=p.engine_result_first)

    async def record_validated(self, rec: ValidationRecord, meta_result: str | None = None) -> dict:
        """Apply validation to live tx (if present) and always persist to the store."""
        p = self.pending.get(rec.txn)
        if p:
            p.state = C.TxState.VALIDATED
            p.validated_ledger = rec.seq
            p.meta_txn_result = meta_result
        else:
            log.debug("record_validated: tx not in pending (race or already finalized): %s", rec.txn)

        await self.store.mark(
            rec.txn,
            state=C.TxState.VALIDATED,
            validated_ledger=rec.seq,
            meta_txn_result=meta_result,
            source=rec.src,
        )
        log.info("txn %s validated at ledger %s via %s", rec.txn, rec.seq, rec.src)
        return {
            "tx_hash": rec.txn,
            "ledger_index": rec.seq,
            "source": rec.src,
            "meta_result": meta_result,
        }


    async def record_expired(self, tx_hash: str):
        if tx_hash in self.pending:
            p = self.pending[tx_hash]
            p.state = C.TxState.EXPIRED
            await self.store.mark(tx_hash, state=C.TxState.EXPIRED)
            # self.pending.pop(tx_hash, None) # see if this gets out of hand?

    def find_by_state(self, *states: C.TxState) -> list[PendingTx]:
        return [p for p in self.pending.values() if p.state in set(states)]

    async def build_sign_and_track(self, txn: Transaction, wallet: Wallet, horizon: int = C.HORIZON) -> PendingTx:
        created_li = (await self._rpc(ServerState(), t=2.0)).result["state"]["validated_ledger"]["seq"] #TODO: Constant
        lls = created_li + horizon
        tx = txn.to_xrpl()
        if tx.get("Flags") == 0:
            del tx["Flags"]

        need_seq = "TicketSequence" not in tx and not tx.get("Sequence")
        need_fee = not tx.get("Fee")

        seq = await self.alloc_seq(wallet.address) if need_seq else tx.get("Sequence")
        fee = await self._open_ledger_fee() if need_fee else int(tx["Fee"])

        rec = self.accounts[wallet.address]
        async with asyncio.timeout(C.LOCK_TIMEOUT):
            log.debug("lock enter %s", wallet.address)
            async with rec.lock:
                if need_seq:
                    tx["Sequence"] = seq
                if need_fee:
                    tx["Fee"] = str(fee)
                tx["SigningPubKey"] = wallet.public_key
                tx["LastLedgerSequence"] = lls

                signing_blob = encode_for_signing(tx)
                to_sign = signing_blob if isinstance(signing_blob, str) else signing_blob.hex()
                tx["TxnSignature"] = sign(to_sign, wallet.private_key)
                signed_blob_hex = encode(tx)
                local_txid = _txid_from_signed_blob_hex(signed_blob_hex)
            log.debug("lock exit %s", wallet.address)

        p = PendingTx(
            tx_hash=local_txid,
            signed_blob_hex=signed_blob_hex,
            account=tx["Account"],
            sequence=tx.get("Sequence"),
            last_ledger_seq=lls,
            transaction_type=tx.get("TransactionType"),
            created_ledger=created_li,
        )
        await self.record_created(p)
        return p

    async def submit_pending(self, p: PendingTx, timeout: float = C.SUBMIT_TIMEOUT) -> dict | None:
        # If the txn is in this state already we've got nothing to do but why are we here in the first place?
        if p.state in {C.TxState.VALIDATED, C.TxState.REJECTED, C.TxState.EXPIRED}:
            log.debug("%s not active txn!", p)
            return None

        try:
            p.attempts += 1
            resp = await asyncio.wait_for(self.client.request(SubmitOnly(tx_blob=p.signed_blob_hex)), timeout=timeout)
            res = resp.result
            er = res.get("engine_result")

            if p.engine_result_first is None:
                p.engine_result_first = er

            if isinstance(er, str) and er.startswith(("tem", "tef")):
                p.state = C.TxState.REJECTED
            else:
                p.state = C.TxState.SUBMITTED
                log.debug("Submitted")
                log.debug("%s", p)

            srv_txid = res.get("tx_json", {}).get("hash")
            if isinstance(srv_txid, str) and srv_txid and srv_txid != p.tx_hash:
                self.pending[srv_txid] = self.pending.pop(p.tx_hash, p)
                p.tx_hash = srv_txid
            await self.record_submitted(p, engine_result=er, srv_txid=srv_txid)
            return res

        except asyncio.TimeoutError:
            p.state = C.TxState.FAILED_NET
            log.error("timeout")
            self.pending[p.tx_hash] = p
            return {"engine_result": "timeout"}
        except Exception as e:
            p.state = C.TxState.FAILED_NET
            self.pending[p.tx_hash] = p
            log.error("submit error tx=%s: %s", p.tx_hash, e)
            return {"engine_result": "error", "message": str(e)}

    def log_validation(self, tx_hash, ledger_index, result, validation_src):
        log.info("Validated via %s tx=%s li=%s result=%s", validation_src, tx_hash, ledger_index, result) # FIX: DEbug only...

    # TODO: Default constants
    async def check_finality(self, p: PendingTx, grace: int = 2) -> tuple[C.TxState, int | None]:
        try:
            txr = await self.client.request(Tx(transaction=p.tx_hash))
            if txr.is_successful() and txr.result.get("validated"):
                li = int(txr.result["ledger_index"])
                result = txr.result["meta"]["TransactionResult"]
                await self.store.mark(
                    p.tx_hash,
                    state=C.TxState.VALIDATED,
                    validated_ledger=li,
                    meta_txn_result=result,
                    )
                p.state, p.validated_ledger, p.meta_txn_result = C.TxState.VALIDATED, li, result
                await self.record_validated(ValidationRecord(p.tx_hash, li, ValidationSrc.POLL), result)
                return p.state, li
        except Exception:
            log.error("Houston, we have a %s", "major problem", exc_info=True)
            pass # TODO: NO!!!

        latest_val = await self._latest_validated_ledger()
        if latest_val > (p.last_ledger_seq + grace):
            await self.store.mark(p.tx_hash, state=C.TxState.EXPIRED)
            p.state = C.TxState.EXPIRED
            return p.state, None

        if p.state != C.TxState.SUBMITTED:
            p.state = C.TxState.RETRYABLE
            await self.store.mark(p.tx_hash, state=p.state)
        return p.state, None

    async def submit_signed_tx_blobs(self, items: list):
        def _to_blob(x):
            if isinstance(x, str): return x
            if isinstance(x, (tuple, list)): return x[0]
            blob = getattr(x, "signed_blob_hex", None)
            if isinstance(blob, str): return blob
            raise TypeError(f"unsupported tx item: {type(x)}")
        blobs = [_to_blob(i) for i in items]
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self.client.request(SubmitOnly(tx_blob=b))) for b in blobs]
        return [t.result().result for t in tasks]

    async def _is_account_active(self, address: str) -> bool:
        try:
            r = await self.client.request(AccountInfo(account=address, ledger_index="validated"))
            return r.is_successful()
        except Exception:
            return False  # TODO: Say something?

    async def _ensure_funded(self, wallet: Wallet, amt_drops: str):
        """Fund the wallet from the funding_wallet account if it isn't active yet."""
        if await self._is_account_active(wallet.address):
            return
        amt_drops = str(amt_drops)
        fund_tx = Payment(
            account=self.funding_wallet.address,
            destination=wallet.address,
            amount=amt_drops,
        )

        p = await self.build_sign_and_track(fund_tx, self.funding_wallet)
        await self.submit_pending(p)
        log.debug(f"Funded {wallet.address} with {int(xrpl.utils.drops_to_xrp(amt_drops))} XRP")

    async def _acctset_flags(self, wallet: Wallet, *, require_auth=False, default_ripple=True):

        flags = []
        if require_auth:
            flags.append(AccountSetAsfFlag.ASF_REQUIRE_AUTH)
        if default_ripple:
            flags.append(AccountSetAsfFlag.ASF_DEFAULT_RIPPLE)
        for f in flags:
            t = AccountSet(account=wallet.address, set_flag=f)
            p = await self.build_sign_and_track(t, wallet)
            await self.submit_pending(p)

    # TODO: Default constants
    async def wait_for_validation(self, tx_hash: str, *, overall: float = 15.0, per_rpc: float = 2.0) -> dict:
        from xrpl.models.requests import Tx
        try:
            async with asyncio.timeout(overall):
                while True:
                    r = await asyncio.wait_for(self.client.request(Tx(transaction=tx_hash)), timeout=per_rpc)
                    if r.result.get("validated"):
                        return r.result
                    await asyncio.sleep(0.5)
        except TimeoutError:
            return {"validated": False, "timeout": True}

    # TODO: Default constants
    async def bootstrap_gateway(self, w, *, drops=1_000_000_000, require_auth=False, default_ripple=False):
        fund = Payment(account=w.address, destination=w.address, amount=str(drops))  # or from funder→w
        p0 = await self.build_sign_and_track(fund, self.funding_wallet)
        await self.submit_pending(p0)

        flags = []
        if require_auth:
            flags.append(AccountSetAsfFlag.ASF_REQUIRE_AUTH)
        if default_ripple:
            flags.append(AccountSetAsfFlag.ASF_DEFAULT_RIPPLE)

        pendings = []
        for f in flags:
            tx = AccountSet(account=w.address, set_flag=f)
            p = await self.build_sign_and_track(tx, w)  # allocator hands next Sequence
            pendings.append(p)
            await self.submit_pending(p)

        for p in [p0, *pendings]:
            _ = await self.wait_for_validation(p.tx_hash, overall=15.0)

    async def _apply_gateway_flags(self, *, req_auth: bool, def_ripple: bool) -> dict[str, Any]:
        """Apply per-gateway account flags. XRPL allows one asf per AccountSet, so we send one tx per flag."""

        flags: list[AccountSetAsfFlag] = []
        if req_auth:
            flags.append(AccountSetAsfFlag.ASF_REQUIRE_AUTH)
        if def_ripple:
            flags.append(AccountSetAsfFlag.ASF_DEFAULT_RIPPLE)

        if not flags or not self.gateways:
            return {"applied": 0, "results": []}

        results: list[dict[str, Any]] = []
        for w in self.gateways:
            for f in flags:
                tx = AccountSet(account=w.address, set_flag=f)
                p = await self.build_sign_and_track(tx, w)
                res = await self.submit_pending(p, timeout=getattr(self, "rpc_timeout", 3.0))
                er = (res or {}).get("engine_result")
                txh = (res or {}).get("tx_json", {}).get("hash") if res else None

                # record outcome snapshot; validation is handled by your periodic checker
                results.append({
                    "address": w.address,
                    "flag": f.name,
                    "engine_result": er,
                    "tx_hash": txh,
                    "state": p.state.name,
                })

                if er != "tesSUCCESS":
                    log.error("AccountSet failed addr=%s flag=%s res=%s", w.address, f.name, res)

        return {"applied": len(flags) * len(self.gateways), "results": results}

    # TODO: Default constants
    async def wait_until_validated(self, tx_hash: str, *, overall: float = 15.0, per_rpc: float = 2.0) -> dict[str, Any]:
        """Block until tx validated, rejected, or timeout. Returns the final Tx result dict."""

        try:
            async with asyncio.timeout(overall):
                while True:
                    r = await asyncio.wait_for(self.client.request(Tx(transaction=tx_hash)), timeout=per_rpc)
                    result = r.result
                    if result.get("validated"):
                        tx_meta = result.get("meta", {})
                        meta_result = tx_meta.get("TransactionResult") if isinstance(tx_meta, dict) else None
                        ledger_index = result.get("ledger_index")
                        await self.record_validated(tx_hash, ledger_index=ledger_index, meta_result=meta_result)
                        return result
                    await asyncio.sleep(0.5)
        except TimeoutError:
            log.warning("Validation timeout tx=%s after %.1fs", tx_hash, overall)
            return {"validated": False, "timeout": True}

    async def submit_random_txn(self, n: int | None = None):
        txn = await generate_txn(self.ctx)
        pending_txn = await self.build_sign_and_track(txn, self.wallets[txn.account])
        x = await self.submit_pending(pending_txn)
        log.debug(f"Submitting random {txn.transaction_type.name.title().replace("_", " ")} txn.")
        log.debug(x)
        return x

    async def create_transaction(self, transacton: str):
        txn = await generate_txn(self.ctx, transacton)
        pending_txn = await self.build_sign_and_track(txn, self.wallets[txn.account])
        res = await self.submit_pending(pending_txn)
        return res

    async def create_account(self, account_info: dict, wait: bool = False) -> dict[str, Any]:

        # TODO: Get from constants
        if not hasattr(self, "default_create_amount"):
            self.default_create_amount = 1_000_000_000
        if not hasattr(self, "max_create_amount"):
            self.max_create_amount = 1_000_000_000_000
        if not hasattr(self, "rpc_timeout"):
            self.rpc_timeout = 3.0

        address = account_info.get("address")
        seed = account_info.get("seed")
        algorithm = account_info.get("algorithm")
        amount = account_info.get("amount")  # drops

        if isinstance(algorithm, CryptoAlgorithm):
            algo = algorithm
        else:
            algo = CryptoAlgorithm(str(algorithm or "secp256k1").lower())

        if address and seed:
            w_check = Wallet.from_seed(seed, algorithm=algo)
            if w_check.classic_address != address:
                raise ValueError("address does not match seed")
            wallet = w_check
        elif seed:
            wallet = Wallet.from_seed(seed, algorithm=algo)
            address = wallet.classic_address
        elif address:
            # Still can track unfunded accounts
            wallet = None
        else:
            wallet = Wallet.from_seed(generate_seed(algorithm=algo), algorithm=algo)
            address = wallet.classic_address

        if wallet:
            self.wallets[address] = wallet
            self._record_for(address)

        drops = int(amount) if amount is not None else C.DEFAULT_CREATE_AMOUNT
        if drops < 0:
            raise ValueError("amount can't be negative!")
        if drops > C.MAX_CREATE_AMOUNT:
            drops = C.MAX_CREATE_AMOUNT

        funded = False
        tx_hash: str | None = None
        validated = False
        ledger_index: int | None = None
        meta_txn_result: str | None = None

        if drops > 0 and getattr(self, "funding_wallet", None):
            pay = Payment(account=self.funding_wallet.address, destination=address, amount=str(drops))
            p = await self.build_sign_and_track(pay, self.funding_wallet)
            submit_res = await self.submit_pending(p, timeout=self.rpc_timeout)
            er = (submit_res or {}).get("engine_result")
            tx_hash = (submit_res or {}).get("tx_json", {}).get("hash")
            funded = (er == "tesSUCCESS")

            if wait and tx_hash:
                log.info("Waiting for transaction to be validated.")
                final = await self.wait_until_validated(tx_hash, overall=15.0, per_rpc=2.0) # TODO: Constantes
                validated = bool(final.get("validated"))
                ledger_index = final.get("ledger_index") or final.get("ledger_index_min")
                meta = final.get("meta")
                if isinstance(meta, dict):
                    meta_txn_result = meta.get("TransactionResult")

        return {
            "address": address,
            "seed": seed,
            "algorithm": algo.name,
            "funded": funded,
            "funding_drops": drops if funded else 0,
            "tx_hash": tx_hash,
            "validated": validated,
            "ledger_index": ledger_index,
            "meta_txn_result": meta_txn_result,
        }
    # ============================================== #
    # ============ Initialization Stuff============= #
    # ============================================== #

    async def init_participants(self, *, gateway_cfg: dict[str, Any], user_cfg: dict[str, Any]) -> dict:
        out_gw, out_us = [], []
        req_auth = gateway_cfg["require_auth"]
        def_ripple = gateway_cfg["default_ripple"]

        log.info(f"Funding {(g := gateway_cfg['number'])} gateways")
        for _ in range(g):
            w = Wallet.create()
            self.wallets[w.address] = w
            self._record_for(w.address)
            await self._ensure_funded(w, self.config["gateways"]["default_balance"])
            self.gateways.append(w)
            out_gw.append(w.address)

        log.info(f"Funding {(u := user_cfg["number"])} users")
        for _ in range(u):
            w = Wallet.create()
            self.wallets[w.address] = w
            self._record_for(w.address)
            await self._ensure_funded(w, self.config["users"]["default_balance"])
            self.users.append(w)
            out_us.append(w.address)

        if req_auth or def_ripple:
            await self._apply_gateway_flags(req_auth=req_auth, def_ripple=def_ripple)
        return {"gateways": out_gw, "users": out_us}

    # ============================================== #
    # =============== Utility Stuff ================ #
    # ============================================== #

    async def _post(self, url: str, payload: dict):
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(url, json=payload)
                response = resp.json()
            except Exception as e:
                pass
                # responses.append({"error": str(e)})
        try:
            response = resp.json()
        except:
            pass
        finally:
            return response

    async def validator_state(self, n: int):
        # FIX: this to work locally, in Docker we'd use the ${val_name}{index}
        import subprocess
        val = f"val{n}"
        cmd = ["docker", "inspect", "-f", "'{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "]
        cmd.append(val)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        val_ip = result.stdout.strip().replace("'","")
        rpc_port = 5005
        return f"http://{val_ip}:{rpc_port}"

    def snapshot_pending(self, *, open_only: bool = True) -> list[dict]:
        OPEN_STATES = {C.TxState.CREATED, C.TxState.SUBMITTED, C.TxState.RETRYABLE, C.TxState.FAILED_NET} # TODO: Move
        out = []
        for txh, p in self.pending.items():
            if open_only and p.state not in OPEN_STATES:
                continue
            out.append({
                "tx_hash": txh,
                "state": p.state.name,
                "account": p.account,
                "sequence": p.sequence,
                "last_ledger_seq": p.last_ledger_seq,
                "created_ledger": p.created_ledger,
                "attempts": p.attempts,
                "engine_result_first": p.engine_result_first,
                "validated_ledger": p.validated_ledger,
                "meta_txn_result": p.meta_txn_result,
            })
        return out

    def snapshot_finalized(self) -> list[dict]:
        FINAL_STATES = {C.TxState.VALIDATED, C.TxState.REJECTED, C.TxState.EXPIRED}
        return [r for r in self.snapshot_pending(open_only=False) if r["state"] in {s.name for s in FINAL_STATES}]

    def snapshot_failed(self) -> list[dict[str, Any]]:
        failed_states = {"REJECTED", "EXPIRED", "FAILED_NET"}
        return [r for r in self.snapshot_pending() if r["state"] in failed_states]

    def snapshot_stats(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for p in self.pending.values():
            state = p.state.name
            by_state[state] = by_state.get(state, 0) + 1

        return {
            "total_tracked": len(self.pending),
            "by_state": by_state,
            "gateways": len(self.gateways),
            "users": len(self.users),
        }

    def snapshot_tx(self, tx_hash: str) -> dict[str, Any]:
        p = self.pending.get(tx_hash)
        if not p:
            return {}
        return {
            "tx_hash": p.tx_hash,
            "state": p.state.name,
            "account": p.account,
            "sequence": p.sequence,
            "last_ledger_seq": p.last_ledger_seq,
            "created_ledger": p.created_ledger,
            "attempts": p.attempts,
            "engine_result_first": p.engine_result_first,
            "validated_ledger": p.validated_ledger,
            "meta_txn_result": p.meta_txn_result,
        }


async def periodic_finality_check(w: Workload, interval: int = 5):
    while True:
        try:
            for p in w.find_by_state(C.TxState.SUBMITTED):
                try:
                    await w.check_finality(p)
                except Exception:
                    log.exception("[finality] check failed for %s", getattr(p, "tx_hash", p))
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[finality] outer loop error; continuing")
            await asyncio.sleep(0.5)
