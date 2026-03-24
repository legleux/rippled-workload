"""Batch and TicketCreate transaction builders."""

from random import choice

from xrpl.models.amounts import IssuedCurrencyAmount
from xrpl.models.transactions import (
    AccountSet,
    Batch,
    BatchFlag,
    Memo,
    NFTokenMint,
    Payment,
    TicketCreate,
    TransactionFlag,
    TrustSet,
)

from workload.constants import TxIntent
from workload.randoms import random, randrange
from workload.txn_factory.context import TxnContext, choice_omit


def build_ticket_create(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build a TicketCreate transaction to create tickets for an account.

    Tickets allow transactions to be submitted out of sequence order.
    """
    src = ctx.rand_account()

    ticket_count = randrange(1, 11)

    return {
        "TransactionType": "TicketCreate",
        "Account": src.address,
        "TicketCount": ticket_count,
    }


async def build_batch(ctx: TxnContext, intent: TxIntent) -> dict:  # TODO: handle TxIntent.INVALID
    """Build a Batch transaction with random inner transactions of various types."""
    src = ctx.rand_account()

    num_inner = randrange(2, 9)

    batch_seq = await ctx.next_sequence(src.address)
    inner_sequences = [await ctx.next_sequence(src.address) for _ in range(num_inner)]

    inner_txns = []
    for seq in inner_sequences:
        txn_type = choice(["Payment", "TrustSet", "AccountSet", "NFTokenMint"])

        if txn_type == "Payment":
            use_xrp = random() < 0.5
            if use_xrp or not ctx.currencies:
                amount = str(randrange(1_000_000, 100_000_000))  # 1-100 XRP in drops
            else:
                currency = ctx.rand_currency()
                amount = IssuedCurrencyAmount(
                    currency=currency.currency,
                    issuer=currency.issuer,
                    value=str(randrange(10, 1000)),
                )

            inner_tx = Payment(
                account=src.address,
                destination=choice_omit(ctx.wallets, [src]).address,
                amount=amount,
                fee="0",
                signing_pub_key="",
                flags=TransactionFlag.TF_INNER_BATCH_TXN,
                sequence=seq,
            )

        elif txn_type == "TrustSet":
            available_cur = [c for c in ctx.currencies if c.issuer != src.address]
            if not available_cur:
                inner_tx = AccountSet(
                    account=src.address,
                    fee="0",
                    signing_pub_key="",
                    flags=TransactionFlag.TF_INNER_BATCH_TXN,
                    sequence=seq,
                )
            else:
                cur = choice(available_cur)
                inner_tx = TrustSet(
                    account=src.address,
                    limit_amount=IssuedCurrencyAmount(
                        currency=cur.currency,
                        issuer=cur.issuer,
                        value=str(ctx.config["transactions"]["trustset"]["limit"]),
                    ),
                    fee="0",
                    signing_pub_key="",
                    flags=TransactionFlag.TF_INNER_BATCH_TXN,
                    sequence=seq,
                )

        elif txn_type == "AccountSet":
            inner_tx = AccountSet(
                account=src.address,
                fee="0",
                signing_pub_key="",
                flags=TransactionFlag.TF_INNER_BATCH_TXN,
                sequence=seq,
            )

        elif txn_type == "NFTokenMint":
            memo = Memo(memo_data="Batch NFT".encode("utf-8").hex())
            inner_tx = NFTokenMint(
                account=src.address,
                nftoken_taxon=0,
                fee="0",
                signing_pub_key="",
                flags=TransactionFlag.TF_INNER_BATCH_TXN,
                sequence=seq,
                memos=[memo],
            )

        inner_txns.append({"RawTransaction": inner_tx})

    batch_mode = choice(
        [
            BatchFlag.TF_ALL_OR_NOTHING,
            BatchFlag.TF_ONLY_ONE,
            BatchFlag.TF_UNTIL_FAILURE,
            BatchFlag.TF_INDEPENDENT,
        ]
    )

    return {
        "TransactionType": "Batch",
        "Account": src.address,
        "Sequence": batch_seq,  # Explicitly set so build_sign_and_track won't allocate a new one
        "Flags": batch_mode,
        "RawTransactions": inner_txns,
    }


BUILDERS = {
    "TicketCreate": (build_ticket_create, TicketCreate),
    "Batch": (build_batch, Batch),
}
