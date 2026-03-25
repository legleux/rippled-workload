"""Batch, TicketCreate, and ticket-consuming transaction builders."""

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


def build_ticket_use(ctx: TxnContext, intent: TxIntent) -> dict | None:
    """Build a Payment using a ticket sequence instead of a regular sequence.

    Pops a ticket from the account's available set at build time to prevent
    double-use within the same submission set. The actual XRPL transaction type
    is Payment — TicketSequence replaces Sequence.
    When forced_account is set, uses that account's tickets.
    """
    if not ctx.tickets:
        return None

    # Use forced_account if set, otherwise find any account with tickets
    if ctx.forced_account:
        src = ctx.forced_account
        if not ctx.tickets.get(src.address):
            return None
    else:
        accounts_with_tickets = [w for w in ctx.wallets if ctx.tickets.get(w.address)]
        if not accounts_with_tickets:
            return None
        src = choice(accounts_with_tickets)

    ticket_seq = ctx.tickets[src.address].pop()
    if not ctx.tickets[src.address]:
        del ctx.tickets[src.address]

    dst = choice_omit(ctx.wallets, [src])
    amount = str(randrange(1_000_000, 100_000_000))  # 1-100 XRP in drops

    return {
        "TransactionType": "Payment",
        "Account": src.address,
        "Destination": dst.address,
        "Amount": amount,
        "Sequence": 0,
        "TicketSequence": ticket_seq,
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
    "TicketUse": (build_ticket_use, Payment),
    "Batch": (build_batch, Batch),
}


def _is_eligible_ticket_use(wallet, ctx) -> bool:
    """Account must have at least one available ticket."""
    return bool(ctx.tickets and ctx.tickets.get(wallet.address))


ELIGIBILITY = {
    "TicketUse": _is_eligible_ticket_use,
}
