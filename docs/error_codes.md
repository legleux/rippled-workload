Read the files in this same directory it's a lot but I'm sure it'll impart a ton of wisdom!!!
docs/Transaction.h
docs/NetworkOPs.cpp
docs/Transactor.cpp
docs/LocalTxs.cpp
docs/TxQ_test.cpp
docs/RobustTransaction_test.cpp
docs/TxQ.cpp

Here's a sampling of errors from docs explaining the some error codes we may come across from online:
https://xrpl.org/docs/references/protocol/transactions/transaction-results/tec-codes
https://xrpl.org/docs/references/protocol/transactions/transaction-results/tef-codes
https://xrpl.org/docs/references/protocol/transactions/transaction-results/tel-codes
https://xrpl.org/docs/references/protocol/transactions/transaction-results/tem-codes
https://xrpl.org/docs/references/protocol/transactions/transaction-results/ter-codes
https://xrpl.org/docs/references/protocol/transactions/transaction-results/tes-success

What we've been dealing with:

terPRE_SEQ	The Sequence number of the current transaction is higher than the current sequence number of the account sending the transaction.


telCAN_NOT_QUEUE	The transaction did not meet the open ledger cost, but this server did not queue this transaction because it did not meet the queuing restrictions. For example, a transaction returns this code when the sender already has 10 other transactions in the queue. You can try again later or sign and submit a replacement transaction with a higher transaction cost in the Fee field.
telCAN_NOT_QUEUE_FULL	The transaction did not meet the open ledger cost and the server did not queue this transaction because this server's transaction queue is full. You could increase the Fee and try again, try again later, or try submitting to a different server. The new transaction must have a higher transaction cost, as measured in fee levels, than the transaction in the queue with the smallest transaction cost.
telCAN_NOT_QUEUE_BALANCE	The transaction did not meet the open ledger cost and also was not added to the transaction queue because the sum of potential XRP costs of already-queued transactions is greater than the expected balance of the account. You can try again later, or try submitting to a different server.
telCAN_NOT_QUEUE_BLOCKS	The transaction did not meet the open ledger cost and also was not added to the transaction queue. This transaction could not replace an existing transaction in the queue because it would block already-queued transactions from the same sender. (For details, see Queuing Restrictions.) You can try again later, or try submitting to a different server.
telCAN_NOT_QUEUE_BLOCKED	The transaction did not meet the open ledger cost and also was not added to the transaction queue because a transaction queued ahead of it from the same sender blocks it. (For details, see Queuing Restrictions.) You can try again later, or try submitting to a different server.
telCAN_NOT_QUEUE_FEE	The transaction did not meet the open ledger cost and also was not added to the transaction queue. This code occurs when a transaction with the same sender and sequence number already exists in the queue and the new one does not pay a large enough transaction cost to replace the existing transaction. To replace a transaction in the queue, the new transaction must have a Fee value that is at least 25% more, as measured in fee levels. You can increase the Fee and try again, send this with a higher Sequence number so it doesn't replace an existing transaction, or try sending to another server.
telCAN_NOT_QUEUE_BALANCE	The transaction did not meet the open ledger cost and also was not added to the transaction queue because the sum of potential XRP costs of already-queued transactions is greater than the expected balance of the account. You can try again later, or try submitting to a different server.

fee stuff:
terINSUF_FEE_B	The account sending the transaction does not have enough XRP to pay the Fee specified in the transaction.
telINSUF_FEE_P	The Fee from the transaction is not high enough to meet the server's current transaction cost requirement, which is derived from its load level and network-level requirements. If the individual server is too busy to process your transaction right now, it may cache the transaction and automatically retry later.
tecINSUFF_FEE	136	The transaction failed because the sending account does not have enough XRP to pay the transaction cost that it specified. (In this case, the transaction processing destroys all of the sender's XRP even though that amount is lower than the specified transaction cost.) This result only occurs if the account's balance decreases after this transaction has been distributed to enough of the network to be included in a consensus set. Otherwise, the transaction fails with terINSUF_FEE_B before being distributed.

terQUEUED	The transaction met the load-scaled transaction cost but did not meet the open ledger requirement, so the transaction has been queued for a future ledger.
terSUBMITTED	Transaction has been submitted, but not yet applied.

telWRONG_NETWORK	The transaction specifies the wrong NetworkID value for the current network. Either specify the correct the NetworkID value for the intended network, or submit the transaction to a server that is connected to the correct network.


tesSUCCESS	The transaction was applied and forwarded to other servers. If this appears in a validated ledger, then the transaction's success is final.

!!! Make these invariants to have a exception/assert if we see these ever! !!!
terLAST	Used internally only. This code should never be returned. -
tecINTERNAL	144	Unspecified internal error, with transaction cost applied. This error code should not normally be returned. If you can reproduce this error, please report an issue.
tecINVARIANT_FAILED	147	An invariant check failed when trying to execute this transaction.
