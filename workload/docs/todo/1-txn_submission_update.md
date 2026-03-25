How close to reality was this description?

Here are the steps I believe are/should be happening.
0. Determine how many txns will be in the submissiont set
1. Choose N txns based on the weights from the config
2. Determine if the txn will be good or bad based on the txn type's ratio of good/bad txns (need to add to config in intuitive manner)
   a. Each txn type will probably need to have certain failure conditions defined in its factory methods.
3. Map available accounts to which txns they'll send
4. Deterine info needed to populate txn i.e. account seq, elevated fee (due to object type or just fee escalation)
5. compose txn (see below for more info  as this needs to be adapted.)
6. sign and submit (beginning of our in-memory lifecycle)


Things that definitely have to change:
Using the xrpl-py library to generate the txn. This is fine for the successful txns but for a intentionally bad txn, we
need to be able to get arround the xrpl-py  librarys txn validation.
We'll do this by constructing the good txn first. Then once we have the a dict of the proper appropriate fields, we can
taint  the txns by modifying values to be out of range, adding invalid keys/values or removing them so they aren't valid.
Then the txn can be encoded to binary and signed for submission.
Since we'll know beforehand if a txn is guaranteed to fail, we can immediately return the account to the pool of available senders
once the initial response of a failure is  received. (maybe also assert the txn doesn't appear in the websocket transactions stream)

Things that may have to change based on this procedure:
Where/how whether a txn will be "successful" or not is determined, as in the numbered list above.

Determine if it might be more efficient to pre-sort the list of accounts into groups based on which types of txns they
will be able to successfully submit. Depends on a number of things, XRP/IOU/MPT balance. pre-existing primitives applied to ledger e.g. can't cash
a check, close an escrow or offer an NFT without already having those on ledger.

We should probably streamline the global txn context object's interaction with the txn_factory builder before it gets too messy and powerful so
any time a method is growing too  long or accepting too many parameters pause and consult me to resassess the direction we need to go.

Is that clear and are there any other issues that should be resovled before this refactor starts?
