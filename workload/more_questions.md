more_questions.txt
How does the batch selected know which accounts are available to submit another txn?
On that "note" replace all uses of the term "batch" in reference to our project's submission of transactions since "Batch" is also a type of xrpld txn. Use the term LTS (ledger transaction set).
How are we validating ledgesr? we should be watching the websocket stream. maybe even  adding the accounts to to accounts stream to  watch when they are validated

The app.py is too large to read all at once.

 After the TaskGroup finishes let's wait till we get a ledger close event from the
 ledger listener? do we  have a ledger listener?  should we?
