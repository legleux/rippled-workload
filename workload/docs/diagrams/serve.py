# /// script
# dependencies = ["fastapi", "uvicorn[standard]"]
# ///
"""
Diagram server — Mermaid.js hot-reload viewer.

Run:
    uv run serve.py
    # or: uvicorn serve:app --reload --port 7700
"""

import os
import threading
import time
import webbrowser
from collections import OrderedDict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# ─────────────────────────────────────────────────────────────────────────────
# Diagram definitions (Mermaid syntax)
# ─────────────────────────────────────────────────────────────────────────────

DIAGRAMS: OrderedDict[str, str] = OrderedDict(
    {
        "Module Dependencies": """
graph TD
    subgraph Entry["Entry Point"]
        app[app.py]
    end

    subgraph Core["Core Logic"]
        core[workload_core.py]
        wsproc[ws_processor.py]
    end

    subgraph Infra["Infrastructure"]
        sqlite[sqlite_store.py]
        ws[ws.py]
    end

    subgraph Factory["Transaction Factory"]
        builder[txn_factory/builder.py]
    end

    subgraph Domain["Domain / Data"]
        constants[constants.py]
        validation[validation.py]
        amm[amm.py]
        balances[balances.py]
        feeinfo[fee_info.py]
        randoms[randoms.py]
        config[config.py]
        logcfg[logging_config.py]
    end

    app --> core
    app --> ws
    app --> wsproc
    app --> constants
    app --> config
    app --> logcfg

    core --> constants
    core --> validation
    core --> amm
    core --> balances
    core --> sqlite
    core --> builder

    sqlite --> constants
    sqlite --> validation

    wsproc --> validation
    wsproc -.->|TYPE_CHECKING only| core

    builder --> randoms

    style ws fill:#f9f9f9,stroke:#aaa
    style constants fill:#f9f9f9,stroke:#aaa
    style validation fill:#f9f9f9,stroke:#aaa
    style amm fill:#f9f9f9,stroke:#aaa
    style balances fill:#f9f9f9,stroke:#aaa
    style feeinfo fill:#f9f9f9,stroke:#aaa
    style randoms fill:#f9f9f9,stroke:#aaa
""",
        "Class Diagram": """
classDiagram
    class TxType {
        <<StrEnum>>
        ACCOUNT_SET
        AMM_CREATE
        AMM_DEPOSIT
        AMM_WITHDRAW
        BATCH
        MPTOKEN_ISSUANCE_CREATE
        MPTOKEN_ISSUANCE_SET
        MPTOKEN_AUTHORIZE
        MPTOKEN_ISSUANCE_DESTROY
        NFTOKEN_MINT
        NFTOKEN_BURN
        NFTOKEN_CREATE_OFFER
        NFTOKEN_CANCEL_OFFER
        NFTOKEN_ACCEPT_OFFER
        OFFER_CREATE
        OFFER_CANCEL
        TICKET_CREATE
        PAYMENT
        TRUSTSET
    }

    class TxState {
        <<StrEnum>>
        CREATED
        SUBMITTED
        RETRYABLE
        VALIDATED
        REJECTED
        EXPIRED
        FAILED_NET
    }

    class ValidationSrc {
        <<StrEnum>>
        POLL
        WS
    }

    class PendingTx {
        <<dataclass>>
        +tx_hash: str
        +signed_blob_hex: str
        +account: str
        +tx_json: dict
        +sequence: int | None
        +last_ledger_seq: int
        +transaction_type: TxType | None
        +created_ledger: int
        +wallet: Wallet | None
        +state: TxState
        +attempts: int
        +engine_result_first: str | None
        +validated_ledger: int | None
        +meta_txn_result: str | None
        +created_at: float
        +finalized_at: float | None
        +__str__() str
    }

    class AccountRecord {
        <<dataclass>>
        +lock: asyncio.Lock
        +next_seq: int | None
    }

    class ValidationRecord {
        <<dataclass>>
        +txn: str
        +seq: int
        +src: str
    }

    class FeeInfo {
        <<dataclass>>
        +expected_ledger_size: int
        +current_ledger_size: int
        +current_queue_size: int
        +max_queue_size: int
        +base_fee: int
        +median_fee: int
        +minimum_fee: int
        +open_ledger_fee: int
        +ledger_current_index: int
        +from_fee_result(result) FeeInfo$
    }

    class DEXMetrics {
        <<dataclass>>
        +pools_created: int
        +total_deposits: int
        +total_withdrawals: int
        +total_offers: int
        +pool_snapshots: list
        +last_poll_ledger: int
        +total_xrp_locked_drops: int
        +active_pools: int
    }

    class TxnContext {
        <<dataclass>>
        +funding_wallet: Wallet
        +wallets: Sequence
        +currencies: Sequence
        +config: dict
        +base_fee_drops: AwaitInt
        +next_sequence: AwaitSeq
        +mptoken_issuance_ids: list | None
        +amm_pools: set | None
        +amm_pool_registry: list | None
        +nfts: dict | None
        +offers: dict | None
        +tickets: dict | None
        +balances: dict | None
        +disabled_types: set | None
        +forced_account: Wallet | None
        +rand_accounts(n, omit) list
        +rand_account(omit) Wallet
        +get_account_currencies(account) list
        +rand_currency() IssuedCurrency
        +rand_mptoken_id() str
        +amm_pool_exists() bool
        +rand_amm_pool() dict
        +build() TxnContext$
    }

    class InMemoryStore {
        -_lock: asyncio.Lock
        -_records: dict
        +validations: deque
        +count_by_state: dict
        +count_by_type: dict
        +validated_by_source: dict
        +get(tx_hash) dict | None
        +mark(tx_hash, source, **fields) None
        +rekey(old_hash, new_hash) None
        +find_by_state(*states) list
        +all_records() list
        -_recount() None
        +snapshot_stats() dict
    }

    class SQLiteStore {
        +db_path: Path
        -_lock: asyncio.Lock
        +validations: deque
        +count_by_state: dict
        +validated_by_source: dict
        -_init_db() None
        -_load_validations() None
        -_recount() None
        +bulk_upsert(records) None
        +has_state() bool
        +load_wallets() list
        +load_currencies() list
        +save_wallet(wallet) None
        +save_currency(currency) None
        +snapshot_stats() dict
    }

    class AMMPoolRegistry {
        -_pools: list
        -_pool_ids: set
        +register(asset1, asset2, creator) None
        +add_lp_holder(asset1, asset2, account) None
        +pools() list
        +pool_ids() set
        +__len__() int
        +__bool__() bool
    }

    class BalanceTracker {
        -_balances: dict
        +get(account, currency, issuer) float
        +set(account, currency, value, issuer) None
        +update(account, currency, delta, issuer) None
        +data() dict
    }

    class Workload {
        +config: dict
        +client: AsyncJsonRpcClient
        +funding_wallet: Wallet
        +accounts: dict
        +wallets: dict
        +gateways: list
        +users: list
        +pending: dict
        +store: InMemoryStore
        +persistent_store: SQLiteStore | None
        +amm: AMMPoolRegistry
        +dex_metrics: DEXMetrics
        +balance_tracker: BalanceTracker
        +ctx: TxnContext
        +target_txns_per_ledger: int
        +workload_started: bool
        +alloc_seq(address) int
        +release_seq(address) None
        +build_sign_and_track(txn, wallet) PendingTx | None
        +submit_pending(pending_tx) None
        +record_validated(rec) None
        +get_fee_info() FeeInfo
        +flush_to_persistent_store() None
    }

    Workload *-- InMemoryStore : store
    Workload *-- AMMPoolRegistry : amm
    Workload *-- DEXMetrics : dex_metrics
    Workload *-- BalanceTracker : balance_tracker
    Workload o-- SQLiteStore : persistent_store
    Workload *-- TxnContext : ctx
    Workload *-- PendingTx : pending dict
    Workload *-- AccountRecord : accounts dict
    TxnContext ..> Workload : callbacks
    PendingTx --> TxState : state
    PendingTx --> TxType : transaction_type
    ValidationRecord --> ValidationSrc : src
""",
        "Transaction State Machine": """
stateDiagram-v2
    [*] --> CREATED : build_sign_and_track()

    CREATED --> SUBMITTED : submit_pending() [tesSUCCESS]
    CREATED --> SUBMITTED : submit_pending() [tel codes, seq released]
    CREATED --> REJECTED : submit_pending() [tem* / tef* terminal error]
    CREATED --> REJECTED : submit_pending() [tefPAST_SEQ] + cascade_expire
    CREATED --> EXPIRED : submit_pending() [terPRE_SEQ] + cascade_expire
    CREATED --> FAILED_NET : submit_pending() [TimeoutError / Exception]

    SUBMITTED --> VALIDATED : record_validated() [WS stream or RPC poll]
    SUBMITTED --> EXPIRED : record_expired() [past LastLedgerSeq + grace]
    SUBMITTED --> RETRYABLE : check_finality() [pending within grace]

    RETRYABLE --> VALIDATED : record_validated()
    RETRYABLE --> EXPIRED : record_expired()

    FAILED_NET --> VALIDATED : record_validated() [WS stream or RPC poll finds it]
    FAILED_NET --> EXPIRED : record_expired() [past LastLedgerSeq + grace]

    VALIDATED --> [*]
    REJECTED --> [*]
    EXPIRED --> [*]

    note right of FAILED_NET
        NOT terminal — tx may still be queued in rippled
        Account stays locked until LLS expires or tx validates
    end note

    note right of VALIDATED
        Side effects on entry:
        Payment → balance_tracker.update()
        Payment (new acct) → wallets dict adopted
        MPTokenIssuanceCreate → _mptoken_issuance_ids
        AMMCreate → amm.register()
        Batch → sync sequence
        DEX types → dex_metrics update
        Primary path: WebSocket tx_validated
        Fallback path: periodic_finality_check (5s)
    end note

    note right of EXPIRED
        HORIZON = 15 ledgers (~45-60s)
        _cascade_expire_account() called
        Higher-seq txns on same account
        also transition to EXPIRED
    end note
""",
        "Transaction Lifecycle Sequence": """
sequenceDiagram
    participant cw as continuous_workload
    participant gen as generate_txn
    participant wl as Workload
    participant ar as AccountRecord lock
    participant mem as InMemoryStore
    participant rpc as rippled RPC
    participant wslist as ws_listener
    participant wsproc as ws_processor
    participant pfc as periodic_finality_check

    Note over cw,rpc: Per-Ledger Batch (triggered on ledger_closed event)

    cw->>wl: get_fee_info()
    wl->>rpc: fee command
    rpc-->>wl: fee result
    wl-->>cw: FeeInfo (open_ledger_fee, ledger_current_index)

    Note over cw: Compute free_accounts: pending_count == 0<br/>Shuffle + slice to target_txns_per_ledger

    par TaskGroup — N concurrent tasks (one per wallet)
        cw->>gen: generate_txn(ctx[forced_account=wallet])
        gen-->>cw: Transaction (xrpl-py model)

        cw->>wl: build_sign_and_track(txn, wallet)
        wl->>ar: alloc_seq(wallet.address)
        Note right of ar: Acquires asyncio.Lock per account.<br/>Fetches from ledger on first call,<br/>increments locally thereafter.
        ar-->>wl: sequence number

        wl->>wl: sign transaction
        wl->>wl: create PendingTx(state=CREATED)
        wl->>mem: mark(tx_hash, state=CREATED)
        mem-->>wl: ok
        wl-->>cw: PendingTx

        cw->>wl: submit_pending(pending_tx)
        wl->>rpc: SubmitOnly(signed_blob_hex)
        rpc-->>wl: engine_result

        alt tesSUCCESS or tel*
            wl->>mem: mark(tx_hash, state=SUBMITTED)
        else tem* / tef* terminal rejection
            wl->>ar: release_seq(wallet.address)
            wl->>mem: mark(tx_hash, state=REJECTED)
        else terPRE_SEQ
            wl->>mem: mark(tx_hash, state=EXPIRED)
            wl->>wl: cascade_expire()
        else TimeoutError / Exception
            wl->>mem: mark(tx_hash, state=FAILED_NET)
        end
    end

    Note over wslist,wsproc: Validation Path A: WebSocket (Primary)

    rpc->>wslist: accounts stream tx_validated message
    wslist->>wslist: _process_message(raw_msg)
    wslist->>wsproc: event_queue.put(tx_validated)
    wsproc->>wl: record_validated(ValidationRecord(hash, ledger, WS))
    wl->>wl: dedup check (ValidationRecord deque)
    wl->>mem: mark(tx_hash, state=VALIDATED, src=ws)
    wl->>wl: side-effects (balance_tracker, amm.register())

    Note over pfc,rpc: Validation Path B: RPC Poll (Fallback — every 5s)

    pfc->>rpc: Tx(hash) for each pending tx
    rpc-->>pfc: tx result

    alt validated=true
        pfc->>wl: record_validated(ValidationRecord(hash, ledger, POLL))
        wl->>mem: mark(tx_hash, state=VALIDATED, src=poll)
    else past last_ledger_seq + grace
        pfc->>wl: record_expired(hash)
        wl->>mem: mark(tx_hash, state=EXPIRED)
        wl->>wl: _cascade_expire_account(fetch_seq_from_ledger=True)
    end

    Note over cw,wl: Shutdown / Flush

    cw->>wl: flush_to_persistent_store()
    wl->>mem: all_records()
    mem-->>wl: list[dict]
    wl->>rpc: SQLiteStore.bulk_upsert(records)
    rpc-->>wl: ok (near-instant — single transaction)
""",
        "Async Runtime": """
graph TD
    subgraph Lifespan["App Lifespan TaskGroup (permanent)"]
        wslist[ws_listener]
        wsproc[ws_processor]
        pfc["periodic_finality_check<br/>every 5s"]
        pdex["periodic_dex_metrics<br/>every N ledgers"]
    end

    subgraph Managed["Managed Task"]
        cw[continuous_workload]
        subgraph Batch["Per-Batch inner TaskGroup (short-lived)"]
            t1["_build_and_submit<br/>wallet_1"]
            t2["_build_and_submit<br/>wallet_2"]
            tN["_build_and_submit<br/>wallet_N"]
        end
    end

    subgraph State["Shared Mutable State"]
        memstore["InMemoryStore\nasyncio.Lock"]
        accrec["AccountRecord × N\nasyncio.Lock per account"]
        sqlstore["SQLiteStore\nasyncio.Lock"]
        queue["asyncio.Queue\nevent_queue"]
    end

    cw -->|spawns| t1
    cw -->|spawns| t2
    cw -->|spawns| tN

    wslist -->|"put(tx_validated\nledger_closed\nserver_status)"| queue
    queue -->|get| wsproc

    wsproc -->|"record_validated()\nsrc=WS"| memstore
    pfc -->|"record_validated()\nsrc=POLL"| memstore

    t1 -->|"alloc_seq()\nrelease_seq()"| accrec
    t2 -->|"alloc_seq()\nrelease_seq()"| accrec
    tN -->|"alloc_seq()\nrelease_seq()"| accrec

    t1 -->|"mark(CREATED)\nmark(SUBMITTED/...)"| memstore
    t2 -->|mark| memstore
    tN -->|mark| memstore

    cw -->|"flush_to_persistent_store()\nbulk_upsert() on shutdown"| sqlstore
    pfc -->|"find_by_state(SUBMITTED, RETRYABLE)"| memstore
    wslist -.->|"accounts_provider() callback"| accrec
""",
        "API Surface": """
graph LR
    wl_state[("app.state.workload: Workload")]

    subgraph Global
        health["GET /health"]
        debugfund["POST /debug/fund"]
        dashboard["GET /state/dashboard HTML"]
    end

    subgraph Accounts["/accounts"]
        acc_create_get["GET /create"]
        acc_create_post["POST /create"]
        acc_create_random["GET /create/random"]
        acc_get["GET /{account_id}"]
        acc_balances["GET /{account_id}/balances"]
        acc_lines["GET /{account_id}/lines"]
    end

    subgraph Payment["/payment  (alias: /pay)"]
        pay_post["POST /payment"]
    end

    subgraph Transaction["/transaction  (alias: /txn)"]
        txn_random["GET /random"]
        txn_create["GET /create/{transaction}"]
        txn_payment["POST /payment"]
        txn_trustset["POST /trustset"]
        txn_accountset["POST /accountset"]
        txn_ammcreate["POST /ammcreate"]
        txn_ammdeposit["POST /ammdeposit"]
        txn_ammwithdraw["POST /ammwithdraw"]
        txn_nftokenmint["POST /nftokenmint"]
        txn_mptokencreate["POST /mptokenissuancecreate"]
        txn_mptokenset["POST /mptokenissuanceset"]
        txn_mptokenauth["POST /mptokenauthorize"]
        txn_mptokendestroy["POST /mptokenissuancedestroy"]
        txn_batch["POST /batch"]
    end

    subgraph State["/state"]
        st_summary["GET /summary"]
        st_pending["GET /pending"]
        st_failed["GET /failed"]
        st_failed_code["GET /failed/{error_code}"]
        st_expired["GET /expired"]
        st_tx["GET /tx/{tx_hash}"]
        st_fees["GET /fees"]
        st_accounts["GET /accounts"]
        st_validations["GET /validations"]
        st_wallets["GET /wallets"]
        st_users["GET /users"]
        st_gateways["GET /gateways"]
        st_currencies["GET /currencies"]
        st_mptokens["GET /mptokens"]
        st_finality["GET /finality"]
        st_ws_stats["GET /ws/stats"]
    end

    subgraph Workload["/workload"]
        wl_start["POST /start"]
        wl_stop["POST /stop"]
        wl_status["GET /status"]
        wl_fill_get["GET /fill-fraction"]
        wl_fill_post["POST /fill-fraction"]
        wl_target_get["GET /target-txns"]
        wl_target_post["POST /target-txns"]
        wl_disabled["GET /disabled-types"]
        wl_toggle["POST /toggle-type"]
    end

    subgraph DEX["/dex"]
        dex_metrics["GET /metrics"]
        dex_pools["GET /pools"]
        dex_pool_idx["GET /pools/{index}"]
        dex_poll["POST /poll"]
    end

    subgraph Network["/network"]
        net_reset["POST /reset"]
    end

    wl_state --- health
    wl_state --- acc_get
    wl_state --- pay_post
    wl_state --- txn_random
    wl_state --- st_summary
    wl_state --- wl_start
    wl_state --- dex_metrics
    wl_state --- net_reset
""",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

_TAB_BUTTONS = "\n".join(
    f'<button class="tab-btn" onclick="showTab({i})" id="btn-{i}">{name}</button>' for i, name in enumerate(DIAGRAMS)
)

_TAB_PANES = "\n".join(
    f'<div class="tab-pane" id="pane-{i}"><div class="mermaid">{diagram.strip()}</div></div>'
    for i, diagram in enumerate(DIAGRAMS.values())
)

HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>rippled-workload Diagrams</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #f8f9fa; color: #212529; }}
  header {{ background: #1a1a2e; color: #e0e0e0; padding: 1rem 2rem; }}
  header h1 {{ font-size: 1.2rem; font-weight: 600; }}
  header small {{ font-size: 0.8rem; color: #888; }}
  .tabs {{ display: flex; flex-wrap: wrap; gap: 0.4rem; padding: 1rem 2rem 0; background: #fff;
           border-bottom: 2px solid #dee2e6; }}
  .tab-btn {{ padding: 0.45rem 1rem; border: 1px solid #dee2e6; border-bottom: none;
              border-radius: 4px 4px 0 0; background: #f1f3f5; cursor: pointer;
              font-size: 0.85rem; transition: background 0.15s; }}
  .tab-btn:hover {{ background: #e9ecef; }}
  .tab-btn.active {{ background: #fff; border-bottom: 2px solid #fff; margin-bottom: -2px;
                     font-weight: 600; color: #2980b9; }}
  .tab-pane {{ display: none; padding: 2rem; overflow: auto; }}
  .tab-pane.active {{ display: block; }}
  .mermaid {{ max-width: 100%; }}
  #reload-indicator {{ position: fixed; bottom: 1rem; right: 1.5rem; font-size: 0.75rem;
                       color: #aaa; }}
</style>
</head>
<body>
<header>
  <h1>rippled-workload — Architecture Diagrams</h1>
  <small>Hot-reload active &mdash; edit serve.py to update</small>
</header>
<nav class="tabs">
{_TAB_BUTTONS}
</nav>
<main>
{_TAB_PANES}
</main>
<div id="reload-indicator">watching for changes...</div>

<script>
  mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});

  let activeTab = parseInt(localStorage.getItem('activeTab') || '0');

  function showTab(i) {{
    document.querySelectorAll('.tab-pane').forEach((p, idx) => {{
      p.classList.toggle('active', idx === i);
    }});
    document.querySelectorAll('.tab-btn').forEach((b, idx) => {{
      b.classList.toggle('active', idx === i);
    }});
    activeTab = i;
    localStorage.setItem('activeTab', i);
  }}

  showTab(activeTab);

  // Hot reload polling
  let lastMtime = null;
  async function pollVersion() {{
    try {{
      const r = await fetch('/version');
      const {{ mtime }} = await r.json();
      if (lastMtime === null) {{ lastMtime = mtime; return; }}
      if (mtime !== lastMtime) {{
        document.getElementById('reload-indicator').textContent = 'reloading...';
        location.reload();
      }}
    }} catch (_) {{ /* server restarting */ }}
  }}
  setInterval(pollVersion, 500);
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="rippled-workload diagrams")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML_TEMPLATE


@app.get("/version")
async def version() -> JSONResponse:
    return JSONResponse({"mtime": os.path.getmtime(__file__)})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = 7700

    def _open_browser() -> None:
        time.sleep(0.8)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_includes=["serve.py"],
    )
