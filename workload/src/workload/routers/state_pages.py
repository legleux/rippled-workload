"""HTML-rendering state endpoints (dashboard, failed page, type page)."""

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from workload.config import cfg

log = logging.getLogger("workload.app")

# Resolve RPC URL (same logic as app.py)
if Path("/.dockerenv").is_file():
    _xrpld = cfg["xrpld"]["docker"]
else:
    _xrpld = cfg["xrpld"]["local"]

_rpc_port = cfg["xrpld"]["rpc_port"]
_xrpld_ip = os.getenv("XRPLD_IP", os.getenv("RIPPLED_IP", _xrpld))
RPC = os.getenv("RPC_URL", f"http://{_xrpld_ip}:{_rpc_port}")

router = APIRouter(prefix="/state", tags=["State"])


@router.get("/dashboard", response_class=HTMLResponse)
async def state_dashboard(request: Request) -> HTMLResponse:
    """HTML dashboard with live stats, explorer embed, and WS terminal."""
    hostname = RPC.split("//")[1].split(":")[0] if "//" in RPC else RPC.split(":")[0]

    # Build node list from compose config for the WS terminal dropdown
    ws_port = cfg["xrpld"]["ws_port"]
    nodes = []
    try:
        from gl.config import ComposeConfig

        cc = ComposeConfig()
        for i in range(cc.num_validators):
            name = f"{cc.validator_name}{i}"
            ws = cc.ws_port + i + cc.num_hubs
            nodes.append({"name": name, "ws": ws})
        for i in range(cc.num_hubs):
            name = cc.hub_name if cc.num_hubs == 1 else f"{cc.hub_name}{i}"
            ws = cc.ws_port + i
            nodes.append({"name": name, "ws": ws})
    except (ImportError, Exception) as e:
        log.debug("gl (generate_ledger) not available (%s), scanning compose file", e)
        # Fallback: scan testnet compose file for services with WS port mappings
        compose_path = Path("testnet/docker-compose.yml")
        if compose_path.exists():
            try:
                import re

                text = compose_path.read_text()
                # Match services with port mappings like "6006:6006" or "6007:6006"
                current_name = None
                for line in text.splitlines():
                    m = re.match(r"\s+container_name:\s*(\S+)", line)
                    if m:
                        current_name = m.group(1)
                    m = re.match(r"\s+-\s+(\d+):(\d+)", line)
                    if m and current_name:
                        host_port, container_port = int(m.group(1)), int(m.group(2))
                        if container_port == ws_port:
                            nodes.append({"name": current_name, "ws": host_port})
            except Exception as e2:
                log.debug("compose file scan failed: %s", e2)
    if not nodes:
        nodes.append({"name": hostname, "ws": ws_port})
    nodes_json = json.dumps(nodes)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Workload Dashboard</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0d1117;
                color: #c9d1d9;
                margin: 0;
                padding: 20px;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            h1 {{ color: #58a6ff; margin-bottom: 10px; }}
            h2 {{ color: #c9d1d9; margin: 0 0 12px 0; font-size: 16px; }}
            .subtitle {{ color: #8b949e; margin-bottom: 20px; }}
            .subtitle a {{ color: #58a6ff; text-decoration: none; }}
            .subtitle a:hover {{ text-decoration: underline; }}

            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 16px;
                margin-bottom: 20px;
            }}
            .stat-card {{
                background: #161b22; border: 1px solid #30363d;
                border-radius: 6px; padding: 16px;
            }}
            .stat-label {{
                color: #8b949e; font-size: 11px;
                text-transform: uppercase; margin-bottom: 6px;
            }}
            .stat-value {{
                font-size: 28px; font-weight: bold; margin-bottom: 2px;
            }}
            .stat-value.success {{ color: #3fb950; }}
            .stat-value.error {{ color: #f85149; }}
            .stat-value.warning {{ color: #d29922; }}
            .stat-value.info {{ color: #58a6ff; }}
            .stat-percentage {{ color: #8b949e; font-size: 13px; }}
            .progress-bar {{
                background: #21262d; border-radius: 6px;
                height: 6px; overflow: hidden; margin-top: 6px;
            }}
            .progress-fill {{ height: 100%; transition: width 0.3s ease; }}
            .progress-fill.success {{ background: #3fb950; }}
            .progress-fill.error {{ background: #f85149; }}
            .progress-fill.info {{ background: #58a6ff; }}

            .panel {{
                background: #161b22; border: 1px solid #30363d;
                border-radius: 6px; padding: 20px; margin-bottom: 20px;
            }}
            .failures-table {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; margin-bottom: 20px; flex: 1; min-width: 300px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #21262d; }}
            th {{ color: #8b949e; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
            tr:last-child td {{ border-bottom: none; }}
            .badge {{
                display: inline-block; padding: 2px 8px; border-radius: 4px;
                font-size: 12px; font-weight: 600;
            }}
            .badge.success {{ background: #3fb9501a; color: #3fb950; }}
            .badge.warning {{ background: #d299221a; color: #d29922; }}
            .badge.error {{ background: #f851491a; color: #f85149; }}
            .badge.info {{ background: #58a6ff1a; color: #58a6ff; }}

            .controls {{
                margin-bottom: 20px; display: flex;
                gap: 10px; align-items: center; flex-wrap: wrap;
            }}
            .btn {{
                padding: 8px 16px; border: none; border-radius: 6px;
                font-size: 13px; font-weight: 600; cursor: pointer;
                transition: opacity 0.2s;
            }}
            .btn:hover {{ opacity: 0.8; }}
            .btn-start {{ background: #3fb950; color: white; }}
            .btn-stop {{ background: #f85149; color: white; }}
            .fill-control {{
                display: flex; align-items: center; gap: 8px;
                background: #161b22; border: 1px solid #30363d;
                border-radius: 6px; padding: 6px 14px;
            }}
            .fill-control label {{
                color: #8b949e; font-size: 12px; font-weight: 600;
                text-transform: uppercase; white-space: nowrap;
            }}
            .fill-control input[type=range] {{ width: 140px; accent-color: #58a6ff; }}
            .fill-control .fill-value {{
                color: #58a6ff; font-weight: 700; font-size: 15px;
                min-width: 36px; text-align: right;
            }}
            .link-btn {{
                padding: 8px 16px; border: 1px solid #30363d; border-radius: 6px;
                font-size: 13px; font-weight: 600; cursor: pointer;
                background: #161b22; color: #58a6ff; text-decoration: none;
                transition: border-color 0.2s;
            }}
            .link-btn:hover {{ border-color: #58a6ff; }}

            .explorer-viewport {{
                position: relative; width: 100%; height: 400px;
                overflow: hidden; border-radius: 4px;
            }}
            .explorer-viewport iframe {{
                position: absolute; top: -385px; left: 0;
                width: 100%; height: calc(100% + 600px); border: none;
            }}

            /* WS Terminal */
            .ws-terminal-bar {{
                display: flex; gap: 8px; align-items: center;
                margin-bottom: 10px; flex-wrap: wrap;
            }}
            .ws-terminal-bar select, .ws-terminal-bar button {{
                background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
                border-radius: 4px; padding: 6px 10px; font-size: 13px;
            }}
            .ws-terminal-bar select {{ min-width: 120px; }}
            .ws-terminal-bar button {{ cursor: pointer; font-weight: 600; }}
            .ws-terminal-bar button:hover {{ border-color: #58a6ff; }}
            .ws-terminal-bar button.active {{ background: #3fb950; color: #0d1117; border-color: #3fb950; }}
            .stream-filters {{
                display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
            }}
            .stream-filters label {{
                display: flex; align-items: center; gap: 4px;
                font-size: 12px; color: #8b949e; cursor: pointer;
                background: #0d1117; border: 1px solid #30363d;
                border-radius: 4px; padding: 4px 8px;
            }}
            .stream-filters label.checked {{
                /* colors set via inline styles from TXN_COLORS */
            }}
            .stream-filters input {{ display: none; }}
            #ws-output {{
                background: #010409; border: 1px solid #21262d; border-radius: 4px;
                height: 400px; overflow-y: auto; padding: 10px;
                font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
                font-size: 12px; line-height: 1.5;
                scroll-behavior: smooth;
            }}
            .ws-line {{ margin: 0; white-space: pre-wrap; word-break: break-all; }}
            .ws-line.ledger {{ color: #58a6ff; }}
            .ws-line.transaction {{ color: #3fb950; }}
            .ws-line.validation {{ color: #d29922; }}
            .ws-line.server {{ color: #8b949e; }}
            .ws-line.consensus {{ color: #bc8cff; }}
            .ws-line.peer {{ color: #f0883e; }}
            .ws-line.error {{ color: #f85149; }}
            .ws-line.info {{ color: #8b949e; font-style: italic; }}
            /* Per-type colors driven by TXN_COLORS JS object via inline styles */
            /* Grouped txn type layout */
            .txn-type-groups {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }}
            .txn-type-group {{ display: flex; flex-direction: column; gap: 4px; }}
            .txn-type-group-label {{ color: #8b949e; font-size: 10px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 2px; }}
            /* Transaction Control pane toggles */
            .txn-control-grid {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }}
            .txn-control-group {{ display: flex; flex-direction: column; gap: 4px; }}
            .txn-control-group-label {{ color: #8b949e; font-size: 10px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 2px; }}
            .txn-toggle {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; cursor: pointer; border: 1px solid #30363d; background: #0d1117; color: #484f58; transition: all 0.15s ease; user-select: none; }}
            .txn-toggle.enabled {{ background: #161b22; }}
            .txn-toggle:hover {{ opacity: 0.85; }}
            .txn-toggle.config-disabled {{ opacity: 0.35; cursor: not-allowed; text-decoration: line-through; }}
            .txn-toggle.config-disabled:hover {{ opacity: 0.35; }}
            .txn-control-group-label {{ cursor: pointer; }}
            .txn-control-group-label:hover {{ color: #c9d1d9; }}
            .ws-status {{ font-size: 12px; margin-left: auto; }}
            .ws-status.connected {{ color: #3fb950; }}
            .ws-status.disconnected {{ color: #f85149; }}
            .msg-count {{ color: #8b949e; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Workload Dashboard</h1>
            <div class="subtitle" id="subtitle">Loading...</div>

            <div class="controls">
                <button class="btn btn-start" onclick="fetch('/workload/start', {{method:'POST'}})">Start</button>
                <button class="btn btn-stop" onclick="fetch('/workload/stop', {{method:'POST'}})">Stop</button>
                <a class="link-btn" id="explorer-link" href="#" target="_blank">XRPL Explorer</a>
                <a class="link-btn" href="/dex/amm-pools" target="_blank">AMM Pools</a>
                <a class="link-btn" href="/state/mpt-issuances" target="_blank">MPTokens</a>
                <a class="link-btn" href="/logs/page" target="_blank">Logs</a>
                <a class="link-btn" href="/docs" target="_blank">API Docs</a>
            </div>

            <!-- Stats cards (updated via JS) -->
            <div class="stats-grid" id="fee-stats"></div>
            <div class="stats-grid" id="txn-stats"></div>

            <!-- Explorer embed -->
            <div class="panel">
                <h2>Ledger Stream</h2>
                <div class="explorer-viewport">
                    <iframe src="" id="explorer-frame"></iframe>
                </div>
            </div>

            <!-- Transaction Control -->
            <div class="panel">
                <h2>Transaction Control</h2>
                <div style="display:flex;align-items:center;gap:20px;margin-bottom:12px;flex-wrap:wrap">
                    <div class="fill-control" style="flex:1;min-width:280px">
                        <label>Target TPS: <span id="target-tps-value" style="font-weight:700;color:#58a6ff">0</span> <span style="color:#8b949e;font-size:11px">(0 = unlimited)</span></label>
                        <input type="range" id="target-tps-input" min="0" max="500" step="5" value="0"
                               style="width:100%;accent-color:#58a6ff;cursor:pointer"
                               oninput="document.getElementById('target-tps-value').textContent=this.value"
                               onchange="sliderCooldown=Date.now()+4000;fetch('/workload/target-tps',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{target_tps:parseFloat(this.value)}})}})">
                    </div>
                    <div class="fill-control" style="flex:1;min-width:280px">
                        <label>Invalid intent: <span id="invalid-intent-value" style="font-weight:700;color:#f85149">10</span>% <span style="color:#8b949e;font-size:11px">(0 = all valid)</span></label>
                        <input type="range" id="invalid-intent-input" min="0" max="100" step="5" value="10"
                               style="width:100%;accent-color:#f85149;cursor:pointer"
                               oninput="document.getElementById('invalid-intent-value').textContent=this.value"
                               onchange="sliderCooldown=Date.now()+4000;fetch('/workload/intent',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{invalid:parseFloat(this.value)/100}})}})">
                    </div>
                </div>
                <div id="effective-rate" style="color:#8b949e;font-size:12px;margin-bottom:12px"></div>
                <div class="txn-control-grid" id="txn-control-pane"></div>
            </div>

            <!-- Transaction types + failures side by side -->
            <div id="tables-container" style="display:flex;gap:20px;flex-wrap:wrap"></div>

            <!-- WS Terminal -->
            <div class="panel">
                <h2>Node WebSocket</h2>
                <div class="ws-terminal-bar">
                    <select id="ws-node"></select>
                    <button id="ws-connect-btn" onclick="toggleWs()">Connect</button>
                    <div class="stream-filters" id="stream-filters"></div>
                    <span class="msg-count" id="ws-msg-count"></span>
                    <span class="ws-status disconnected" id="ws-status">disconnected</span>
                </div>
                <div class="ws-terminal-bar" style="margin-top:0;align-items:flex-start">
                    <span style="color:#8b949e;font-size:11px;text-transform:uppercase;font-weight:600;padding-top:4px">Txn types</span>
                    <div class="txn-type-groups" id="txn-type-filters"></div>
                </div>
                <div id="ws-output"></div>
            </div>
        </div>

        <script>
        // --- Explorer + WS use browser hostname so dashboard works from remote machines ---
        const _h = location.hostname;
        const _explorerBase = 'http://custom.xrpl.org/' + _h + ':6006';
        document.getElementById('explorer-frame').src = _explorerBase;
        document.getElementById('explorer-link').href = _explorerBase;

        // --- Nodes ---
        const NODES = {nodes_json};
        const STREAMS = [
            {{name:'ledger', label:'Ledger', on:true}},
            {{name:'transactions', label:'Transactions', on:true}},
            {{name:'validations', label:'Validations', on:false}},
            {{name:'consensus', label:'Consensus', on:false}},
            {{name:'peer_status', label:'Peers', on:false}},
            {{name:'server', label:'Server', on:false}},
        ];
        let ws = null;
        let msgCount = 0;
        let sliderCooldown = 0;  // timestamp: skip slider overwrites until this time
        let activeStreams = new Set(STREAMS.filter(s=>s.on).map(s=>s.name));
        const MAX_LINES = 500;

        // Populate node dropdown
        const nodeSelect = document.getElementById('ws-node');
        NODES.forEach(n => {{
            const opt = document.createElement('option');
            opt.value = n.ws;
            opt.textContent = n.name + ' (:' + n.ws + ')';
            nodeSelect.appendChild(opt);
        }});

        // Populate stream filter checkboxes
        const filtersEl = document.getElementById('stream-filters');
        STREAMS.forEach(s => {{
            const lbl = document.createElement('label');
            lbl.className = s.on ? 'checked' : '';
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = s.on;
            cb.dataset.stream = s.name;
            cb.onchange = function() {{
                if (this.checked) activeStreams.add(s.name);
                else activeStreams.delete(s.name);
                lbl.className = this.checked ? 'checked' : '';
                if (ws && ws.readyState === WebSocket.OPEN) resubscribe();
            }};
            lbl.appendChild(cb);
            lbl.appendChild(document.createTextNode(s.label));
            filtersEl.appendChild(lbl);
        }});

        // Shared txn type definitions
        const TXN_TYPE_GROUPS = [
            {{label: 'Core', types: ['Payment','TrustSet','AccountSet','TicketCreate','TicketUse']}},
            {{label: 'DEX', types: ['OfferCreate','OfferCancel']}},
            {{label: 'AMM', types: ['AMMCreate','AMMDeposit','AMMWithdraw']}},
            {{label: 'NFT', types: ['NFTokenMint','NFTokenBurn','NFTokenCreateOffer','NFTokenCancelOffer','NFTokenAcceptOffer']}},
            {{label: 'MPT', types: ['MPTokenIssuanceCreate','MPTokenIssuanceSet','MPTokenAuthorize','MPTokenIssuanceDestroy']}},
            {{label: 'Check', types: ['CheckCreate','CheckCash','CheckCancel']}},
            {{label: 'Escrow', types: ['EscrowCreate','EscrowFinish','EscrowCancel']}},
            {{label: 'Credential', types: ['CredentialCreate','CredentialAccept','CredentialDelete']}},
            {{label: 'Domain', types: ['PermissionedDomainSet','PermissionedDomainDelete']}},
            {{label: 'Vault', types: ['VaultCreate','VaultSet','VaultDelete','VaultDeposit','VaultWithdraw','VaultClawback']}},
            {{label: 'Other', types: ['DelegateSet']}},
        ];
        const TXN_TYPES = TXN_TYPE_GROUPS.flatMap(g => g.types);
        const TXN_COLORS = {{
            Payment:'#3fb950', TrustSet:'#d29922', AccountSet:'#8b949e',
            TicketCreate:'#7ee787', TicketUse:'#56d364',
            OfferCreate:'#58a6ff', OfferCancel:'#6cb6ff',
            AMMCreate:'#f778ba', AMMDeposit:'#da70d6', AMMWithdraw:'#b8527a',
            NFTokenMint:'#bc8cff', NFTokenBurn:'#986ee2', NFTokenCreateOffer:'#a371f7',
            NFTokenCancelOffer:'#8957e5', NFTokenAcceptOffer:'#c297ff',
            MPTokenIssuanceCreate:'#f0883e', MPTokenIssuanceSet:'#d4762c',
            MPTokenAuthorize:'#e09b4f', MPTokenIssuanceDestroy:'#c45e1a',
            CheckCreate:'#79c0ff', CheckCash:'#58a6ff', CheckCancel:'#388bfd',
            EscrowCreate:'#f0883e', EscrowFinish:'#d29922', EscrowCancel:'#c45e1a',
            CredentialCreate:'#a5d6ff', CredentialAccept:'#79c0ff', CredentialDelete:'#388bfd',
            PermissionedDomainSet:'#d2a8ff', PermissionedDomainDelete:'#a371f7',
            VaultCreate:'#7ee787', VaultSet:'#56d364', VaultDelete:'#3fb950',
            VaultDeposit:'#2ea043', VaultWithdraw:'#238636', VaultClawback:'#196c2e',
            DelegateSet:'#8b949e',
        }};

        // WS terminal display filters (client-side only)
        let activeTxnTypes = new Set(TXN_TYPES);
        const txnFiltersEl = document.getElementById('txn-type-filters');
        TXN_TYPE_GROUPS.forEach(group => {{
            const groupDiv = document.createElement('div');
            groupDiv.className = 'txn-type-group';
            const groupLabel = document.createElement('div');
            groupLabel.className = 'txn-type-group-label';
            groupLabel.textContent = group.label;
            groupDiv.appendChild(groupLabel);
            group.types.forEach(tt => {{
                const lbl = document.createElement('label');
                const color = TXN_COLORS[tt] || '#8b949e';
                lbl.className = 'checked';
                lbl.style.borderColor = color;
                lbl.style.color = color;
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.checked = true;
                cb.onchange = function() {{
                    if (this.checked) activeTxnTypes.add(tt);
                    else activeTxnTypes.delete(tt);
                    lbl.className = this.checked ? 'checked' : '';
                    lbl.style.borderColor = this.checked ? color : '#30363d';
                    lbl.style.color = this.checked ? color : '#8b949e';
                }};
                lbl.appendChild(cb);
                lbl.appendChild(document.createTextNode(tt));
                groupDiv.appendChild(lbl);
            }});
            txnFiltersEl.appendChild(groupDiv);
        }});

        // Transaction Control pane (controls actual generation)
        const txnControlEl = document.getElementById('txn-control-pane');
        let enabledGenTypes = new Set(TXN_TYPES);
        let configDisabledTypes = new Set();

        function renderTxnControl() {{
            txnControlEl.innerHTML = '';
            TXN_TYPE_GROUPS.forEach(group => {{
                const groupDiv = document.createElement('div');
                groupDiv.className = 'txn-control-group';
                const groupLabel = document.createElement('div');
                groupLabel.className = 'txn-control-group-label';
                groupLabel.textContent = group.label;
                // Group toggle: click label to toggle all toggleable types in group
                const toggleable = group.types.filter(t => !configDisabledTypes.has(t));
                if (toggleable.length > 0) {{
                    groupLabel.onclick = async () => {{
                        const anyOn = toggleable.some(t => enabledGenTypes.has(t));
                        const newEnabled = !anyOn;
                        for (const tt of toggleable) {{
                            try {{
                                await fetch('/workload/toggle-type', {{
                                    method: 'POST',
                                    headers: {{'Content-Type': 'application/json'}},
                                    body: JSON.stringify({{txn_type: tt, enabled: newEnabled}}),
                                }});
                                if (newEnabled) enabledGenTypes.add(tt);
                                else enabledGenTypes.delete(tt);
                            }} catch(e) {{ console.error('Group toggle failed:', e); }}
                        }}
                        renderTxnControl();
                    }};
                }}
                groupDiv.appendChild(groupLabel);
                group.types.forEach(tt => {{
                    const btn = document.createElement('div');
                    const isConfigDisabled = configDisabledTypes.has(tt);
                    const isOn = enabledGenTypes.has(tt);
                    btn.className = 'txn-toggle' + (isOn ? ' enabled' : '') + (isConfigDisabled ? ' config-disabled' : '');
                    btn.textContent = tt;
                    const color = TXN_COLORS[tt] || '#8b949e';
                    if (isConfigDisabled) {{
                        btn.title = tt + ' — disabled in config.toml (amendment not available)';
                    }} else {{
                        if (isOn) {{ btn.style.borderColor = color; btn.style.color = color; }}
                        btn.onclick = async () => {{
                            const newEnabled = !enabledGenTypes.has(tt);
                            try {{
                                await fetch('/workload/toggle-type', {{
                                    method: 'POST',
                                    headers: {{'Content-Type': 'application/json'}},
                                    body: JSON.stringify({{txn_type: tt, enabled: newEnabled}}),
                                }});
                                if (newEnabled) enabledGenTypes.add(tt);
                                else enabledGenTypes.delete(tt);
                                renderTxnControl();
                            }} catch(e) {{ console.error('Toggle failed:', e); }}
                        }};
                    }}
                    groupDiv.appendChild(btn);
                }});
                txnControlEl.appendChild(groupDiv);
            }});
        }}

        async function refreshDisabledTypes() {{
            try {{
                const res = await fetch('/workload/disabled-types');
                const data = await res.json();
                enabledGenTypes = new Set(data.enabled_types);
                configDisabledTypes = new Set(data.config_disabled || []);
                renderTxnControl();
            }} catch(e) {{ /* workload not ready yet */ }}
        }}
        refreshDisabledTypes();
        setInterval(refreshDisabledTypes, 10000);

        function wsLog(text, cls) {{
            const out = document.getElementById('ws-output');
            const line = document.createElement('div');
            line.className = 'ws-line ' + (cls || '');
            if (cls && cls.startsWith('txn-')) {{
                const tt = cls.slice(4);
                if (TXN_COLORS[tt]) line.style.color = TXN_COLORS[tt];
            }}
            const ts = new Date().toLocaleTimeString('en-US', {{hour12:false}});
            line.textContent = ts + '  ' + text;
            out.appendChild(line);
            while (out.children.length > MAX_LINES) out.removeChild(out.firstChild);
            out.scrollTop = out.scrollHeight;
        }}

        // Returns [text, cssClass, txnType|null]
        function formatMsg(data) {{
            const t = data.type || '';
            if (t === 'ledgerClosed') {{
                return [
                    `LEDGER #${{data.ledger_index}}  txns=${{data.txn_count}}  close=${{data.ledger_time}}`,
                    'ledger', null
                ];
            }}
            if (t === 'transaction') {{
                const tx = data.transaction || {{}};
                const tt = tx.TransactionType || '?';
                const v = data.validated ? 'validated' : 'proposed';
                const r = data.engine_result || data.meta?.TransactionResult || '';
                return [
                    `${{tt}}  ${{tx.Account?.slice(0,12)}}..  ${{r}}  [${{v}}]  ${{tx.hash?.slice(0,16)}}..`,
                    'txn-' + tt, tt
                ];
            }}
            if (t === 'validationReceived') {{
                return [
                    `VALIDATION  ledger=${{data.ledger_index}}  key=${{data.validation_public_key?.slice(0,16)}}..`,
                    'validation', null
                ];
            }}
            if (t === 'serverStatus') {{
                return [
                    `SERVER  load=${{data.load_factor}}  state=${{data.server_status}}`,
                    'server', null
                ];
            }}
            if (t === 'consensusPhase') {{
                return [`CONSENSUS  phase=${{data.consensus}}`, 'consensus', null];
            }}
            if (t === 'peerStatusChange') {{
                return [`PEER  ${{data.action}}  ${{data.address || ''}}`, 'peer', null];
            }}
            return [JSON.stringify(data).slice(0, 200), '', null];
        }}

        function resubscribe() {{
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            // Unsubscribe all, then resubscribe active
            ws.send(JSON.stringify({{command:'unsubscribe', streams:STREAMS.map(s=>s.name)}}));
            const streams = Array.from(activeStreams);
            if (streams.length > 0) {{
                ws.send(JSON.stringify({{command:'subscribe', streams}}));
                wsLog('Subscribed: ' + streams.join(', '), 'info');
            }}
        }}

        function toggleWs() {{
            const btn = document.getElementById('ws-connect-btn');
            const statusEl = document.getElementById('ws-status');
            if (ws && ws.readyState <= WebSocket.OPEN) {{
                ws.close();
                return;
            }}
            const port = nodeSelect.value;
            const url = 'ws://' + location.hostname + ':' + port;
            wsLog('Connecting to ' + url + '...', 'info');
            ws = new WebSocket(url);
            ws.onopen = () => {{
                statusEl.textContent = 'connected';
                statusEl.className = 'ws-status connected';
                btn.textContent = 'Disconnect';
                btn.className = 'active';
                const streams = Array.from(activeStreams);
                ws.send(JSON.stringify({{command:'subscribe', streams}}));
                wsLog('Connected. Subscribed: ' + streams.join(', '), 'info');
            }};
            ws.onmessage = (ev) => {{
                const data = JSON.parse(ev.data);
                if (data.type === 'response') return; // subscribe ack
                const [text, cls, txnType] = formatMsg(data);
                // Filter: if it's a transaction, check txn type filter
                if (txnType && !activeTxnTypes.has(txnType)) return;
                wsLog(text, cls);
                msgCount++;
                document.getElementById('ws-msg-count').textContent = msgCount + ' msgs';
            }};
            ws.onclose = () => {{
                statusEl.textContent = 'disconnected';
                statusEl.className = 'ws-status disconnected';
                btn.textContent = 'Connect';
                btn.className = '';
                wsLog('Disconnected', 'info');
            }};
            ws.onerror = () => wsLog('WebSocket error', 'error');
        }}

        // --- Stats polling (no page reload, keeps WS alive) ---
        function fmt(n) {{ return n == null ? '—' : Number(n).toLocaleString(); }}
        function pct(a, b) {{ return b > 0 ? (a/b*100).toFixed(1) : '0.0'; }}
        function fmtUptime(sec) {{
            if (!sec) return '—';
            const d = Math.floor(sec / 86400), h = Math.floor(sec % 86400 / 3600),
                  m = Math.floor(sec % 3600 / 60), s = sec % 60;
            if (d > 0) return d + 'd ' + h + 'h ' + m + 'm';
            if (h > 0) return h + 'h ' + m + 'm ' + s + 's';
            if (m > 0) return m + 'm ' + s + 's';
            return s + 's';
        }}

        function statCard(label, value, cls, extra, barPct) {{
            let html = '<div class="stat-card"><div class="stat-label">' + label + '</div>';
            html += '<div class="stat-value ' + (cls||'') + '">' + value + '</div>';
            if (extra) html += '<div class="stat-percentage">' + extra + '</div>';
            if (barPct != null) {{
                html += '<div class="progress-bar"><div class="progress-fill ' + (cls||'') + '" style="width:' + barPct + '%"></div></div>';
            }}
            return html + '</div>';
        }}

        async function refreshStats() {{
            try {{
                const [statsRes, feeRes, rateRes, failCodesRes, intentRes] = await Promise.all([
                    fetch('/state/summary').then(r=>r.json()),
                    fetch('/state/fees').then(r=>r.json()),
                    fetch('/workload/rate-controls').then(r=>r.json()),
                    fetch('/state/failure-codes').then(r=>r.json()),
                    fetch('/workload/intent').then(r=>r.json()),
                ]);
                const s = statsRes;
                const f = feeRes;
                const bs = s.by_state || {{}};
                const total = s.total_tracked || 0;
                const validated = bs.VALIDATED || 0;
                const rejected = bs.REJECTED || 0;
                const submitted = bs.SUBMITTED || 0;
                const created = bs.CREATED || 0;
                const retryable = bs.RETRYABLE || 0;
                const expired = bs.EXPIRED || 0;

                // Subtitle
                document.getElementById('subtitle').innerHTML =
                    'Live monitoring &bull; Ledger ' + f.ledger_current_index + ' @ {hostname}' +
                    ' &bull; ' + fmt(s.ledgers_elapsed) + ' ledgers (' + fmtUptime(s.uptime_seconds) + ')';

                // Rate control sliders (don't overwrite while user is dragging or during cooldown)
                const cool = Date.now() < sliderCooldown;
                const tpsInput = document.getElementById('target-tps-input');
                if (!cool && document.activeElement !== tpsInput) {{
                    tpsInput.value = rateRes.target_tps;
                    document.getElementById('target-tps-value').textContent = rateRes.target_tps;
                }}


                const intentInput = document.getElementById('invalid-intent-input');
                const intentPct = Math.round((intentRes.invalid || 0) * 100);
                if (!cool && document.activeElement !== intentInput) {{
                    intentInput.value = intentPct;
                    document.getElementById('invalid-intent-value').textContent = intentPct;
                }}

                // Effective rate display
                const actualTpl = s.ledgers_elapsed > 0 ? (validated / s.ledgers_elapsed).toFixed(1) : '—';
                const actualTps = s.uptime_seconds > 0 ? (validated / s.uptime_seconds).toFixed(1) : '—';
                document.getElementById('effective-rate').innerHTML =
                    'Actual: <b>' + actualTps + '</b> txns/sec &bull; <b>' + actualTpl + '</b> validated/ledger';

                // Fee stats
                const feeWarn = f.minimum_fee > f.base_fee ? 'warning' : 'success';
                const qPct = f.max_queue_size > 0 ? (f.current_queue_size/f.max_queue_size*100) : 0;
                const lastClosed = f.last_closed_txn_count || 0;
                const lPct = f.expected_ledger_size > 0 ? (lastClosed/f.expected_ledger_size*100) : 0;
                document.getElementById('fee-stats').innerHTML =
                    statCard('Fee (min/open/base)', f.minimum_fee+'/'+f.open_ledger_fee+'/'+f.base_fee, feeWarn, 'drops') +
                    statCard('Queue Utilization', f.current_queue_size+'/'+f.max_queue_size, 'info', qPct.toFixed(1)+'%', qPct) +
                    statCard('Ledger Utilization', lastClosed+'/'+f.expected_ledger_size, 'info', lPct.toFixed(1)+'%', lPct);

                // Txn stats
                document.getElementById('txn-stats').innerHTML =
                    statCard('Total Transactions', fmt(total), 'info') +
                    statCard('Validated', fmt(validated), 'success', pct(validated,total)+'%', pct(validated,total)) +
                    statCard('Rejected', fmt(rejected), 'error', pct(rejected,total)+'%', pct(rejected,total)) +
                    statCard('In-Flight', fmt(submitted+created), 'warning', 'Submitted: '+submitted+' | Created: '+created) +
                    statCard('Expired', fmt(expired), '');

                // Tables
                let tablesHtml = '';

                // Transaction type breakdown (left)
                const byTypeTotal = s.by_type_total || {{}};
                const byTypeValidated = s.by_type_validated || {{}};
                const temDisabledTypes = new Set(s.tem_disabled_types || []);
                const typePct = (e) => e[1] > 0 ? (byTypeValidated[e[0]]||0)/e[1] : -1;
                const sortedTypes = Object.entries(byTypeTotal).sort((a,b) => typePct(b)-typePct(a) || b[1]-a[1]);
                if (sortedTypes.length) {{
                    tablesHtml += '<div class="failures-table"><h2>Transaction Types ('+sortedTypes.length+')</h2><table><thead><tr><th>Type</th><th>Success Rate</th></tr></thead><tbody>';
                    sortedTypes.forEach(([t,total]) => {{
                        const validated = byTypeValidated[t] || 0;
                        const pct = total > 0 ? Math.round(validated/total*100) : 0;
                        const color = pct >= 80 ? '#3fb950' : pct >= 50 ? '#d29922' : '#f85149';
                        const disabled = configDisabledTypes.has(t);
                        const amendmentDisabled = temDisabledTypes.has(t);
                        const link = '<a href="/state/types/'+t+'" target="_blank" style="text-decoration:none;color:inherit">'+t+'</a>';
                        const nameHtml = disabled
                            ? '<s style="color:#484f58">'+link+'</s> <span style="color:#484f58;font-size:11px">disabled</span>'
                            : amendmentDisabled
                            ? '<span style="opacity:0.4">'+link+'</span> <span style="color:#484f58;font-size:11px">temDISABLED</span>'
                            : link;
                        tablesHtml += '<tr><td>'+nameHtml+'</td><td><span style="color:'+color+';font-weight:600">'+fmt(validated)+'/'+fmt(total)+' ('+pct+'%)</span></td></tr>';
                    }});
                    tablesHtml += '</tbody></table></div>';
                }}

                // Top failures (right) — uses cumulative counters (survives pending cleanup)
                const topFail = (failCodesRes.failure_codes || []).slice(0, 10);
                if (topFail.length) {{
                    tablesHtml += '<div class="failures-table"><h2><a href="/state/failures" target="_blank" style="text-decoration:none;color:inherit;cursor:pointer">Top Failures &#8599;</a></h2><table><thead><tr><th>Error Code</th><th>Count</th></tr></thead><tbody>';
                    topFail.forEach(([r,c]) => {{
                        tablesHtml += '<tr><td><a href="/state/failures/'+r+'" target="_blank" style="text-decoration:none"><span class="badge error" style="cursor:pointer">'+r+'</span></a></td><td>'+fmt(c)+'</td></tr>';
                    }});
                    tablesHtml += '</tbody></table></div>';
                }}

                document.getElementById('tables-container').innerHTML = tablesHtml;

            }} catch(e) {{
                console.error('Stats refresh error:', e);
            }}
        }}

        // Initial load + periodic refresh
        refreshStats();
        setInterval(refreshStats, 3000);

</script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.get("/failures")
async def state_all_failures_page(request: Request) -> HTMLResponse:
    """HTML page showing all failure codes with cumulative counts."""
    failure_codes = request.app.state.workload.snapshot_failure_codes()
    sorted_codes = sorted(failure_codes.items(), key=lambda x: x[1], reverse=True)
    total = sum(failure_codes.values())

    rows = ""
    for code, count in sorted_codes:
        rows += (
            f"<tr>"
            f'<td><a href="/state/failures/{code}" style="text-decoration:none">'
            f'<span class="badge">{code}</span></a></td>'
            f"<td>{count:,}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
    <html><head>
    <title>All Failures</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; padding: 20px; }}
        h1 {{ color: #f85149; }}
        a {{ color: #58a6ff; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 600px; margin-top: 16px; }}
        th, td {{ padding: 8px 12px; border: 1px solid #30363d; text-align: left; font-size: 13px; }}
        th {{ background: #161b22; color: #8b949e; text-transform: uppercase; font-size: 11px; }}
        tr:hover {{ background: #161b22; }}
        .badge {{ padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; background: #3d1f1f; color: #f85149; }}
        .count {{ color: #8b949e; font-size: 14px; }}
    </style>
    </head><body>
    <a href="/state/dashboard">&larr; Dashboard</a>
    <h1>All Failures <span class="count">{total:,} total across {len(sorted_codes)} codes</span></h1>
    <table>
        <thead><tr><th>Error Code</th><th>Count</th></tr></thead>
        <tbody>{rows if rows else '<tr><td colspan="2" style="text-align:center;color:#8b949e">No failures recorded</td></tr>'}</tbody>
    </table>
    <p style="margin-top:16px;color:#8b949e">JSON: <a href="/state/failure-codes">/state/failure-codes</a></p>
    </body></html>"""
    return HTMLResponse(content=html)


@router.get("/failures/{error_code}")
async def state_failed_page(request: Request, error_code: str) -> HTMLResponse:
    """HTML page showing failed transactions for a specific error code."""
    all_failed = request.app.state.workload.snapshot_failed()
    filtered = [
        f
        for f in all_failed
        if f.get("engine_result_final") == error_code or f.get("engine_result_first") == error_code
    ]

    hostname = RPC.split("//")[1].split(":")[0] if "//" in RPC else RPC.split(":")[0]
    explorer_base = f"http://custom.xrpl.org/{hostname}:6006"
    # tec codes are applied to the ledger and have an on-chain hash
    on_ledger = error_code.startswith("tec") or error_code.startswith("tes")
    rows = ""
    for f in filtered:
        account = f.get("account", "")
        tx_hash = f.get("tx_hash", "")
        account_cell = (
            f'<a href="{explorer_base}/accounts/{account}" target="_blank"><code>{account}</code></a>'
            if account
            else ""
        )
        if on_ledger and tx_hash:
            hash_cell = f'<a href="{explorer_base}/transactions/{tx_hash}" target="_blank"><code>{tx_hash}</code></a>'
        elif tx_hash:
            hash_cell = f'<a href="/state/tx/{tx_hash}" target="_blank"><code>{tx_hash}</code></a>'
        else:
            hash_cell = f"<code>{tx_hash}</code>"
        msg = f.get("engine_result_message") or ""
        rows += (
            f"<tr>"
            f"<td>{hash_cell}</td>"
            f"<td>{f.get('transaction_type', '')}</td>"
            f"<td>{account_cell}</td>"
            f"<td>{f.get('sequence', '')}</td>"
            f"<td>{f.get('state', '')}</td>"
            f"<td>{f.get('created_ledger', '')}</td>"
            f"<td>{f.get('last_ledger_seq', '')}</td>"
            f"<td style='font-size:11px;color:#8b949e'>{msg}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
    <html><head>
    <title>Failed: {error_code}</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; padding: 20px; }}
        h1 {{ color: #f85149; }}
        a {{ color: #58a6ff; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
        th, td {{ padding: 8px 12px; border: 1px solid #30363d; text-align: left; font-size: 13px; }}
        th {{ background: #161b22; color: #8b949e; text-transform: uppercase; font-size: 11px; cursor: pointer; user-select: none; }}
        th:hover {{ color: #c9d1d9; }}
        th .sort-arrow {{ font-size: 9px; margin-left: 4px; }}
        tr:hover {{ background: #161b22; }}
        code {{ color: #58a6ff; }}
        .badge {{ padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; background: #3d1f1f; color: #f85149; }}
        .count {{ color: #8b949e; font-size: 14px; }}
    </style>
    </head><body>
    <a href="/state/dashboard">&larr; Dashboard</a>
    <h1><span class="badge">{error_code}</span> <span class="count">{len(filtered)} transactions</span></h1>
    <table id="results-table">
        <thead><tr>
            <th data-col="0">Hash <span class="sort-arrow"></span></th>
            <th data-col="1">Type <span class="sort-arrow"></span></th>
            <th data-col="2">Account <span class="sort-arrow"></span></th>
            <th data-col="3" data-type="num">Seq <span class="sort-arrow"></span></th>
            <th data-col="4">State <span class="sort-arrow"></span></th>
            <th data-col="5" data-type="num">Created Ledger <span class="sort-arrow"></span></th>
            <th data-col="6" data-type="num">Last Ledger Seq <span class="sort-arrow"></span></th>
            <th data-col="7">Message <span class="sort-arrow"></span></th>
        </tr></thead>
        <tbody>{rows if rows else '<tr><td colspan="8" style="text-align:center;color:#8b949e">No transactions found</td></tr>'}</tbody>
    </table>
    <p style="margin-top:16px;color:#8b949e">JSON: <a href="/state/failed/{error_code}">/state/failed/{error_code}</a></p>
    <script>
    (function() {{
        const table = document.getElementById('results-table');
        const headers = table.querySelectorAll('th');
        let sortCol = -1, sortAsc = true;

        headers.forEach(th => {{
            th.addEventListener('click', () => {{
                const col = parseInt(th.dataset.col);
                const isNum = th.dataset.type === 'num';
                if (sortCol === col) {{ sortAsc = !sortAsc; }}
                else {{ sortCol = col; sortAsc = true; }}

                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                rows.sort((a, b) => {{
                    const aText = a.cells[col]?.textContent.trim() || '';
                    const bText = b.cells[col]?.textContent.trim() || '';
                    if (isNum) {{
                        const aNum = parseInt(aText) || 0;
                        const bNum = parseInt(bText) || 0;
                        return sortAsc ? aNum - bNum : bNum - aNum;
                    }}
                    return sortAsc ? aText.localeCompare(bText) : bText.localeCompare(aText);
                }});
                rows.forEach(r => tbody.appendChild(r));

                headers.forEach(h => h.querySelector('.sort-arrow').textContent = '');
                th.querySelector('.sort-arrow').textContent = sortAsc ? ' ▲' : ' ▼';
            }});
        }});
    }})();
    </script>
    </body></html>"""
    return HTMLResponse(content=html)


@router.get("/types/{txn_type}")
async def state_type_page(request: Request, txn_type: str) -> HTMLResponse:
    """HTML page showing transactions for a specific type."""
    wl = request.app.state.workload
    filtered = [r for r in wl.snapshot_pending(open_only=False) if r.get("transaction_type") == txn_type]

    hostname = RPC.split("//")[1].split(":")[0] if "//" in RPC else RPC.split(":")[0]
    explorer_base = f"http://custom.xrpl.org/{hostname}:6006"
    rows = ""
    for f in filtered:
        account = f.get("account", "")
        tx_hash = f.get("tx_hash", "")
        state = f.get("state", "")
        account_cell = (
            f'<a href="{explorer_base}/accounts/{account}" target="_blank"><code>{account}</code></a>'
            if account
            else ""
        )
        on_ledger = state == "VALIDATED" or f.get("engine_result_final", "").startswith("tec")
        hash_cell = (
            f'<a href="{explorer_base}/transactions/{tx_hash}" target="_blank"><code>{tx_hash}</code></a>'
            if on_ledger and tx_hash
            else f"<code>{tx_hash}</code>"
        )
        er = f.get("engine_result_final") or f.get("engine_result_first") or ""
        state_cls = "success" if state == "VALIDATED" else "error" if state in ("REJECTED", "EXPIRED") else "warning"
        msg = f.get("engine_result_message") or ""
        rows += (
            f"<tr>"
            f"<td>{hash_cell}</td>"
            f"<td>{account_cell}</td>"
            f"<td>{f.get('sequence', '')}</td>"
            f"<td><span class='badge {state_cls}'>{state}</span></td>"
            f"<td><span class='badge'>{er}</span></td>"
            f"<td>{f.get('created_ledger', '')}</td>"
            f"<td>{f.get('validated_ledger') or ''}</td>"
            f"<td style='font-size:11px;color:#8b949e'>{msg}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
    <html><head>
    <title>{txn_type}</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; padding: 20px; }}
        h1 {{ color: #58a6ff; }}
        a {{ color: #58a6ff; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
        th, td {{ padding: 8px 12px; border: 1px solid #30363d; text-align: left; font-size: 13px; }}
        th {{ background: #161b22; color: #8b949e; text-transform: uppercase; font-size: 11px; cursor: pointer; user-select: none; }}
        th:hover {{ color: #c9d1d9; }}
        th .sort-arrow {{ font-size: 9px; margin-left: 4px; }}
        tr:hover {{ background: #161b22; }}
        code {{ color: #58a6ff; }}
        .badge {{ padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
        .badge.success {{ background: #1f3d2a; color: #3fb950; }}
        .badge.error {{ background: #3d1f1f; color: #f85149; }}
        .badge.warning {{ background: #3d2f1f; color: #d29922; }}
        .count {{ color: #8b949e; font-size: 14px; }}
    </style>
    </head><body>
    <a href="/state/dashboard">&larr; Dashboard</a>
    <h1>{txn_type} <span class="count">{len(filtered)} transactions</span></h1>
    <table id="results-table">
        <thead><tr>
            <th data-col="0">Hash <span class="sort-arrow"></span></th>
            <th data-col="1">Account <span class="sort-arrow"></span></th>
            <th data-col="2" data-type="num">Seq <span class="sort-arrow"></span></th>
            <th data-col="3">State <span class="sort-arrow"></span></th>
            <th data-col="4">Result <span class="sort-arrow"></span></th>
            <th data-col="5" data-type="num">Created Ledger <span class="sort-arrow"></span></th>
            <th data-col="6" data-type="num">Validated Ledger <span class="sort-arrow"></span></th>
            <th data-col="7">Message <span class="sort-arrow"></span></th>
        </tr></thead>
        <tbody>{rows if rows else '<tr><td colspan="8" style="text-align:center;color:#8b949e">No transactions found</td></tr>'}</tbody>
    </table>
    <p style="margin-top:16px;color:#8b949e">JSON: <a href="/state/type/{txn_type}">/state/type/{txn_type}</a></p>
    <script>
    (function() {{{{
        const table = document.getElementById('results-table');
        const headers = table.querySelectorAll('th');
        let sortCol = -1, sortAsc = true;

        headers.forEach(th => {{{{
            th.addEventListener('click', () => {{{{
                const col = parseInt(th.dataset.col);
                const isNum = th.dataset.type === 'num';
                if (sortCol === col) {{{{ sortAsc = !sortAsc; }}}}
                else {{{{ sortCol = col; sortAsc = true; }}}}

                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                rows.sort((a, b) => {{{{
                    const aText = a.cells[col]?.textContent.trim() || '';
                    const bText = b.cells[col]?.textContent.trim() || '';
                    if (isNum) {{{{
                        const aNum = parseInt(aText) || 0;
                        const bNum = parseInt(bText) || 0;
                        return sortAsc ? aNum - bNum : bNum - aNum;
                    }}}}
                    return sortAsc ? aText.localeCompare(bText) : bText.localeCompare(aText);
                }}}});
                rows.forEach(r => tbody.appendChild(r));

                headers.forEach(h => h.querySelector('.sort-arrow').textContent = '');
                th.querySelector('.sort-arrow').textContent = sortAsc ? ' ▲' : ' ▼';
            }}}});
        }}}});
    }}}})();
    </script>
    </body></html>"""
    return HTMLResponse(content=html)


@router.get("/mpt-issuances", response_class=HTMLResponse)
async def state_mptokens_page(request: Request) -> HTMLResponse:
    """HTML page listing all tracked MPToken issuance IDs."""
    wl = request.app.state.workload
    mptoken_ids = getattr(wl, "_mptoken_issuance_ids", {})

    rows = ""
    for i, (mpt_id, issuer) in enumerate(mptoken_ids.items()):
        rows += f"<tr><td>{i}</td><td><code>{mpt_id}</code></td><td><code>{issuer}</code></td></tr>"

    html = f"""<!DOCTYPE html>
    <html><head>
    <title>MPToken Issuances</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; padding: 20px; }}
        h1 {{ color: #f0883e; }}
        a {{ color: #58a6ff; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
        th, td {{ padding: 8px 12px; border: 1px solid #30363d; text-align: left; font-size: 13px; }}
        th {{ background: #161b22; color: #8b949e; text-transform: uppercase; font-size: 11px; }}
        tr:hover {{ background: #161b22; }}
        code {{ color: #f0883e; }}
        .count {{ color: #8b949e; font-size: 14px; }}
    </style>
    </head><body>
    <a href="/state/dashboard">&larr; Dashboard</a>
    <h1>MPToken Issuances <span class="count">{len(mptoken_ids)} issuances</span></h1>
    <table>
        <thead><tr><th>#</th><th>MPToken Issuance ID</th><th>Issuer</th></tr></thead>
        <tbody>{rows if rows else '<tr><td colspan="3" style="text-align:center;color:#8b949e">No MPToken issuances tracked</td></tr>'}</tbody>
    </table>
    <p style="margin-top:16px;color:#8b949e">JSON: <a href="/state/mptokens">/state/mptokens</a></p>
    </body></html>"""
    return HTMLResponse(content=html)
