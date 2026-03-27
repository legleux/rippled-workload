from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from workload.workload_core import Workload

router = APIRouter(prefix="/dex", tags=["DEX"])


@router.get("/metrics")
async def dex_metrics(request: Request):
    """Get DEX metrics including AMM pool states, trading activity counts."""
    return request.app.state.workload.snapshot_dex_metrics()


@router.get("/pools")
async def dex_pools(request: Request):
    """List all tracked AMM pools."""
    w: Workload = request.app.state.workload
    return {
        "total_pools": len(w.amm.pools),
        "pools": w.amm.pools,
    }


def _fmt_asset(a: dict) -> str:
    if a.get("currency") == "XRP":
        return "XRP"
    return f"{a['currency']}/{a.get('issuer', '?')[:12]}..."


@router.get("/amm-pools", response_class=HTMLResponse)
async def dex_pools_page(request: Request) -> HTMLResponse:
    """HTML page listing all tracked AMM pools."""
    w: Workload = request.app.state.workload
    pools = w.amm.pools

    rows = ""
    for i, pool in enumerate(pools):
        a1 = _fmt_asset(pool["asset1"])
        a2 = _fmt_asset(pool["asset2"])
        creator = pool.get("creator", "")
        rows += (
            f"<tr>"
            f'<td><a href="/dex/pools/{i}" target="_blank">{i}</a></td>'
            f"<td>{a1}</td>"
            f"<td>{a2}</td>"
            f"<td><code>{creator}</code></td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
    <html><head>
    <title>AMM Pools</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; padding: 20px; }}
        h1 {{ color: #f778ba; }}
        a {{ color: #58a6ff; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
        th, td {{ padding: 8px 12px; border: 1px solid #30363d; text-align: left; font-size: 13px; }}
        th {{ background: #161b22; color: #8b949e; text-transform: uppercase; font-size: 11px; cursor: pointer; user-select: none; }}
        th:hover {{ color: #c9d1d9; }}
        th .sort-arrow {{ font-size: 9px; margin-left: 4px; }}
        tr:hover {{ background: #161b22; }}
        code {{ color: #58a6ff; }}
        .count {{ color: #8b949e; font-size: 14px; }}
    </style>
    </head><body>
    <a href="/state/dashboard">&larr; Dashboard</a>
    <h1>AMM Pools <span class="count">{len(pools)} pools</span></h1>
    <table id="results-table">
        <thead><tr>
            <th data-col="0" data-type="num"># <span class="sort-arrow"></span></th>
            <th data-col="1">Asset 1 <span class="sort-arrow"></span></th>
            <th data-col="2">Asset 2 <span class="sort-arrow"></span></th>
            <th data-col="3">Creator <span class="sort-arrow"></span></th>
        </tr></thead>
        <tbody>{rows if rows else '<tr><td colspan="4" style="text-align:center;color:#8b949e">No AMM pools tracked</td></tr>'}</tbody>
    </table>
    <p style="margin-top:16px;color:#8b949e">JSON: <a href="/dex/pools">/dex/pools</a> &bull; Metrics: <a href="/dex/metrics">/dex/metrics</a></p>
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
                    if (isNum) return sortAsc ? (parseInt(aText)||0) - (parseInt(bText)||0) : (parseInt(bText)||0) - (parseInt(aText)||0);
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


@router.get("/pools/{index}")
async def dex_pool_detail(index: int, request: Request):
    """Get detailed amm_info for a specific pool by index."""
    from xrpl.models.currencies import XRP as XRPCurrency
    from xrpl.models.requests import AMMInfo

    w: Workload = request.app.state.workload
    if index >= len(w.amm.pools):
        raise HTTPException(status_code=404, detail=f"Pool index {index} not found")

    pool = w.amm.pools[index]
    asset1, asset2 = pool["asset1"], pool["asset2"]

    if asset1.get("currency") == "XRP":
        a1 = XRPCurrency()
    else:
        from xrpl.models import IssuedCurrency

        a1 = IssuedCurrency(currency=asset1["currency"], issuer=asset1["issuer"])

    if asset2.get("currency") == "XRP":
        a2 = XRPCurrency()
    else:
        from xrpl.models import IssuedCurrency

        a2 = IssuedCurrency(currency=asset2["currency"], issuer=asset2["issuer"])

    resp = await w._rpc(AMMInfo(asset=a1, asset2=a2))
    return resp.result


@router.post("/poll")
async def dex_poll_now(request: Request):
    """Manually trigger a DEX metrics poll."""
    return await request.app.state.workload.poll_dex_metrics()
