from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["Logs"])

_LOG_FILE = Path("/tmp/workload.log")
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@router.get("/logs")
async def get_logs(n: int = 200, level: str | None = None):
    """Return the last n log lines, optionally filtered to a minimum level."""
    if not _LOG_FILE.exists():
        return {"lines": [], "file": str(_LOG_FILE)}

    level = level.upper() if level else None
    if level and level not in _LOG_LEVELS:
        raise HTTPException(status_code=400, detail=f"level must be one of {_LOG_LEVELS}")

    lines = _LOG_FILE.read_text(errors="replace").splitlines()

    if level:
        priority = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        min_idx = priority.index(level)
        lines = [l for l in lines if any(f" {lvl} " in l or f" {lvl:<8}" in l for lvl in priority[min_idx:])]

    return {"lines": lines[-n:], "total_matching": len(lines), "file": str(_LOG_FILE)}


@router.get("/logs/page", response_class=HTMLResponse)
async def logs_page(n: int = 300, level: str = "WARNING"):
    """Live log viewer — auto-refreshes every 3 seconds."""
    level_opts = "".join(
        f'<option value="{lv}"{" selected" if lv == level else ""}>{lv}</option>'
        for lv in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>workload logs</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font: 13px/1.5 "Fira Mono", "Cascadia Code", monospace; background: #0d1117; color: #c9d1d9; }}
  header {{ display: flex; align-items: center; gap: 1rem; padding: 0.6rem 1rem;
            background: #161b22; border-bottom: 1px solid #30363d; }}
  header h1 {{ font-size: 0.95rem; color: #58a6ff; }}
  select, input {{ background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
                   border-radius: 4px; padding: 0.2rem 0.4rem; font: inherit; }}
  label {{ font-size: 0.8rem; color: #8b949e; }}
  #log {{ padding: 0.8rem 1rem; white-space: pre-wrap; word-break: break-all; }}
  .WARNING {{ color: #d29922; }}
  .ERROR   {{ color: #f85149; }}
  .CRITICAL {{ color: #ff7b72; font-weight: bold; }}
  .DEBUG   {{ color: #6e7681; }}
  .INFO    {{ color: #c9d1d9; }}
  #status  {{ margin-left: auto; font-size: 0.75rem; color: #6e7681; }}
</style>
</head>
<body>
<header>
  <h1>workload logs</h1>
  <label>Min level
    <select id="lvl" onchange="reload()">{level_opts}</select>
  </label>
  <label>Lines
    <input type="number" id="nlines" value="{n}" min="10" max="2000" style="width:5rem" onchange="reload()">
  </label>
  <span id="status">loading...</span>
</header>
<div id="log"></div>
<script>
  function levelClass(line) {{
    for (const lv of ['CRITICAL','ERROR','WARNING','INFO','DEBUG'])
      if (line.includes(' ' + lv + ' ') || line.includes(lv.padEnd(8))) return lv;
    return '';
  }}
  async function reload() {{
    const lv = document.getElementById('lvl').value;
    const n  = document.getElementById('nlines').value;
    try {{
      const r = await fetch(`/logs?n=${{n}}&level=${{lv}}`);
      const d = await r.json();
      const html = d.lines.map(l => {{
        const cls = levelClass(l);
        const esc = l.replace(/&/g,'&amp;').replace(/</g,'&lt;');
        return cls ? `<span class="${{cls}}">${{esc}}</span>` : esc;
      }}).join('\\n');
      document.getElementById('log').innerHTML = html;
      document.getElementById('status').textContent =
        `${{d.lines.length}} lines shown / ${{d.total_matching}} matching — ${{new Date().toLocaleTimeString()}}`;
      window.scrollTo(0, document.body.scrollHeight);
    }} catch(e) {{ document.getElementById('status').textContent = 'fetch error: ' + e; }}
  }}
  reload();
  setInterval(reload, 3000);
</script>
</body>
</html>"""
