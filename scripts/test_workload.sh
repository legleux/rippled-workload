#!/usr/bin/env bash
# Self-contained workload test: clean → gen → up → monitor → report
# Usage: ./scripts/test_workload.sh [duration_seconds]
#   duration_seconds: how long to monitor (default: 300 = 5 minutes)
set -euo pipefail

DURATION="${1:-300}"
WORKLOAD_DIR="$(cd "$(dirname "$0")/../workload" && pwd)"
COMPOSE_FILE="$WORKLOAD_DIR/docker-compose.yml"
API="http://localhost:8000"
TIMESTAMP="$(date -u +%Y-%m-%d-%H%M)"
REPORT="$WORKLOAD_DIR/docs/todo/${TIMESTAMP}-test-results.md"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# ── Step 1: Stop existing network ──────────────────────────────────
log "Stopping existing network (if any)..."
docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true

# ── Step 2: Clean testnet artifacts ────────────────────────────────
log "Cleaning testnet artifacts..."
rm -rf "$WORKLOAD_DIR/testnet"

# ── Step 3: Generate testnet ───────────────────────────────────────
log "Generating testnet config..."
cd "$WORKLOAD_DIR"
uv run --group gen workload gen

# ── Step 4: Build and start ────────────────────────────────────────
log "Building and starting network..."
docker compose -f "$COMPOSE_FILE" up -d --build

# ── Step 5: Wait for workload bootstrap ────────────────────────────
log "Waiting for workload to bootstrap..."
MAX_WAIT=180
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -sf "$API/health" >/dev/null 2>&1; then
        # Health is up, but check if workload has started submitting
        STATUS=$(curl -sf "$API/state/summary" 2>/dev/null || echo "")
        if echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('ledger_index',0)>5 else 1)" 2>/dev/null; then
            log "Workload is submitting (ledger > 5)"
            break
        fi
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    log "  waiting... (${ELAPSED}s/${MAX_WAIT}s)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    log "ERROR: Workload did not start within ${MAX_WAIT}s"
    docker compose -f "$COMPOSE_FILE" logs workload | tail -30
    exit 1
fi

# ── Step 6: Set intent to 0% invalid ──────────────────────────────
log "Setting intent to 0% invalid (all valid)..."
curl -sf -X POST "$API/workload/intent" \
    -H 'Content-Type: application/json' \
    -d '{"invalid": 0.0}' | python3 -m json.tool

# Wait 15s for pipeline to flush pre-change txns
log "Flushing pipeline (15s)..."
sleep 15

# ── Step 7: Take baseline snapshot ─────────────────────────────────
log "Taking baseline snapshot..."
BASELINE_CODES=$(curl -sf "$API/state/failure-codes")
BASELINE_SUMMARY=$(curl -sf "$API/state/summary")

# ── Step 8: Monitor for $DURATION seconds ──────────────────────────
log "Monitoring for ${DURATION}s..."
sleep "$DURATION"

# ── Step 9: Collect final snapshots ────────────────────────────────
log "Collecting final snapshots..."
FINAL_CODES=$(curl -sf "$API/state/failure-codes")
FINAL_SUMMARY=$(curl -sf "$API/state/summary")
FINAL_WS=$(curl -sf "$API/state/ws/stats")
FINAL_DIAG=$(curl -sf "$API/state/diagnostics")
FINAL_INTENT=$(curl -sf "$API/workload/intent")

# ── Step 10: Generate report ───────────────────────────────────────
log "Generating report at $REPORT..."

python3 - "$BASELINE_CODES" "$BASELINE_SUMMARY" "$FINAL_CODES" "$FINAL_SUMMARY" "$FINAL_WS" "$FINAL_DIAG" "$FINAL_INTENT" "$DURATION" "$REPORT" <<'PYEOF'
import json, sys

b_codes_raw, b_summary_raw, f_codes_raw, f_summary_raw, f_ws_raw, f_diag_raw, f_intent_raw, duration, report_path = sys.argv[1:]

b_codes = json.loads(b_codes_raw)
b_summary = json.loads(b_summary_raw)
f_codes = json.loads(f_codes_raw)
f_summary = json.loads(f_summary_raw)
f_ws = json.loads(f_ws_raw)
f_diag = json.loads(f_diag_raw)
f_intent = json.loads(f_intent_raw)

# Compute deltas
b_s = b_summary.get("by_state", {})
f_s = f_summary.get("by_state", {})
d_val = f_s.get("VALIDATED", 0) - b_s.get("VALIDATED", 0)
d_rej = f_s.get("REJECTED", 0) - b_s.get("REJECTED", 0)
d_exp = f_s.get("EXPIRED", 0) - b_s.get("EXPIRED", 0)
d_total = d_val + d_rej + d_exp
d_rate = d_val / d_total * 100 if d_total else 0

# Failure code deltas
b_fc = {c[0]: c[1] for c in b_codes.get("failure_codes", [])}
f_fc = {c[0]: c[1] for c in f_codes.get("failure_codes", [])}
all_codes = set(list(b_fc.keys()) + list(f_fc.keys()))
deltas = {}
for code in all_codes:
    d = f_fc.get(code, 0) - b_fc.get(code, 0)
    if d > 0:
        deltas[code] = d

ws_counters = f_ws.get("ws_event_counters", {})
val_by_src = f_summary.get("validated_by_source", {})

lines = []
lines.append(f"# Workload Test Results")
lines.append(f"")
lines.append(f"**Date:** {report_path.split('/')[-1][:16]}")
lines.append(f"**Duration:** {duration}s (monitoring window only, excludes bootstrap)")
lines.append(f"**Intent:** {f_intent}")
lines.append(f"**Ledger range:** {b_summary.get('ledger_index', '?')} → {f_summary.get('ledger_index', '?')}")
lines.append(f"")
lines.append(f"## Results (delta during monitoring window)")
lines.append(f"")
lines.append(f"| Metric | Value |")
lines.append(f"|--------|-------|")
lines.append(f"| Validated | {d_val} |")
lines.append(f"| Rejected | {d_rej} |")
lines.append(f"| Expired | {d_exp} |")
lines.append(f"| **Success rate** | **{d_rate:.1f}%** |")
lines.append(f"| tefPAST_SEQ (delta) | {deltas.get('tefPAST_SEQ', 0)} |")
lines.append(f"| terPRE_SEQ (delta) | {deltas.get('terPRE_SEQ', 0)} |")
lines.append(f"")
lines.append(f"## Failure Codes (delta)")
lines.append(f"")
lines.append(f"| Code | Count |")
lines.append(f"|------|-------|")
for code, count in sorted(deltas.items(), key=lambda x: -x[1]):
    lines.append(f"| {code} | {count} |")
lines.append(f"| **Total** | **{sum(deltas.values())}** |")
lines.append(f"")
lines.append(f"## WS Stats (cumulative)")
lines.append(f"")
lines.append(f"| Metric | Value |")
lines.append(f"|--------|-------|")
lines.append(f"| validated_matched | {ws_counters.get('validated_matched', 0)} |")
lines.append(f"| validated_unmatched | {ws_counters.get('validated_unmatched', 0)} |")
lines.append(f"| proposed_matched | {ws_counters.get('proposed_matched', 0)} |")
lines.append(f"| proposed_unmatched | {ws_counters.get('proposed_unmatched', 0)} |")
lines.append(f"| Validated by WS | {val_by_src.get('ws', 0)} |")
lines.append(f"| Validated by POLL | {val_by_src.get('poll', 0)} |")
lines.append(f"")
lines.append(f"## Diagnostics (final)")
lines.append(f"")
lines.append(f"| Metric | Value |")
lines.append(f"|--------|-------|")
lines.append(f"| Blocked accounts | {f_diag.get('blocked_accounts', '?')} |")
lines.append(f"| Free accounts | {f_diag.get('free_accounts', '?')} |")
lines.append(f"| Oldest pending age | {f_diag.get('oldest_pending_age_ledgers', '?')} ledgers |")
lines.append(f"| Pending states | {f_diag.get('pending_by_state', {})} |")

with open(report_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\n{'='*60}")
print(f"  SUCCESS RATE: {d_rate:.1f}%")
print(f"  validated={d_val} rejected={d_rej} expired={d_exp}")
print(f"  tefPAST_SEQ={deltas.get('tefPAST_SEQ', 0)}")
print(f"  WS matched={ws_counters.get('validated_matched', 0)} unmatched={ws_counters.get('validated_unmatched', 0)}")
print(f"  Poll validations={val_by_src.get('poll', 0)}")
print(f"  Report: {report_path}")
print(f"{'='*60}")
PYEOF

log "Done. Network is still running."
log "Dashboard: $API/state/dashboard"
