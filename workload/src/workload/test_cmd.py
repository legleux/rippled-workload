"""Lifecycle test: clean → gen → up → monitor → report."""

import datetime
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

POLL_INTERVAL = 30  # seconds between monitoring progress prints
BOOT_POLL_INTERVAL = 5


@dataclass
class TestConfig:
    duration: int = 300
    do_clean: bool = False
    do_gen: bool = False
    do_up: bool = False
    api_url: str = "http://localhost:8000"
    output_dir: str = "testnet"
    validators: int = 5
    boot_timeout: int = 180
    flush_seconds: int = 15
    focus: list[str] = field(default_factory=list)
    intent_invalid: float = 0.0


def _log(msg: str) -> None:
    ts = datetime.datetime.now(datetime.UTC).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _compose_file() -> Path:
    return Path("docker-compose.yml")


# ── Steps ─────────────────────────────────────────────────────────


def step_stop() -> None:
    _log("Stopping existing network (if any)...")
    subprocess.run(
        ["docker", "compose", "-f", str(_compose_file()), "down"],
        capture_output=True,
    )


def step_clean(cfg: TestConfig) -> None:
    _log(f"Cleaning {cfg.output_dir}/...")
    shutil.rmtree(cfg.output_dir, ignore_errors=True)


def step_gen(cfg: TestConfig) -> None:
    _log("Generating testnet config...")
    try:
        from workload.gen_cmd import run_gen
    except ModuleNotFoundError:
        print(
            "Error: 'workload test --gen' requires the generate_ledger package.\n"
            "Install it with:\n"
            "  uv pip install -e /path/to/generate_ledger\n\n"
            "If you already have a testnet/ directory, drop --gen."
        )
        raise SystemExit(1) from None
    run_gen(output_dir=cfg.output_dir, num_validators=cfg.validators)


def step_up() -> None:
    _log("Building and starting network...")
    subprocess.run(
        ["docker", "compose", "-f", str(_compose_file()), "up", "-d", "--build"],
        check=True,
    )


def step_wait_bootstrap(cfg: TestConfig) -> None:
    _log("Waiting for workload to bootstrap...")
    elapsed = 0
    with httpx.Client(timeout=5) as client:
        while elapsed < cfg.boot_timeout:
            try:
                resp = client.get(f"{cfg.api_url}/health")
                if resp.is_success:
                    summary = client.get(f"{cfg.api_url}/state/summary").json()
                    if summary.get("ledger_index", 0) > 5:
                        _log("Workload is submitting (ledger > 5)")
                        return
            except httpx.HTTPError:
                pass
            time.sleep(BOOT_POLL_INTERVAL)
            elapsed += BOOT_POLL_INTERVAL
            _log(f"  waiting... ({elapsed}s/{cfg.boot_timeout}s)")

    _log(f"ERROR: Workload did not start within {cfg.boot_timeout}s")
    subprocess.run(
        ["docker", "compose", "-f", str(_compose_file()), "logs", "workload", "--tail", "30"],
    )
    raise SystemExit(1)


def step_set_intent(cfg: TestConfig) -> None:
    _log(f"Setting intent to {cfg.intent_invalid:.0%} invalid...")
    with httpx.Client(timeout=10) as client:
        resp = client.post(
            f"{cfg.api_url}/workload/intent",
            json={"invalid": cfg.intent_invalid},
        )
        print(json.dumps(resp.json(), indent=2))

    _log(f"Flushing pipeline ({cfg.flush_seconds}s)...")
    time.sleep(cfg.flush_seconds)


def step_snapshot(cfg: TestConfig) -> dict:
    endpoints = {
        "summary": "/state/summary",
        "failure_codes": "/state/failure-codes",
        "ws_stats": "/state/ws/stats",
        "diagnostics": "/state/diagnostics",
        "intent": "/workload/intent",
    }
    for extra in cfg.focus:
        key = extra.replace("/", "_")
        endpoints[key] = f"/{extra.lstrip('/')}"

    snap: dict = {}
    with httpx.Client(timeout=10) as client:
        for key, path in endpoints.items():
            try:
                resp = client.get(f"{cfg.api_url}{path}")
                snap[key] = resp.json()
            except Exception:
                snap[key] = {}
    return snap


def step_monitor(cfg: TestConfig) -> None:
    _log(f"Monitoring for {cfg.duration}s...")
    elapsed = 0
    with httpx.Client(timeout=5) as client:
        while elapsed < cfg.duration:
            sleep_for = min(POLL_INTERVAL, cfg.duration - elapsed)
            time.sleep(sleep_for)
            elapsed += sleep_for
            try:
                s = client.get(f"{cfg.api_url}/state/summary").json()
                by_state = s.get("by_state", {})
                _log(
                    f"  ledger={s.get('ledger_index', '?')} "
                    f"validated={by_state.get('VALIDATED', 0)} "
                    f"rejected={by_state.get('REJECTED', 0)} "
                    f"expired={by_state.get('EXPIRED', 0)} "
                    f"({elapsed}/{cfg.duration}s)"
                )
            except Exception:
                _log(f"  (API unreachable) ({elapsed}/{cfg.duration}s)")


# ── Report ─────────────────────────────────────────────────────────


def generate_report(baseline: dict, final: dict, cfg: TestConfig) -> Path:
    b_s = baseline.get("summary", {}).get("by_state", {})
    f_s = final.get("summary", {}).get("by_state", {})
    d_val = f_s.get("VALIDATED", 0) - b_s.get("VALIDATED", 0)
    d_rej = f_s.get("REJECTED", 0) - b_s.get("REJECTED", 0)
    d_exp = f_s.get("EXPIRED", 0) - b_s.get("EXPIRED", 0)
    d_total = d_val + d_rej + d_exp
    d_rate = d_val / d_total * 100 if d_total else 0

    # Failure code deltas
    b_fc = {c[0]: c[1] for c in baseline.get("failure_codes", {}).get("failure_codes", [])}
    f_fc = {c[0]: c[1] for c in final.get("failure_codes", {}).get("failure_codes", [])}
    deltas = {}
    for code in set(b_fc) | set(f_fc):
        d = f_fc.get(code, 0) - b_fc.get(code, 0)
        if d > 0:
            deltas[code] = d

    ws_counters = final.get("ws_stats", {}).get("ws_event_counters", {})
    val_by_src = final.get("summary", {}).get("validated_by_source", {})
    f_diag = final.get("diagnostics", {})
    pending_by_state = f_diag.get("pending_by_state", {})

    lines = [
        "# Workload Test Results",
        "",
        f"**Date:** {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Duration:** {cfg.duration}s (monitoring window only, excludes bootstrap)",
        f"**Intent:** {json.dumps(final.get('intent', {}))}",
        f"**Ledger range:** {baseline.get('summary', {}).get('ledger_index', '?')}"
        f" → {final.get('summary', {}).get('ledger_index', '?')}",
        "",
        "## Results (delta during monitoring window)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Validated | {d_val} |",
        f"| Rejected | {d_rej} |",
        f"| Expired | {d_exp} |",
        f"| **Success rate** | **{d_rate:.1f}%** |",
        f"| tefPAST_SEQ (delta) | {deltas.get('tefPAST_SEQ', 0)} |",
        f"| terPRE_SEQ (delta) | {deltas.get('terPRE_SEQ', 0)} |",
        "",
        "## Failure Codes (delta)",
        "",
        "| Code | Count |",
        "|------|-------|",
    ]
    for code, count in sorted(deltas.items(), key=lambda x: -x[1]):
        lines.append(f"| {code} | {count} |")
    lines.append(f"| **Total** | **{sum(deltas.values())}** |")

    lines += [
        "",
        "## WS Stats (cumulative)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| validated_matched | {ws_counters.get('validated_matched', 0)} |",
        f"| validated_unmatched | {ws_counters.get('validated_unmatched', 0)} |",
        f"| proposed_matched | {ws_counters.get('proposed_matched', 0)} |",
        f"| proposed_unmatched | {ws_counters.get('proposed_unmatched', 0)} |",
        f"| Validated by WS | {val_by_src.get('ws', 0)} |",
        f"| Validated by POLL | {val_by_src.get('poll', 0)} |",
        "",
        "## Diagnostics (final)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Blocked accounts | {f_diag.get('blocked_accounts', '?')} |",
        f"| Free accounts | {f_diag.get('free_accounts', '?')} |",
        f"| Oldest pending age | {f_diag.get('oldest_pending_age_ledgers', '?')} ledgers |",
        f"| Pending states | {pending_by_state} |",
    ]

    # Focus endpoint data
    standard_keys = {"summary", "failure_codes", "ws_stats", "diagnostics", "intent"}
    for key, data in final.items():
        if key not in standard_keys and data:
            lines += [
                "",
                f"## Focus: {key}",
                "",
                "```json",
                json.dumps(data, indent=2),
                "```",
            ]

    report_dir = Path("docs/todo")
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d-%H%M")
    report_path = report_dir / f"{stamp}-test-results.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


# ── Orchestrator ───────────────────────────────────────────────────


def run_test(cfg: TestConfig) -> None:
    try:
        _run_test_inner(cfg)
    except KeyboardInterrupt:
        print()
        _log("Interrupted. Network is still running.")
        sys.exit(130)


def _run_test_inner(cfg: TestConfig) -> None:
    if cfg.do_clean:
        step_stop()
        step_clean(cfg)

    if cfg.do_gen:
        step_gen(cfg)

    if cfg.do_up:
        step_up()
        step_wait_bootstrap(cfg)

    # Set intent and flush
    step_set_intent(cfg)

    # Baseline
    _log("Taking baseline snapshot...")
    baseline = step_snapshot(cfg)

    # Monitor
    step_monitor(cfg)

    # Final
    _log("Collecting final snapshots...")
    final = step_snapshot(cfg)

    # Report
    report_path = generate_report(baseline, final, cfg)

    # Summary
    b_s = baseline.get("summary", {}).get("by_state", {})
    f_s = final.get("summary", {}).get("by_state", {})
    d_val = f_s.get("VALIDATED", 0) - b_s.get("VALIDATED", 0)
    d_rej = f_s.get("REJECTED", 0) - b_s.get("REJECTED", 0)
    d_exp = f_s.get("EXPIRED", 0) - b_s.get("EXPIRED", 0)
    d_total = d_val + d_rej + d_exp
    d_rate = d_val / d_total * 100 if d_total else 0

    ws_counters = final.get("ws_stats", {}).get("ws_event_counters", {})
    val_by_src = final.get("summary", {}).get("validated_by_source", {})

    print(f"\n{'=' * 60}")
    print(f"  SUCCESS RATE: {d_rate:.1f}%")
    print(f"  validated={d_val} rejected={d_rej} expired={d_exp}")
    ws_match = ws_counters.get("validated_matched", 0)
    ws_unmatch = ws_counters.get("validated_unmatched", 0)
    print(f"  WS matched={ws_match} unmatched={ws_unmatch}")
    print(f"  Poll validations={val_by_src.get('poll', 0)}")
    print(f"  Report: {report_path}")
    print(f"{'=' * 60}")

    _log("Done. Network is still running.")
    _log(f"Dashboard: {cfg.api_url}/state/dashboard")
