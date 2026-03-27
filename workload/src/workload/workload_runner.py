"""Workload runner — continuous transaction submission loop and lifecycle controls.

Extracted from app.py to keep routing separate from workload execution logic.
"""

import asyncio
import logging
import time
from dataclasses import replace
from time import perf_counter

from xrpl.models import Transaction
from xrpl.wallet import Wallet

import workload.constants as C
from workload.constants import TxIntent
from workload.txn_factory import build_txn_dict, compose_submission_set, txn_model_cls
from workload.txn_factory.taint import taint_txn
from workload.workload_core import PendingTx, Workload

log = logging.getLogger("workload.runner")

# ── Module-level state ──────────────────────────────────────────────────────

running = False
_stop_event: asyncio.Event | None = None
_task: asyncio.Task | None = None
stats = {"submitted": 0, "validated": 0, "failed": 0, "started_at": None}


def _log_task_exception(task: asyncio.Task) -> None:
    """Done callback: log unhandled exceptions from fire-and-forget tasks."""
    if not task.cancelled() and (exc := task.exception()):
        log.error("Task %r crashed: %s: %s", task.get_name(), type(exc).__name__, exc, exc_info=exc)


# ── Core loop ────────────────────────────────────────────────────────────────


async def continuous_workload(wl: Workload) -> None:
    """Continuously build and submit transactions, rate-limited by a token bucket.

    Single loop that:
    1. Finds free accounts (pending_count < max_pending_per_account)
    2. Splits into clean (0 pending, eligible for any intent) and partial pools
    3. Applies token-bucket rate limiting if target_tps > 0
    4. Builds + signs txns for up to submission_set_size accounts
    5. Submits them all in parallel via TaskGroup
    6. Loops back — sleeps only if rate-limited or no free accounts

    Rate control: target_tps=0 means unlimited (firehose). Any positive value
    uses a token bucket that accumulates tokens at target_tps/sec, capped at
    2 seconds of burst.
    """
    global stats

    log.debug("Continuous workload started")
    stats["started_at"] = perf_counter()

    consecutive_empty = 0

    # Token bucket state
    token_budget: float = 0.0
    last_tick = perf_counter()

    # Fetch fee once, then rely on WS ledgerClosed updates (or re-fetch on telINSUF_FEE_P)
    if wl._cached_fee is None:
        try:
            fee_info = await asyncio.wait_for(wl.get_fee_info(), timeout=5.0)
            wl._cached_fee = fee_info.minimum_fee
            log.info("workload: fetched base fee = %d drops", wl._cached_fee)
        except Exception as e:
            log.warning("workload: initial fee fetch failed: %s", e)

    try:
        while not _stop_event.is_set():
            try:
                # Wait for first ledger index from WS
                batch_ledger = wl.latest_ledger_index
                if batch_ledger == 0:
                    await asyncio.sleep(0.5)
                    continue
                batch_lls = batch_ledger + C.HORIZON

                # Re-fetch fee if invalidated (telINSUF_FEE_P sets _cached_fee = None)
                if wl._cached_fee is None:
                    try:
                        fee_info = await asyncio.wait_for(wl.get_fee_info(), timeout=2.0)
                        wl._cached_fee = fee_info.minimum_fee
                        log.info("workload: refreshed fee = %d drops (fee escalation detected)", wl._cached_fee)
                    except Exception:
                        pass
                batch_fee = wl._cached_fee or 10

                # ── Token bucket rate limiting ──────────────────────────
                now = perf_counter()
                elapsed = now - last_tick
                last_tick = now
                tps = wl.target_tps

                if tps > 0:
                    token_budget = min(token_budget + elapsed * tps, tps * 2)  # cap burst at 2s
                    if token_budget < 1:
                        await asyncio.sleep(0.05)
                        continue

                # ── Find free accounts ──────────────────────────────────
                max_p = wl.max_pending_per_account
                pending_counts = wl.get_pending_txn_counts_by_account()
                # Clean accounts (0 pending) can receive any intent including INVALID
                clean_accounts = [addr for addr in wl.wallets if pending_counts.get(addr, 0) == 0]
                # Partial accounts (>0 but below max) can only receive VALID intent
                partial_accounts = [addr for addr in wl.wallets if 0 < pending_counts.get(addr, 0) < max_p]
                free_accounts = clean_accounts + partial_accounts
                n_free = len(free_accounts)

                if n_free == 0:
                    consecutive_empty += 1
                    if consecutive_empty == 1 or consecutive_empty % 5 == 0:
                        log.warning(
                            "workload: 0 free accounts (consecutive=%d, total=%d, pending=%d)",
                            consecutive_empty,
                            len(wl.wallets),
                            sum(pending_counts.values()),
                        )
                    # Self-healing: force-expire txns past their LastLedgerSequence
                    if consecutive_empty >= 3:
                        expired = wl.expire_past_lls(batch_ledger)
                        if expired:
                            log.warning(
                                "workload: self-heal expired %d stale txns at ledger %d",
                                expired,
                                batch_ledger,
                            )
                            pending_counts = wl.get_pending_txn_counts_by_account()
                            clean_accounts = [addr for addr in wl.wallets if pending_counts.get(addr, 0) == 0]
                            partial_accounts = [addr for addr in wl.wallets if 0 < pending_counts.get(addr, 0) < max_p]
                            free_accounts = clean_accounts + partial_accounts
                            n_free = len(free_accounts)
                            if n_free > 0:
                                log.info("workload: self-heal freed %d accounts", n_free)
                        elif consecutive_empty % 10 == 0:
                            diag = wl.diagnostics_snapshot()
                            log.warning(
                                "workload: STUCK %d iterations — blocked=%d free=%d states=%s age=%d",
                                consecutive_empty,
                                diag["blocked_accounts"],
                                diag["free_accounts"],
                                diag["pending_by_state"],
                                diag["oldest_pending_age_ledgers"],
                            )
                    if n_free == 0:
                        await asyncio.sleep(0.5)
                        continue
                else:
                    if consecutive_empty >= 3:
                        log.info(
                            "workload: recovered — %d free accounts after %d empty iterations",
                            n_free,
                            consecutive_empty,
                        )
                    consecutive_empty = 0

                # ── Determine target for this iteration ─────────────────
                if tps > 0:
                    target = min(int(token_budget), wl.submission_set_size, n_free)
                else:
                    target = min(wl.submission_set_size, n_free)

                # Two-phase build: compose dicts (sync), then parallel alloc_seq + sign
                build_start = perf_counter()

                # Type-first: roll types, assign eligible accounts, determine intent
                assignments = compose_submission_set(free_accounts, clean_accounts, target, wl.ctx, wl.config)

                # Phase 1: Build txn dicts (sync, no RPC)
                built: list[tuple[Wallet, Transaction, bool]] = []
                for addr, txn_type, intent in assignments:
                    wallet = wl.wallets[addr]
                    is_invalid = intent == TxIntent.INVALID
                    try:
                        ctx = replace(wl.ctx, forced_account=wallet)
                        composed = build_txn_dict(txn_type, ctx, intent)
                        if composed is None:
                            continue
                        if is_invalid:
                            composed = taint_txn(composed, txn_type)
                        txn = txn_model_cls(txn_type).from_xrpl(composed)
                        built.append((wallet, txn, is_invalid))
                    except Exception as e:
                        log.debug("build %s/%s: %s", addr, txn_type, e)

                if not built:
                    await asyncio.sleep(0)
                    continue

                # Phase 2: Parallel alloc_seq + sign (concurrent RPC + crypto)
                async def _alloc_and_sign(
                    w: Wallet,
                    t: Transaction,
                    inv: bool,
                    _fee: int = batch_fee,
                    _ledger: int = batch_ledger,
                    _lls: int = batch_lls,
                ) -> PendingTx | None:
                    try:
                        seq = await wl.alloc_seq(w.address)
                    except Exception as e:
                        log.debug("alloc_seq %s: %s", w.address, e)
                        return None
                    try:
                        return await wl.build_sign_and_track(
                            t,
                            w,
                            fee_drops=_fee,
                            created_ledger=_ledger,
                            last_ledger_seq=_lls,
                            preallocated_seq=seq,
                            expect_rejection=inv,
                        )
                    except Exception as e:
                        await wl.release_seq(w.address, seq)
                        log.debug("sign %s: %s", w.address, e)
                        return None

                async with asyncio.TaskGroup() as tg:
                    sign_tasks = [tg.create_task(_alloc_and_sign(w, t, inv)) for w, t, inv in built]
                submission_set: list[PendingTx] = [t.result() for t in sign_tasks if t.result() is not None]

                if not submission_set:
                    await asyncio.sleep(0)
                    continue

                build_ms = (perf_counter() - build_start) * 1000
                log.info(
                    "Submission set: %d txns, ledger=%d, wallets=%d, build=%.0fms",
                    len(submission_set),
                    batch_ledger,
                    len(wl.wallets),
                    build_ms,
                )

                # Submit all in parallel — no waiting, fire immediately
                try:
                    async with asyncio.TaskGroup() as tg:
                        tasks = [tg.create_task(wl.submit_pending(p)) for p in submission_set]

                    submitted = 0
                    failed = 0
                    errors: dict[str, int] = {}
                    for task in tasks:
                        result = task.result()
                        if result is None:
                            failed += 1
                            errors["build_failed"] = errors.get("build_failed", 0) + 1
                        else:
                            er = result.get("engine_result")
                            if er and er not in ("terQUEUED",) and er.startswith(("ter", "tem", "tef", "tel")):
                                failed += 1
                                errors[er] = errors.get(er, 0) + 1
                            else:
                                submitted += 1
                    stats["submitted"] += submitted
                    stats["failed"] += failed
                    if "telINSUF_FEE_P" in errors:
                        wl._cached_fee = None
                        log.warning("Fee escalation detected — invalidating cached fee")
                    if failed:
                        log.info("Submit result: %d ok, %d failed — errors=%s", submitted, failed, errors)
                    else:
                        log.info("Submit result: %d ok", submitted)
                except* Exception as eg:
                    for exc in eg.exceptions:
                        log.error("Submit error: %s: %s", type(exc).__name__, exc)
                    stats["failed"] += len(submission_set)

                # Deduct from token bucket after submission
                if tps > 0:
                    token_budget -= len(submission_set)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("WORKLOAD CRASH (recovering): %s: %s", type(e).__name__, e, exc_info=True)
                await asyncio.sleep(1)

    except asyncio.CancelledError:
        log.debug("Continuous workload cancelled")
        raise
    finally:
        log.debug("Continuous workload stopped — stats: %s", stats)


# ── Lifecycle ────────────────────────────────────────────────────────────────


async def start(wl: Workload) -> dict:
    """Start continuous random transaction workload."""
    global running, _stop_event, _task, stats

    if running:
        raise RuntimeError("Workload already running")

    stats = {"submitted": 0, "validated": 0, "failed": 0, "started_at": perf_counter()}

    log.info("Starting workload")
    _stop_event = asyncio.Event()
    _task = asyncio.create_task(continuous_workload(wl), name="continuous_workload")
    _task.add_done_callback(_log_task_exception)
    running = True
    wl.workload_started = True

    return {
        "status": "started",
        "message": "Continuous workload started",
    }


async def stop(wl: Workload) -> dict:
    """Stop continuous workload."""
    global running

    if not running:
        raise RuntimeError("Workload not running")

    log.info("Stopping workload")
    _stop_event.set()
    await _task
    stop_ledger = await wl._current_ledger_index()
    log.info("Stopped workload at ledger %s", stop_ledger)
    running = False
    wl.workload_started = False

    return {"status": "stopped", "stats": stats}


async def force_stop() -> None:
    """Force-stop workload during shutdown or reset. Does not raise if not running."""
    global running

    if running and _stop_event:
        _stop_event.set()
        if _task:
            try:
                await asyncio.wait_for(_task, timeout=3.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                log.info("Workload task didn't stop cleanly, continuing")
        running = False


def status(wl: Workload) -> dict:
    """Get current workload status and statistics."""
    return {
        "running": running,
        "stats": stats,
        "uptime_seconds": round(time.time() - wl.started_at),
        "started_at": wl.started_at,
    }
