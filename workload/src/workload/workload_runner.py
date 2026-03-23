"""Workload runner — continuous transaction submission loop and lifecycle controls.

Extracted from app.py to keep routing separate from workload execution logic.
"""

import asyncio
import logging
import time
from dataclasses import replace
from time import perf_counter

import workload.constants as C
from workload.constants import TxIntent
from workload.randoms import choices, random
from workload.txn_factory.builder import build_txn_dict, pick_eligible_txn_type, txn_model_cls
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
    """Continuously build and submit transactions as fast as possible.

    No queue, no producer-consumer split. Single loop that:
    1. Finds free accounts (no in-flight txns)
    2. Builds + signs txns for up to `target_txns_per_ledger` accounts
    3. Submits them all in parallel via TaskGroup
    4. Immediately loops back — no waiting for ledger close

    The ledger close is the tick for *validation tracking*, not for submission.
    Real-world users submit whenever they want; so do we.

    Sequence safety: an account is only picked when pending_count == 0.
    record_created() marks it pending at build time, preventing double-allocation.
    """
    global stats

    log.debug("Continuous workload started")
    stats["started_at"] = perf_counter()

    consecutive_empty = 0
    last_account_create_ledger = 0

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

                # Find free accounts
                pending_counts = wl.get_pending_txn_counts_by_account()
                free_accounts = [addr for addr in wl.wallets if pending_counts.get(addr, 0) == 0]

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
                            log.warning("workload: self-heal expired %d stale txns at ledger %d", expired, batch_ledger)
                            pending_counts = wl.get_pending_txn_counts_by_account()
                            free_accounts = [addr for addr in wl.wallets if pending_counts.get(addr, 0) == 0]
                            n_free = len(free_accounts)
                            if n_free > 0:
                                log.info("workload: self-heal freed %d accounts", n_free)
                        elif consecutive_empty % 10 == 0:
                            diag = wl.diagnostics_snapshot()
                            log.warning(
                                "workload: STUCK for %d iterations — blocked=%d free=%d pending_by_state=%s oldest_age=%d",
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
                            "workload: recovered — %d free accounts after %d empty iterations", n_free, consecutive_empty
                        )
                    consecutive_empty = 0

                # Occasionally grow the account pool (once per ledger)
                current_ledger = wl.latest_ledger_index
                if (
                    current_ledger > last_account_create_ledger
                    and random() < 0.50
                    and "Payment" not in wl.disabled_txn_types
                ):
                    funding_pending = pending_counts.get(wl.funding_wallet.address, 0)
                    if funding_pending == 0:
                        try:
                            default_balance = wl.config["users"]["default_balance"]
                            large_balance = str(int(default_balance) * 10)
                            await wl.create_account(initial_xrp_drops=large_balance)
                            stats["submitted"] += 1
                            last_account_create_ledger = current_ledger
                        except Exception as e:
                            log.debug("Failed to create new account: %s", e)
                            stats["failed"] += 1

                # Build + sign txns for free accounts
                # TODO: Parallelize this loop — alloc_seq RPCs and signing are sequential.
                # With 400 accounts, build phase takes 2-3s. Use TaskGroup for alloc_seq,
                # then sign concurrently. This is the main throughput bottleneck.
                target = wl.target_txns_per_ledger
                intent_cfg = wl.config.get("transactions", {}).get("intent", {})
                intent_weights = [intent_cfg.get("valid", 0.90), intent_cfg.get("invalid", 0.10)]
                intent_choices = [TxIntent.VALID, TxIntent.INVALID]
                build_start = perf_counter()
                batch: list[PendingTx] = []
                for addr in free_accounts[:target]:
                    wallet = wl.wallets[addr]
                    intent = choices(intent_choices, weights=intent_weights, k=1)[0]
                    txn_type = pick_eligible_txn_type(wallet, wl.ctx, intent)
                    if txn_type is None:
                        continue

                    try:
                        ctx = replace(wl.ctx, forced_account=wallet)
                        composed = build_txn_dict(txn_type, ctx, intent)
                        if composed is None:
                            continue
                        txn = txn_model_cls(txn_type).from_xrpl(composed)
                    except Exception as e:
                        log.debug("build %s/%s: %s", addr, txn_type, e)
                        continue

                    try:
                        seq = await wl.alloc_seq(wallet.address)
                    except Exception as e:
                        log.debug("alloc_seq %s: %s", addr, e)
                        continue

                    try:
                        pending = await wl.build_sign_and_track(
                            txn,
                            wallet,
                            fee_drops=batch_fee,
                            created_ledger=batch_ledger,
                            last_ledger_seq=batch_lls,
                            preallocated_seq=seq,
                        )
                        batch.append(pending)
                    except Exception as e:
                        await wl.release_seq(wallet.address, seq)
                        log.debug("sign %s/%s: %s", addr, txn_type, e)
                        continue

                if not batch:
                    await asyncio.sleep(0)
                    continue

                build_ms = (perf_counter() - build_start) * 1000
                log.info(
                    "Batch: %d txns, ledger=%d, wallets=%d, build=%.0fms",
                    len(batch), batch_ledger, len(wl.wallets), build_ms,
                )

                # Submit all in parallel — no waiting, fire immediately
                try:
                    async with asyncio.TaskGroup() as tg:
                        tasks = [tg.create_task(wl.submit_pending(p)) for p in batch]

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
                        log.info("Batch result: %d ok, %d failed — errors=%s", submitted, failed, errors)
                    else:
                        log.info("Batch result: %d ok", submitted)
                except* Exception as eg:
                    for exc in eg.exceptions:
                        log.error("Batch error: %s: %s", type(exc).__name__, exc)
                    stats["failed"] += len(batch)

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
        "message": "Continuous workload started - submitting random transactions at expected_ledger_size + 1 per ledger (max 200)",
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
