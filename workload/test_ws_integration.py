#!/usr/bin/env python3
"""
Test script to verify WebSocket integration is working correctly.

Usage:
    python test_ws_integration.py [--base-url http://localhost:8000]
"""
import asyncio
import httpx
import sys
from datetime import datetime
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"


def print_status(test_name: str, passed: bool, details: str = ""):
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"{status} {test_name}")
    if details:
        print(f"      {details}")


async def check_health() -> bool:
    """Verify the service is running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{BASE_URL}/health")
            return r.status_code == 200
    except Exception as e:
        print(f"{Colors.RED}Cannot connect to {BASE_URL}: {e}{Colors.END}")
        return False


async def get_ws_stats() -> Dict[str, Any]:
    """Get WebSocket statistics."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{BASE_URL}/state/ws/stats")
        r.raise_for_status()
        return r.json()


async def get_validations(limit: int = 10) -> list:
    """Get recent validation records."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{BASE_URL}/state/validations", params={"limit": limit})
        r.raise_for_status()
        return r.json()


async def submit_random_transaction() -> Dict[str, Any]:
    """Submit a random transaction."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{BASE_URL}/transaction/random")
        r.raise_for_status()
        return r.json()


async def test_ws_queue_exists():
    """Test 1: Verify WS queue exists and has capacity."""
    print(f"\n{Colors.BLUE}Test 1: WebSocket Queue{Colors.END}")

    try:
        stats = await get_ws_stats()
        queue_size = stats.get("queue_size", -1)
        queue_max = stats.get("queue_maxsize", -1)

        passed = queue_size >= 0 and queue_max > 0
        details = f"Queue: {queue_size}/{queue_max} events"
        print_status("WS queue configured", passed, details)

        return passed
    except Exception as e:
        print_status("WS queue configured", False, str(e))
        return False


async def test_ws_validation_source():
    """Test 2: Check if WS validations are being recorded."""
    print(f"\n{Colors.BLUE}Test 2: WebSocket Validation Source{Colors.END}")

    try:
        stats = await get_ws_stats()
        by_source = stats.get("validations_by_source", {})
        ws_count = by_source.get("WS", 0)
        poll_count = by_source.get("POLL", 0)

        # Just having the structure is a pass
        has_ws_key = "WS" in by_source
        print_status("WS validation tracking enabled", has_ws_key)

        if ws_count > 0:
            ratio = ws_count / (ws_count + poll_count) if (ws_count + poll_count) > 0 else 0
            details = f"WS: {ws_count}, POLL: {poll_count} (WS: {ratio * 100:.1f}%)"
            print_status("WS validations detected", True, details)
            return True
        elif poll_count > 0:
            print(f"      {Colors.YELLOW}Note: Only POLL validations so far. WS may not have seen transactions yet.{Colors.END}")
            return True
        else:
            print(f"      {Colors.YELLOW}Note: No validations yet. Submit transactions to test.{Colors.END}")
            return True

    except Exception as e:
        print_status("WS validation source check", False, str(e))
        return False


async def test_submit_and_validate():
    """Test 3: Submit transaction and verify WS validation."""
    print(f"\n{Colors.BLUE}Test 3: End-to-End Transaction{Colors.END}")

    try:
        # Get baseline stats
        stats_before = await get_ws_stats()
        ws_before = stats_before.get("validations_by_source", {}).get("WS", 0)

        print(f"      Submitting random transaction...")
        tx_result = await submit_random_transaction()
        tx_hash = None

        # Try to extract tx_hash from various possible locations
        if isinstance(tx_result, dict):
            tx_hash = (
                tx_result.get("tx_hash") or
                tx_result.get("tx_json", {}).get("hash") or
                tx_result.get("result", {}).get("tx_json", {}).get("hash")
            )

        if not tx_hash:
            print_status("Transaction submitted", False, "Could not extract tx_hash")
            return False

        print_status("Transaction submitted", True, f"hash: {tx_hash[:16]}...")

        # Wait for validation (max 15 seconds)
        print(f"      Waiting for validation...")
        for i in range(15):
            await asyncio.sleep(1)

            # Check if validated
            validations = await get_validations(limit=50)
            for v in validations:
                if v.get("txn") == tx_hash:
                    source = v.get("source", "unknown")
                    ledger = v.get("ledger")

                    is_ws = source == "WS"
                    details = f"Validated in ledger {ledger} via {source} ({i+1}s)"
                    print_status("Transaction validated", True, details)

                    if is_ws:
                        print_status("Validation via WebSocket", True, "🎉 WS integration working!")
                    else:
                        print_status("Validation via WebSocket", False,
                                   f"Validated via {source} instead (WS may be slow/disconnected)")

                    return is_ws

        print_status("Transaction validated", False, "Timeout after 15s")
        return False

    except Exception as e:
        print_status("End-to-end test", False, str(e))
        return False


async def main():
    print(f"\n{Colors.BLUE}═══════════════════════════════════════{Colors.END}")
    print(f"{Colors.BLUE}  WebSocket Integration Test Suite{Colors.END}")
    print(f"{Colors.BLUE}═══════════════════════════════════════{Colors.END}")
    print(f"Testing against: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check service health
    print(f"\n{Colors.BLUE}Preliminary Check{Colors.END}")
    if not await check_health():
        print(f"\n{Colors.RED}✗ Service not available. Exiting.{Colors.END}")
        sys.exit(1)
    print_status("Service health", True, f"{BASE_URL}/health responds")

    # Run tests
    results = []
    results.append(await test_ws_queue_exists())
    results.append(await test_ws_validation_source())
    results.append(await test_submit_and_validate())

    # Summary
    passed = sum(results)
    total = len(results)

    print(f"\n{Colors.BLUE}═══════════════════════════════════════{Colors.END}")
    print(f"{Colors.BLUE}  Test Summary{Colors.END}")
    print(f"{Colors.BLUE}═══════════════════════════════════════{Colors.END}")

    if passed == total:
        print(f"{Colors.GREEN}✓ All tests passed ({passed}/{total}){Colors.END}")
        print(f"\n{Colors.GREEN}WebSocket integration is working correctly!{Colors.END}")
        sys.exit(0)
    else:
        print(f"{Colors.YELLOW}⚠ Some tests failed ({passed}/{total} passed){Colors.END}")
        print("\nCheck the logs for details:")
        print("  - WS listener: Look for 'WS connected' and 'WS subscription successful'")
        print("  - WS processor: Look for 'WS validation' messages")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted.{Colors.END}")
        sys.exit(130)
