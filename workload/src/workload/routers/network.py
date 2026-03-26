import os
from pathlib import Path

from fastapi import APIRouter

from workload import workload_runner

router = APIRouter(prefix="/network", tags=["Network"])


@router.post("/reset")
async def network_reset():
    """Reset the network: stop workload, regenerate testnet, restart containers, restart workload.

    This calls `gen auto` to regenerate the testnet, then `docker compose down/up`.
    Requires gen CLI and docker compose to be available on the host.
    """
    import shutil
    import subprocess

    # 1. Stop workload if running
    await workload_runner.force_stop()

    testnet_dir = os.getenv("TESTNET_DIR", str(Path(__file__).resolve().parents[4] / "prepare-workload" / "testnet"))
    gen_bin = os.getenv("GEN_BIN", "gen")

    steps = []

    # 2. Docker compose down
    try:
        r = subprocess.run(
            ["docker", "compose", "down"],
            cwd=testnet_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        steps.append({"step": "docker compose down", "returncode": r.returncode, "stderr": r.stderr.strip()})
    except Exception as e:
        steps.append({"step": "docker compose down", "error": str(e)})

    # 3. Clean old artifacts
    for name in ["volumes", "ledger.json", "accounts.json", "docker-compose.yml"]:
        target = Path(testnet_dir) / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.is_file():
            target.unlink()
    # Clean state.db (check both workload cwd and parent)
    for db_path in [Path("state.db"), Path(__file__).resolve().parents[4] / "state.db"]:
        if db_path.is_file():
            db_path.unlink()
    steps.append({"step": "clean artifacts", "status": "ok"})

    # 4. Regenerate with library API
    try:
        from workload.gen_cmd import run_gen

        num_validators = int(os.getenv("TESTNET_VALIDATORS", "5"))
        run_gen(output_dir=testnet_dir, num_validators=num_validators)
        steps.append({"step": "gen (library)", "status": "ok"})
    except Exception as e:
        steps.append({"step": "gen (library)", "error": str(e)})

    # 5. Docker compose up
    try:
        r = subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=testnet_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        steps.append({"step": "docker compose up", "returncode": r.returncode, "stderr": r.stderr.strip()[-200:]})
    except Exception as e:
        steps.append({"step": "docker compose up", "error": str(e)})

    return {
        "status": "reset complete — restart the workload process to reconnect",
        "steps": steps,
    }
