"""Bridge between workload config.toml and gl (generate_ledger) library API."""

from pathlib import Path

import tomllib

from gl.compose import ComposeConfig, write_compose_file
from gl.ledger import LedgerConfig, write_ledger_file
from gl.rippled_cfg import RippledConfigSpec

from workload.compose import write_workload_compose

CONFIG_PATH = Path(__file__).parent / "config.toml"


def _load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def run_gen(
    output_dir: str = "testnet",
    num_validators: int = 5,
    amendment_profile: str | None = None,
    amendment_source: str | None = None,
    enable_amendments: list[str] | None = None,
    disable_amendments: list[str] | None = None,
) -> None:
    """Read config.toml and call generate_ledger to produce a testnet."""
    cfg = _load_config()

    gw_number = cfg["gateways"]["number"]
    user_number = cfg["users"]["number"]
    total_accounts = gw_number + user_number

    user_balance = str(cfg["users"]["default_balance"])

    # Currencies for gateway trustlines
    genesis_currencies = cfg.get("genesis", {}).get("currencies")
    if genesis_currencies is None:
        genesis_currencies = cfg["currencies"]["codes"][:4]
    assets_per_gateway = len(genesis_currencies) // gw_number if gw_number else 4

    base_dir = Path(output_dir)

    # --- Ledger config (pass nested configs as dicts for pydantic-settings) ---
    ledger_kwargs: dict = {
        "account_cfg": {
            "num_accounts": total_accounts,
            "balance": user_balance,
        },
        "gateway_cfg": {
            "num_gateways": gw_number,
            "assets_per_gateway": assets_per_gateway,
            "currencies": genesis_currencies,
        },
        "base_dir": base_dir,
    }
    if amendment_profile is not None:
        ledger_kwargs["amendment_profile"] = amendment_profile
    if amendment_source is not None:
        ledger_kwargs["amendment_profile_source"] = amendment_source
    if enable_amendments:
        ledger_kwargs["enable_amendments"] = enable_amendments
    if disable_amendments:
        ledger_kwargs["disable_amendments"] = disable_amendments

    ledger_cfg = LedgerConfig(**ledger_kwargs)

    # --- Write ledger.json + accounts.json ---
    write_ledger_file(config=ledger_cfg)

    # --- Rippled configs ---
    rippled_cfg = RippledConfigSpec(
        num_validators=num_validators,
        base_dir=base_dir / "volumes",
        reference_fee=ledger_cfg.fee_cfg.base_fee_drops,
        account_reserve=ledger_cfg.fee_cfg.reserve_base_drops,
        owner_reserve=ledger_cfg.fee_cfg.reserve_increment_drops,
    )
    rippled_cfg.write()

    # --- testnet docker-compose.yml ---
    compose_cfg = ComposeConfig(
        num_validators=num_validators,
        base_dir=base_dir,
    )
    write_compose_file(config=compose_cfg)

    # --- outer docker-compose.yml (includes testnet + workload service) ---
    write_workload_compose(output_dir)

    print(f"\nTestnet generated in {base_dir.resolve()}/")
    print(f"  Accounts: {total_accounts} ({gw_number} gateways + {user_number} users)")
    print(f"  Validators: {num_validators}")
    print("\nTo start: docker compose up -d --build")
