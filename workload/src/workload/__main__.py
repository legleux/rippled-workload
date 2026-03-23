import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="workload", description="XRPL workload generator")
    subs = parser.add_subparsers(dest="command")

    # --- run (explicit server start, native) ---
    run_p = subs.add_parser("run", help="Start the workload server")
    run_p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    run_p.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")

    # --- gen (generate testnet + docker-compose.yml) ---
    gen_p = subs.add_parser("gen", help="Generate testnet configs from config.toml")
    gen_p.add_argument("-o", "--output-dir", default="testnet", help="Output directory (default: testnet)")
    gen_p.add_argument("-v", "--validators", type=int, default=5, help="Number of validators (default: 5)")
    gen_p.add_argument(
        "--amendment-profile",
        default=None,
        help='Amendment profile: "release", "develop", or "custom"',
    )

    # --- compose (write docker-compose.yml for existing testnet) ---
    compose_p = subs.add_parser("compose", help="Write docker-compose.yml for existing testnet")
    compose_p.add_argument("-o", "--output-dir", default="testnet", help="Testnet directory (default: testnet)")

    args = parser.parse_args()

    match args.command:
        case "gen":
            try:
                from workload.gen_cmd import run_gen
            except ModuleNotFoundError:
                print(
                    "Error: 'workload gen' requires the generate_ledger package.\n"
                    "Install it with:\n"
                    "  uv pip install -e /path/to/generate_ledger\n\n"
                    "If you already have a testnet/ directory, use 'workload compose' instead."
                )
                raise SystemExit(1)

            run_gen(
                output_dir=args.output_dir,
                num_validators=args.validators,
                amendment_profile=args.amendment_profile,
            )
        case "compose":
            from pathlib import Path

            from workload.compose import write_workload_compose

            testnet = Path(args.output_dir)
            if not testnet.exists():
                print(f"Error: {testnet}/ does not exist. Run 'workload gen' first or provide the right path.")
                raise SystemExit(1)
            write_workload_compose(args.output_dir)
            print(f"Wrote docker-compose.yml (includes {args.output_dir}/)")
            print("\nTo start: docker compose up -d --build")
        case "run" | None:
            import uvicorn

            host = getattr(args, "host", "0.0.0.0")
            port = getattr(args, "port", 8000)
            uvicorn.run("workload.app:app", host=host, port=port, lifespan="on")


if __name__ == "__main__":
    main()
