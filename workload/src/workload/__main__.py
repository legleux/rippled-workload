import argparse

from workload import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="workload", description="XRPL workload generator")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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

    # --- test (lifecycle: clean -> gen -> up -> monitor -> report) ---
    test_p = subs.add_parser("test", help="Run lifecycle test: monitor (default), opt-in to clean/gen/up")
    test_p.add_argument("--clean", action="store_true", help="Stop network and delete testnet/ before starting")
    test_p.add_argument("--gen", action="store_true", help="Generate testnet config (implies --clean)")
    test_p.add_argument("--up", action="store_true", help="Build and start network (implies --gen --clean)")
    test_p.add_argument("--no-clean", action="store_true", help="Override: skip clean even with --gen/--up")
    test_p.add_argument("--duration", type=int, default=300, help="Monitoring window in seconds (default: 300)")
    test_p.add_argument("--api-url", default="http://localhost:8000", help="Workload API URL")
    test_p.add_argument("-v", "--validators", type=int, default=5, help="Number of validators (default: 5)")
    test_p.add_argument("--boot-timeout", type=int, default=180, help="Max seconds to wait for bootstrap")
    test_p.add_argument("--intent-invalid", type=float, default=0.0, help="Invalid txn ratio (default: 0.0)")
    test_p.add_argument("--focus", nargs="*", default=[], help="Extra API endpoints to snapshot")

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
        case "test":
            from workload.test_cmd import TestConfig, run_test

            # Implication chain: --up implies --gen implies --clean (unless --no-clean)
            do_gen = args.gen or args.up
            do_clean = (args.clean or do_gen) and not args.no_clean

            cfg = TestConfig(
                duration=args.duration,
                do_clean=do_clean,
                do_gen=do_gen,
                do_up=args.up,
                api_url=args.api_url,
                output_dir="testnet",
                validators=args.validators,
                boot_timeout=args.boot_timeout,
                intent_invalid=args.intent_invalid,
                focus=args.focus or [],
            )
            run_test(cfg)
        case "run" | None:
            import uvicorn

            host = getattr(args, "host", "0.0.0.0")
            port = getattr(args, "port", 8000)
            uvicorn.run("workload.app:app", host=host, port=port, lifespan="on")


if __name__ == "__main__":
    main()
