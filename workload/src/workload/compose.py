"""Generate the outer docker-compose.yml that includes the testnet and adds the workload service."""

from pathlib import Path

from ruamel.yaml import YAML


def _find_p2p_service(testnet_dir: str) -> str:
    """Find the p2p node service name from the testnet docker-compose.yml."""
    testnet_compose = Path(testnet_dir) / "docker-compose.yml"
    if testnet_compose.exists():
        yaml = YAML()
        with open(testnet_compose) as f:
            tc = yaml.load(f)
        for name, svc in (tc.get("services") or {}).items():
            # p2p node exposes RPC port 5005
            ports = svc.get("ports", [])
            if any("5005" in str(p) for p in ports):
                return name
    return "xrpld"


def write_workload_compose(testnet_dir: str = "testnet") -> Path:
    """Write docker-compose.yml that includes testnet/ and adds the workload service."""
    p2p_service = _find_p2p_service(testnet_dir)
    compose = {
        "include": [f"{testnet_dir}/docker-compose.yml"],
        "services": {
            "workload": {
                "image": "workload:latest",
                "build": {"context": ".", "dockerfile": "Dockerfile"},
                "container_name": "workload",
                "hostname": "workload",
                "ports": ["8000:8000"],
                "init": True,
                "restart": "on-failure",
                "depends_on": {p2p_service: {"condition": "service_started"}},
                "networks": {"xrpl_net": {}},
                "volumes": [f"./{testnet_dir}/accounts.json:/workload/testnet/accounts.json:ro"],
            },
        },
    }
    out = Path("docker-compose.yml")
    yaml = YAML()
    yaml.default_flow_style = False
    with open(out, "w") as f:
        yaml.dump(compose, f)
    return out
