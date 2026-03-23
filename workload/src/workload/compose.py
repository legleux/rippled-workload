"""Generate the outer docker-compose.yml that includes the testnet and adds the workload service."""

from pathlib import Path

from ruamel.yaml import YAML


def write_workload_compose(testnet_dir: str = "testnet") -> Path:
    """Write docker-compose.yml that includes testnet/ and adds the workload service."""
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
                "depends_on": {"rippled": {"condition": "service_started"}},
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
