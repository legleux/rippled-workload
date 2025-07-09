from dataclasses import dataclass
from enum import StrEnum
from ruamel.yaml import YAML
from pathlib import Path
import os

yaml = YAML()
yaml.indent(offset=2, sequence=4)

num_validators = int(os.environ.get("NUM_VALIDATORS", 5))
validator_name = os.environ.get("VALIDATOR_NAME", "val")
rippled_name = os.environ.get("RIPPLED_NAME", "rippled")

network_name = "antithesis_net"
ledger_file = "ledger.json"
image = "rippled:latest"
name = "val"
entrypoint = "rippled"
load_command = {"command": ["--ledgerfile", ledger_file]}
start_command = "--start"
init = True
healthcheck = {
      "test": ["CMD", "/usr/bin/curl", "--insecure", "https://localhost:51235/health"],
      "interval": 5,
}
port = {
    "rpc": 5005,
    "ws": 6006,
}

compose_data = {
    "services": {
        (name:=rippled_name if i >= num_validators else f"{validator_name}{i}"): {
        # f"val{i}": {
            "image": image,
            "container_name": f"{name}",
            "hostname": f"{name}",
            "entrypoint": [entrypoint],  # ✅ use the variable here
            **({"ports": [f'{port["ws"]}:{port["ws"]}']} if i >= num_validators else {}),
            **(load_command if i == 0 else {}),
            "volumes": [
                f"./volumes/{name}:/etc/opt/ripple",
                *([f"./{ledger_file}:/{ledger_file}"] if i == 0 else [])
                # "./ledger.json:/ledger.json" if i == 0 else None,
            ],
            "networks": [network_name]
        }
        for i in range(num_validators + 1 )
    },
    "networks": {
        network_name : {
            "name": network_name
        },
    },
}

test_net_dir = Path("test_network")
compose_yml = test_net_dir / "docker-compose.yml"
test_net_dir.mkdir(exist_ok=True, parents=True)
with compose_yml.open("w") as f:
    yaml.dump(compose_data, f)


@dataclass
class Config(StrEnum):
    pass

def generate_validator_config():
    services = {
    }
def generate_rippled_config():
    pass
def generate_config():
    pass
