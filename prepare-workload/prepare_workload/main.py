import argparse
import tomllib
from pathlib import Path
import prepare_workload.generate_unl as gl
from prepare_workload.settings import settings as get_settings
from prepare_workload.node_config.render_config import config_data
from prepare_workload.compose import render_compose
from prepare_workload.topo2cfg import get_peers
from prepare_workload.generate_unl import generate_unl_data
import json
import shutil

ALL = object() # peers are all connected to each other

def read_network(network_spec_file: Path) -> dict:
    with open(network_spec_file, "rb") as f:
        network = tomllib.load(f)
    return network["edges"]

def write_config(node_data, settings):
    s = settings
    net = s.network
    node = s.node_config
    val_name = net.validator_name
    name = node_data.get("name")
    num_val = net.num_validators
    # FIX: Will not handle multiple peers!
    if net.peers is ALL:
        peers = {name:[f"{val_name}{j}" for j in range(num_val) if f"{val_name}{j}" != name]}
    else:
        peers = settings.network.peers
    is_validator = node_data["is_validator"]
    signing_support = "false" if is_validator else "true"

    node_config_data = {
        "is_validator": is_validator,
        "ports": node.ports,
        "signing_support": signing_support,
        "node_config_template": s.node_config_template,
        "validator_public_keys": "\n".join(node_data["public_keys"]),
        "ips_fixed": "\n".join([f"{p} {settings.node_config.ports["peer"]}" for p in peers[name]])
    }

    if is_validator:
        node_config_data["seed"] = node_data["seed"]

    if node_data["use_unl"]:
        node_config_data.update(validator_list_sites=node_data["validator_list_sites"])
        node_config_data.update(validator_list_keys=node_data["validator_list_keys"])
    node_config_data["use_unl"] = node_data["use_unl"]
    cfg_data = config_data(node_config_data)
    config_dir = s.network_dir_path / s.config_dir / name
    Path(config_dir).mkdir(parents=True, exist_ok=True)
    config_file =  config_dir / s.node_config_file
    config_file.write_text(cfg_data)

def write_compose(settings):
    compose_file = settings.network_dir_path / settings.compose_yml_file
    compose_data = render_compose.render_compose_data(settings)
    compose_file.write_text(compose_data)

def parse_args():
    # create the top-level parser
    s = get_settings()
    parser = argparse.ArgumentParser(prog="PROG")
    parser.add_argument("-t", "--testnet-dir",
                        type=Path,
                        help="Output dir for network config.",
                        )
    parser.add_argument("-n", "--network",
                        type=Path,
                        help="Path to network spec file.",
                        )
    parser.add_argument("-v", "--num-validators",
                        type=int,
                        help="Number of validators to create.",
                        )
    return parser.parse_args()

def overrides(a) -> dict:
    o: dict = {}
    net = {}
    if a.testnet_dir is not None:
        o["testnet_dir"] = a.testnet_dir
    if a.num_validators is not None:
        net["num_validators"] = a.num_validators
    if net: o["network"] = net
    return o

def generate_compose_data(validators: list, use_unl: bool) -> dict:
    return {}

def main():
    args = parse_args()
    s = get_settings(**overrides(args))
    s.network_dir_path.mkdir(parents=True, exist_ok=True)
    if s.network_file.is_file():
        network_spec = read_network(s.network_file)
        s.network.peers = get_peers(network_spec)
        # TODO: assert that the network is fully defined here?, no do it in read_network()
        # assert at least 1 non-validator node present?
        # assert len(s.network.peers) == s.network.num_peers + s.network.num_validators
    else:
        # Everyone is connected to everyone else
        s.network.peers = ALL
    # Generate some validators
    validators = [gl.gen_validator() for _ in range(s.network.num_validators)]
    seeds = [v["master_seed"] for v in validators]
    public_keys = [v["node_public_key"] for v in validators]

    use_unl = s.network.use_unl
    config_data = {
        "public_keys": public_keys,
        "use_unl": use_unl,
    }

    # If we want to use a UNL, generate a publisher and the list
    if use_unl:
        publisher = gl.gen_validator()
        unl_data = gl.generate_unl_data(validators, publisher)
        config_data["validator_list_keys"] = publisher["master_pubkey"]
        config_data["validator_list_sites"] = s.network.validator_list_sites
        # We need to add the server to the compose file

    for idx, seed in enumerate(seeds):
        val_name =  s.network.validator_name
        val_data = {
            "name": f"{val_name}{idx}",
            "seed": seed,
            "is_validator": True,
            **config_data,
        }
        write_config(val_data, s)

    # TODO: Handle more than one peer
    peer_data = {
        "name": "rippled",
        "is_validator": False,
        **config_data,
    }
    write_config(peer_data, s)

    # Generate the compose file
    write_compose(s)

    # Write the UNL
    if use_unl:
        unl_data = generate_unl_data(validators, publisher)
        unl_json_file = s.network_dir_path / s.unl_file
        unl_json_file.write_text(json.dumps(unl_data))
        # copy the server to the network dir
        shutil.copy(s.unl_server, s.network_dir_path)
        pass
if __name__ == "__main__":
    main()
