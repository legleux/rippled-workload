import argparse
import json
import shutil
import sys
from pathlib import Path

from mako.template import Template

import prepare_workload.generate_unl as gl
from prepare_workload import node_config as nc
from prepare_workload.compose import render_compose
from prepare_workload.generate_unl import generate_unl_data
from prepare_workload.settings import settings as get_settings


def write_config(node_data, settings):

    s = settings
    node = s.node_config
    is_validator = node_data["is_validator"]

    node_config_data = {
        "ports": node.ports,
        "node_config_template": s.node_config_template,
        "validator_public_keys": "\n".join(node_data["validator_public_keys"]),
        # REVIEW: Do we want peers in ips_fixed also?
        "ips_fixed": "\n".join([f"{p} {settings.node_config.ports['peer']}" for p in node_data["peers"]]),
        **node_data,
    }
    if is_validator:
        # Assume validators don't sign
        node_config_data["validation_seed"] = node_data["keys"]["master_seed"]
        node_config_data["voting"] = s.node_config.voting
    node_config_data["signing_support"] = str(is_validator).lower()

    config_template = Template(filename=str(s.node_config_template))
    node_config = config_template.render(**node_config_data)
    config_dir = s.network_dir_path / s.config_dir / node_data["name"]
    Path(config_dir).mkdir(parents=True, exist_ok=True)
    config_file = config_dir / s.node_config_file
    config_file.write_text(node_config)


def write_compose(settings):
    compose_file = settings.network_dir_path / settings.compose_yml_file
    compose_data = render_compose.render_compose_data(settings)
    compose_file.write_text(compose_data)


def parse_args():
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
    if net:
        o["network"] = net
    return o


def generate_compose_data(validators: list, use_unl: bool) -> dict:
    return {}


# def get_node_configs(settings):
#     # If there is network customization, determine the node configs from it, otherwise use all the defaults.
#     if settings.network_file.is_file():
#         spec = parse_network.read_network_spec_file(settings.network_file)
#         edge_list = spec.get("edges")
#         private_peers = spec.get("private_peers")

#     # Get all the nodes defined
#     # nodes = parse_network.get_nodes(spec)
#     if edge_list is None:

#     peers = parse_network.get_peers(edge_list)
#     node_configs = {}
#     for p in peers:
#         node_configs[p] = {
#             "peers": peers[p],
#             "is_validator": p.startswith(settings.network.validator_name),
#             "peer_private": p in spec["private_peers"],
#         }
#     # Get all the edges specified
#     return node_configs


def main():
    args = parse_args()
    s = get_settings(**overrides(args))
    s.network_dir_path.mkdir(parents=True, exist_ok=True)

    # Parse the network config (if any) to get some default configs.
    node_configs = nc.get_node_configs(s)

    if s.network.use_unl:
        publisher = gl.gen_validator()

    validator_public_keys = []
    validators = []
    for config in node_configs.values():

        if config["is_validator"]:
            config["keys"] = gl.gen_validator()
            # config["validation_seed"] = config["keys"]["master_seed"]
            validators.append(config["keys"])
            validator_public_keys.append(config["keys"]["node_public_key"])

        config["use_unl"] = s.network.use_unl
        if s.network.use_unl:
            config["validator_list_keys"] = publisher["master_pubkey"]
            config["validator_list_sites"] = s.network.validator_list_sites

    # Write eacho node's config file
    for config in node_configs.values():
        config["validator_public_keys"] = "\n".join(validator_public_keys)
        write_config(config, s)

    # Write the compose file
    write_compose(s)

    # Write the UNL
    if s.network.use_unl:
        unl_data = generate_unl_data(validators, publisher)
        unl_json_file = s.network_dir_path / s.unl_file
        unl_json_file.write_text(json.dumps(unl_data))
        # copy the server to the network dir
        shutil.copy(s.unl_server, s.network_dir_path)


if __name__ == "__main__":
    main()
