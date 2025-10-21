from mako.template import Template
service_template = "service.yml.mako"
compose_template = "compose.yml.mako"

container_config_path = "/opt/ripple/etc"

def render_node(data):
    pass
    name = data["name"]
    d = {
        "is_validtar": False,
        "service_name": f"{name}",
        "container_name": f"{name}",
        "hostname": f"{name}",
        # "entrypoint": "rippled",
        "image": data["compose_config"].image,
        "network_name": data["compose_config"].network_name,
        "volumes": [f"{data['volumes_path']}/{name}:{container_config_path}"],
    }
    rpc_port = data["node_data"].ports["rpc_admin_local"]
    ws_port = data["node_data"].ports["ws_admin_local"]
    d["ports"] = [f"{rpc_port}:{rpc_port}", f"{ws_port}:{ws_port}"]
    node_config_template = Template(filename=str(data["template"]))
    return node_config_template.render(**d)

def render_validators(validator_data):
    # node_data = render_node(validator_data)
    name = validator_data["name"]
    validators = []
    for i in range(validator_data["num_nodes"]):
        val_data = {
            "is_validator": True,
            "service_name": f"{name}{i}",
            "container_name": f"{name}{i}",
            "hostname": f"{name}{i}",
            # "entrypoint": "rippled",
            "command": "--start",
            "image": validator_data["compose_config"].image,
            "network_name": validator_data["compose_config"].network_name,
            "volumes": [f"{validator_data['volumes_path']}/{name}{i}:{container_config_path}"],
        }
        # if use_ledger ... append to volumes here.
        # Assume rippled takes the default port locally...
        rpc_port = validator_data["node_data"].ports["rpc_admin_local"]
        ws_port = validator_data["node_data"].ports["ws_admin_local"]
        val_data["ports"] = [f"{rpc_port + i + 1}:{rpc_port}", f"{ws_port + i + 1}:{ws_port}"]
        validators.append(val_data)

    # node_config_template = Template(filename=str(data["template"]))
    validator_field = []
    t = Template(filename=str(validator_data["template"]))
    for v in validators:
        validator_field.append(t.render(**v))
    return "\n".join(validator_field)

def render_peers(peer_data):
    # TODO: Handle multiple peers
    return render_node(peer_data)

def render_unl_server(unl_data):
    unl_template = Template(filename=str(unl_data["template"]))
    return unl_template.render(**unl_data)

def render_network(name):
    return

def render_compose_data(settings):
    service_template_file_path = settings.template_dir_path / service_template
    compose_template_file_path = settings.template_dir_path / compose_template
    compose_yml_path = settings.network_dir_path / settings.compose_yml_file
    unl_service_template = settings.template_dir_path / "unl_service.yml.mako"
    network_dir_name = settings.network.network_dir_name
    s_data = {
        "node_data": settings.node_config,
        "network_dir_name": settings.network.network_dir_name,
        "volumes_path": f"./{settings.config_dir}",
        "compose_config": settings.compose_config,

    }
    validator_data = {
        "num_nodes": settings.network.num_validators,
        "name": settings.network.validator_name,
        "template": service_template_file_path,
        **s_data,
    }
    peer_data = {
        "num_nodes": settings.network.num_peers,
        "name": settings.network.peer_name,
        "template": service_template_file_path,
        **s_data,
    }
    data = {
        # "templates": templates,
        "validators": render_validators(validator_data),
        "peers": render_peers(peer_data),
        "use_unl": settings.network.use_unl,
        "network_name": settings.compose_config.network_name,
    }

    if settings.network.use_unl:
        name = "unl"
        unl_data= {
            "template": unl_service_template,
            "name": name,
            "unl_file": settings.unl_file,
            "network_name": settings.compose_config.network_name,
        }
        unl_service = render_unl_server(unl_data)
        data["unl_service"] = unl_service

    compose_tmpl = Template(filename=str(compose_template_file_path))
    return compose_tmpl.render(**data)
