from mako.template import Template

def config_data(data):
    config_template = Template(filename=str(data["node_config_template"]))

    if data["is_validator"]:
        data.update(is_validator=True)
        data.update(validation_seed=data["seed"])
    else:
        pass

    return config_template.render(**data)
