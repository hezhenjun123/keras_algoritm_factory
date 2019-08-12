import yaml
import os


def read_config(args):
    MODULE_CONFIG_FILE = 'config/{}'.format(args.config)
    if os.path.exists(MODULE_CONFIG_FILE) is False:
        raise Exception("config file does not exist: {}".format(MODULE_CONFIG_FILE))
    with open(MODULE_CONFIG_FILE) as f:
        module_config = yaml.safe_load(f)

    if module_config["EXPERIMENT"]["VALID_TRANSFORM"] is None:
        module_config["EXPERIMENT"]["VALID_TRANSFORM"] = module_config["EXPERIMENT"][
            "TRAIN_TRANSFORM"]
    if module_config["EXPERIMENT"]["VALID_GENERATOR"] is None:
        module_config["EXPERIMENT"]["VALID_GENERATOR"] = module_config["EXPERIMENT"][
            "TRAIN_GENERATOR"]
    return module_config
