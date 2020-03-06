import yaml
import os


def read_config(config_path):
    # MODULE_CONFIG_FILE = 'config/{}'.format(config_path)
    MODULE_CONFIG_FILE = config_path
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
    if module_config[module_config["INFERENCE_ENGINE"]].get("TRANSFORM",None) is None:
        module_config[module_config["INFERENCE_ENGINE"]]["TRANSFORM"] = module_config["EXPERIMENT"][
            "VALID_TRANSFORM"]
    if module_config[module_config["INFERENCE_ENGINE"]].get("GENERATOR",None) is None:
        module_config[module_config["INFERENCE_ENGINE"]]["GENERATOR"] = module_config["EXPERIMENT"][
            "VALID_GENERATOR"]
    if module_config[module_config["INFERENCE_ENGINE"]].get("MODEL_NAME",None)  is None:
        module_config[module_config["INFERENCE_ENGINE"]]["MODEL_NAME"] = module_config["EXPERIMENT"][
            "MODEL_NAME"]
    return module_config
