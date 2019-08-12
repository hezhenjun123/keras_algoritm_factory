import os
import argparse
import logging
import yaml

from experiment.experiment_factory import ExperimentFactory

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
logging.getLogger().setLevel(logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['S3_REQUEST_TIMEOUT_MSEC'] = '6000000'


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
    if module_config["LOAD_MODEL"] is None or "LOAD_MODEL" not in module_config:
        module_config["LOAD_MODEL"]=False
    if "LOAD_MODEL_DIRECORY" not in module_config:
        module_config["LOAD_MODEL_DIRECORY"]=None
    return module_config


def run_experiments(config):
    experiment_factory = ExperimentFactory(config)
    experiment = experiment_factory.create_experiment(config["EXPERIMENT"]["EXPERIMENT_NAME"])
    experiment.run_experiment()


def main(args):
    module_config = read_config(args)
    run_experiments(module_config)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
