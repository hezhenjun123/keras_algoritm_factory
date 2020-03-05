import os
import argparse
import logging

from utilities.config import read_config
from utilities.helper import allow_memory_growth
from experiment.experiment_factory import ExperimentFactory

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
logging.getLogger().setLevel(logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['S3_REQUEST_TIMEOUT_MSEC'] = '6000000'


def run_experiments(config):
    experiment_factory = ExperimentFactory(config)
    experiment = experiment_factory.create_experiment(config["EXPERIMENT"]["EXPERIMENT_NAME"])
    experiment.run_experiment()


def main(args):
    allow_memory_growth()
    module_config = read_config(args.config)
    run_experiments(module_config)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
