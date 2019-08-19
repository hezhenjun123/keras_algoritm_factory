import os
import argparse
import logging
from inference.inference_factory import InferenceFactory
from utilities.config import read_config

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
logging.getLogger().setLevel(logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['S3_REQUEST_TIMEOUT_MSEC'] = '6000000'


def run_inference(config):
    inference_factory = InferenceFactory(config)
    inference = inference_factory.create_inference(config["INFERENCE"]["INFERENCE_NAME"])
    inference.run_inference()


def main(args):
    module_config = read_config(args.config)
    run_inference(module_config)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
