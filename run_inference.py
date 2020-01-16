import os
import argparse
import logging
from inference.inference_factory import InferenceFactory
from utilities.config import read_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 0 for GPU; -1 for CPU

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--freeze_to_pb_path", type=str, required=False)
logging.getLogger().setLevel(logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['S3_REQUEST_TIMEOUT_MSEC'] = '6000000'


def run_inference(config, freeze_to_pb_path=None):
    inference_factory = InferenceFactory(config)
    inference = inference_factory.create_inference(config["INFERENCE"]["INFERENCE_NAME"])

    if freeze_to_pb_path != None:
        inference.freeze_to_pb(freeze_to_pb_path)
    else:
        inference.run_inference()
def main(args):
    module_config = read_config(args.config)
    run_inference(module_config, args.freeze_to_pb_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
