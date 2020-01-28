import os
from absl import app
from absl import flags
import logging
from inference.inference_factory import InferenceFactory
from utilities.config import read_config
import numpy as np

logging.getLogger().setLevel(logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 0 for GPU; -1 for CPU

flags.DEFINE_string("freeze_to_pb_path", None, "path to store pb file")

flags.DEFINE_string("config", None, "model config file")
flags.DEFINE_boolean("debug", False, "run tf to compare results")
flags.DEFINE_boolean("create_trt_engine", False, "run tf to compare results")

flags.mark_flag_as_required('config')

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['S3_REQUEST_TIMEOUT_MSEC'] = '6000000'


def run_inference(config, freeze_to_pb_path=None):
    inference_factory = InferenceFactory(config)
    inference = inference_factory.create_inference(config["INFERENCE"]["INFERENCE_NAME"])

    if freeze_to_pb_path != None:
        inference.freeze_to_pb(freeze_to_pb_path)

    else:
        output = inference.run_inference()
        if FLAGS.debug:
            output_tf = inference.run_inference_tf()
            np.testing.assert_array_almost_equal(np.array(output).ravel(), np.array(output_tf).ravel(), decimal=4)

def main(_):
    module_config = read_config(FLAGS.config)
    if FLAGS.create_trt_engine:
        module_config["INFERENCE"]["CREATE_ENGINE"] = True
    run_inference(module_config, FLAGS.freeze_to_pb_path)


if __name__ == "__main__":
    app.run(main)
