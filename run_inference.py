import os
import time
from absl import app
from absl import flags
import logging
from inference.inference_factory import InferenceFactory
from utilities.config import read_config
from utilities.s3context import ZL_CACHE
import numpy as np
import yaml

logging.getLogger().setLevel(logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 0 for GPU; -1 for CPU

flags.DEFINE_string("freeze_to_pb_path", None, "path to store pb file")

flags.DEFINE_string("config", None, "model config file")
flags.DEFINE_string("load_test_config", None, "config for multiple models")
flags.DEFINE_boolean("debug", False, "run tf to compare results")
flags.DEFINE_boolean("create_trt_engine", False, "run tf to compare results")
flags.DEFINE_boolean("upload", False, "upload pb file to s3 bucket")

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['S3_REQUEST_TIMEOUT_MSEC'] = '6000000'


def run_inference(config, freeze_to_pb_path=None):
    inference_factory = InferenceFactory(config)
    inference = inference_factory.create_inference(config["INFERENCE"]["INFERENCE_NAME"])

    if freeze_to_pb_path != None:
        inference.freeze_to_pb(freeze_to_pb_path)
        if FLAGS.upload:
            ZL_CACHE.upload("{}/frozen_model.pb".format(freeze_to_pb_path))

    else:
        output = inference.run_inference()
        if FLAGS.debug:
            output_tf = inference.run_inference_tf()
            np.testing.assert_array_almost_equal(np.array(output).ravel(), np.array(output_tf).ravel(), decimal=4)

def run_load_test(load_test_config):
    print(load_test_config)

    workers= []
    timers = []
    configs = load_test_config["deploy_configs"]["models"]
    for config in configs:
        model_config_path = config["model_config"]
        model_config = read_config(model_config_path)
        if FLAGS.create_trt_engine:
            model_config["INFERENCE"]["CREATE_ENGINE"] = True
        inference_factory = InferenceFactory(model_config)
        workers.append(inference_factory.create_inference(model_config["INFERENCE"]["INFERENCE_NAME"]))
        timers.append(0)
    num_run = load_test_config["load_test"]["num_run"]
    input_size = load_test_config["load_test"]["input_size"]

    for i in range(num_run):
        worker_idx = i % len(workers)
        img = np.random.randint(255, size=input_size).astype('float32')
        start_time = time.time()
        workers[worker_idx].get_image_pred(img, True)
        timers[worker_idx] += time.time() - start_time
    timers = list(map(lambda x: x / num_run, timers))
    print("Average Latency for {} runs per model: {}".format(num_run, timers))

def main(_):
    if FLAGS.load_test_config:
        with open(FLAGS.load_test_config) as f:
            config = yaml.safe_load(f)
            run_load_test(config)
    else:
        module_config = read_config(FLAGS.config)
        if FLAGS.create_trt_engine:
            module_config["INFERENCE"]["CREATE_ENGINE"] = True
        run_inference(module_config, FLAGS.freeze_to_pb_path)

    


if __name__ == "__main__":
    app.run(main)
