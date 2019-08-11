import os
import tensorflow as tf
from datetime import datetime
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)


class DataGeneratorBase:

    def __init__(self, config):
        output_shape = config["DATA_GENERATOR"]["OUTPUT_SHAPE"]
        self.output_shape = (output_shape[0], output_shape[1])
        self.output_image_channels = config["DATA_GENERATOR"]["OUTPUT_IMAGE_CHANNELS"]
        self.output_image_type = tf.dtypes.as_dtype(
            np.dtype(config["DATA_GENERATOR"]["OUTPUT_IMAGE_TYPE"]))
        self.data_dir = config["DATA_DIR"]
        self.batch_size = config["BATCH_SIZE"]
        self.drop_remainder = config["DATA_GENERATOR"]["DROP_REMAINDER"]
        self.cache_dir = config["DATA_GENERATOR"]["CACHE_DIR"]
        self.repeat = config["DATA_GENERATOR"]["REPEAT"]
        self.num_parallel_calls = config["DATA_GENERATOR"]["NUM_PARALLEL_CALLS"]
        self.segmentation_path = config["TRAINING_DATA_INFO"]["SEGMENTATION_PATH"]
        self.image_path = config["TRAINING_DATA_INFO"]["IMAGE_PATH"]
        self.label_name = config["TRAINING_DATA_INFO"]["LABEL_NAME"]
        self.image_level_label = config["TRAINING_DATA_INFO"]["IMAGE_LEVEL_LABEL"]
        self.split = config["TRAINING_DATA_INFO"]["SPLIT"]
        self.n_classes = config["NUM_CLASSES"]
        if config["RUN_MODE"] == "inference":
            self.__inference_override(config)
        if config["RUN_ENV"] == "local":
            self.__local_override_config(config)

    def __local_override_config(self, config):
        if config["LOCAL_OVERRIDE"]["DATA_DIR"] is not None and \
                len(config["LOCAL_OVERRIDE"]["DATA_DIR"].strip()) > 0:
            self.data_dir = config["LOCAL_OVERRIDE"]["DATA_DIR"]
            logging.info(f"=====Override data_dir with {self.data_dir}")

    def __inference_override(self, config):
        logging.info("=====Override config with inference configs=====")
        self.repeat = False
        self.data_dir = config["INFERENCE"]["INFERENCE_DATA_DIR"]

    def create_dataset(self, df, transforms):
        raise NotImplementedError()

    def get_join_root_dir_map(self, root_dir):
        """This function builds a map that takes a `relative_path` and joins it with
        `root_dir`.
        """

        def join_root_dir_map(relative_path):
            total_path = ""
            if relative_path:
                total_path = os.path.join(root_dir, relative_path)
            return total_path

        return join_root_dir_map

    def load_image(self, path):
        """Loads an image located in `path`.

        Parameters
        ----------
        path : Tensor of dtype tf.string.
            The full path to the label `png` file. It assumes the path is valid.

        Returns
        -------
        tf.Tensor
            A Tensor of type `tf.uint8` and shape `(H, W, C)`.
        """
        logging.info(path)
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image)
        return image

    def cache_file_location(self, cache_dir):
        if os.path.exists(cache_dir) is False:
            os.makedirs(cache_dir)
        curr_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
        cache_file = os.path.join(cache_dir, f"cache_{curr_time}")
        num = 1
        while os.path.exists(cache_file):
            cache_file = f"{cache_file}_{num}"
            num += 1
        return cache_file
