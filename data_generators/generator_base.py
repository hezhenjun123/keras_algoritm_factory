import os
import tensorflow as tf
from datetime import datetime
import numpy as np


class DataGeneratorBase:
    def __init__(self, config):
        output_shape = config["DATA_GENERATOR"]["OUTPUT_SHAPE"]
        self.output_shape = (output_shape[0], output_shape[1])
        self.output_image_channels = config["DATA_GENERATOR"][
            "OUTPUT_IMAGE_CHANNELS"]
        self.output_image_type = tf.dtypes.as_dtype(
            np.dtype(config["DATA_GENERATOR"]["OUTPUT_IMAGE_TYPE"]))
        if config["RUN_ENV"] == "aws":
            self.data_dir = config["AWS_PARA"]["DATA_DIR"]
        elif config["RUN_ENV"] == "local":
            self.data_dir = config["LOCAL_PARA"]["DATA_DIR"]
        else:
            self.run_env = config["RUN_ENV"]
            raise Exception(f"Incorrect run env: {self.run_env}")
        self.batch_size = config["BATCH_SIZE"]
        self.drop_remainder = config["DATA_GENERATOR"]["DROP_REMAINDER"]
        self.cache_dir = config["DATA_GENERATOR"]["CACHE_DIR"]
        self.repeat = config["DATA_GENERATOR"]["REPEAT"]
        self.num_parallel_calls = config["DATA_GENERATOR"][
            "NUM_PARALLEL_CALLS"]
        self.segmentation_path = config["TRAINING_DATA_CSV_SCHEMA"][
            "SEGMENTATION_PATH"]
        self.image_path = config["TRAINING_DATA_CSV_SCHEMA"]["IMAGE_PATH"]
        self.label_name = config["TRAINING_DATA_CSV_SCHEMA"]["LABEL_NAME"]
        self.image_level_label = config["TRAINING_DATA_CSV_SCHEMA"][
            "IMAGE_LEVEL_LABEL"]
        self.split = config["TRAINING_DATA_CSV_SCHEMA"]["SPLIT"]
        self.n_classes = config["NUM_CLASSES"]

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
        image = tf.read_file(path)
        image = tf.image.decode_image(image)
        return image

    def cache_file(self, cache_dir):
        if os.path.exists(cache_dir) is False:
            os.makedirs(cache_dir)
        curr_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
        cache_file = os.path.join(cache_dir, f"cache_{curr_time}")
        return cache_file
