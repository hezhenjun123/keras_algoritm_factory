import platform
import os
import logging
from transforms.transform_factory import TransformFactory
from model.model_factory import ModelFactory
import tensorflow as tf
from data_generators.generator_factory import DataGeneratorFactory

if platform.machine() != 'aarch64':
    import pandas as pd
logging.getLogger().setLevel(logging.INFO)


#FIXME: May want to combine Inference Base with Experiment base. Experiment can handle both training and inference
class InferenceBase:

    def __init__(self, config):
        self.config = config
        self.split_col = config["INFERENCE"]["SPLIT"]
        self.split_val = config["INFERENCE"]["SPLIT_VAL"]
        self.inference_csv = config["INFERENCE"]["INFERENCE_CSV_FILE"]
        self.csv_separator = config["INFERENCE"]["SEPARATOR"]
        self.inference_transform_name = self.config["INFERENCE"]["TRANSFORM"]
        self.inference_generator_name = self.config["INFERENCE"]["GENERATOR"]
        self.model_name = self.config["INFERENCE"]["MODEL_NAME"]
        self.num_classes = config["NUM_CLASSES"]
        self.save_dir = config["DIR_OUT"]
        self.load_model_path = self.config["LOAD_MODEL_DIRECTORY"]
        if config["RUN_ENV"] == "local":
            self.__local_override_config(config)

    def __local_override_config(self, config):
        self.save_dir = self.__create_run_dir(config["LOCAL_OVERRIDE"]["DIR_OUT"])

    def generate_transform(self):
        transform_factory = TransformFactory(self.config)
        inference_transform = transform_factory.create_transform(self.inference_transform_name)
        return inference_transform

    def read_train_csv(self):
        data_from_inference_csv = pd.read_csv(self.inference_csv, sep=self.csv_separator).fillna("")
        logging.info(data_from_inference_csv.head())
        logging.info("#" * 15 + "Reading inference data" + "#" * 15)
        if self.split_val not in ["all", "train", "valid"]:	
            raise ValueError(f" spilt_value='{self.split_val}' is not allowed, only ['train', 'valid', 'all'] are supported.")
        if self.split_val == "all":
            return data_from_inference_csv
        inference_data_filter = data_from_inference_csv[self.split_col] == self.split_val
        data_inference_split = data_from_inference_csv[inference_data_filter]
        return data_inference_split

    def generate_dataset(self, data_inference_split, inference_transform, evaluate):
        self.config['BATCH_SIZE'] = 1
        generator_factory = DataGeneratorFactory(self.config)
        inference_generator = generator_factory.create_generator(self.inference_generator_name)
        if evaluate:
            inference_dataset = inference_generator.create_dataset(
            df=data_inference_split, transforms=inference_transform)
        else:
            inference_dataset = inference_generator.create_inference_dataset(
            df=data_inference_split, transforms=inference_transform)
        return inference_dataset

    def load_model(self, create_raw_model = False):
        if not self.config["LOAD_MODEL"]:
            raise ValueError('LOAD_MODEL config must be set to true for inference')
        if create_raw_model:
            self.config["LOAD_MODEL"] = False
        model_factory = ModelFactory(self.config)
        model = model_factory.create_model(self.model_name)
        return model

    def freeze_to_pb(self, save_dir):
        raise NotImplementedError

    def run_inference(self):
        raise NotImplementedError

    def __create_run_dir(self, save_dir):
        """Creates a numbered directory named "run1". If directory "run1" already
        exists then creates directory "run2", and so on.

        Parameters
        ----------
        save_dir : str
            The root directory where to create the "run{number}" folder.

        Returns
        -------
        str
            The full path of the newly created "run{number}" folder.
        """
        tf.io.gfile.mkdir(save_dir)
        list_of_files = tf.io.gfile.listdir(save_dir)
        i = 1
        while f"inference{i}" in list_of_files:
            i += 1
        run_dir = os.path.join(save_dir, f"inference{i}")
        tf.io.gfile.mkdir(run_dir)
        logging.info("#" * 40)
        logging.info(f"Saving inference on {run_dir}")
        logging.info("#" * 40)
        return run_dir
