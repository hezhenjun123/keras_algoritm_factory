import os
import logging
import pandas as pd
from transforms.transform_factory import TransformFactory
from data_generators.generator_factory import DataGeneratorFactory
from model.model_factory import ModelFactory
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)


#FIXME: May want to combine Inference Base with Experiment base. Experiment can handle both training and inference
class InferenceBase:

    def __init__(self, config):
        self.config = config
        self.split_col = config["INFERENCE"]["SPLIT"]
        self.split_val = config["INFERENCE"]["SPLIT_VAL"]
        self.inference_csv = config["INFERENCE"]["INFERENCE_CSV_FILE"]
        self.csv_separator = config["INFERENCE"]["SEPARATOR"]
        self.inference_transform_name = self.config["EXPERIMENT"]["VALID_TRANSFORM"]
        self.inference_generator_name = self.config["EXPERIMENT"]["VALID_GENERATOR"]
        self.model_name = self.config["EXPERIMENT"]["MODEL_NAME"]
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
        inference_data_filter = data_from_inference_csv[self.split_col] == self.split_val
        data_inference_split = data_from_inference_csv[inference_data_filter]
        return data_inference_split

    def generate_dataset(self, data_inference_split, inference_transform):
        generator_factory = DataGeneratorFactory(self.config)
        inference_generator = generator_factory.create_generator(self.inference_generator_name)
        inference_dataset = inference_generator.create_inference_dataset(
            df=data_inference_split, transforms=inference_transform)
        return inference_dataset

    def load_model(self):
        if not self.config["LOAD_MODEL"]:
            raise ValueError('LOAD_MODEL config must be set to true for inference')
        model_factory = ModelFactory(self.config)
        model = model_factory.create_model(self.model_name)
        return model

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
