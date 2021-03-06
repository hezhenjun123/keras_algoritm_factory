import os
import logging
import pandas as pd
from transforms.transform_factory import TransformFactory
from data_generators.generator_factory import DataGeneratorFactory
from model.model_factory import ModelFactory
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)


class ExperimentBase:

    def __init__(self, config):
        self.config = config
        self.split_col = config["TRAINING_DATA_INFO"]["SPLIT"]
        self.split_train_val = config["TRAINING_DATA_INFO"]["SPLIT_TRAIN_VAL"]
        self.split_valid_val = config["TRAINING_DATA_INFO"]["SPLIT_VALID_VAL"]
        self.train_csv = config["TRAINING_DATA_INFO"]["TRAIN_CSV_FILE"]
        self.valid_csv = config["TRAINING_DATA_INFO"]["VALID_CSV_FILE"]
        self.csv_separator = config["TRAINING_DATA_INFO"]["SEPARATOR"]
        self.train_transform_name = self.config["EXPERIMENT"]["TRAIN_TRANSFORM"]
        self.valid_transform_name = self.config["EXPERIMENT"]["VALID_TRANSFORM"]
        self.train_generator_name = self.config["EXPERIMENT"]["TRAIN_GENERATOR"]
        self.valid_generator_name = self.config["EXPERIMENT"]["VALID_GENERATOR"]
        self.save_dir = config["DIR_OUT"]
        self.model_name = self.config["EXPERIMENT"]["MODEL_NAME"]
        if config["RUN_ENV"] == "local":
            self.__local_override_config(config)

    def __local_override_config(self, config):
        self.save_dir = self.__create_run_dir(config["LOCAL_OVERRIDE"]["DIR_OUT"])

    def generate_transform(self):
        transform_factory = TransformFactory(self.config)
        train_transform = transform_factory.create_transform(self.train_transform_name)
        valid_transform = transform_factory.create_transform(self.valid_transform_name)
        return [train_transform, valid_transform]

    def read_train_csv(self):
        data_from_train_csv = pd.read_csv(self.train_csv, sep=self.csv_separator).fillna("")
        logging.info(data_from_train_csv.head())
        logging.info("#" * 15 + "Reading training data" + "#" * 15)
        train_data_filter = data_from_train_csv[self.split_col] == self.split_train_val
        data_train_split = data_from_train_csv[train_data_filter].sample(frac=1)

        data_from_valid_csv = pd.read_csv(self.valid_csv, sep=self.csv_separator).fillna("")
        logging.info("#" * 15 + "Reading valid data" + "#" * 15)
        valid_data_filter = data_from_valid_csv[self.split_col] == self.split_valid_val
        data_valid_split = data_from_valid_csv[valid_data_filter].sample(frac=1)
        return [data_train_split, data_valid_split]

    def generate_dataset(self, data_train_split, data_valid_split, train_transform,
                         valid_transform):
        generator_factory = DataGeneratorFactory(self.config)
        train_generator = generator_factory.create_generator(self.train_generator_name)
        valid_generator = generator_factory.create_generator(self.valid_generator_name)
        train_dataset = train_generator.create_dataset(df=data_train_split,
                                                       transforms=train_transform)
        valid_dataset = valid_generator.create_dataset(df=data_valid_split,
                                                       transforms=valid_transform)
        return [train_dataset, valid_dataset]

    def generate_model(self):
        model_factory = ModelFactory(self.config)
        model = model_factory.create_model(self.model_name)
        compile_para = self.model_compile_para()
        model.compile_model(**compile_para)
        return model

    def model_compile_para(self):
        raise NotImplementedError

    def run_experiment(self):
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
        while f"run{i}" in list_of_files:
            i += 1
        run_dir = os.path.join(save_dir, f"run{i}")
        tf.io.gfile.mkdir(run_dir)
        logging.info("#" * 40)
        logging.info(f"Saving summaries on {run_dir}")
        logging.info("#" * 40)
        return run_dir
