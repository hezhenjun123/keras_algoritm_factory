import os
import logging
import pandas as pd
from transforms.transform_factory import TransformFactory
from data_generators.generator_factory import DataGeneratorFactory
from model.model_factory import ModelFactory
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

logging.getLogger().setLevel(logging.INFO)


class ExperimentBase:

    def __init__(self, config):
        self.config = config
        self.split_col = config["TRAINING_DATA_CSV_SCHEMA"]["SPLIT"]
        self.split_train_val = config["TRAINING_DATA_CSV_SCHEMA"][
            "SPLIT_TRAIN_VAL"]
        self.split_valid_val = config["TRAINING_DATA_CSV_SCHEMA"][
            "SPLIT_VALID_VAL"]
        self.data_csv = config["DATA_CSV"]
        self.csv_separator = config["TRAINING_DATA_CSV_SCHEMA"]["SEPARATOR"]
        self.train_transform_name = self.config["EXPERIMENT"]["TRAIN_TRANSFORM"]
        self.valid_transform_name = self.config["EXPERIMENT"]["VALID_TRANSFORM"]
        self.train_generator_name = self.config["EXPERIMENT"]["TRAIN_GENERATOR"]
        self.valid_generator_name = self.config["EXPERIMENT"]["VALID_GENERATOR"]
        if config["RUN_ENV"] == "aws":
            self.save_dir = config["AWS_PARA"]["DIR_OUT"]
        elif config["RUN_ENV"] == "local":
            self.save_dir = self.__create_run_dir(
                config["LOCAL_PARA"]["DIR_OUT"])
        else:
            run_env = config["RUN_ENV"]
            raise Exception(f"Incorrect RUN_ENV: {run_env}")
        self.model_name = self.config["EXPERIMENT"]["MODEL_NAME"]

    def generate_transform(self):
        transform_factory = TransformFactory(self.config)
        train_transform = transform_factory.create_transform(
            self.train_transform_name)
        valid_transform = transform_factory.create_transform(
            self.valid_transform_name)
        return [train_transform, valid_transform]

    def read_train_csv(self):
        data_from_train_csv = pd.read_csv(self.data_csv,
                                          sep=self.csv_separator).fillna("")
        logging.info(data_from_train_csv.head())
        logging.info("#" * 15 + "Reading training data" + "#" * 15)
        data_train_split = data_from_train_csv[data_from_train_csv[
            self.split_col] == self.split_train_val].sample(frac=1)
        logging.info("#" * 15 + "Reading valid data" + "#" * 15)
        data_valid_split = data_from_train_csv[data_from_train_csv[
            self.split_col] == self.split_valid_val].sample(frac=1)
        return [data_train_split, data_valid_split]

    def generate_dataset(self, data_train_split, data_valid_split,
                         train_transform, valid_transform):
        generator_factory = DataGeneratorFactory(self.config)
        train_generator = generator_factory.create_generator(
            self.train_generator_name)
        valid_generator = generator_factory.create_generator(
            self.valid_generator_name)
        train_dataset = train_generator.create_dataset(
            df=data_train_split, transforms=train_transform)
        valid_dataset = valid_generator.create_dataset(
            df=data_valid_split, transforms=valid_transform)
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
        tf.gfile.MakeDirs(save_dir)
        list_of_files = tf.gfile.ListDirectory(save_dir)
        i = 1
        while f"run{i}" in list_of_files:
            i += 1
        run_dir = os.path.join(save_dir, f"run{i}")
        tf.gfile.MakeDirs(run_dir)
        print("#" * 40)
        print(f"Saving summaries on {run_dir}")
        print("#" * 40)
        return run_dir
