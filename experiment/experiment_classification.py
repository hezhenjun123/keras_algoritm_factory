import logging
import pandas as pd
from experiment.experiment_base import ExperimentBase
from transforms.transform_factory import TransformFactory
from data_generators.generator_factory import DataGeneratorFactory
from model.model_factory import ModelFactory

logging.getLogger().setLevel(logging.INFO)


class ExperimentClassification(ExperimentBase):
    def __init__(self, config):
        super().__init__(config)
        self.split_col = config["TRAINING_DATA_CSV_SCHEMA"]["SPLIT"]
        self.split_train_val = config["TRAINING_DATA_CSV_SCHEMA"][
            "SPLIT_TRAIN_VAL"]
        self.split_valid_val = config["TRAINING_DATA_CSV_SCHEMA"][
            "SPLIT_VALID_VAL"]
        self.data_csv = config["DATA_CSV"]

    def generate_transform(self):
        transform_factory = TransformFactory(self.config)
        self.train_transform = transform_factory.create_transform(
            self.config["EXPERIMENT"]["TRAIN_TRANSFORM"])
        self.valid_transform = transform_factory.create_transform(
            self.config["EXPERIMENT"]["VALID_TRANSFORM"])

    def __read_train_csv(self):
        data_from_train_csv = pd.read_csv(self.data_csv, sep='\t').fillna("")
        logging.info(data_from_train_csv.head())
        logging.info("#" * 15 + "Reading training data" + "#" * 15)
        self.data_train_split = data_from_train_csv[data_from_train_csv[
            self.split_col] == self.split_train_val].sample(frac=1)
        logging.info("#" * 15 + "Reading valid data" + "#" * 15)
        self.data_valid_split = data_from_train_csv[data_from_train_csv[
            self.split_col] == self.split_valid_val].sample(frac=1)

    def generate_dataset(self):
        self.generate_transform()
        self.__read_train_csv()
        generator_factory = DataGeneratorFactory(self.config)
        train_generator = generator_factory.create_generator(
            self.config["EXPERIMENT"]["TRAIN_GENERATOR"])
        valid_generator = generator_factory.create_generator(
            self.config["EXPERIMENT"]["VALID_GENERATOR"])
        self.train_dataset = train_generator.create_dataset(
            df=self.data_train_split, transforms=self.train_transform)
        self.valid_dataset = valid_generator.create_dataset(
            df=self.data_valid_split, transforms=self.valid_transform)

    def train(self):
        self.generate_dataset()
        model_factory = ModelFactory(self.config)
        model = model_factory.create_model(
            self.config["EXPERIMENT"]["MODEL_NAME"])
        kwarg_para = {
            "num_train_data": len(self.data_train_split),
            "num_valid_data": len(self.data_valid_split)
        }
        model.model_fit(self.train_dataset, self.valid_dataset, **kwarg_para)
