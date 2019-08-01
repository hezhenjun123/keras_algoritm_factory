import logging
from math import ceil

logging.getLogger().setLevel(logging.INFO)


class ModelBase:

    def __init__(self, config):
        self.config = config
        self.batch_size = config["BATCH_SIZE"]
        self.data_csv = config["DATA_CSV"]
        self.learning_rate = config["LEARNING_RATE"]
        self.num_classes = config["NUM_CLASSES"]
        self.epochs = config["EPOCHS"]
        self.train_steps_per_epoch = config["TRAIN_STEPS_PER_EPOCH"]
        self.valid_steps_per_epoch = config["VALID_STEPS_PER_EPOCH"]
        resize_config = config["TRANSFORM"]["RESIZE"]
        self.resize = (resize_config[0], resize_config[1])
        logging.info(config)

    def create_model(self):
        raise NotImplementedError

    def compile_model(self, **kwargs):
        self.model.compile(**kwargs)

    def set_runtime_parameters(self, **kwargs):
        if "num_train_data" in kwargs:
            self.num_train_data = kwargs["num_train_data"]
            if self.train_steps_per_epoch == -1:
                self.train_steps_per_epoch = ceil(self.num_train_data / self.batch_size)

        if "num_valid_data" in kwargs:
            self.num_valid_data = kwargs["num_valid_data"]
            if self.valid_steps_per_epoch == -1:
                self.valid_steps_per_epoch = ceil(self.num_valid_data / self.batch_size)

    def fit_model(self, data_train, data_valid, callbacks, **kwargs):
        raise NotImplementedError
