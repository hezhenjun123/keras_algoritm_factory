import logging
from math import ceil
import tensorflow as tf
import os
from utilities import file_system_manipulation as fsm
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelBase:

    def __init__(self, config):
        self.config = config
        self.batch_size = config["BATCH_SIZE"]
        self.learning_rate = config["LEARNING_RATE"]
        self.num_classes = config["NUM_CLASSES"]
        self.epochs = config["EPOCHS"]
        self.train_steps_per_epoch = config["TRAIN_STEPS_PER_EPOCH"]
        self.valid_steps_per_epoch = config["VALID_STEPS_PER_EPOCH"]
        self.model_preload = self.config["LOAD_MODEL"]
        self.model_directory = self.config["LOAD_MODEL_DIRECTORY"]
        resize_config = config["TRANSFORM"]["RESIZE"]
        self.resize = (resize_config[0], resize_config[1])
        self.generate_custom_objects()
        logging.info(config)

    def get_or_load_model(self):
        if self.model_preload:
            return self.load_model()
        return self.create_model()

    def load_model(self):
        logger.debug("Loading model from {}".format(self.model_directory))
        if fsm.is_s3_path(self.model_directory):
            self.model_directory = fsm.s3_to_local(self.model_directory, './model.hdf5')[0]
        if not os.path.exists(self.model_directory):
            raise ValueError("Incorrect model path")
        return tf.keras.models.load_model(self.model_directory, compile=True, custom_objects=self.custom_objects)

    def create_model(self):
        raise NotImplementedError

    def generate_custom_objects(self):
        self.custom_objects = {}

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

    def fit_model(self, train_dataset, valid_dataset, callbacks, **kwargs):
        self.set_runtime_parameters(**kwargs)
        logging.info('STARTING TRAINING, {} train steps, {} valid steps'.format(
            self.train_steps_per_epoch, self.valid_steps_per_epoch))
        if valid_dataset is None:
            self.model.fit(train_dataset,
                           epochs=self.epochs,
                           steps_per_epoch=self.train_steps_per_epoch,
                           validation_data=valid_dataset,
                           verbose=2,
                           callbacks=callbacks)
        else:
            self.model.fit(train_dataset,
                           epochs=self.epochs,
                           steps_per_epoch=self.train_steps_per_epoch,
                           validation_data=valid_dataset,
                           validation_steps=self.valid_steps_per_epoch,
                           verbose=2,
                           callbacks=callbacks)

    def predict(self, predict_data_source, steps=None):
        if not tf.executing_eagerly():
            steps = 1
            logging.info("=========================predict_data_source=============")
            logging.info(predict_data_source)
        return self.model.predict(x=predict_data_source, steps=steps)

    def evaluate(self, predict_data_source, steps=None):
        return self.model.evaluate(x=predict_data_source, steps=steps)
