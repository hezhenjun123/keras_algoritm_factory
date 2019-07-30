import logging
from experiment.experiment_base import ExperimentBase
from model.model_factory import ModelFactory
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

logging.getLogger().setLevel(logging.INFO)


class DummyLoss(tf.keras.losses.Loss):
  def __init__(self,
               name=None,
               **kwargs):
    super(tf.keras.losses.Loss, self).__init__()
    self.name=name

    self.reduction=losses_utils.ReductionV2.AUTO

  def call(self, y_true, y_pred):
    return tf.zeros(tf.shape(y_true))

class ExperimentSegmentationUnetBackbone(ExperimentBase):
    def __init__(self, config):
        super().__init__(config)
        self.learning_rate = config["LEARNING_RATE"]
        self.log_directory = self.config["MODEL"]["LOG_DIRECTORY"]
        self.num_classes = config["NUM_CLASSES"]
        self.model_name = self.config["EXPERIMENT"]["MODEL_NAME"]


    def run_experiment(self):
        train_transform, valid_transform = self.generate_transform()
        data_train_split, data_valid_split = self.read_train_csv()
        train_dataset, valid_dataset = self.generate_dataset(data_train_split, data_valid_split, train_transform,
                                                             valid_transform)
        model_factory = ModelFactory(self.config)
        model = model_factory.create_model(self.model_name)

        compile_para = self.__model_compile_para()
        model.compile_model(**compile_para)


        callbacks = self.__compile_callbacks()
        model.fit_model(train_dataset, valid_dataset, callbacks)

    def __model_compile_para(self):
        compile_para = dict()
        compile_para["optimizer"] = tf.keras.optimizers.Adam(lr=self.learning_rate)
        compile_para["loss"] = {"probabilities": tf.keras.losses.BinaryCrossentropy(), "predictions": DummyLoss()}
        compile_para["metrics"] = {"probabilities": 'accuracy',
                                   "predictions": tf.keras.metrics.MeanIoU(num_classes=self.num_classes)}
        return compile_para


    def __compile_callbacks(self):
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.log_directory, update_freq='batch')
        ]
        return callbacks