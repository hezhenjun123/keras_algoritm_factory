import logging
from experiment.experiment_base import ExperimentBase
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

logging.getLogger().setLevel(logging.INFO)


class DummyLoss(tf.keras.losses.Loss):

    def __init__(self, name=None, **kwargs):
        super(tf.keras.losses.Loss, self).__init__()
        self.name = name

        self.reduction = losses_utils.ReductionV2.AUTO

    def call(self, y_true, y_pred):
        return tf.zeros(tf.shape(y_true))


class ExperimentSegmentationTF2Unet(ExperimentBase):

    def __init__(self, config):
        super().__init__(config)
        self.learning_rate = config["LEARNING_RATE"]
        self.log_directory = self.config["MODEL"]["LOG_DIRECTORY"]
        self.num_classes = config["NUM_CLASSES"]

    def run_experiment(self):
        train_transform, valid_transform = self.generate_transform()
        data_train_split, data_valid_split = self.read_train_csv()
        train_dataset, valid_dataset = self.generate_dataset(data_train_split, data_valid_split,
                                                             train_transform, valid_transform)
        model = self.generate_model()

        callbacks = self.__compile_callbacks()

        kwarg_para = {
            "num_train_data": len(data_train_split),
            "num_valid_data": len(data_valid_split),
        }

        model.fit_model(train_dataset, valid_dataset, callbacks, **kwarg_para)

    def model_compile_para(self):
        compile_para = dict()
        compile_para["optimizer"] = tf.keras.optimizers.Adam(lr=self.learning_rate)
        compile_para["loss"] = {
            "probabilities": tf.keras.losses.BinaryCrossentropy(),
            "predictions": DummyLoss()
        }
        compile_para["metrics"] = {
            "probabilities": 'accuracy',
            "predictions": tf.keras.metrics.MeanIoU(num_classes=self.num_classes)
        }
        return compile_para

    def __compile_callbacks(self):
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.log_directory, update_freq='batch')
        ]
        return callbacks
