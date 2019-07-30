import os
import logging
from experiment.experiment_base import ExperimentBase
from utilities.smart_checkpoint import SmartCheckpoint
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

logging.getLogger().setLevel(logging.INFO)


class ExperimentClassification(ExperimentBase):

    def __init__(self, config):
        super().__init__(config)
        self.learning_rate = config["LEARNING_RATE"]

    def run_experiment(self):
        train_transform, valid_transform = self.generate_transform()
        data_train_split, data_valid_split = self.read_train_csv()
        train_dataset, valid_dataset = self.generate_dataset(
            data_train_split, data_valid_split, train_transform,
            valid_transform)
        model = self.generate_model()

        callbacks = self.__compile_callback()
        kwarg_para = {
            "num_train_data": len(data_train_split),
            "num_valid_data": len(data_valid_split)
        }
        model.fit_model(train_dataset, valid_dataset, callbacks, **kwarg_para)

    def __compile_callback(self):
        summaries_dir = os.path.join(self.save_dir, "summaries")
        checkpoints_dir = os.path.join(self.save_dir, "checkpoints")
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=summaries_dir),
            SmartCheckpoint(checkpoints_dir,
                            file_format='epoch_{epoch:04d}/cp.ckpt',
                            period=10)
        ]
        return callbacks

    def model_compile_para(self):
        compile_para = dict()
        compile_para["optimizer"] = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)
        compile_para["loss"] = 'categorical_crossentropy'
        compile_para["metrics"] = ['accuracy']
        return compile_para
