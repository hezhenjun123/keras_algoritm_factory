import logging
import os
from experiment.experiment_base import ExperimentBase
import tensorflow as tf
from metric.mean_iou import MeanIOU
from tensorflow.python.keras.utils import losses_utils
from loss.dice import Dice
from utilities.image_summary import ImageSummary
from utilities.color import generate_colormap
from utilities.cos_anneal import CosineAnnealingScheduler
from utilities.smart_checkpoint import SmartCheckpoint
from utilities.helper import get_plot_data


logging.getLogger().setLevel(logging.INFO)


class ExperimentSegmentationTF2Unet(ExperimentBase):

    def __init__(self, config):
        super().__init__(config)
        self.learning_rate = config["LEARNING_RATE"]
        self.num_classes = config["NUM_CLASSES"]

    def run_experiment(self):
        train_transform, valid_transform = self.generate_transform()
        data_train_split, data_valid_split = self.read_train_csv()
        train_dataset, valid_dataset = self.generate_dataset(data_train_split, data_valid_split,
                                                             train_transform, valid_transform)
        model = self.generate_model()


        callbacks = self.__compile_callbacks(data_valid_split, valid_transform)

        kwarg_para = {
            "num_train_data": len(data_train_split),
            "num_valid_data": len(data_valid_split),
        }

        model.fit_model(train_dataset, valid_dataset, callbacks, **kwarg_para)

    def model_compile_para(self):
        compile_para = dict()
        compile_para["optimizer"] = tf.keras.optimizers.Adam(lr=self.learning_rate)
        compile_para["loss"] = Dice()
        compile_para["metrics"] = [
            'accuracy',
            MeanIOU(num_classes=self.num_classes)
        ]
        return compile_para

    def __compile_callbacks(self, valid_data_dataframe, valid_transforms):
        # plot_df = valid_data_dataframe.sample(n=self.num_plots, random_state=69)
        # data_to_plot = get_plot_data(plot_df, self.config)
        summaries_dir = os.path.join(self.save_dir, "summaries")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=summaries_dir)
        checkpoints_dir = os.path.join(self.save_dir, "checkpoints/")
        # if self.num_classes == 2:
        #     cmap = "viridis"
        # else:
        #     cmap = generate_colormap(self.num_classes, "ADE20K")
        callbacks = [
            tensorboard_callback,
            # ImageSummary(
            #     tensorboard_callback,
            #     data_to_plot,
            #     update_freq=10,
            #     transforms=valid_transforms,
            #     cmap=cmap,
            # ),
            CosineAnnealingScheduler(20, self.learning_rate),
            SmartCheckpoint(destination_path=checkpoints_dir,
                            file_format='epoch_{epoch:04d}/cp.hdf5',
                            save_weights_only=False,
                            verbose=1,
                            monitor='val_mean_iou',
                            mode='max',
                            save_best_only=False),
        ]
        return callbacks
