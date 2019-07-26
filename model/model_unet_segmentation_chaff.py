import os
import logging
import tensorflow as tf
from math import ceil
from model.model_base import ModelBase
from loss.dice import Dice
from metric.mean_iou import MeanIOU
from model.architecture_unet import UNet
from utilities.image_summary import ImageSummary
from utilities.color import generate_colormap
from utilities.crop_patches import CropAndExpand
from utilities.cos_anneal import CosineAnnealingScheduler
from utilities.smart_checkpoint import SmartCheckpoint
from utilities.helper import get_plot_data

logging.getLogger().setLevel(logging.INFO)


class ModelUnetSegmentationChaff(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.num_plots = config["NUM_PLOTS"]

    def __model_compile(self):
        model = UNet(
            input_shape=(*self.resize, 3),
            channel_list=self.channel_list,
            num_classes=self.num_classes,
            return_logits=False,
            activation=self.activation,
            dropout_prob=self.dropout,
            dropout_type="spatial",
            name="unet",
            input_name="images",
            output_name="seg_map",
            conv_block="default",
            normalization="layernorm",
        )
        model.summary()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=Dice(),
            metrics=[MeanIOU(num_classes=self.num_classes)],
        )
        self.model = model

    def __callback_compile(self):
        plot_df = self.valid_data_dataframe.sample(n=self.num_plots,
                                                   random_state=69)
        data_to_plot = get_plot_data(plot_df, self.config)
        summaries_dir = os.path.join(self.save_dir, "summaries")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=summaries_dir)
        checkpoints_dir = os.path.join(self.save_dir, "checkpoints/")
        if self.num_classes == 2:
            cmap = "viridis"
        else:
            cmap = generate_colormap(self.num_classes, "ADE20K")
        self.callbacks = [
            tensorboard_callback,
            ImageSummary(
                tensorboard_callback,
                data_to_plot,
                update_freq=10,
                transforms=self.valid_transforms,
                cmap=cmap,
            ),
            CosineAnnealingScheduler(20, self.learning_rate),
            SmartCheckpoint(destination_path=checkpoints_dir,
                            file_format='epoch_{epoch:04d}/cp.ckpt',
                            save_weights_only=False,
                            verbose=1,
                            monitor='val_mean_iou',
                            mode='max',
                            save_best_only=True),
        ]

    def __set_model_parameters(self, **kwargs):
        if "num_train_data" not in kwargs:
            self.num_train_data = 1000
        else:
            self.num_train_data = kwargs["num_train_data"]

        if "num_valid_data" not in kwargs:
            self.num_valid_data = 100
        else:
            self.num_valid_data = kwargs["num_valid_data"]

        if "valid_transforms" not in kwargs:
            raise Exception("Need valid_transforms for plot")
        else:
            self.valid_transforms = kwargs["valid_transforms"]

        if "valid_data_dataframe" not in kwargs:
            raise Exception("Need valid_data_dataframe for plot")
        else:
            self.valid_data_dataframe = kwargs["valid_data_dataframe"]

    def model_fit(self, train_dataset, valid_dataset, **kwargs):
        self.__set_model_parameters(**kwargs)
        self.__model_compile()
        self.__callback_compile()
        if self.steps_per_epoch == -1:
            steps_per_epoch = ceil(self.num_train_data / self.batch_size)
        valid_steps = ceil(self.num_valid_data / self.batch_size)

        logging.info(
            'STARTING TRAINING, {} train steps, {} valid steps'.format(
                steps_per_epoch, valid_steps))
        self.model.fit(train_dataset,
                       epochs=self.epochs,
                       steps_per_epoch=steps_per_epoch,
                       validation_data=valid_dataset,
                       validation_steps=valid_steps,
                       verbose=2,
                       callbacks=self.callbacks)
