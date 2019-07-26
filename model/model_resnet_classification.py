import os
import tensorflow as tf
from math import ceil
from model.model_base import ModelBase
from model.architecture_resnet import ResNet
from utilities.smart_checkpoint import SmartCheckpoint


class ModelResnetClassification(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.pretrained = config["PRETRAINED"]

    def __model_compile(self):
        model = ResNet(input_shape=(*self.resize, 3),
                       pretrained=self.pretrained,
                       channel_list=self.channel_list,
                       num_classes=self.num_classes,
                       activation=self.activation,
                       dropout_prob=self.dropout,
                       name='unet',
                       input_name='images',
                       output_name='logits')
        model.summary()
        model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def __callback_compile(self):
        summaries_dir = os.path.join(self.save_dir, "summaries")
        checkpoints_dir = os.path.join(self.save_dir, "checkpoints")
        self.callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=summaries_dir),
            SmartCheckpoint(checkpoints_dir,
                            file_format='epoch_{epoch:04d}/cp.ckpt',
                            period=10)
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

    def model_fit(self, train_dataset, valid_dataset, **kwargs):
        self.__set_model_parameters(**kwargs)
        self.__model_compile()
        self.__callback_compile()
        if self.steps_per_epoch == -1:
            steps_per_epoch = ceil(self.num_train_data / self.batch_size)
        else:
            steps_per_epoch = self.steps_per_epoch
        valid_steps = ceil(self.num_valid_data / self.batch_size)
        self.model.fit(train_dataset,
                       epochs=self.epochs,
                       steps_per_epoch=steps_per_epoch,
                       validation_data=valid_dataset,
                       validation_steps=valid_steps,
                       verbose=2,
                       callbacks=self.callbacks)
