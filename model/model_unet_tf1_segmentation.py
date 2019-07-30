import logging
from math import ceil
from model.model_base import ModelBase
from model.architecture_unet import UNet


logging.getLogger().setLevel(logging.INFO)


class ModelUnetTF1Segmentation(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.num_plots = config["NUM_PLOTS"]
        self.dropout = config["DROPOUT"]
        self.channel_list = config["CHANNEL_LIST"]
        self.activation = config["ACTIVATION"]
        self.model = self.create_model()


    def create_model(self):
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
        return model


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

    def fit_model(self, train_dataset, valid_dataset, callbacks, **kwargs):
        self.__set_model_parameters(**kwargs)
        if self.steps_per_epoch == -1:
            steps_per_epoch = ceil(self.num_train_data / self.batch_size)
        else:
            steps_per_epoch = self.steps_per_epoch
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
                       callbacks=callbacks)
