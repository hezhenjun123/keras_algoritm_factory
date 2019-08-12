import logging
from model.model_base import ModelBase
from model.architecture_unet import UNet

logging.getLogger().setLevel(logging.INFO)


class ModelTF1UnetSegmentation(ModelBase):

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

    def set_runtime_parameters(self, **kwargs):
        super().set_runtime_parameters(**kwargs)

        if "valid_transforms" not in kwargs:
            raise Exception("Need valid_transforms for plot")
        else:
            self.valid_transforms = kwargs["valid_transforms"]

        if "valid_data_dataframe" not in kwargs:
            raise Exception("Need valid_data_dataframe for plot")
        else:
            self.valid_data_dataframe = kwargs["valid_data_dataframe"]
