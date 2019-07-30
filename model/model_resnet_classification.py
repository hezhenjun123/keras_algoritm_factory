from math import ceil
from model.model_base import ModelBase
from model.architecture_resnet import ResNet


class ModelResnetClassification(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.pretrained = config["PRETRAINED"]
        self.dropout = config["DROPOUT"]
        self.channel_list = config["CHANNEL_LIST"]
        self.activation = config["ACTIVATION"]
        self.model = self.create_model()

    def create_model(self):
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

    def fit_model(self, train_dataset, valid_dataset, callbacks, **kwargs):
        self.__set_model_parameters(**kwargs)
        if self.train_steps_per_epoch == -1:
            train_steps_per_epoch = ceil(self.num_train_data / self.batch_size)
        else:
            train_steps_per_epoch = self.train_steps_per_epoch
        if self.valid_steps_per_epoch == -1:
            valid_steps_per_epoch = ceil(self.num_valid_data / self.batch_size)
        else:
            valid_steps_per_epoch = self.valid_steps_per_epoch
        self.model.fit(train_dataset,
                       epochs=self.epochs,
                       steps_per_epoch=train_steps_per_epoch,
                       validation_data=valid_dataset,
                       validation_steps=valid_steps_per_epoch,
                       verbose=2,
                       callbacks=callbacks)
