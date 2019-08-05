from math import ceil
from model.model_base import ModelBase
from model.architecture_resnet import ResNet
import tensorflow as tf
import tensorflow.keras.layers as l
class ModelResnetClassificationYield(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.pretrained = config["PRETRAINED"]
        self.dropout = config["DROPOUT"]
        self.channel_list = config["CHANNEL_LIST"]
        self.dense_list = config["DENSE_LIST"]

        self.activation = config["ACTIVATION"]
        self.model = self.create_model()

    def create_model(self):
        inp = l.Input((*self.resize, 6))
        x = inp
        for channel in self.channel_list:
            for i in range(2):
                x = l.Conv2D(filters=channel,
                                        kernel_size=(3, 3),
                                        activation=self.activation,
                                        padding="same",
                                        use_bias=False)(x)
                x = l.BatchNormalization()(x)
                if self.dropout!=0:
                    x = l.Dropout(self.dropout)(x)
            x = l.MaxPool2D(pool_size=2)(x)
            
        x= l.GlobalMaxPooling2D()(x)
        for dense in self.dense_list:
            x = l.Dense(dense, activation=self.activation)(x)
            if self.dropout != 0.0:
                x = l.Dropout(self.dropout)(x)

        preds = l.Dense(self.num_classes, activation='softmax')(x)
        model =  tf.keras.models.Model([inp], [preds], name='conv_net')
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
        self.model.fit(train_dataset,
                       epochs=self.epochs,
                       steps_per_epoch=self.train_steps_per_epoch,
                       validation_data=valid_dataset,
                       validation_steps=self.valid_steps_per_epoch,
                       verbose=2,
                       callbacks=callbacks)
