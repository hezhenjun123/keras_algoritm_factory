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

    def fit_model(self, train_dataset, valid_dataset, callbacks, **kwargs):
        self.set_runtime_parameters(**kwargs)
        self.model.fit(train_dataset,
                       epochs=self.epochs,
                       steps_per_epoch=self.train_steps_per_epoch,
                       validation_data=valid_dataset,
                       validation_steps=self.valid_steps_per_epoch,
                       verbose=2,
                       callbacks=callbacks)
