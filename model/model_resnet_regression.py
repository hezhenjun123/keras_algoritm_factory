from model.model_base import ModelBase
from model.architecture_resnet import ResNet


class ModelResnetRegression(ModelBase):

    def __init__(self, config):
        super().__init__(config)
        self.pretrained = config["PRETRAINED"]
        self.dropout = config["DROPOUT"]
        self.channel_list = config["CHANNEL_LIST"]
        self.activation = config["ACTIVATION"]
        self.model = self.get_or_load_model()

    def create_model(self):
        model = ResNet(input_shape=(*self.resize, 3),
                       pretrained=self.pretrained,
                       channel_list=self.channel_list,
                       num_classes=1,
                       activation=self.activation,
                       dropout_prob=self.dropout,
                       name='resnet',
                       input_name='images',
                       output_name='logits',
                       output_activation='sigmoid')
        model.summary()
        return model
