from model.model_base import ModelBase
from model.architecture_resnet import ResNet


class ModelResnetClassification(ModelBase):

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
                       num_classes=self.num_classes,
                       activation=self.activation,
                       dropout_prob=self.dropout,
                       name='unet',
                       input_name='images',
                       output_name='logits')
        model.summary()
        return model
