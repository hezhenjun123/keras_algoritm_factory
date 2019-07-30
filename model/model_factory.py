from model.model_resnet_classification import ModelResnetClassification
from model.model_tf1unet_segmentation import ModelTF1UnetSegmentation
from model.model_tf2unet_segmentation import ModelTF2UnetSegmentation


class ModelFactory:

    model_registry = {
        "ModelResnetClassification": ModelResnetClassification,
        "ModelTF1UnetSegmentation": ModelTF1UnetSegmentation,
        "ModelTF2UnetSegmentation": ModelTF2UnetSegmentation
    }

    def __init__(self, config):
        self.config = config

    def create_model(self, name):
        if name not in self.model_registry:
            raise Exception(f"model type is not supported: {name}")
        return self.model_registry[name](self.config)
