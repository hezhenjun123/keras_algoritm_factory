from model.model_resnet_classification import ModelResnetClassification
from model.model_unet_segmentation_uv import ModelUnetSegmentationUV


class ModelFactory:

    model_registry = {
        "ModelResnetClassification": ModelResnetClassification,
        "ModelUnetSegmentationUV": ModelUnetSegmentationUV
    }

    def __init__(self, config):
        self.config = config

    def create_model(self, name):
        if name not in self.model_registry:
            raise Exception(f"model type is not supported: {name}")
        return self.model_registry[name](self.config)
